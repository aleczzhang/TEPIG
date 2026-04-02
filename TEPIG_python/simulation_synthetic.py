"""
simulation_synthetic.py
-----------------------
Full simulation study with synthetic data (CLUSSO Section 4.1 style).

Usage:
    python simulation_synthetic.py --n 300

Data generation:
  - Slide-specific distributions:
      Slide 1: cluster 1 ~ N(2,1),  cluster 2 ~ N(5,3)
      Slide 2: cluster 1 ~ N(4,2),  cluster 2 ~ N(8,1)
  - For each subject i, for each slide s:
      Draw X*_{i,s} in R^(G x q) from slide-specific distributions
      Draw w_{i,s} ~ Uniform{0.2, ..., 0.8}
      X**_{i,s} = diag(w, 1-w) @ X*_{i,s}
  - Stack S=2 slides: X_i in R^(G=2, q, S=2)

True model (full tensor, no low-rank):
    y_i = intercept + <B_true, X_i> + eps_i,   eps_i ~ N(0, 1)
    B_true[:, j, :] = B_TRUE_BLOCK  for j in nonzero_features
    B_true[:, j, :] = 0             otherwise

Estimators:
    tepig_grad : proximal gradient + group lasso (correct model)
    tepig      : alternating rank-1 structured lasso (misspecified)
    naive      : average over slides, run CLUSSO (misspecified)
    oracle     : OLS on true nonzero features only (benchmark)

Lambda grids (fixed across all simulation settings):
    tepig_grad : {0.25, 0.50, ..., 2.50}  (CLUSSO grid divided by 2 for S=2 slides)
    tepig/naive: logspace(-4, 0, 15)
"""

import os
import sys
import argparse
import pickle
import warnings
import numpy as np
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CLUSSO_python'))
from Mainfunction_albet import Mainfunction_albet, _glmnet_lasso
from SLasso_MSE import lambda_CV_mse

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

# ── Parse args ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, required=True, help='Number of subjects')
args = parser.parse_args()
N = args.n

# ── Config (fixed across all settings) ────────────────────────────────────────
Q       = 10     # number of features
G       = 2      # number of clusters
S       = 2      # number of slides
N_NZ    = 2      # number of nonzero features (sparsity = 1 - N_NZ/Q = 0.8)
SIGMA   = 1.0    # outcome noise std
B_SIMS  = 200    # simulation repetitions
N_JOBS  = -1     # joblib: -1 uses all available cores
RANDOM_SEED = 42

MAX_ITER = 50
TOL      = 1e-2
M_INIT   = 3

# Fixed lambda grids (same across all n, q, sparsity settings)
PROXGRAD_LAM_GRID = np.arange(0.25, 2.75, 0.25)   # CLUSSO {0.5,...,5.0} / 2
LAM_GRID          = np.logspace(-4, 0, 15)          # for tepig / naive

# True B block: (G=2, S=2), rank-2 (det != 0)
B_TRUE_BLOCK = np.array([[1.0, 2.0],
                          [3.0, 4.0]])

# ── Build B_true ───────────────────────────────────────────────────────────────
nonzero_idx = np.arange(N_NZ)
B_true      = np.zeros((G, Q, S))
for j in nonzero_idx:
    B_true[:, j, :] = B_TRUE_BLOCK

beta_star = np.array([float(np.linalg.norm(B_true[:, j, :])) for j in range(Q)])

print(f"simulation_synthetic.py | n={N}, q={Q}, G={G}, S={S}, n_nonzero={N_NZ}")
print(f"  B_TRUE_BLOCK det={float(np.linalg.det(B_TRUE_BLOCK)):.3f}  (rank-2)")
print(f"  PROXGRAD lambda grid: {PROXGRAD_LAM_GRID[0]:.2f} to {PROXGRAD_LAM_GRID[-1]:.2f}")
print(f"  Running {B_SIMS} reps with n_jobs={N_JOBS}...")


# ── Synthetic data generation ──────────────────────────────────────────────────

def generate_synthetic(rng, n, q):
    """
    Generate (G, q, S, n) tensor of synthetic subject data.
    Slide 1: cluster1~N(2,1), cluster2~N(5,3)
    Slide 2: cluster1~N(4,2), cluster2~N(8,1)
    """
    weight_choices = np.arange(0.2, 0.9, 0.1)
    slide_params = [
        (2.0, 1.0,          5.0, np.sqrt(3.0)),
        (4.0, np.sqrt(2.0), 8.0, 1.0         ),
    ]

    tensors = []
    for _ in range(n):
        slides = []
        for s in range(S):
            mu1, sig1, mu2, sig2 = slide_params[s]
            X_star       = np.zeros((G, q))
            X_star[0, :] = rng.normal(mu1, sig1, q)
            X_star[1, :] = rng.normal(mu2, sig2, q)
            w            = rng.choice(weight_choices)
            weights      = np.array([w, 1.0 - w])
            slides.append(weights[:, np.newaxis] * X_star)
        tensors.append(np.stack(slides, axis=2))

    return np.stack(tensors, axis=3)   # (G, q, S, n)


# ── Helpers ────────────────────────────────────────────────────────────────────

def compute_metrics(beta_hat, y, y_pred):
    nz     = beta_star != 0
    z      = ~nz
    hat_nz = np.abs(beta_hat) > 1e-6

    tpr = float(hat_nz[nz].mean()) if nz.sum() > 0 else float('nan')
    fpr = float(hat_nz[z].mean())  if z.sum()  > 0 else float('nan')

    bsn = beta_star / np.sum(np.abs(beta_star))
    bhn = (beta_hat / np.sum(np.abs(beta_hat))
           if np.sum(np.abs(beta_hat)) > 0 else beta_hat.copy())
    l1  = float(np.sum(np.abs(bhn - bsn)))
    mse = float(np.mean((y - y_pred) ** 2))
    return tpr, fpr, l1, mse


def _normalize_vec(v):
    if np.max(np.abs(v)) > 0:
        sig = np.sign(v[np.argmax(np.abs(v))])
        return sig * v / np.sum(np.abs(v))
    return v


def _make_folds(n, rng):
    idx  = rng.permutation(n).tolist()
    size = n // 5
    folds = [sorted(idx[k * size:(k + 1) * size]) for k in range(4)]
    folds.append(sorted(idx[4 * size:]))
    return folds


# ── proxgrad_fit ───────────────────────────────────────────────────────────────

def proxgrad_fit(X, y, lam, max_iter=2000, tol=1e-6):
    G, q, S, n_tr = X.shape
    d = G * q * S

    gnorms = np.array([
        float(np.sqrt(np.mean(np.sum(X[:, j, :, :] ** 2, axis=(0, 1)))))
        for j in range(q)
    ])
    gnorms = np.where(gnorms > 1e-10, gnorms, 1.0)
    Xs     = np.clip(X / gnorms[np.newaxis, :, np.newaxis, np.newaxis], -1e3, 1e3)
    X_flat = Xs.reshape(d, n_tr).T

    sigma_max = np.linalg.svd(X_flat, compute_uv=False)[0]
    L         = float(sigma_max ** 2) / n_tr
    eta       = 1.0 / L
    threshold = eta * lam

    x         = np.zeros(d)
    z         = np.zeros(d)
    t         = 1.0
    intercept = float(np.mean(y))

    for _ in range(max_iter):
        x_old = x.copy()
        if not np.isfinite(z).all():
            break
        pred_z    = X_flat @ z
        intercept = float(np.mean(y - pred_z))
        residuals = y - intercept - pred_z
        grad      = -(1.0 / n_tr) * (X_flat.T @ residuals)
        if not np.isfinite(grad).all():
            break

        v = (z - eta * grad).reshape(G, q, S)
        for j in range(q):
            block = v[:, j, :]
            norm  = float(np.linalg.norm(block))
            if norm > threshold:
                v[:, j, :] = (1.0 - threshold / norm) * block
            else:
                v[:, j, :] = 0.0
        x = v.reshape(-1)

        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z     = x + ((t - 1.0) / t_new) * (x - x_old)
        t     = t_new

        if np.max(np.abs(x - x_old)) < tol:
            break

    B_std       = x.reshape(G, q, S)
    B           = B_std / gnorms[np.newaxis, :, np.newaxis]
    X_flat_orig = X.reshape(d, n_tr).T
    intercept   = float(np.mean(y - X_flat_orig @ B.reshape(-1)))
    return intercept, B


# ── tepig_fit ──────────────────────────────────────────────────────────────────

def tepig_fit(X, y, lam, alpha_init, beta_init, gamma_init):
    G, q, S, n = X.shape
    alpha    = _normalize_vec(np.asarray(alpha_init, dtype=float).copy())
    beta     = np.asarray(beta_init,  dtype=float).copy()
    gamma    = np.asarray(gamma_init, dtype=float).copy()
    nr_alpha = max(0.0, float(np.sum(np.abs(alpha))))

    for _ in range(MAX_ITER):
        alpha0, beta0, gamma0 = alpha.copy(), beta.copy(), gamma.copy()

        Z       = np.einsum('gjsn,g,s->jn', X, alpha, gamma)
        eff_lam = float(lam) * np.sum(np.abs(alpha)) * np.sum(np.abs(gamma))
        beta    = _glmnet_lasso(Z.T, y, eff_lam)
        if np.max(np.abs(beta)) > 0:
            beta = _normalize_vec(beta)

        if np.max(np.abs(beta)) > 0:
            W     = np.einsum('gjsn,j,s->gn', X, beta, gamma)
            W_int = np.column_stack([np.ones(n), W.T])
            coeffs, _, _, _ = np.linalg.lstsq(W_int, y, rcond=None)
            alpha = coeffs[1:]

        nr_alpha = max(0.0, float(np.sum(np.abs(alpha))))
        if np.max(np.abs(alpha)) > 0:
            alpha = _normalize_vec(alpha)

        if np.max(np.abs(alpha)) > 0 and np.max(np.abs(beta)) > 0:
            V     = np.einsum('gjsn,g,j->sn', X, alpha, beta)
            V_int = np.column_stack([np.ones(n), V.T])
            coeffs, _, _, _ = np.linalg.lstsq(V_int, y, rcond=None)
            gamma = coeffs[1:]

        dif = (np.linalg.norm(alpha - alpha0) +
               np.linalg.norm(beta  - beta0)  +
               np.linalg.norm(gamma - gamma0))
        if dif < TOL:
            break

    alpha[np.abs(alpha) < 0.001] = 0.0
    beta[np.abs(beta)   < 0.001] = 0.0
    gamma[np.abs(gamma) < 0.001] = 0.0
    return nr_alpha * alpha, beta, gamma


# ── Single simulation rep ──────────────────────────────────────────────────────

def run_one_sim(seed):
    rng = np.random.default_rng(seed)
    n   = N
    q   = Q

    X      = generate_synthetic(rng, n, q)        # (G, q, S, n)
    y_true = np.einsum('gjs,gjsn->n', B_true, X)
    y      = y_true + rng.normal(0, SIGMA, n)

    results  = {}
    all_idx  = list(range(n))
    folds    = _make_folds(n, rng)

    # ── tepig_grad ────────────────────────────────────────────────────────────
    d_full   = G * q * S
    cv_mse   = []
    for lam in PROXGRAD_LAM_GRID:
        fold_mse = []
        for k in range(5):
            te = folds[k]; tr = [i for i in all_idx if i not in te]
            ic, B  = proxgrad_fit(X[:, :, :, tr], y[tr], lam)
            ypred  = ic + X[:, :, :, te].reshape(d_full, len(te)).T @ B.reshape(-1)
            fold_mse.append(float(np.mean((y[te] - ypred) ** 2)))
        cv_mse.append(float(np.mean(fold_mse)))

    best_lam      = PROXGRAD_LAM_GRID[int(np.argmin(cv_mse))]
    ic_f, B_f     = proxgrad_fit(X, y, best_lam)
    beta_hat_f    = np.array([float(np.linalg.norm(B_f[:, j, :])) for j in range(q)])
    y_pred_f      = ic_f + X.reshape(d_full, n).T @ B_f.reshape(-1)
    results['tepig_grad'] = compute_metrics(beta_hat_f, y, y_pred_f)

    # ── tepig ─────────────────────────────────────────────────────────────────
    cv_mse_t = []
    for lam in LAM_GRID:
        fold_mse = []
        for k in range(5):
            te = folds[k]; tr = [i for i in all_idx if i not in te]
            a0 = np.ones(G) / G; b0 = np.ones(q) / q; g0 = np.ones(S) / S
            a, b, g = tepig_fit(X[:, :, :, tr], y[tr], lam, a0, b0, g0)
            yp = np.einsum('gjsn,g,j,s->n', X[:, :, :, te], a, b, g)
            fold_mse.append(float(np.mean((y[te] - yp) ** 2)))
        cv_mse_t.append(float(np.mean(fold_mse)))

    best_lam_t = LAM_GRID[int(np.argmin(cv_mse_t))]
    best_mse_t = np.inf
    best_res_t = (np.ones(G) / G, np.zeros(q), np.ones(S) / S)
    for _ in range(M_INIT):
        a0 = rng.dirichlet(np.ones(G))
        b0 = rng.uniform(-1, 1, q)
        g0 = rng.dirichlet(np.ones(S))
        a, b, g = tepig_fit(X, y, best_lam_t, a0, b0, g0)
        yp  = np.einsum('gjsn,g,j,s->n', X, a, b, g)
        mse = float(np.mean((y - yp) ** 2))
        if mse < best_mse_t:
            best_mse_t = mse; best_res_t = (a, b, g)

    a_hat, b_hat, g_hat = best_res_t
    y_pred_t = np.einsum('gjsn,g,j,s->n', X, a_hat, b_hat, g_hat)
    results['tepig'] = compute_metrics(b_hat, y, y_pred_t)

    # ── naive ─────────────────────────────────────────────────────────────────
    X_naive      = X.mean(axis=2)   # (G, q, n)
    cv_mse_naive = [
        lambda_CV_mse(X_naive, y, np.ones(G) / G, np.ones(q) / q, lam)
        for lam in LAM_GRID
    ]
    best_lam_naive = LAM_GRID[int(np.argmin(cv_mse_naive))]

    best_mse_n = np.inf
    best_b_n   = np.zeros(q)
    best_a_n   = np.ones(G) / G
    for _ in range(M_INIT):
        a0  = rng.dirichlet(np.ones(G))
        b0  = rng.uniform(-1, 1, q)
        res = Mainfunction_albet(X_naive, y, a0, b0, best_lam_naive)
        a_n, b_n = res['alpha'], res['bet']
        yp_n = np.array([a_n @ X_naive[:, :, i] @ b_n for i in range(n)])
        mse_n = float(np.mean((y - yp_n) ** 2))
        if mse_n < best_mse_n:
            best_mse_n = mse_n; best_b_n = b_n; best_a_n = a_n

    y_pred_naive = np.array([best_a_n @ X_naive[:, :, i] @ best_b_n for i in range(n)])
    results['naive'] = compute_metrics(best_b_n, y, y_pred_naive)

    # ── oracle ────────────────────────────────────────────────────────────────
    n_nz     = len(nonzero_idx)
    X_oracle = X[:, nonzero_idx, :, :].reshape(G * n_nz * S, n).T
    X_int    = np.column_stack([np.ones(n), X_oracle])
    coeffs, _, _, _ = np.linalg.lstsq(X_int, y, rcond=None)
    ic_oracle     = coeffs[0]
    B_oracle_flat = coeffs[1:]
    B_oracle      = B_oracle_flat.reshape(G, n_nz, S)
    beta_hat_oracle = np.zeros(q)
    for k, j in enumerate(nonzero_idx):
        beta_hat_oracle[j] = float(np.linalg.norm(B_oracle[:, k, :]))
    y_pred_oracle = ic_oracle + X_oracle @ B_oracle_flat
    results['oracle'] = compute_metrics(beta_hat_oracle, y, y_pred_oracle)

    return results


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    seed_seq  = np.random.SeedSequence(RANDOM_SEED)
    seed_ints = [int(s.generate_state(1)[0]) for s in seed_seq.spawn(B_SIMS)]

    all_results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(run_one_sim)(s) for s in seed_ints
    )

    estimators  = ['tepig_grad', 'tepig', 'naive', 'oracle']
    metric_keys = ['tpr', 'fpr', 'l1', 'mse']

    summary = {est: {m: [] for m in metric_keys} for est in estimators}
    for rep in all_results:
        for est in estimators:
            tpr, fpr, l1, mse = rep[est]
            summary[est]['tpr'].append(tpr)
            summary[est]['fpr'].append(fpr)
            summary[est]['l1'].append(l1)
            summary[est]['mse'].append(mse)

    # Save pickle
    pkl_path = os.path.join(OUT_DIR, f'simulation_synthetic_n{N}_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'summary': summary,
            'config': {
                'B': B_SIMS, 'N': N, 'Q': Q, 'G': G, 'S': S, 'N_NZ': N_NZ,
                'B_TRUE_BLOCK': B_TRUE_BLOCK.tolist(),
                'PROXGRAD_LAM_GRID': PROXGRAD_LAM_GRID.tolist(),
                'SIGMA': SIGMA,
            }
        }, f)

    # Save summary text
    txt_path = os.path.join(OUT_DIR, f'simulation_synthetic_n{N}_summary.txt')
    with open(txt_path, 'w') as f:
        f.write("=" * 65 + "\n")
        f.write("TEPIG SYNTHETIC SIMULATION SUMMARY\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"B={B_SIMS} reps | n={N} | q={Q} | G={G} | S={S} | n_nonzero={N_NZ}\n")
        f.write(f"Sparsity={1 - N_NZ/Q:.1f} | sigma={SIGMA}\n")
        f.write(f"B_TRUE_BLOCK: {B_TRUE_BLOCK.tolist()}\n\n")
        f.write(f"{'Estimator':<14} {'TPR':>8} {'FPR':>8} {'L1 bias':>10} {'MSE':>10}\n")
        f.write("-" * 54 + "\n")
        for est in estimators:
            tpr = float(np.nanmean(summary[est]['tpr']))
            fpr = float(np.nanmean(summary[est]['fpr']))
            l1  = float(np.nanmean(summary[est]['l1']))
            mse = float(np.nanmean(summary[est]['mse']))
            f.write(f"  {est:<12} {tpr:>8.3f} {fpr:>8.3f} {l1:>10.3f} {mse:>10.3f}\n")

    print(f"\nSaved: {pkl_path}")
    print(f"Saved: {txt_path}")

    # Print summary to stdout
    print(f"\n{'Estimator':<14} {'TPR':>8} {'FPR':>8} {'L1 bias':>10} {'MSE':>10}")
    print("-" * 54)
    for est in estimators:
        tpr = float(np.nanmean(summary[est]['tpr']))
        fpr = float(np.nanmean(summary[est]['fpr']))
        l1  = float(np.nanmean(summary[est]['l1']))
        mse = float(np.nanmean(summary[est]['mse']))
        print(f"  {est:<12} {tpr:>8.3f} {fpr:>8.3f} {l1:>10.3f} {mse:>10.3f}")

"""
simulation_bootstrap.py
-----------------------
Bootstrap-based simulation for TEPIG with full tensor model (no low-rank assumption).

Key differences from simulation_rankone.py:
  - Only uses subjects with two true WSIs (n_real=35)
  - Each rep bootstraps N_TARGET subjects from those 35 with replacement
  - Adds Gaussian noise (sigma=SIGMA_R) to each resampled tensor
  - Outcome generated as: y_i = <B_true, X_i> + eps_i  (full tensor inner product)
  - B_true is a full (G, q, S) tensor — B_true[:,j,:] = B_TRUE_BLOCK for nonzero j
  - B_TRUE_BLOCK is rank-2 (det != 0), so no low-rank assumption is imposed
  - Oracle knows the true nonzero features, fits OLS on those features only

True generating model:
    y_i = intercept + <B_true, X_i**> + eps_i,   eps_i ~ N(0, SIGMA_SQ)
    B_true[:, j, :] = B_TRUE_BLOCK  for j in BETA_FEATURES
    B_true[:, j, :] = 0             otherwise

Estimators
----------
TEPIG       : alternating 3-mode structured lasso (rank-1 assumption)
TEPIG_JOINT : collapse cluster+slide -> (G*S, q), run CLUSSO solver
TEPIG_GRAD  : proximal gradient descent (FISTA) + group lasso, full B tensor
Naive CLUSSO: average X_i across slides -> (G, q), run existing CLUSSO solver
Oracle      : knows true nonzero features, fits OLS on those features only

Outputs saved to outputs/
    simulation_bootstrap_results.pkl  : raw per-repetition metrics + config
    simulation_bootstrap_summary.txt  : human-readable table
"""

import os
import sys
import pickle
import numpy as np
from joblib import Parallel, delayed

from Mainfunction_albet import Mainfunction_albet, _glmnet_lasso
from SLasso_MSE import lambda_CV_mse

# ── Config ─────────────────────────────────────────────────────────────────────
_BASE    = os.path.join(os.path.dirname(__file__), '..', 'outputs')
OUT_REF  = os.path.join(_BASE, 'reference')
OUT_DATA = os.path.join(_BASE, 'data')
OUT_SUMM = os.path.join(_BASE, 'summaries')

BETA_FEATURES = [
    'Standard Deviation Red Nuclei',
    'Average TBM Thickness',
    'Total Object Aspect Ratio',
]

# True B tensor block for nonzero features: (G=2, S=2) rank-2 matrix.
# B_true[:, j, :] = B_TRUE_BLOCK for nonzero j, zeros otherwise.
# det = 6*9 - 4*3 = 42 != 0  =>  rank-2 (not low-rank / rank-1)
B_TRUE_BLOCK = np.array([[6.0, 4.0],
                          [3.0, 9.0]])

SIGMA_SQ          = 1.0   # outcome noise variance
SIGMA_R           = 1.0   # Gaussian noise added to bootstrapped tensors
N_TARGET          = 300   # bootstrap sample size per simulation rep
B_SIMS            = 200   # simulation repetitions
LAM_GRID          = np.logspace(-4, 0, 15)          # lambda grid for tepig/naive/joint
PROXGRAD_LAM_GRID = np.arange(0.005, 0.055, 0.005)  # lambda grid for tepig_grad CV
M_INIT            = 3     # random initialisations per final TEPIG fit
MAX_ITER          = 50    # max alternating iterations in tepig_fit
TOL               = 1e-2  # convergence tolerance
N_JOBS            = -1    # joblib: -1 uses all available cores
RANDOM_SEED       = 42

# ── Load pre-computed cluster results ──────────────────────────────────────────
print("Loading cluster results...")
with open(os.path.join(OUT_DATA, 'cluster_results.pkl'), 'rb') as f:
    cluster_results = pickle.load(f)

with open(os.path.join(OUT_REF, 'remaining_features.txt')) as f:
    features = [line.strip() for line in f]

q = len(features)
S = 2
G = 2

# Filter to subjects that truly have two WSIs (exclude those with simulated slide)
true_2slide = [s for s in sorted(cluster_results.keys())
               if len(cluster_results[s]['slides']) == 2]
n_real = len(true_2slide)
print(f"  True 2-slide subjects: {n_real}, bootstrap target: N_TARGET={N_TARGET}")
print(f"  q={q} features, G={G} clusters, S={S} slides")

# ── Build B_true: full (G, q, S) tensor ───────────────────────────────────────
B_true = np.zeros((G, q, S))
for feat in BETA_FEATURES:
    if feat in features:
        B_true[:, features.index(feat), :] = B_TRUE_BLOCK
    else:
        print(f"  WARNING: '{feat}' not found in remaining features")

nonzero_idx = np.array([j for j in range(q) if np.any(B_true[:, j, :] != 0)])

# beta_star[j] = Frobenius norm of B_true[:,j,:] — used in compute_metrics
beta_star = np.array([float(np.linalg.norm(B_true[:, j, :])) for j in range(q)])

print(f"  beta_star: {len(nonzero_idx)} nonzero features")
print(f"  Nonzero: {[features[i] for i in nonzero_idx]}")
print(f"  B_TRUE_BLOCK Frobenius norm: {float(np.linalg.norm(B_TRUE_BLOCK)):.3f}")


# ── Bootstrap sample ───────────────────────────────────────────────────────────

def bootstrap_sample(rng):
    """
    Resample N_TARGET subjects with replacement from true_2slide subjects.
    For each resampled subject, add Gaussian noise (sigma=SIGMA_R) to their
    (G, q, S) tensor — following the CLUSSO paper resampling approach.

    Returns X_boot : (G, q, S, N_TARGET)
    """
    idx = rng.integers(0, n_real, size=N_TARGET)
    tensors = []
    for i in idx:
        subj   = true_2slide[i]
        tensor = cluster_results[subj]['X_tensor'].copy()   # (G, q, S)
        tensors.append(tensor + rng.normal(0, SIGMA_R, tensor.shape))
    return np.stack(tensors, axis=3)   # (G, q, S, N_TARGET)


# ── TEPIG: alternating 3-mode structured lasso ────────────────────────────────

def _normalize_vec(v):
    """L1-normalise with positive-sign convention (largest |entry| is positive)."""
    if np.max(np.abs(v)) > 0:
        sig = np.sign(v[np.argmax(np.abs(v))])
        return sig * v / np.sum(np.abs(v))
    return v


def tepig_fit(X, y, lam, alpha_init, beta_init, gamma_init):
    """
    Fit the TEPIG rank-1 tensor model by alternating structured lasso.
    Model: y_i = sum_{g,j,s} alpha[g] * beta[j] * gamma[s] * X[g,j,s,i]
    """
    G, q, S, n = X.shape
    alpha = np.asarray(alpha_init, dtype=float).copy()
    beta  = np.asarray(beta_init,  dtype=float).copy()
    gamma = np.asarray(gamma_init, dtype=float).copy()

    alpha    = _normalize_vec(alpha)
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
               np.linalg.norm(beta  - beta0 ) +
               np.linalg.norm(gamma - gamma0))
        if dif < TOL:
            break

    alpha[np.abs(alpha) < 0.001] = 0.0
    beta[np.abs(beta)   < 0.001] = 0.0
    gamma[np.abs(gamma) < 0.001] = 0.0
    alpha = nr_alpha * alpha

    return alpha, beta, gamma


def _make_folds(n, rng):
    """Partition n subject indices into 5 folds."""
    idx  = rng.permutation(n).tolist()
    size = n // 5
    folds = [sorted(idx[k * size:(k + 1) * size]) for k in range(4)]
    folds.append(sorted(idx[4 * size:]))
    return folds


def tepig_select_and_fit(X, y, rng):
    """Select lambda by 5-fold CV, re-fit with M_INIT random starts."""
    G, q, S, n = X.shape
    all_idx = list(range(n))

    cv_mse = []
    for lam in LAM_GRID:
        folds    = _make_folds(n, rng)
        fold_mse = []
        for k in range(5):
            te = folds[k]
            tr = [i for i in all_idx if i not in te]
            a0 = np.ones(G) / G
            b0 = np.ones(q) / q
            g0 = np.ones(S) / S
            a, b, g = tepig_fit(X[:, :, :, tr], y[tr], lam, a0, b0, g0)
            y_pred  = np.einsum('gjsn,g,j,s->n', X[:, :, :, te], a, b, g)
            fold_mse.append(float(np.mean((y[te] - y_pred) ** 2)))
        cv_mse.append(float(np.mean(fold_mse)))

    best_lam = LAM_GRID[int(np.argmin(cv_mse))]

    best_mse = np.inf
    best_res = (np.ones(G) / G, np.zeros(q), np.ones(S) / S)
    for _ in range(M_INIT):
        a0 = rng.dirichlet(np.ones(G))
        b0 = rng.uniform(-1, 1, q)
        g0 = rng.dirichlet(np.ones(S))
        a, b, g = tepig_fit(X, y, best_lam, a0, b0, g0)
        y_pred  = np.einsum('gjsn,g,j,s->n', X, a, b, g)
        mse     = float(np.mean((y - y_pred) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_res = (a, b, g)

    return best_res


# ── TEPIG Full: proximal gradient descent (FISTA) with group lasso ────────────

def proxgrad_fit(X, y, lam, max_iter=2000, tol=1e-6):
    """
    Fit full B tensor model (no low-rank assumption) via FISTA + group lasso.
    Model:   y_i = intercept + <X[:,:,:,i], B> + eps_i
    Penalty: lam * sum_j ||B[:,j,:]||_F  (group lasso — zeros out whole features)
    """
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
        x_old     = x.copy()
        pred_z    = X_flat @ z
        intercept = float(np.mean(y - pred_z))
        residuals = y - intercept - pred_z
        grad      = -(1.0 / n_tr) * (X_flat.T @ residuals)

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


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(beta_hat, y, y_pred):
    """TPR, FPR, L1 bias, MSE. Uses module-level beta_star and nonzero_idx."""
    nz     = beta_star != 0
    z      = ~nz
    hat_nz = np.abs(beta_hat) > 1e-6

    tpr = float(hat_nz[nz].mean()) if nz.sum() > 0 else float('nan')
    fpr = float(hat_nz[z].mean())  if z.sum()  > 0 else float('nan')

    beta_star_norm = beta_star / np.sum(np.abs(beta_star))
    beta_hat_norm  = (beta_hat / np.sum(np.abs(beta_hat))
                      if np.sum(np.abs(beta_hat)) > 0 else beta_hat.copy())
    l1  = float(np.sum(np.abs(beta_hat_norm - beta_star_norm)))
    mse = float(np.mean((y - y_pred) ** 2))
    return tpr, fpr, l1, mse


# ── Single simulation repetition ──────────────────────────────────────────────

def run_one_sim(seed):
    """
    One simulation repetition:
    1. Bootstrap N_TARGET subjects from true 2-slide subjects.
    2. Generate y = <B_true, X_i> + eps  (full tensor, no low-rank assumption).
    3. Fit all estimators on (X, y).
    4. Return metrics.
    """
    rng = np.random.default_rng(seed)

    # ── Bootstrap data ────────────────────────────────────────────────────────
    X = bootstrap_sample(rng)   # (G, q, S, N_TARGET)
    n = N_TARGET

    # ── Generate outcome (full tensor inner product, no low-rank assumption) ──
    y_true = np.einsum('gjs,gjsn->n', B_true, X)
    y      = y_true + rng.normal(0, np.sqrt(SIGMA_SQ), n)

    results = {}

    # ── TEPIG ─────────────────────────────────────────────────────────────────
    a_hat, b_hat, g_hat = tepig_select_and_fit(X, y, rng)
    y_pred_tepig = np.einsum('gjsn,g,j,s->n', X, a_hat, b_hat, g_hat)
    results['tepig'] = compute_metrics(b_hat, y, y_pred_tepig)

    # ── Naive CLUSSO ──────────────────────────────────────────────────────────
    X_naive      = X.mean(axis=2)   # (G, q, n)
    cv_mse_naive = [
        lambda_CV_mse(X_naive, y, np.ones(G) / G, np.ones(q) / q, lam)
        for lam in LAM_GRID
    ]
    best_lam_naive = LAM_GRID[int(np.argmin(cv_mse_naive))]

    best_mse_naive = np.inf
    best_b_naive   = np.zeros(q)
    best_a_naive   = np.ones(G) / G
    for _ in range(M_INIT):
        a0  = rng.dirichlet(np.ones(G))
        b0  = rng.uniform(-1, 1, q)
        res = Mainfunction_albet(X_naive, y, a0, b0, best_lam_naive)
        a_n, b_n = res['alpha'], res['bet']
        y_pred_n = np.array([a_n @ X_naive[:, :, i] @ b_n for i in range(n)])
        mse_n    = float(np.mean((y - y_pred_n) ** 2))
        if mse_n < best_mse_naive:
            best_mse_naive = mse_n
            best_b_naive   = b_n
            best_a_naive   = a_n

    y_pred_naive = np.array([best_a_naive @ X_naive[:, :, i] @ best_b_naive
                             for i in range(n)])
    results['naive'] = compute_metrics(best_b_naive, y, y_pred_naive)

    # ── Joint W TEPIG ─────────────────────────────────────────────────────────
    GS          = G * S
    X_collapsed = X.transpose(0, 2, 1, 3).reshape(GS, q, n)   # (4, q, n)
    cv_mse_joint = [
        lambda_CV_mse(X_collapsed, y, np.ones(GS) / GS, np.ones(q) / q, lam)
        for lam in LAM_GRID
    ]
    best_lam_joint = LAM_GRID[int(np.argmin(cv_mse_joint))]

    best_mse_joint = np.inf
    best_b_joint   = np.zeros(q)
    best_w_joint   = np.ones(GS) / GS
    for _ in range(M_INIT):
        w0  = rng.dirichlet(np.ones(GS))
        b0  = rng.uniform(-1, 1, q)
        res = Mainfunction_albet(X_collapsed, y, w0, b0, best_lam_joint)
        w_j, b_j = res['alpha'], res['bet']
        y_pred_j = np.array([w_j @ X_collapsed[:, :, i] @ b_j for i in range(n)])
        mse_j    = float(np.mean((y - y_pred_j) ** 2))
        if mse_j < best_mse_joint:
            best_mse_joint = mse_j
            best_b_joint   = b_j
            best_w_joint   = w_j

    y_pred_joint = np.array([best_w_joint @ X_collapsed[:, :, i] @ best_b_joint
                             for i in range(n)])
    results['tepig_joint'] = compute_metrics(best_b_joint, y, y_pred_joint)

    # ── TEPIG Full (proximal gradient, no low-rank assumption) ───────────────
    d_full      = G * q * S
    X_flat_full = X.reshape(d_full, n).T

    folds_full   = _make_folds(n, rng)
    all_idx_full = list(range(n))
    cv_mse_full  = []
    for lam in PROXGRAD_LAM_GRID:
        fold_mse = []
        for k in range(5):
            te  = folds_full[k]
            tr  = [i for i in all_idx_full if i not in te]
            ic, B = proxgrad_fit(X[:, :, :, tr], y[tr], lam)
            y_pred_te = ic + X[:, :, :, te].reshape(d_full, len(te)).T @ B.reshape(-1)
            fold_mse.append(float(np.mean((y[te] - y_pred_te) ** 2)))
        cv_mse_full.append(float(np.mean(fold_mse)))

    best_lam_full     = PROXGRAD_LAM_GRID[int(np.argmin(cv_mse_full))]
    ic_full, B_full   = proxgrad_fit(X, y, best_lam_full)
    beta_hat_full     = np.array([float(np.linalg.norm(B_full[:, j, :])) for j in range(q)])
    y_pred_full       = ic_full + X_flat_full @ B_full.reshape(-1)
    results['tepig_grad'] = compute_metrics(beta_hat_full, y, y_pred_full)

    # ── Oracle ────────────────────────────────────────────────────────────────
    # Knows the true nonzero feature indices; fits OLS on those features only.
    n_nz      = len(nonzero_idx)
    X_oracle  = X[:, nonzero_idx, :, :].reshape(G * n_nz * S, n).T   # (n, G*n_nz*S)
    X_int     = np.column_stack([np.ones(n), X_oracle])
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


# ── Main: run B_SIMS repetitions in parallel ──────────────────────────────────
if __name__ == '__main__':
    print(f"\nRunning {B_SIMS} simulation repetitions (n_jobs={N_JOBS})...")

    seed_seq  = np.random.SeedSequence(RANDOM_SEED)
    seed_ints = [int(s.generate_state(1)[0]) for s in seed_seq.spawn(B_SIMS)]

    all_results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(run_one_sim)(s) for s in seed_ints
    )

    print("\nSummarising results...")
    estimators  = ['tepig', 'tepig_joint', 'tepig_grad', 'naive', 'oracle']
    metric_keys = ['tpr', 'fpr', 'l1', 'mse']

    summary = {est: {m: [] for m in metric_keys} for est in estimators}
    for rep in all_results:
        for est in estimators:
            tpr, fpr, l1, mse = rep[est]
            summary[est]['tpr'].append(tpr)
            summary[est]['fpr'].append(fpr)
            summary[est]['l1'].append(l1)
            summary[est]['mse'].append(mse)

    results_path = os.path.join(OUT_DATA, 'simulation_bootstrap_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump({
            'summary': summary,
            'config': {
                'B': B_SIMS, 'n_real': n_real, 'N_TARGET': N_TARGET,
                'q': q, 'G': G, 'S': S,
                'B_TRUE_BLOCK': B_TRUE_BLOCK.tolist(),
                'BETA_FEATURES': BETA_FEATURES,
                'SIGMA_SQ': SIGMA_SQ, 'SIGMA_R': SIGMA_R,
            }
        }, f)

    summary_path = os.path.join(OUT_SUMM, 'simulation_bootstrap_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("TEPIG BOOTSTRAP SIMULATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"B={B_SIMS} repetitions | n_real={n_real} subjects "
                f"(bootstrapped to N={N_TARGET}) | q={q} features "
                f"| G={G} clusters | S={S} slides\n")
        f.write(f"Nonzero features: {', '.join(features[i] for i in nonzero_idx)}\n")
        f.write(f"B_TRUE_BLOCK (G x S):\n")
        f.write(f"  {B_TRUE_BLOCK[0].tolist()}\n")
        f.write(f"  {B_TRUE_BLOCK[1].tolist()}\n")
        f.write(f"True model: full tensor (no low-rank assumption)\n\n")
        f.write(f"{'Estimator':<16} {'TPR':>8} {'FPR':>8} "
                f"{'L1 bias':>10} {'MSE':>10}\n")
        f.write("-" * 56 + "\n")
        for est in estimators:
            tpr = float(np.nanmean(summary[est]['tpr']))
            fpr = float(np.nanmean(summary[est]['fpr']))
            l1  = float(np.nanmean(summary[est]['l1']))
            mse = float(np.nanmean(summary[est]['mse']))
            f.write(f"  {est:<14} {tpr:>8.3f} {fpr:>8.3f} "
                    f"{l1:>10.3f} {mse:>10.3f}\n")

    print(f"  Saved: {results_path}")
    print(f"  Saved: {summary_path}")
    print("\nDone.")

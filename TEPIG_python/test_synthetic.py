"""
test_synthetic.py
-----------------
Single-rep synthetic data test for tepig_grad (and other estimators).

Data generation follows CLUSSO Section 4.1, adapted for TEPIG's tensor structure:
  - For each subject i and each slide s:
      X*_{i,s}[g, j] ~ N(2, 1)  if g == 0  (cluster 1)
      X*_{i,s}[g, j] ~ N(5, 3)  if g == 1  (cluster 2)
      w_{i,s} ~ Uniform{0.2, 0.3, ..., 0.8}
      X**_{i,s}[g, j] = w_{i,s}^g * (1 - w_{i,s})^(1-g) * X*_{i,s}[g, j]
                       = diag(w, 1-w) @ X*_{i,s}   ->  (G=2, q) per slide
  - Stack S=2 slides: X_i in R^(G=2, q, S=2)
  - All subjects stacked: X in R^(G=2, q, S=2, n=300)

True model (full tensor, no low-rank):
    y_i = intercept + <B_true, X_i> + eps_i,   eps_i ~ N(0, 1)
    B_true[:, j, :] = B_TRUE_BLOCK  for j in nonzero_features (2 out of q=10)
    B_true[:, j, :] = 0             otherwise

Estimators tested:
    tepig_grad : proximal gradient + group lasso (correct model)
    tepig      : alternating rank-1 structured lasso (misspecified)
    naive      : average over slides, run CLUSSO (misspecified)
    oracle     : OLS on true nonzero features only (benchmark)
"""

import os
import sys
import warnings
import numpy as np
warnings.filterwarnings('ignore')

from Mainfunction_albet import Mainfunction_albet, _glmnet_lasso
from SLasso_MSE import lambda_CV_mse

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'results')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
N       = 300    # number of subjects
Q       = 10     # number of features
G       = 2      # number of clusters
S       = 2      # number of slides
N_NZ    = 2      # number of nonzero features
SIGMA   = 1.0    # outcome noise std
SEED    = 42

# True B block for each nonzero feature: (G=2, S=2), entries from {1,2,3,4}
# det = 1*4 - 2*3 = 4-6 = -2 != 0  =>  rank-2 (not low-rank)
B_TRUE_BLOCK = np.array([[1.0, 2.0],
                          [3.0, 4.0]])

# Lambda grids
PROXGRAD_LAM_GRID = np.arange(0.25, 2.75, 0.25)   # CLUSSO grid {0.5,...,5.0} divided by 2 (two slides)
LAM_GRID          = np.logspace(-4, 0, 15)       # for tepig / naive

MAX_ITER = 50
TOL      = 1e-2
M_INIT   = 3


# ── Synthetic data generation (CLUSSO Section 4.1, adapted for 2 slides) ──────

def generate_synthetic(rng):
    """
    Generate (G=2, q, S=2, n) tensor of synthetic subject data.

    Slide-specific distributions (distinct to prevent naive averaging from trivially winning):
      Slide 1: cluster 1 ~ N(2,1),  cluster 2 ~ N(5,3)
      Slide 2: cluster 1 ~ N(4,2),  cluster 2 ~ N(8,1)

    For each subject i, for each slide s:
      - Draw X*_{i,s} in R^(G x q) from slide-specific distributions
      - Draw w_{i,s} ~ Uniform{0.2, 0.3, ..., 0.8}
      - Form X**_{i,s} = diag(w_{i,s}, 1-w_{i,s}) @ X*_{i,s}

    Returns X : (G, q, S, n)
    """
    weight_choices = np.arange(0.2, 0.9, 0.1)   # {0.2, 0.3, ..., 0.8}

    # Slide-specific distributions: (mean_cluster1, std_cluster1, mean_cluster2, std_cluster2)
    slide_params = [
        (2.0, 1.0, 5.0, np.sqrt(3.0)),   # slide 1: cluster1~N(2,1), cluster2~N(5,3)
        (4.0, np.sqrt(2.0), 8.0, 1.0),   # slide 2: cluster1~N(4,2), cluster2~N(8,1)
    ]

    tensors = []
    for _ in range(N):
        slides = []
        for s in range(S):
            mu1, sig1, mu2, sig2 = slide_params[s]

            # Raw features: (G, q)
            X_star = np.zeros((G, Q))
            X_star[0, :] = rng.normal(mu1, sig1, Q)   # cluster 1
            X_star[1, :] = rng.normal(mu2, sig2, Q)   # cluster 2

            # Weight for this subject-slide
            w = rng.choice(weight_choices)
            weights = np.array([w, 1.0 - w])          # (G,)

            # Weighted average: diag(w, 1-w) @ X_star
            X_star_weighted = weights[:, np.newaxis] * X_star   # (G, q)
            slides.append(X_star_weighted)

        # Stack slides: (G, q, S)
        tensors.append(np.stack(slides, axis=2))

    return np.stack(tensors, axis=3)   # (G, q, S, n)


# ── True B tensor ──────────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)

nonzero_idx = np.arange(N_NZ)    # first N_NZ features are nonzero
B_true = np.zeros((G, Q, S))
for j in nonzero_idx:
    B_true[:, j, :] = B_TRUE_BLOCK

beta_star = np.array([float(np.linalg.norm(B_true[:, j, :])) for j in range(Q)])

print("=" * 70)
print("SYNTHETIC DATA SIMULATION TEST  (CLUSSO Section 4.1 style)")
print("=" * 70)
print(f"n={N}, q={Q}, G={G}, S={S}, n_nonzero={N_NZ}")
print(f"B_TRUE_BLOCK det={float(np.linalg.det(B_TRUE_BLOCK)):.3f}")
print(f"Nonzero features: {nonzero_idx.tolist()}")
print()


# ── Generate data ──────────────────────────────────────────────────────────────
X = generate_synthetic(rng)          # (G, q, S, n)

y_true = np.einsum('gjs,gjsn->n', B_true, X)
y      = y_true + rng.normal(0, SIGMA, N)

print(f"y_true std={float(np.std(y_true)):.3f},  noise std={SIGMA:.3f}")
print()

out_lines = []
out_lines.append("=" * 70)
out_lines.append("SYNTHETIC DATA SIMULATION TEST  (CLUSSO Section 4.1 style)")
out_lines.append("=" * 70)
out_lines.append(f"n={N}, q={Q}, G={G}, S={S}, n_nonzero={N_NZ}")
out_lines.append(f"B_TRUE_BLOCK det={float(np.linalg.det(B_TRUE_BLOCK)):.3f}  (rank-2)")
out_lines.append(f"y_true std={float(np.std(y_true)):.3f},  noise std={SIGMA:.3f}")
out_lines.append("")


# ── Helpers ────────────────────────────────────────────────────────────────────

def compute_metrics(beta_hat, y, y_pred):
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


def _normalize_vec(v):
    if np.max(np.abs(v)) > 0:
        sig = np.sign(v[np.argmax(np.abs(v))])
        return sig * v / np.sum(np.abs(v))
    return v


# ── tepig_grad: proximal gradient + group lasso ────────────────────────────────

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


# ── CV for tepig_grad ──────────────────────────────────────────────────────────
out_lines.append("── tepig_grad CV lambda selection ──")
out_lines.append(f"Grid: {PROXGRAD_LAM_GRID[0]:.3f} to {PROXGRAD_LAM_GRID[-1]:.3f}")

folds    = [list(range(k * (N // 5), min((k + 1) * (N // 5), N))) for k in range(5)]
all_idx  = list(range(N))
cv_mse   = []

for lam in PROXGRAD_LAM_GRID:
    fold_mse = []
    for k in range(5):
        te = folds[k]; tr = [i for i in all_idx if i not in te]
        ic, B  = proxgrad_fit(X[:, :, :, tr], y[tr], lam)
        ypred  = ic + X[:, :, :, te].reshape(G * Q * S, len(te)).T @ B.reshape(-1)
        fold_mse.append(float(np.mean((y[te] - ypred) ** 2)))
    cv_mse.append(float(np.mean(fold_mse)))

best_idx = int(np.argmin(cv_mse))
best_lam = PROXGRAD_LAM_GRID[best_idx]

out_lines.append("CV MSE per lambda:")
for lam, mse in zip(PROXGRAD_LAM_GRID, cv_mse):
    marker = " <-- best" if lam == best_lam else ""
    out_lines.append(f"  lambda={lam:.3f}  cv_mse={mse:.4f}{marker}")

boundary = "BOUNDARY" if best_idx == 0 or best_idx == len(PROXGRAD_LAM_GRID) - 1 else "OK (interior)"
out_lines.append(f"CV best lambda: {best_lam:.4f}  {boundary}")

ic_full, B_full = proxgrad_fit(X, y, best_lam)
beta_hat_full   = np.array([float(np.linalg.norm(B_full[:, j, :])) for j in range(Q)])
d_full          = G * Q * S
y_pred_full     = ic_full + X.reshape(d_full, N).T @ B_full.reshape(-1)
tpr, fpr, l1, mse = compute_metrics(beta_hat_full, y, y_pred_full)

out_lines.append(f"tepig_grad:  TPR={tpr:.3f}  FPR={fpr:.3f}  L1={l1:.3f}  MSE={mse:.3f}")
sel_names = [str(j) for j in range(Q) if beta_hat_full[j] > 1e-6]
out_lines.append(f"Selected features (indices): {sel_names}")
out_lines.append("")


# ── tepig: alternating rank-1 structured lasso ────────────────────────────────

def tepig_fit(X, y, lam, alpha_init, beta_init, gamma_init):
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


out_lines.append("── tepig ──")
cv_mse_tepig = []
for lam in LAM_GRID:
    fold_mse = []
    for k in range(5):
        te = folds[k]; tr = [i for i in all_idx if i not in te]
        a0 = np.ones(G) / G; b0 = np.ones(Q) / Q; g0 = np.ones(S) / S
        a, b, g = tepig_fit(X[:, :, :, tr], y[tr], lam, a0, b0, g0)
        yp = np.einsum('gjsn,g,j,s->n', X[:, :, :, te], a, b, g)
        fold_mse.append(float(np.mean((y[te] - yp) ** 2)))
    cv_mse_tepig.append(float(np.mean(fold_mse)))

best_lam_tepig = LAM_GRID[int(np.argmin(cv_mse_tepig))]
best_mse_t = np.inf
best_res_t  = (np.ones(G) / G, np.zeros(Q), np.ones(S) / S)
for _ in range(M_INIT):
    a0 = rng.dirichlet(np.ones(G))
    b0 = rng.uniform(-1, 1, Q)
    g0 = rng.dirichlet(np.ones(S))
    a, b, g = tepig_fit(X, y, best_lam_tepig, a0, b0, g0)
    yp  = np.einsum('gjsn,g,j,s->n', X, a, b, g)
    mse = float(np.mean((y - yp) ** 2))
    if mse < best_mse_t:
        best_mse_t = mse; best_res_t = (a, b, g)

a_hat, b_hat, g_hat = best_res_t
y_pred_tepig = np.einsum('gjsn,g,j,s->n', X, a_hat, b_hat, g_hat)
tpr, fpr, l1, mse = compute_metrics(b_hat, y, y_pred_tepig)
out_lines.append(f"tepig:       TPR={tpr:.3f}  FPR={fpr:.3f}  L1={l1:.3f}  MSE={mse:.3f}")
out_lines.append("")


# ── naive: average slides, run CLUSSO ─────────────────────────────────────────
out_lines.append("── naive ──")
X_naive = X.mean(axis=2)   # (G, q, n)

cv_mse_naive = [
    lambda_CV_mse(X_naive, y, np.ones(G) / G, np.ones(Q) / Q, lam)
    for lam in LAM_GRID
]
best_lam_naive = LAM_GRID[int(np.argmin(cv_mse_naive))]

best_mse_n = np.inf
best_b_n   = np.zeros(Q)
best_a_n   = np.ones(G) / G
for _ in range(M_INIT):
    a0  = rng.dirichlet(np.ones(G))
    b0  = rng.uniform(-1, 1, Q)
    res = Mainfunction_albet(X_naive, y, a0, b0, best_lam_naive)
    a_n, b_n = res['alpha'], res['bet']
    yp_n = np.array([a_n @ X_naive[:, :, i] @ b_n for i in range(N)])
    mse_n = float(np.mean((y - yp_n) ** 2))
    if mse_n < best_mse_n:
        best_mse_n = mse_n; best_b_n = b_n; best_a_n = a_n

y_pred_naive = np.array([best_a_n @ X_naive[:, :, i] @ best_b_n for i in range(N)])
tpr, fpr, l1, mse = compute_metrics(best_b_n, y, y_pred_naive)
out_lines.append(f"naive:       TPR={tpr:.3f}  FPR={fpr:.3f}  L1={l1:.3f}  MSE={mse:.3f}")
out_lines.append("")


# ── oracle: OLS on true nonzero features only ──────────────────────────────────
out_lines.append("── oracle ──")
n_nz     = len(nonzero_idx)
X_oracle = X[:, nonzero_idx, :, :].reshape(G * n_nz * S, N).T
X_int    = np.column_stack([np.ones(N), X_oracle])
coeffs, _, _, _ = np.linalg.lstsq(X_int, y, rcond=None)
ic_oracle     = coeffs[0]
B_oracle_flat = coeffs[1:]
B_oracle      = B_oracle_flat.reshape(G, n_nz, S)
beta_hat_oracle = np.zeros(Q)
for k, j in enumerate(nonzero_idx):
    beta_hat_oracle[j] = float(np.linalg.norm(B_oracle[:, k, :]))
y_pred_oracle = ic_oracle + X_oracle @ B_oracle_flat
tpr, fpr, l1, mse = compute_metrics(beta_hat_oracle, y, y_pred_oracle)
out_lines.append(f"oracle:      TPR={tpr:.3f}  FPR={fpr:.3f}  L1={l1:.3f}  MSE={mse:.3f}")
out_lines.append("")


# ── Write output ───────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'test_synthetic_output.txt')
with open(out_path, 'w') as f:
    f.write('\n'.join(out_lines) + '\n')
print('\n'.join(out_lines))
print(f"Saved to {out_path}")

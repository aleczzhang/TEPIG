"""
test_bootstrap.py
-----------------
Local test for simulation_bootstrap.py.
Runs a single simulation repetition (instead of 200) so you can verify:
  - Bootstrap is working correctly
  - Outcome generation uses full tensor model (no low-rank)
  - All estimators run without errors
  - tepig_grad CV picks lambda in the interior of the grid

Writes output to outputs/test_bootstrap_output.txt
"""

import os
import sys
import pickle
import warnings
import numpy as np
warnings.filterwarnings('ignore')

from Mainfunction_albet import Mainfunction_albet, _glmnet_lasso
from SLasso_MSE import lambda_CV_mse

# ── Same config as simulation_bootstrap.py ────────────────────────────────────
_BASE    = os.path.join(os.path.dirname(__file__), '..', 'outputs')
OUT_REF  = os.path.join(_BASE, 'reference')
OUT_DATA = os.path.join(_BASE, 'data')
OUT_SUMM = os.path.join(_BASE, 'results')

BETA_FEATURES = [
    'Standard Deviation Red Nuclei',
    'Average TBM Thickness',
    'Total Object Aspect Ratio',
]
B_TRUE_BLOCK = np.array([[6.0, 4.0],
                          [3.0, 9.0]])
SIGMA_SQ          = 1.0
SIGMA_R           = 1.0
N_TARGET          = 300
LAM_GRID          = np.logspace(-4, 0, 15)
PROXGRAD_LAM_GRID = np.arange(0.005, 0.055, 0.005)
M_INIT            = 3
MAX_ITER          = 50
TOL               = 1e-2

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
with open(os.path.join(OUT_DATA, 'cluster_results.pkl'), 'rb') as f:
    cluster_results = pickle.load(f)
with open(os.path.join(OUT_REF, 'remaining_features.txt')) as f:
    features = [line.strip() for line in f]

q, G, S = len(features), 2, 2

true_2slide = [s for s in sorted(cluster_results.keys())
               if len(cluster_results[s]['slides']) == 2]
n_real = len(true_2slide)
print(f"  True 2-slide subjects: {n_real}")
print(f"  Bootstrap target: N_TARGET={N_TARGET}")

# ── Build B_true ──────────────────────────────────────────────────────────────
B_true = np.zeros((G, q, S))
for feat in BETA_FEATURES:
    if feat in features:
        B_true[:, features.index(feat), :] = B_TRUE_BLOCK

nonzero_idx = np.array([j for j in range(q) if np.any(B_true[:, j, :] != 0)])
beta_star   = np.array([float(np.linalg.norm(B_true[:, j, :])) for j in range(q)])

print(f"\nTrue nonzero features: {[features[i] for i in nonzero_idx]}")
print(f"B_TRUE_BLOCK:\n{B_TRUE_BLOCK}")
print(f"B_TRUE_BLOCK det: {float(np.linalg.det(B_TRUE_BLOCK)):.3f}  (nonzero = rank-2)")
print(f"B_TRUE_BLOCK Frobenius norm: {float(np.linalg.norm(B_TRUE_BLOCK)):.3f}")

# ── Bootstrap ─────────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

def bootstrap_sample(rng):
    idx = rng.integers(0, n_real, size=N_TARGET)
    tensors = []
    for i in idx:
        subj   = true_2slide[i]
        tensor = cluster_results[subj]['X_tensor'].copy()
        tensors.append(tensor + rng.normal(0, SIGMA_R, tensor.shape))
    return np.stack(tensors, axis=3)

print(f"\nBootstrapping {N_TARGET} subjects from {n_real} true 2-slide subjects...")
X = bootstrap_sample(rng)
n = N_TARGET
print(f"  X shape: {X.shape}  (G={G}, q={q}, S={S}, n={n})")

# Check bootstrap: verify unique subjects and noise
unique_sampled = len(set(
    true_2slide[i] for i in rng.integers(0, n_real, size=N_TARGET)
))
print(f"  Unique subjects sampled (approx): ~{int(n_real * (1 - (1-1/n_real)**N_TARGET))}")

# ── Generate outcome ──────────────────────────────────────────────────────────
y_true = np.einsum('gjs,gjsn->n', B_true, X)
y      = y_true + rng.normal(0, np.sqrt(SIGMA_SQ), n)
print(f"\nOutcome stats:")
print(f"  y_true mean={float(np.mean(y_true)):.3f}, std={float(np.std(y_true)):.3f}")
print(f"  y      mean={float(np.mean(y)):.3f},      std={float(np.std(y)):.3f}")
print(f"  SNR (std_signal / std_noise) = {float(np.std(y_true)):.3f}")

out_lines = []
out_lines.append("=" * 70)
out_lines.append("BOOTSTRAP SIMULATION TEST")
out_lines.append("=" * 70)
out_lines.append(f"n_real={n_real}, N_TARGET={N_TARGET}, q={q}, G={G}, S={S}")
out_lines.append(f"B_TRUE_BLOCK det={float(np.linalg.det(B_TRUE_BLOCK)):.3f}  (rank-2)")
out_lines.append(f"y_true std={float(np.std(y_true)):.3f},  noise std={np.sqrt(SIGMA_SQ):.3f}")
out_lines.append("")

def _normalize_vec(v):
    if np.max(np.abs(v)) > 0:
        sig = np.sign(v[np.argmax(np.abs(v))])
        return sig * v / np.sum(np.abs(v))
    return v

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
        dif = (np.linalg.norm(alpha - alpha0) + np.linalg.norm(beta - beta0) +
               np.linalg.norm(gamma - gamma0))
        if dif < TOL:
            break
    alpha[np.abs(alpha) < 0.001] = 0.0
    beta[np.abs(beta)   < 0.001] = 0.0
    gamma[np.abs(gamma) < 0.001] = 0.0
    alpha = nr_alpha * alpha
    return alpha, beta, gamma

def _make_folds(n, rng):
    idx  = rng.permutation(n).tolist()
    size = n // 5
    folds = [sorted(idx[k * size:(k + 1) * size]) for k in range(4)]
    folds.append(sorted(idx[4 * size:]))
    return folds

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
    x, z, t   = np.zeros(d), np.zeros(d), 1.0
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
        x     = v.reshape(-1)
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

# ── Run each estimator ────────────────────────────────────────────────────────

# tepig_grad CV grid check
out_lines.append("── tepig_grad CV lambda selection ──")
out_lines.append(f"Grid: {PROXGRAD_LAM_GRID[0]:.3f} to {PROXGRAD_LAM_GRID[-1]:.3f}")
folds_pg   = _make_folds(n, rng)
all_idx    = list(range(n))
d_full     = G * q * S
cv_mse_pg  = []
for lam in PROXGRAD_LAM_GRID:
    fold_mse = []
    for k in range(5):
        te  = folds_pg[k]
        tr  = [i for i in all_idx if i not in te]
        ic, B = proxgrad_fit(X[:, :, :, tr], y[tr], lam)
        ypred = ic + X[:, :, :, te].reshape(d_full, len(te)).T @ B.reshape(-1)
        fold_mse.append(float(np.mean((y[te] - ypred) ** 2)))
    cv_mse_pg.append(float(np.mean(fold_mse)))

cv_mse_arr = np.array(cv_mse_pg)
best_lam_pg = PROXGRAD_LAM_GRID[int(np.argmin(cv_mse_arr))]
out_lines.append("CV MSE per lambda:")
for lam, mse in zip(PROXGRAD_LAM_GRID, cv_mse_arr):
    marker = " <-- best" if lam == best_lam_pg else ""
    out_lines.append(f"  lambda={lam:.3f}  cv_mse={mse:.4f}{marker}")
boundary = (best_lam_pg == PROXGRAD_LAM_GRID[0] or best_lam_pg == PROXGRAD_LAM_GRID[-1])
out_lines.append(f"CV best lambda: {best_lam_pg:.4f}  {'*** BOUNDARY — adjust grid ***' if boundary else 'OK (interior)'}")

ic_pg, B_pg   = proxgrad_fit(X, y, best_lam_pg)
beta_hat_pg   = np.array([float(np.linalg.norm(B_pg[:, j, :])) for j in range(q)])
y_pred_pg     = ic_pg + X.reshape(d_full, n).T @ B_pg.reshape(-1)
tpr, fpr, l1, mse = compute_metrics(beta_hat_pg, y, y_pred_pg)
out_lines.append(f"tepig_grad:  TPR={tpr:.3f}  FPR={fpr:.3f}  L1={l1:.3f}  MSE={mse:.3f}")
out_lines.append(f"Selected: {[features[i] for i in np.where(beta_hat_pg > 1e-6)[0]]}")
out_lines.append("")

# tepig (1 init only for speed)
print("Running tepig...")
out_lines.append("── tepig ──")
cv_mse_t = []
for lam in LAM_GRID:
    folds_t  = _make_folds(n, rng)
    fold_mse = []
    for k in range(5):
        te = folds_t[k]; tr = [i for i in all_idx if i not in te]
        a, b, g = tepig_fit(X[:,:,:,tr], y[tr], lam,
                            np.ones(G)/G, np.ones(q)/q, np.ones(S)/S)
        yp = np.einsum('gjsn,g,j,s->n', X[:,:,:,te], a, b, g)
        fold_mse.append(float(np.mean((y[te] - yp)**2)))
    cv_mse_t.append(float(np.mean(fold_mse)))
best_lam_t = LAM_GRID[int(np.argmin(cv_mse_t))]
a_hat, b_hat, g_hat = tepig_fit(X, y, best_lam_t,
                                 np.ones(G)/G, np.ones(q)/q, np.ones(S)/S)
y_pred_t = np.einsum('gjsn,g,j,s->n', X, a_hat, b_hat, g_hat)
tpr, fpr, l1, mse = compute_metrics(b_hat, y, y_pred_t)
out_lines.append(f"tepig:       TPR={tpr:.3f}  FPR={fpr:.3f}  L1={l1:.3f}  MSE={mse:.3f}")
out_lines.append("")

# naive
print("Running naive...")
out_lines.append("── naive ──")
X_naive      = X.mean(axis=2)
cv_mse_naive = [lambda_CV_mse(X_naive, y, np.ones(G)/G, np.ones(q)/q, lam)
                for lam in LAM_GRID]
best_lam_n   = LAM_GRID[int(np.argmin(cv_mse_naive))]
res          = Mainfunction_albet(X_naive, y, np.ones(G)/G, np.ones(q)/q, best_lam_n)
a_n, b_n     = res['alpha'], res['bet']
y_pred_n     = np.array([a_n @ X_naive[:, :, i] @ b_n for i in range(n)])
tpr, fpr, l1, mse = compute_metrics(b_n, y, y_pred_n)
out_lines.append(f"naive:       TPR={tpr:.3f}  FPR={fpr:.3f}  L1={l1:.3f}  MSE={mse:.3f}")
out_lines.append("")

# oracle
print("Running oracle...")
out_lines.append("── oracle ──")
n_nz      = len(nonzero_idx)
X_oracle  = X[:, nonzero_idx, :, :].reshape(G * n_nz * S, n).T
X_int     = np.column_stack([np.ones(n), X_oracle])
coeffs, _, _, _ = np.linalg.lstsq(X_int, y, rcond=None)
ic_oracle     = coeffs[0]
B_oracle_flat = coeffs[1:]
B_oracle      = B_oracle_flat.reshape(G, n_nz, S)
beta_hat_oracle = np.zeros(q)
for k, j in enumerate(nonzero_idx):
    beta_hat_oracle[j] = float(np.linalg.norm(B_oracle[:, k, :]))
y_pred_oracle = ic_oracle + X_oracle @ B_oracle_flat
tpr, fpr, l1, mse = compute_metrics(beta_hat_oracle, y, y_pred_oracle)
out_lines.append(f"oracle:      TPR={tpr:.3f}  FPR={fpr:.3f}  L1={l1:.3f}  MSE={mse:.3f}")

# ── Write output ──────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_SUMM, 'test_bootstrap_output.txt')
with open(out_path, 'w') as f:
    f.write('\n'.join(out_lines) + '\n')
print('\n'.join(out_lines))
print(f"\nSaved to {out_path}")

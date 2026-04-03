"""Quick local test for tepig_grad (proximal gradient descent estimator)."""

import os
import sys
import pickle
import warnings
import numpy as np
warnings.filterwarnings('ignore')  # suppress overflow/NaN warnings from near-zero features

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CLUSSO_python'))

# ── Load data (same setup as simulation.py) ────────────────────────────────
_BASE    = os.path.join(os.path.dirname(__file__), '..', 'outputs')
OUT_REF  = os.path.join(_BASE, 'reference')
OUT_DATA = os.path.join(_BASE, 'data')
OUT_SUMM = os.path.join(_BASE, 'results')

with open(os.path.join(OUT_DATA, 'cluster_results.pkl'), 'rb') as f:
    cluster_results = pickle.load(f)
with open(os.path.join(OUT_REF, 'remaining_features.txt')) as f:
    features = [line.strip() for line in f]

subjects = sorted(cluster_results.keys())
n, q, G, S = len(subjects), len(features), 2, 2

X_all = np.stack([cluster_results[s]['X_tensor'] for s in subjects], axis=3)

BETA_VAL      = 15.0
BETA_FEATURES = ['Standard Deviation Red Nuclei', 'Average TBM Thickness',
                 'Total Object Aspect Ratio']
beta_star = np.zeros(q)
for feat in BETA_FEATURES:
    if feat in features:
        beta_star[features.index(feat)] = BETA_VAL

nonzero_idx = np.where(beta_star != 0)[0]
print(f"True nonzero features: {[features[i] for i in nonzero_idx]}")

alpha_norm = np.array([1.0, 2.0]) / 3.0
gamma_norm = np.array([2.0, 1.0]) / 3.0
y_true = np.einsum('gjsn,g,j,s->n', X_all, alpha_norm, beta_star, gamma_norm)

# ── One simulation rep ─────────────────────────────────────────────────────
rng = np.random.default_rng(42)
y   = y_true + rng.normal(0, 1.0, n)

# ── Copy proxgrad_fit from simulation.py ──────────────────────────────────
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
    L, eta    = float(sigma_max ** 2) / n_tr, 0.0
    eta       = 1.0 / (float(sigma_max ** 2) / n_tr)
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

# ── Lambda grid ────────────────────────────────────────────────────────────
lam_grid = np.arange(0.005, 0.055, 0.005)  # {0.005, 0.010, ..., 0.050} — shifted down further

out_lines = []
out_lines.append(f"Lambda grid: {lam_grid[0]:.4f} to {lam_grid[-1]:.4f}")
out_lines.append("")
out_lines.append(f"{'lambda':>10}  {'n_sel':>5}  {'TPR':>6}  {'FPR':>6}  features selected")
out_lines.append("-" * 100)

for lam in lam_grid:
    ic, B  = proxgrad_fit(X_all, y, lam)
    imp    = np.array([float(np.linalg.norm(B[:, j, :])) for j in range(q)])
    sel    = np.where(imp > 1e-6)[0]
    tpr    = float(np.isin(nonzero_idx, sel).mean())
    n_fp   = sum(1 for s in sel if s not in nonzero_idx)
    fpr    = n_fp / max(q - len(nonzero_idx), 1)
    names  = [features[i][:20] for i in sel]
    out_lines.append(f"{lam:10.4f}  {len(sel):>5}  {tpr:>6.2f}  {fpr:>6.2f}  {names}")

# ── CV lambda selection ────────────────────────────────────────────────────
out_lines.append("")
out_lines.append("── CV lambda selection ──")
folds    = [list(range(k * 23, min((k+1)*23, n))) for k in range(5)]
all_idx  = list(range(n))
cv_mse   = []
for lam in lam_grid:
    fold_mse = []
    for k in range(5):
        te = folds[k]; tr = [i for i in all_idx if i not in te]
        ic, B  = proxgrad_fit(X_all[:, :, :, tr], y[tr], lam)
        ypred  = ic + X_all[:, :, :, te].reshape(G*q*S, len(te)).T @ B.reshape(-1)
        fold_mse.append(float(np.mean((y[te] - ypred)**2)))
    cv_mse.append(float(np.mean(fold_mse)))

cv_mse_arr = np.array(cv_mse)
best_lam   = lam_grid[int(np.argmin(cv_mse_arr))]
ic_cv, B_cv = proxgrad_fit(X_all, y, best_lam)
imp_cv   = np.array([float(np.linalg.norm(B_cv[:, j, :])) for j in range(q)])
sel_cv   = np.where(imp_cv > 1e-6)[0]
tpr_cv   = float(np.isin(nonzero_idx, sel_cv).mean())
n_fp_cv  = sum(1 for s in sel_cv if s not in nonzero_idx)
fpr_cv   = n_fp_cv / max(q - len(nonzero_idx), 1)
out_lines.append("CV MSE per lambda:")
for lam, mse in zip(lam_grid, cv_mse_arr):
    out_lines.append(f"  lambda={lam:.1f}  cv_mse={mse:.4f}")
out_lines.append(f"CV best lambda: {best_lam:.4f}")
out_lines.append(f"TPR: {tpr_cv:.3f}  FPR: {fpr_cv:.3f}  n_selected: {len(sel_cv)}")
out_lines.append(f"Selected: {[features[i] for i in sel_cv]}")

# ── Write output ───────────────────────────────────────────────────────────
out_path = os.path.join(OUT_SUMM, 'test_proxgrad_output.txt')
with open(out_path, 'w') as f:
    f.write('\n'.join(out_lines) + '\n')
print('\n'.join(out_lines))
print(f"\nSaved to {out_path}")

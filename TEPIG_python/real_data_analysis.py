"""
real_data_analysis.py
---------------------
Apply TEPIG, CLUSSO, naive, and oracle estimators to the real Coimbra
kidney transplant data.

Only subjects with 2 real biopsy slides are used (confirmed by professor).

Inputs:
  - outputs/data/cluster_results.pkl          : per-subject GMM cluster tensors
  - outputs/reference/remaining_features.txt  : feature names after pruning
  - Object_level_data/.../coimbra_clinical_outcomes.csv : outcome data

Outcome: eGFR_CKD_EPI_12M (1-year eGFR via CKD-EPI formula)

Estimators:
  TEPIG  : proximal gradient + group lasso on full (G, q, S, n) tensor
  clusso : Mainfunction_albet on mega-slide (G, q, n) — tubules pooled
           across both slides per cluster
  naive  : Mainfunction_albet on slide-averaged (G, q, n) matrix
  oracle : OLS on features selected by TEPIG (post-selection, no penalty)
           — not a true oracle (no ground truth), but removes lasso bias

Lambda selection: 5-fold CV on MSE for TEPIG, CLUSSO, and naive.

Outputs saved to outputs/results/:
  - real_data_results.pkl  : predictions, selected features, coefficients
  - real_data_summary.txt  : human-readable results
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CLUSSO_python'))
from Mainfunction_albet import Mainfunction_albet, _glmnet_lasso
from SLasso_MSE import lambda_CV_mse

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
_BASE    = os.path.join(_HERE, '..', 'outputs')
OUT_DATA = os.path.join(_BASE, 'data')
OUT_REF  = os.path.join(_BASE, 'reference')
OUT_RES  = os.path.join(_BASE, 'results')
COIMBRA  = os.path.join(_HERE, '..', 'Object_level_data',
                        'Donors_included_after_biopsy_QCed',
                        'coimbra_clinical_outcomes.csv')
os.makedirs(OUT_RES, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
G           = 2
S           = 2
M_INIT      = 3     # random initialisations for TEPIG / CLUSSO
MAX_ITER    = 50
TOL         = 1e-2
RANDOM_SEED = 42

# Lambda grid built data-adaptively after X is constructed (see below)

# ── Load cluster results and feature names ─────────────────────────────────────
print("Loading cluster results...")
with open(os.path.join(OUT_DATA, 'cluster_results.pkl'), 'rb') as f:
    cluster_results = pickle.load(f)

with open(os.path.join(OUT_REF, 'remaining_features.txt')) as f:
    features = [line.strip() for line in f if line.strip()]

q = len(features)

# Filter to subjects with 2 real slides (professor's requirement)
two_slide_subjects = sorted([
    s for s in cluster_results
    if len(cluster_results[s]['slides']) == 2
])
print(f"  Total subjects in cluster_results : {len(cluster_results)}")
print(f"  Subjects with 2 real slides       : {len(two_slide_subjects)}")
print(f"  Features (q)                      : {q}")

# ── Load outcome data ──────────────────────────────────────────────────────────
print("\nLoading outcome data...")
coimbra = pd.read_csv(COIMBRA)

# Extract subject ID: 'H11-02415 - 1' -> 'H11-02415'
coimbra['subject_id'] = (coimbra['Slide_number'].astype(str)
                         .str.split(' - ').str[0].str.strip())

# Per subject: take first non-null 1-year eGFR
outcome_map = (
    coimbra.groupby('subject_id')['eGFR_CKD_EPI_12M']
    .first()
    .dropna()
    .to_dict()
)
print(f"  Subjects with 1-yr eGFR outcome: {len(outcome_map)}")

# ── Match subjects ─────────────────────────────────────────────────────────────
subjects = []
y_list   = []
for subj in two_slide_subjects:
    key = subj
    if key not in outcome_map:
        key = subj.replace(' ', '-')   # 'H20 02897' -> 'H20-02897'
    if key in outcome_map:
        subjects.append(subj)
        y_list.append(outcome_map[key])

n = len(subjects)
y = np.array(y_list, dtype=float)
print(f"\n  Subjects with 2 slides + outcome: {n}")
print(f"  eGFR  mean={y.mean():.1f}  std={y.std():.1f}  "
      f"min={y.min():.1f}  max={y.max():.1f}")

# ── Build design tensors ───────────────────────────────────────────────────────
# X_tepig : (G, q, S, n) — per-slide weighted cluster averages
X_tepig = np.stack(
    [cluster_results[s]['X_tensor'] for s in subjects], axis=3
)

# X_naive : (G, q, n) — average over S slides
X_naive = X_tepig.mean(axis=2)

# X_clusso : (G, q, n) — mega-slide: pool tubules from both slides per cluster.
# For each subject i and cluster g:
#   n_g_s    = number of tubules in cluster g for slide s (from stored labels)
#   n_g_tot  = sum of n_g_s across slides
#   n_tot    = total tubules across all slides
#   mean_g   = weighted average of per-slide cluster means by n_g_s
#   X_clusso[g, :, i] = (n_g_tot / n_tot) * mean_g
X_clusso = np.zeros((G, q, n))
for i, subj in enumerate(subjects):
    res     = cluster_results[subj]
    labels  = res['labels']          # list of arrays, one per real slide
    c_avgs  = res['cluster_avgs']    # list of (G, q) arrays, one per real slide
    n_tubs  = res['n_tubules']       # list of total tubule counts per slide
    n_tot   = sum(n_tubs)
    for g in range(G):
        n_g_per_slide = np.array([(lbl == g + 1).sum() for lbl in labels])
        n_g_tot       = n_g_per_slide.sum()
        if n_g_tot > 0:
            mean_g = np.sum(
                [n_g_per_slide[s] * c_avgs[s][g] for s in range(len(labels))],
                axis=0
            ) / n_g_tot
        else:
            mean_g = np.zeros(q)
        X_clusso[g, :, i] = (n_g_tot / n_tot) * mean_g

print(f"\n  X_tepig  shape : {X_tepig.shape}   (G, q, S, n)")
print(f"  X_naive  shape : {X_naive.shape}  (G, q, n)")
print(f"  X_clusso shape : {X_clusso.shape}  (G, q, n)")

# ── Lambda grid ───────────────────────────────────────────────────────────────
# Same fixed grid as simulation_synthetic.py. Both _glmnet_lasso (CLUSSO/naive)
# and proxgrad_fit (TEPIG) standardize/normalize X internally, so the lambda
# scale is independent of raw feature magnitude.
LAM_GRID = np.array([0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75,
                     2.00, 2.25, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00])

print(f"\n  Lambda grid: {LAM_GRID[0]:.2f} to {LAM_GRID[-1]:.2f}  "
      f"({len(LAM_GRID)} values)")

# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalize_vec(v):
    if np.max(np.abs(v)) > 0:
        sig = np.sign(v[np.argmax(np.abs(v))])
        return sig * v / np.sum(np.abs(v))
    return v


def _make_folds(n, rng, k=5):
    idx  = rng.permutation(n).tolist()
    size = n // k
    folds = [sorted(idx[i * size:(i + 1) * size]) for i in range(k - 1)]
    folds.append(sorted(idx[(k - 1) * size:]))
    return folds


def proxgrad_fit(X, y, lam, max_iter=2000, tol=1e-6):
    """Proximal gradient descent with group lasso on (G, q, S, n) tensor."""
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
    eta       = 1.0 / (float(sigma_max ** 2) / n_tr)
    threshold = eta * lam

    x, z, t  = np.zeros(d), np.zeros(d), 1.0
    intercept = float(np.mean(y))

    for _ in range(max_iter):
        x_old     = x.copy()
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


def clusso_select_and_fit(X_mat, y, rng, lam_grid):
    """
    Select lambda by 5-fold CV on MSE then refit with M_INIT random starts.
    X_mat : (G, q, n)
    """
    n_loc = X_mat.shape[2]
    q_loc = X_mat.shape[1]

    cv_mse = [
        lambda_CV_mse(X_mat, y, np.ones(G) / G, np.ones(q_loc) / q_loc, lam)
        for lam in lam_grid
    ]
    best_lam    = lam_grid[int(np.argmin(cv_mse))]
    best_cv_mse = float(np.min(cv_mse))

    best_mse = np.inf
    best_b   = np.zeros(q_loc)
    best_a   = np.ones(G) / G
    for _ in range(M_INIT):
        a0  = rng.dirichlet(np.ones(G))
        b0  = rng.uniform(-1, 1, q_loc)
        res = Mainfunction_albet(X_mat, y, a0, b0, best_lam)
        a_, b_ = res['alpha'], res['bet']
        yp  = np.array([a_ @ X_mat[:, :, i] @ b_ for i in range(n_loc)])
        mse = float(np.mean((y - yp) ** 2))
        if mse < best_mse:
            best_mse = mse; best_b = b_; best_a = a_

    y_pred = np.array([best_a @ X_mat[:, :, i] @ best_b for i in range(n_loc)])
    return best_a, best_b, best_lam, best_cv_mse, y_pred


# ── Fit estimators ─────────────────────────────────────────────────────────────
rng     = np.random.default_rng(RANDOM_SEED)
folds   = _make_folds(n, rng)
all_idx = list(range(n))
results = {}

# ── TEPIG ──────────────────────────────────────────────────────────────────────
print("\nFitting TEPIG...")
d_full = G * q * S
cv_mse_tg = []
for lam in LAM_GRID:
    fold_mse = []
    for k in range(5):
        te = folds[k]; tr = [i for i in all_idx if i not in te]
        ic, B  = proxgrad_fit(X_tepig[:, :, :, tr], y[tr], lam)
        ypred  = ic + X_tepig[:, :, :, te].reshape(d_full, len(te)).T @ B.reshape(-1)
        fold_mse.append(float(np.mean((y[te] - ypred) ** 2)))
    cv_mse_tg.append(float(np.mean(fold_mse)))

best_lam_tg = LAM_GRID[int(np.argmin(cv_mse_tg))]
best_cv_tg  = float(np.min(cv_mse_tg))
ic_tg, B_tg = proxgrad_fit(X_tepig, y, best_lam_tg)
imp_tg      = np.array([float(np.linalg.norm(B_tg[:, j, :])) for j in range(q)])
sel_tg      = [features[j] for j in range(q) if imp_tg[j] > 1e-6]
sel_idx_tg  = [j for j in range(q) if imp_tg[j] > 1e-6]
y_pred_tg   = ic_tg + X_tepig.reshape(d_full, n).T @ B_tg.reshape(-1)

results['TEPIG'] = {
    'lambda': best_lam_tg, 'cv_mse': best_cv_tg,
    'B': B_tg, 'intercept': ic_tg,
    'importance': imp_tg, 'selected': sel_tg, 'selected_idx': sel_idx_tg,
    'y_pred': y_pred_tg,
}
print(f"  lambda={best_lam_tg:.4f}  CV-MSE={best_cv_tg:.2f}  "
      f"features selected={len(sel_tg)}")

# ── CLUSSO ─────────────────────────────────────────────────────────────────────
print("\nFitting CLUSSO...")
a_cl, b_cl, lam_cl, cv_mse_cl, y_pred_cl = clusso_select_and_fit(X_clusso, y, rng, LAM_GRID)
sel_cl = [features[j] for j in range(q) if abs(b_cl[j]) > 1e-6]

results['clusso'] = {
    'lambda': lam_cl, 'cv_mse': cv_mse_cl,
    'alpha': a_cl, 'beta': b_cl,
    'selected': sel_cl, 'y_pred': y_pred_cl,
}
print(f"  lambda={lam_cl:.4f}  CV-MSE={cv_mse_cl:.2f}  "
      f"features selected={len(sel_cl)}")
print(f"  alpha (cluster weights): {np.round(a_cl, 3)}")

# ── Naive ──────────────────────────────────────────────────────────────────────
print("\nFitting naive...")
a_nv, b_nv, lam_nv, cv_mse_nv, y_pred_nv = clusso_select_and_fit(X_naive, y, rng, LAM_GRID)
sel_nv = [features[j] for j in range(q) if abs(b_nv[j]) > 1e-6]

results['naive'] = {
    'lambda': lam_nv, 'cv_mse': cv_mse_nv,
    'alpha': a_nv, 'beta': b_nv,
    'selected': sel_nv, 'y_pred': y_pred_nv,
}
print(f"  lambda={lam_nv:.4f}  CV-MSE={cv_mse_nv:.2f}  "
      f"features selected={len(sel_nv)}")
print(f"  alpha (cluster weights): {np.round(a_nv, 3)}")

# ── Oracle: OLS on TEPIG-selected features ─────────────────────────────────────
# With n=35 subjects and G*q=86 total features, OLS on all features is
# underdetermined. Oracle is defined as OLS on the features selected by TEPIG,
# removing the lasso shrinkage bias on those features.
print("\nFitting oracle (OLS on TEPIG-selected features)...")
if len(sel_idx_tg) > 0 and len(sel_idx_tg) < n:
    # Use TEPIG X_tepig restricted to selected features: (G, q_sel, S, n)
    X_or_tensor = X_tepig[:, sel_idx_tg, :, :]               # (G, q_sel, S, n)
    d_or        = G * len(sel_idx_tg) * S
    X_or        = X_or_tensor.reshape(d_or, n).T              # (n, G*q_sel*S)
    X_int       = np.column_stack([np.ones(n), X_or])
    coeffs_or, _, _, _ = np.linalg.lstsq(X_int, y, rcond=None)
    ic_or       = coeffs_or[0]
    b_or_flat   = coeffs_or[1:]
    y_pred_or   = ic_or + X_or @ b_or_flat

    # CV-MSE for oracle
    cv_mse_or = []
    for k in range(5):
        te = folds[k]; tr = [i for i in all_idx if i not in te]
        X_tr = np.column_stack([np.ones(len(tr)), X_or[tr]])
        X_te = np.column_stack([np.ones(len(te)), X_or[te]])
        c, _, _, _ = np.linalg.lstsq(X_tr, y[tr], rcond=None)
        cv_mse_or.append(float(np.mean((y[te] - X_te @ c) ** 2)))
    cv_mse_or_mean = float(np.mean(cv_mse_or))
else:
    # No features selected by TEPIG — fall back to intercept-only
    cv_mse_or_mean = float(np.var(y))
    y_pred_or      = np.full(n, y.mean())
    b_or_flat      = np.array([])

results['oracle'] = {
    'cv_mse': cv_mse_or_mean,
    'selected': sel_tg,   # same features as TEPIG
    'y_pred': y_pred_or,
}
print(f"  OLS on {len(sel_tg)} TEPIG-selected features  "
      f"CV-MSE={cv_mse_or_mean:.2f}")

# ── Save results ───────────────────────────────────────────────────────────────
results['meta'] = {
    'subjects': subjects, 'y': y,
    'features': features,
    'n': n, 'q': q, 'G': G, 'S': S,
    'lam_grid': LAM_GRID.tolist(),
}

pkl_path = os.path.join(OUT_RES, 'real_data_results.pkl')
with open(pkl_path, 'wb') as f:
    pickle.dump(results, f)
print(f"\n  Saved: {pkl_path}")

# ── Summary ────────────────────────────────────────────────────────────────────
summary_path = os.path.join(OUT_RES, 'real_data_summary.txt')
with open(summary_path, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("REAL DATA ANALYSIS SUMMARY\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Subjects (n)  : {n}  (2 real slides only)\n")
    f.write(f"Features (q)  : {q}\n")
    f.write(f"Outcome       : eGFR_CKD_EPI_12M (1-year eGFR)\n")
    f.write(f"eGFR  mean={y.mean():.1f}  std={y.std():.1f}  "
            f"min={y.min():.1f}  max={y.max():.1f}\n\n")

    f.write(f"{'Estimator':<12} {'CV-MSE':>10} {'N selected':>12}\n")
    f.write("-" * 36 + "\n")
    for est in ['TEPIG', 'clusso', 'naive', 'oracle']:
        cv  = results[est]['cv_mse']
        sel = results[est].get('selected', [])
        n_sel = len(sel) if sel else 'N/A'
        f.write(f"  {est:<10} {cv:>10.4f} {str(n_sel):>12}\n")

    f.write("\n")
    for est in ['TEPIG', 'clusso', 'naive']:
        f.write("-" * 70 + "\n")
        f.write(f"Estimator: {est}\n")
        f.write(f"  Lambda   : {results[est]['lambda']:.6f}\n")
        f.write(f"  CV-MSE   : {results[est]['cv_mse']:.4f}\n")
        if 'alpha' in results[est]:
            f.write(f"  Alpha (cluster weights): "
                    f"{np.round(results[est]['alpha'], 4).tolist()}\n")
        f.write(f"  Features selected ({len(results[est]['selected'])}):\n")
        for feat in results[est]['selected']:
            f.write(f"    {feat}\n")
        f.write("\n")

    f.write("-" * 70 + "\n")
    f.write("Estimator: oracle\n")
    f.write(f"  CV-MSE   : {results['oracle']['cv_mse']:.4f}\n")
    f.write(f"  OLS on TEPIG-selected features ({len(sel_tg)}) — "
            f"removes lasso shrinkage bias\n")

print(f"  Saved: {summary_path}")
print("\nDone.")

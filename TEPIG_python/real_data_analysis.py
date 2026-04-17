"""
real_data_analysis.py
---------------------
Apply TEPIG, CLUSSO, and naive estimators to Coimbra data.

All subjects with eGFR outcomes are used (both 1-slide and 2-slide).
1-slide subjects have their single slide duplicated as slide 2
(exact copy — treats the WSI as the true population average).

80/20 train/test split. Lambda selected by 5-fold CV on training set only.
Test MSE reported on held-out subjects never seen during fitting.

TEPIG beta: mean of (G, S) block per feature → (q,) vector → L1-normalized
for feature importance reporting (comparable to CLUSSO/naive's (q,) beta).
Oracle not included — only meaningful in simulation with known true coefficients.

Inputs:
  - outputs/data/cluster_results.pkl
  - outputs/reference/remaining_features.txt
  - Object_level_data/.../coimbra_clinical_outcomes.csv

Outputs:
  - outputs/results/real_data_results.pkl
  - outputs/results/real_data_summary.txt
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

from sklearn.linear_model import LassoCV
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
M_INIT      = 3
RANDOM_SEED = 42
N_BOOTSTRAP = 300   # resample with replacement from all subjects with outcomes
TEST_FRAC   = 0.2   # 80/20 train/test split on bootstrapped data

LAM_GRID = np.array([0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75,
                     2.00, 2.25, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00])

# ── Load cluster results and feature names ─────────────────────────────────────
print("Loading cluster results...")
with open(os.path.join(OUT_DATA, 'cluster_results.pkl'), 'rb') as f:
    cluster_results = pickle.load(f)

with open(os.path.join(OUT_REF, 'remaining_features.txt')) as f:
    features = [line.strip() for line in f if line.strip()]

q = len(features)

all_subjects = sorted(cluster_results.keys())
print(f"  Total subjects in cluster_results : {len(all_subjects)}")
print(f"  Features (q)                      : {q}")

# ── Load outcome data ──────────────────────────────────────────────────────────
print("\nLoading outcome data...")
coimbra = pd.read_csv(COIMBRA)
coimbra['subject_id'] = (coimbra['Slide_number'].astype(str)
                         .str.split(' - ').str[0].str.strip())
outcome_map = (
    coimbra.groupby('subject_id')['eGFR_CKD_EPI_12M']
    .first().dropna().to_dict()
)

# ── Match all subjects (1-slide and 2-slide) to outcomes ──────────────────────
subjects_orig, y_orig_list, n_real_slides = [], [], []
for subj in all_subjects:
    key = subj
    if key not in outcome_map:
        key = subj.replace(' ', '-')
    if key in outcome_map:
        subjects_orig.append(subj)
        y_orig_list.append(outcome_map[key])
        n_real_slides.append(len(cluster_results[subj]['slides']))

n_orig         = len(subjects_orig)
y_orig         = np.array(y_orig_list, dtype=float)
n_two_slide    = sum(s == 2 for s in n_real_slides)
n_one_slide    = sum(s == 1 for s in n_real_slides)
print(f"  Subjects with outcome : {n_orig}  "
      f"(2-slide: {n_two_slide}, 1-slide duplicated: {n_one_slide})")
print(f"  eGFR  mean={y_orig.mean():.1f}  std={y_orig.std():.1f}  "
      f"min={y_orig.min():.1f}  max={y_orig.max():.1f}")

# ── Build design tensors for original n_orig subjects ──────────────────────────
print("\nBuilding design tensors...")
X_tepig_orig = np.stack(
    [cluster_results[s]['X_tensor'] for s in subjects_orig], axis=3
)  # (G, q, S, n_orig)

X_naive_orig = X_tepig_orig.mean(axis=(0, 2)).T  # (n_orig, q) — averaged over G and S

X_clusso_orig = np.zeros((G, q, n_orig))
for i, subj in enumerate(subjects_orig):
    res    = cluster_results[subj]
    labels = res['labels']
    c_avgs = res['cluster_avgs']
    n_tubs = res['n_tubules']
    n_tot  = sum(n_tubs)
    for g in range(G):
        n_g = np.array([(lbl == g + 1).sum() for lbl in labels])
        n_g_tot = n_g.sum()
        if n_g_tot > 0:
            mean_g = np.sum(
                [n_g[s] * c_avgs[s][g] for s in range(len(labels))], axis=0
            ) / n_g_tot
        else:
            mean_g = np.zeros(q)
        X_clusso_orig[g, :, i] = (n_g_tot / n_tot) * mean_g

# ── Bootstrap then 80/20 train/test split ─────────────────────────────────────
print(f"\nBootstrapping: {n_orig} → {N_BOOTSTRAP} subjects (resample with replacement)")
rng_boot = np.random.default_rng(RANDOM_SEED)
boot_idx = rng_boot.choice(n_orig, size=N_BOOTSTRAP, replace=True)

n        = N_BOOTSTRAP
y        = y_orig[boot_idx]
X_tepig  = X_tepig_orig[:, :, :, boot_idx]   # (G, q, S, n)
X_naive  = X_naive_orig[boot_idx, :]          # (n, q)
X_clusso = X_clusso_orig[:, :, boot_idx]      # (G, q, n)

print(f"  Bootstrapped eGFR  mean={y.mean():.1f}  std={y.std():.1f}")

print(f"\n80/20 train/test split on bootstrapped data (seed={RANDOM_SEED})")
rng_split = np.random.default_rng(RANDOM_SEED + 10)
shuffled  = rng_split.permutation(n)
n_test    = max(1, int(round(n * TEST_FRAC)))
n_train   = n - n_test
test_idx  = sorted(shuffled[:n_test].tolist())
train_idx = sorted(shuffled[n_test:].tolist())

print(f"  n_train={n_train}  n_test={n_test}")
print(f"  Train eGFR  mean={y[train_idx].mean():.1f}  std={y[train_idx].std():.1f}")
print(f"  Test  eGFR  mean={y[test_idx].mean():.1f}  std={y[test_idx].std():.1f}")
print(f"  Lambda grid: {LAM_GRID[0]:.2f} to {LAM_GRID[-1]:.2f}  ({len(LAM_GRID)} values)")

# ── Helpers ────────────────────────────────────────────────────────────────────

def _l1_normalize(v):
    """L1-normalize; sign convention: largest-magnitude entry is positive."""
    s = np.sum(np.abs(v))
    if s > 0:
        sig = np.sign(v[np.argmax(np.abs(v))])
        return sig * v / s
    return v


def _make_folds(indices, rng, k=5):
    """Split a list of indices into k folds."""
    idx  = rng.permutation(indices).tolist()
    size = len(idx) // k
    folds = [sorted(idx[i * size:(i + 1) * size]) for i in range(k - 1)]
    folds.append(sorted(idx[(k - 1) * size:]))
    return folds


def proxgrad_fit(X, y, lam, max_iter=2000, tol=1e-6):
    """Proximal gradient + group lasso on (G, q, S, n) tensor."""
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
    """5-fold CV to select lambda, refit with M_INIT random starts. X_mat: (G, q, n).
    Mainfunction_albet drops the intercept from returned alpha/beta, so we shift
    predictions to match mean(y) — matching the intercept correction in simulation_synthetic.py.
    Returns intercept so it can be applied to held-out test predictions."""
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
        yp_raw = np.array([a_ @ X_mat[:, :, i] @ b_ for i in range(n_loc)])
        yp     = yp_raw + (np.mean(y) - np.mean(yp_raw))
        mse    = float(np.mean((y - yp) ** 2))
        if mse < best_mse:
            best_mse = mse; best_b = b_; best_a = a_

    y_pred_raw  = np.array([best_a @ X_mat[:, :, i] @ best_b for i in range(n_loc)])
    intercept   = float(np.mean(y) - np.mean(y_pred_raw))
    y_pred      = y_pred_raw + intercept
    return best_a, best_b, best_lam, best_cv_mse, y_pred, intercept


def naive_lasso_fit(X_tr, y_tr, X_te=None):
    """
    X_tr: (n_train, q) — training set, averaged over G and S.
    X_te: (n_test, q)  — test set (optional).
    Standardize X, center y, LassoCV with 100-point glmnet-style lambda path.
    Returns cv_mse, beta (q,), y_pred_train (n_train,), y_pred_test (n_test,) or None
    """
    n_loc   = X_tr.shape[0]
    X_mu    = X_tr.mean(axis=0)
    X_c     = X_tr - X_mu
    X_scale = np.sqrt((X_c ** 2).mean(axis=0))
    X_scale = np.where(X_scale > 1e-10, X_scale, 1.0)
    X_std   = X_c / X_scale
    y_mu    = y_tr.mean(); y_c = y_tr - y_mu
    lambda_max = float(np.max(np.abs(X_std.T @ y_c)) / n_loc)
    lambdas_R  = np.exp(np.linspace(np.log(lambda_max),
                                    np.log(lambda_max * 1e-4), 100))
    model = LassoCV(cv=5, alphas=lambdas_R / 2.0,
                    fit_intercept=False, max_iter=100_000)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(X_std, y_c)
    beta        = model.coef_ / X_scale
    intercept   = float(y_mu - X_mu @ beta)
    y_pred_tr   = intercept + X_tr @ beta
    cv_mse      = float(model.mse_path_[
        list(model.alphas_).index(model.alpha_)].mean())
    y_pred_te   = (intercept + X_te @ beta) if X_te is not None else None
    return cv_mse, beta, y_pred_tr, y_pred_te


# ── Fit estimators ─────────────────────────────────────────────────────────────
rng    = np.random.default_rng(RANDOM_SEED + 1)
# CV folds are built over training indices only
folds  = _make_folds(train_idx, rng)
results = {}

# ── TEPIG ──────────────────────────────────────────────────────────────────────
print("\nFitting TEPIG...")
d_full    = G * q * S
cv_mse_tg = []
for lam in LAM_GRID:
    fold_mse = []
    for k in range(5):
        te_cv = folds[k]
        tr_cv = [i for fold in folds for i in fold if fold is not folds[k]]
        _, B      = proxgrad_fit(X_tepig[:, :, :, tr_cv], y[tr_cv], lam)
        X_te      = X_tepig[:, :, :, te_cv]
        X_te_mean = X_te.mean(axis=3, keepdims=True)
        y_te_c    = y[te_cv] - y[te_cv].mean()
        X_te_c    = (X_te - X_te_mean).reshape(d_full, len(te_cv)).T
        ypred_c   = X_te_c @ B.reshape(-1)
        fold_mse.append(float(np.mean((y_te_c - ypred_c) ** 2)))
    cv_mse_tg.append(float(np.mean(fold_mse)))

best_lam_tg = LAM_GRID[int(np.argmin(cv_mse_tg))]
best_cv_tg  = float(np.min(cv_mse_tg))

# Refit on full training set
ic_tg, B_tg = proxgrad_fit(X_tepig[:, :, :, train_idx], y[train_idx], best_lam_tg)

# Beta for reporting: mean of (G, S) block → (q,) for dimensional comparability
beta_tg_raw = B_tg.mean(axis=(0, 2))   # (q,)

# Predict using centered approach (matches simulation_synthetic.py / slasso_mse):
#   center X by training mean, add back training mean of y
X_tr_mean     = X_tepig[:, :, :, train_idx].mean(axis=3, keepdims=True)  # (G,q,S,1)
y_tr_mean     = float(np.mean(y[train_idx]))
y_pred_tg_tr  = (X_tepig[:, :, :, train_idx] - X_tr_mean).reshape(d_full, n_train).T @ B_tg.reshape(-1) + y_tr_mean
y_pred_tg_test = (X_tepig[:, :, :, test_idx] - X_tr_mean).reshape(d_full, n_test).T @ B_tg.reshape(-1) + y_tr_mean

# Selection: Frobenius norm + adaptive threshold (computed on training residuals)
sigma_hat_tg  = float(np.std(y[train_idx] - y_pred_tg_tr, ddof=1))
tau_tg        = sigma_hat_tg * np.sqrt(2.0 * np.log(q) / n_train)
block_norms   = np.sqrt(np.sum(B_tg ** 2, axis=(0, 2)))   # (q,)
beta_tg_raw[block_norms < tau_tg] = 0.0

sel_idx_tg  = [j for j in range(q) if abs(beta_tg_raw[j]) > 1e-6]
sel_tg      = [features[j] for j in sel_idx_tg]
beta_tg_sel = beta_tg_raw[sel_idx_tg]
beta_tg_l1  = _l1_normalize(beta_tg_sel) if len(sel_idx_tg) > 0 else np.array([])

y_pred_tg_train = y_pred_tg_tr
train_mse_tg = float(np.mean((y[train_idx] - y_pred_tg_train) ** 2))
test_mse_tg  = float(np.mean((y[test_idx]  - y_pred_tg_test)  ** 2))

results['TEPIG'] = {
    'lambda': best_lam_tg, 'cv_mse': best_cv_tg,
    'train_mse': train_mse_tg, 'test_mse': test_mse_tg,
    'B': B_tg, 'beta_raw': beta_tg_raw, 'beta_l1': beta_tg_l1,
    'selected': sel_tg, 'selected_idx': sel_idx_tg,
    'y_pred_train': y_pred_tg_train, 'y_pred_test': y_pred_tg_test,
}
print(f"  lambda={best_lam_tg:.2f}  CV-MSE={best_cv_tg:.2f}  "
      f"train MSE={train_mse_tg:.2f}  test MSE={test_mse_tg:.2f}  "
      f"features selected={len(sel_tg)}")

# ── CLUSSO ─────────────────────────────────────────────────────────────────────
print("\nFitting CLUSSO...")
a_cl, b_cl, lam_cl, cv_mse_cl, y_pred_cl_tr, intercept_cl = clusso_select_and_fit(
    X_clusso[:, :, train_idx], y[train_idx], rng, LAM_GRID)

sel_idx_cl = [j for j in range(q) if abs(b_cl[j]) > 1e-6]
sel_cl     = [features[j] for j in sel_idx_cl]
beta_cl_l1 = _l1_normalize(b_cl[sel_idx_cl]) if len(sel_idx_cl) > 0 else np.array([])

y_pred_cl_test_raw = np.array([a_cl @ X_clusso[:, :, i] @ b_cl for i in test_idx])
y_pred_cl_test     = y_pred_cl_test_raw + intercept_cl
train_mse_cl   = float(np.mean((y[train_idx] - y_pred_cl_tr)   ** 2))
test_mse_cl    = float(np.mean((y[test_idx]  - y_pred_cl_test) ** 2))

results['clusso'] = {
    'lambda': lam_cl, 'cv_mse': cv_mse_cl,
    'train_mse': train_mse_cl, 'test_mse': test_mse_cl,
    'alpha': a_cl, 'beta': b_cl, 'beta_l1': beta_cl_l1,
    'selected': sel_cl, 'selected_idx': sel_idx_cl,
    'y_pred_train': y_pred_cl_tr, 'y_pred_test': y_pred_cl_test,
}
print(f"  lambda={lam_cl:.2f}  CV-MSE={cv_mse_cl:.2f}  "
      f"train MSE={train_mse_cl:.2f}  test MSE={test_mse_cl:.2f}  "
      f"features selected={len(sel_cl)}")
print(f"  alpha (cluster weights): {np.round(a_cl, 3)}")

# ── Naive ──────────────────────────────────────────────────────────────────────
print("\nFitting naive...")
cv_mse_nv, b_nv, y_pred_nv_tr, y_pred_nv_te = naive_lasso_fit(
    X_naive[train_idx], y[train_idx], X_te=X_naive[test_idx])

sel_idx_nv = [j for j in range(q) if abs(b_nv[j]) > 1e-6]
sel_nv     = [features[j] for j in sel_idx_nv]
beta_nv_l1 = _l1_normalize(b_nv[sel_idx_nv]) if len(sel_idx_nv) > 0 else np.array([])
train_mse_nv = float(np.mean((y[train_idx] - y_pred_nv_tr) ** 2))
test_mse_nv  = float(np.mean((y[test_idx]  - y_pred_nv_te) ** 2))

results['naive'] = {
    'cv_mse': cv_mse_nv,
    'train_mse': train_mse_nv, 'test_mse': test_mse_nv,
    'beta': b_nv, 'beta_l1': beta_nv_l1,
    'selected': sel_nv, 'selected_idx': sel_idx_nv,
    'y_pred_train': y_pred_nv_tr, 'y_pred_test': y_pred_nv_te,
}
print(f"  CV-MSE={cv_mse_nv:.2f}  train MSE={train_mse_nv:.2f}  "
      f"test MSE={test_mse_nv:.2f}  features selected={len(sel_nv)}")

# ── Save results ───────────────────────────────────────────────────────────────
results['meta'] = {
    'subjects_orig': subjects_orig, 'y_orig': y_orig,
    'n_real_slides': n_real_slides,
    'boot_idx': boot_idx,
    'train_idx': train_idx, 'test_idx': test_idx,
    'y': y, 'features': features,
    'n_orig': n_orig, 'n_bootstrap': N_BOOTSTRAP,
    'n_train': n_train, 'n_test': n_test,
    'q': q, 'G': G, 'S': S,
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
    f.write(f"Original subjects (n_orig) : {n_orig}  "
            f"(2-slide: {n_two_slide}, 1-slide duplicated: {n_one_slide})\n")
    f.write(f"Bootstrapped subjects (n)  : {n}  (resampled with replacement)\n")
    f.write(f"Train / Test               : {n_train} / {n_test}  (80/20 split on bootstrapped)\n")
    f.write(f"Features (q)               : {q}\n")
    f.write(f"Outcome                    : eGFR_CKD_EPI_12M (1-year eGFR)\n")
    f.write(f"eGFR  mean={y.mean():.1f}  std={y.std():.1f}  "
            f"min={y.min():.1f}  max={y.max():.1f}\n\n")

    f.write(f"{'Estimator':<10} {'Lambda':>8} {'CV-MSE':>10} "
            f"{'Train MSE':>11} {'Test MSE':>10} {'N selected':>12}\n")
    f.write("-" * 65 + "\n")
    for est in ['TEPIG', 'clusso', 'naive']:
        r = results[est]
        lam_str = f"{r['lambda']:>8.2f}" if 'lambda' in r else f"{'N/A':>8}"
        f.write(f"  {est:<8} {lam_str} {r['cv_mse']:>10.4f} "
                f"{r['train_mse']:>11.4f} {r['test_mse']:>10.4f} "
                f"{len(r['selected']):>12}\n")

    f.write("\n")
    for est in ['TEPIG', 'clusso', 'naive']:
        r = results[est]
        f.write("-" * 70 + "\n")
        f.write(f"Estimator: {est}\n")
        if 'lambda' in r:
            f.write(f"  Lambda        : {r['lambda']:.4f}\n")
        f.write(f"  CV-MSE (train): {r['cv_mse']:.4f}\n")
        f.write(f"  Train MSE     : {r['train_mse']:.4f}\n")
        f.write(f"  Test MSE      : {r['test_mse']:.4f}\n")
        if 'alpha' in r:
            f.write(f"  Alpha (cluster weights): {np.round(r['alpha'], 4).tolist()}\n")
        sel  = r['selected']
        bl1  = r['beta_l1']
        f.write(f"  Features selected ({len(sel)}) — L1-normalized coefficients:\n")
        for feat, coef in zip(sel, bl1):
            f.write(f"    {coef:+.4f}  {feat}\n")
        if len(sel) == 0:
            f.write(f"    (none)\n")
        f.write("\n")

print(f"  Saved: {summary_path}")
print("\nDone.")

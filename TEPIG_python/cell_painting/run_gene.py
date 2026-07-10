"""
run_gene.py
-----------
Apply TEPIG, CLUSSO, and naive estimators to the LINCS-Pilot1 Cell Painting
tensor (channel as the third mode), predicting one L1000 gene's expression
(or the cell-cycle gene-set score).

Tensor per observation:  compartment (G=3) x feature-core (q) x channel (S=5).
Observation unit      :  one compound at a SINGLE dose level (i.i.d. subjects).
Outcome               :  that subject's L1000 gene expression, standardized.

We analyze one dose level at a time (--dose 1..6). Within a dose level each
compound contributes exactly one profile, so the ~1,400 subjects are i.i.d.
(unlike pooling all doses, where a compound's 6 doses are correlated). Run all
six dose levels to see robustness across doses.

Estimators (mirror TEPIG_python/real_data_analysis.py):
  TEPIG   : full (G, q, S, n) tensor, group lasso over the q feature-cores
  CLUSSO  : (G, q, n) = tensor averaged over channels
  naive   : (n, q)    = tensor averaged over compartments and channels

Metrics: test MSE (primary, matches the simulations) and R^2, on a plain 80/20
split (subjects are i.i.d., so no grouping needed).

Usage:
    python run_gene.py --gene GLRX --dose 6
    python run_gene.py --gene CELLCYCLE --dose 6   # cell-cycle gene-set score
"""

import os
import sys
import argparse
import pickle
import warnings
import numpy as np
warnings.filterwarnings('ignore')

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))     # estimator deps in TEPIG_python

from sklearn.linear_model import LassoCV                    # noqa: E402
from Mainfunction_albet import Mainfunction_albet           # noqa: E402
from SLasso_MSE import lambda_CV_mse                         # noqa: E402
from genes import CELL_CYCLE                                 # noqa: E402

CACHE_PKL = os.path.join(_HERE, 'cache', 'cp_lincs_tensor.pkl')
RES_DIR   = os.path.join(_HERE, 'results')
os.makedirs(RES_DIR, exist_ok=True)

RANDOM_SEED = 42
TEST_FRAC   = 0.2
M_INIT      = 8                                   # CLUSSO random restarts
N_LAM_TEPIG = 12
CLUSSO_LAM_GRID = np.geomspace(1.0, 0.005, 9)     # glmnet scale, standardized y


# ── Helpers (ported from real_data_analysis.py) ─────────────────────────────
def _l1_normalize(v):
    s = np.sum(np.abs(v))
    if s > 0:
        return np.sign(v[np.argmax(np.abs(v))]) * v / s
    return v


def _spectral_norm(A, iters=100, seed=0):
    """Largest singular value of A via power iteration (fast vs full SVD)."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(A.shape[1]); v /= np.linalg.norm(v)
    for _ in range(iters):
        v = A.T @ (A @ v)
        nv = np.linalg.norm(v)
        if nv == 0:
            return 0.0
        v /= nv
    return float(np.linalg.norm(A @ v))


def proxgrad_fit(X, y, lam, max_iter=2000, tol=1e-6):
    """FISTA + group lasso on (G, q, S, n) tensor. Group = the (G,S) block of
    each feature-core j; identical to real_data_analysis.py."""
    G, q, S, n_tr = X.shape
    d = G * q * S
    gnorms = np.array([
        float(np.sqrt(np.mean(np.sum(X[:, j, :, :] ** 2, axis=(0, 1)))))
        for j in range(q)])
    gnorms = np.where(gnorms > 1e-10, gnorms, 1.0)
    Xs     = np.clip(X / gnorms[np.newaxis, :, np.newaxis, np.newaxis], -1e3, 1e3)
    X_flat = Xs.reshape(d, n_tr).T

    eta       = 1.0 / (_spectral_norm(X_flat) ** 2 / n_tr)
    threshold = eta * lam
    x, z, t   = np.zeros(d), np.zeros(d), 1.0

    for _ in range(max_iter):
        x_old  = x.copy()
        pred_z = X_flat @ z
        ic     = float(np.mean(y - pred_z))
        grad   = -(1.0 / n_tr) * (X_flat.T @ (y - ic - pred_z))
        if not np.isfinite(grad).all():
            break
        v = (z - eta * grad).reshape(G, q, S)
        for j in range(q):
            blk  = v[:, j, :]
            nrm  = float(np.linalg.norm(blk))
            v[:, j, :] = (1.0 - threshold / nrm) * blk if nrm > threshold else 0.0
        x = v.reshape(-1)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z = x + ((t - 1.0) / t_new) * (x - x_old)
        t = t_new
        if np.max(np.abs(x - x_old)) < tol:
            break

    B = x.reshape(G, q, S) / gnorms[np.newaxis, :, np.newaxis]
    ic = float(np.mean(y - X.reshape(d, n_tr).T @ B.reshape(-1)))
    return ic, B


def tepig_lambda_grid(X, y, num=N_LAM_TEPIG):
    """Data-driven grid from lambda_max (smallest lam zeroing all blocks) down to
    lambda_max * 1e-3. Guarantees an interior CV optimum."""
    G, q, S, n = X.shape
    gnorms = np.array([
        float(np.sqrt(np.mean(np.sum(X[:, j, :, :] ** 2, axis=(0, 1)))))
        for j in range(q)])
    gnorms = np.where(gnorms > 1e-10, gnorms, 1.0)
    Xs = X / gnorms[np.newaxis, :, np.newaxis, np.newaxis]
    yc = y - y.mean()
    lam_max = max(
        float(np.linalg.norm((Xs[:, j, :, :] * yc[np.newaxis, np.newaxis, :]).sum(axis=2) / n))
        for j in range(q))
    return lam_max * np.geomspace(1.0, 1e-3, num)


def clusso_select_and_fit(X_mat, y, rng, lam_grid):
    """5-fold CV to select lambda, refit with M_INIT random starts. X_mat (G,q,n)."""
    G_loc, q_loc, n_loc = X_mat.shape
    cv = [lambda_CV_mse(X_mat, y, np.ones(G_loc) / G_loc, np.ones(q_loc) / q_loc, lam)
          for lam in lam_grid]
    best_lam = lam_grid[int(np.argmin(cv))]
    best_mse, best_b, best_a = np.inf, np.zeros(q_loc), np.ones(G_loc) / G_loc
    for _ in range(M_INIT):
        res = Mainfunction_albet(X_mat, y, rng.dirichlet(np.ones(G_loc)),
                                 rng.uniform(-1, 1, q_loc), best_lam)
        a_, b_ = res['alpha'], res['bet']
        yp = np.array([a_ @ X_mat[:, :, i] @ b_ for i in range(n_loc)])
        yp = yp + (np.mean(y) - np.mean(yp))
        m = float(np.mean((y - yp) ** 2))
        if m < best_mse:
            best_mse, best_b, best_a = m, b_, a_
    yp_raw = np.array([best_a @ X_mat[:, :, i] @ best_b for i in range(n_loc)])
    ic = float(np.mean(y) - np.mean(yp_raw))
    return best_a, best_b, best_lam, float(np.min(cv)), yp_raw + ic, ic, cv


def naive_lasso_fit(X_tr, y_tr, X_te):
    """LassoCV on (n, q) profiles averaged over compartments and channels."""
    n_loc   = X_tr.shape[0]
    X_mu    = X_tr.mean(axis=0)
    X_c     = X_tr - X_mu
    X_scale = np.sqrt((X_c ** 2).mean(axis=0))
    X_scale = np.where(X_scale > 1e-10, X_scale, 1.0)
    X_std   = X_c / X_scale
    y_mu    = y_tr.mean(); y_c = y_tr - y_mu
    lam_max = float(np.max(np.abs(X_std.T @ y_c)) / n_loc)
    lambdas = np.exp(np.linspace(np.log(lam_max), np.log(lam_max * 1e-4), 100))
    model = LassoCV(cv=5, alphas=lambdas / 2.0, fit_intercept=False, max_iter=100_000)
    model.fit(X_std, y_c)
    beta = model.coef_ / X_scale
    ic   = float(y_mu - X_mu @ beta)
    cv   = float(model.mse_path_[list(model.alphas_).index(model.alpha_)].mean())
    return cv, beta, ic + X_tr @ beta, ic + X_te @ beta


def r2(y_true, y_pred):
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / ss_tot if ss_tot > 0 else float('nan')


# ── Plain 80/20 split / folds (subjects are i.i.d. within a dose level) ──────
def plain_split(n, test_frac, seed):
    idx = np.random.default_rng(seed).permutation(n)
    n_test = int(round(n * test_frac))
    return np.sort(idx[n_test:]), np.sort(idx[:n_test])


def plain_folds(train_idx, seed, k=5):
    idx = np.random.default_rng(seed).permutation(train_idx)
    return [sorted(g.tolist()) for g in np.array_split(idx, k)]


def build_outcome(gene, expr, probes, sym2probe):
    """Return raw outcome vector (n,) for a single gene or the cell-cycle set."""
    if gene == 'CELLCYCLE':
        cols = [probes.index(sym2probe[g]) for g in CELL_CYCLE if g in sym2probe]
        Z = expr[:, cols]
        Z = (Z - Z.mean(axis=0)) / np.where(Z.std(axis=0) > 0, Z.std(axis=0), 1.0)
        return Z.mean(axis=1), len(cols)          # mean of z-scored member genes
    if gene not in sym2probe:
        raise SystemExit(f"gene {gene} not in L1000 panel")
    return expr[:, probes.index(sym2probe[gene])].astype(float), 1


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gene', default='GLRX')
    ap.add_argument('--dose', type=int, default=6,
                    help='dose rank 1..6 (one dose level = i.i.d. subjects)')
    args = ap.parse_args()
    gene, dose = args.gene, args.dose

    cache = pickle.load(open(CACHE_PKL, 'rb'))
    G, q, S = cache['G'], cache['q'], cache['S']
    features = cache['features']

    # ── Filter to one dose level -> one row per compound (i.i.d. subjects) ────
    mask  = cache['obs_dose_rank'] == dose
    X_all = cache['X'][:, :, :, mask]              # (G, q, S, n)
    expr  = cache['expr'][mask]                    # (n, 978)
    n = X_all.shape[3]

    y_raw, n_members = build_outcome(gene, expr, cache['probes'], cache['sym2probe'])
    print(f"gene={gene}  dose_rank={dose}  n_subjects={n}  (G={G}, q={q}, S={S})"
          + (f"  set_size={n_members}" if gene == 'CELLCYCLE' else ''))

    # plain 80/20 split (subjects are i.i.d.: one compound each at this dose)
    train_idx, test_idx = plain_split(n, TEST_FRAC, RANDOM_SEED)
    n_train, n_test = len(train_idx), len(test_idx)
    y_mu, y_sd = y_raw[train_idx].mean(), y_raw[train_idx].std()
    y = (y_raw - y_mu) / (y_sd if y_sd > 0 else 1.0)
    print(f"  split: n_train={n_train}  n_test={n_test}")

    X_clusso = X_all.mean(axis=2)                  # (G, q, n) avg over channels
    X_naive  = X_all.mean(axis=(0, 2)).T           # (n, q) avg over compartments+channels
    folds = plain_folds(train_idx, RANDOM_SEED + 1)
    results = {}

    # ── TEPIG ────────────────────────────────────────────────────────────────
    print("Fitting TEPIG...")
    LAM = tepig_lambda_grid(X_all[:, :, :, train_idx], y[train_idx])
    d = G * q * S
    cv_tg = []
    for lam in LAM:
        fm = []
        for kk in range(5):
            te = folds[kk]
            tr = [i for f in folds for i in f if f is not folds[kk]]
            _, B = proxgrad_fit(X_all[:, :, :, tr], y[tr], lam)
            Xte = X_all[:, :, :, te]
            Xc  = (Xte - Xte.mean(axis=3, keepdims=True)).reshape(d, len(te)).T
            yc  = y[te] - y[te].mean()
            fm.append(float(np.mean((yc - Xc @ B.reshape(-1)) ** 2)))
        cv_tg.append(float(np.mean(fm)))
    best_lam = LAM[int(np.argmin(cv_tg))]
    _, B_tg = proxgrad_fit(X_all[:, :, :, train_idx], y[train_idx], best_lam)
    Xtr_mean = X_all[:, :, :, train_idx].mean(axis=3, keepdims=True)
    ytr_mean = float(np.mean(y[train_idx]))
    pred_tr = (X_all[:, :, :, train_idx] - Xtr_mean).reshape(d, n_train).T @ B_tg.reshape(-1) + ytr_mean
    pred_te = (X_all[:, :, :, test_idx]  - Xtr_mean).reshape(d, n_test).T  @ B_tg.reshape(-1) + ytr_mean
    sigma = float(np.std(y[train_idx] - pred_tr, ddof=1))
    tau   = sigma * np.sqrt(2.0 * np.log(q) / n_train)
    bn    = np.sqrt(np.sum(B_tg ** 2, axis=(0, 2)))
    sel_tg = [features[j] for j in range(q) if bn[j] > tau]
    results['TEPIG'] = {
        'lambda': float(best_lam), 'cv_curve': (LAM.tolist(), cv_tg),
        'train_mse': float(np.mean((y[train_idx] - pred_tr) ** 2)),
        'test_mse':  float(np.mean((y[test_idx]  - pred_te) ** 2)),
        'test_r2': r2(y[test_idx], pred_te),
        'n_fitted': int((bn > 1e-6).sum()), 'n_selected': len(sel_tg), 'selected': sel_tg}
    r = results['TEPIG']
    print(f"  lambda={best_lam:.4g} (argmin {int(np.argmin(cv_tg))}/{len(LAM)-1})  "
          f"test MSE={r['test_mse']:.3f}  R2={r['test_r2']:.3f}  "
          f"fitted={r['n_fitted']} selected={r['n_selected']}")

    # ── CLUSSO ───────────────────────────────────────────────────────────────
    print("Fitting CLUSSO...")
    rng = np.random.default_rng(RANDOM_SEED + 2)
    a_cl, b_cl, lam_cl, cv_cl, pred_cl_tr, ic_cl, cl_curve = clusso_select_and_fit(
        X_clusso[:, :, train_idx], y[train_idx], rng, CLUSSO_LAM_GRID)
    pred_cl_te = np.array([a_cl @ X_clusso[:, :, i] @ b_cl for i in test_idx]) + ic_cl
    sel_cl = [features[j] for j in range(q) if abs(b_cl[j]) > 1e-6]
    results['clusso'] = {
        'lambda': float(lam_cl), 'alpha': a_cl.tolist(),
        'train_mse': float(np.mean((y[train_idx] - pred_cl_tr) ** 2)),
        'test_mse':  float(np.mean((y[test_idx]  - pred_cl_te) ** 2)),
        'test_r2': r2(y[test_idx], pred_cl_te),
        'n_selected': len(sel_cl), 'selected': sel_cl}
    r = results['clusso']
    print(f"  lambda={lam_cl:.4g}  test MSE={r['test_mse']:.3f}  R2={r['test_r2']:.3f}  "
          f"selected={r['n_selected']}  alpha={np.round(a_cl,3)}")

    # ── naive ────────────────────────────────────────────────────────────────
    print("Fitting naive...")
    cv_nv, b_nv, pred_nv_tr, pred_nv_te = naive_lasso_fit(
        X_naive[train_idx], y[train_idx], X_naive[test_idx])
    sel_nv = [features[j] for j in range(q) if abs(b_nv[j]) > 1e-6]
    results['naive'] = {
        'train_mse': float(np.mean((y[train_idx] - pred_nv_tr) ** 2)),
        'test_mse':  float(np.mean((y[test_idx]  - pred_nv_te) ** 2)),
        'test_r2': r2(y[test_idx], pred_nv_te),
        'n_selected': len(sel_nv), 'selected': sel_nv}
    r = results['naive']
    print(f"  test MSE={r['test_mse']:.3f}  R2={r['test_r2']:.3f}  selected={r['n_selected']}")

    print("\n" + "=" * 60)
    print(f"SUMMARY  gene={gene}  dose={dose}  n={n} (train {n_train}/test {n_test})  q={q}")
    print(f"{'method':<8}{'test MSE':>10}{'test R2':>10}{'#sel':>7}")
    for m in ['TEPIG', 'clusso', 'naive']:
        rr = results[m]
        print(f"{m:<8}{rr['test_mse']:>10.3f}{rr['test_r2']:>10.3f}{rr['n_selected']:>7}")

    out = os.path.join(RES_DIR, f'gene_{gene}_dose{dose}.pkl')
    pickle.dump({'gene': gene, 'dose': dose, 'results': results, 'n': n,
                 'n_train': n_train, 'n_test': n_test, 'q': q, 'G': G, 'S': S}, open(out, 'wb'))
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()

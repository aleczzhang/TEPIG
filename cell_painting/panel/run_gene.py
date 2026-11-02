"""
run_gene.py
-----------
Estimator wrappers shared by the Cell Painting pipeline (stability_check.one_seed,
run_panel.py, scan_genes.py ...) plus a legacy driver for the LINCS-Pilot1 well-level
cache (one dose level at a time).

Estimators
  TEPIG   : core/tepig.py  -- group lasso on the full (G, q, S, n) tensor (THE solver;
            this file holds no copy of it any more)
  CLUSSO  : Mainfunction_albet (Jeremy's structured lasso) on (G, q, n) = tensor
            averaged over sites, lambda by CV with random restarts
  naive   : sklearn LassoCV on (n, q) profiles averaged over clusters and sites

All three take the SAME inner folds (lists of global well indices into the training
set) so lambda is tuned on identical held-out wells; with platemap-grouped folds no
compound is both in an inner training and an inner validation fold.

Usage (legacy LINCS run):
    python run_gene.py --gene GLRX --dose 6
"""

import os
import sys
import argparse
import pickle
import warnings
import numpy as np
warnings.filterwarnings('ignore')


from sklearn.linear_model import LassoCV                    # noqa: E402
_HERE = os.path.dirname(os.path.abspath(__file__))
_CP = os.path.dirname(_HERE)                       # cell_painting/
_ROOT = os.path.dirname(_CP)                       # repository root
sys.path[:0] = [_ROOT, _CP, os.path.join(_CP, 'panel'), os.path.join(_CP, 'screen')]
from tepig.clusso.Mainfunction_albet import Mainfunction_albet           # noqa: E402
from genes import CELL_CYCLE                                 # noqa: E402
from tepig import tepig as T                                            # noqa: E402

CACHE_PKL = os.path.join(_CP, 'cache', 'cp_lincs_tensor.pkl')
RES_DIR   = os.path.join(_CP, 'results')
os.makedirs(RES_DIR, exist_ok=True)

RANDOM_SEED = 42
TEST_FRAC   = 0.2
M_INIT      = 8                                   # CLUSSO random restarts (final fit)
N_CV_INIT   = 4                                   # CLUSSO random restarts per CV fold
N_LAM_TEPIG = 15
LAM_RATIO   = 1e-4                                # TEPIG grid: lambda_max -> ratio * lambda_max
CLUSSO_LAM_GRID = np.geomspace(1.0, 0.005, 9)     # glmnet scale, standardized y
# Selection rule for TEPIG's reported set (see core/tepig.py select_features). Every
# block statistic is stored per split, so other rules can be applied post hoc.
SELECT_RULE = 'col_std_refit'
SELECT_GROUP_TERM = True                          # tau uses (G*S + 2 log q)/n


# ── Helpers ─────────────────────────────────────────────────────────────────
def _l1_normalize(v):
    s = np.sum(np.abs(v))
    if s > 0:
        return np.sign(v[np.argmax(np.abs(v))]) * v / s
    return v


def proxgrad_fit(X, y, lam, **kw):
    """Backward-compatible wrapper: (intercept, B) from the shared core solver."""
    f = T.fit_one(X, y, lam, **kw)
    return f.intercept, f.B


def tepig_lambda_grid(X, y, num=N_LAM_TEPIG, ratio=LAM_RATIO):
    return T.lambda_grid(X, y, num=num, ratio=ratio)


def tepig_select_and_fit(X, y, folds, num=N_LAM_TEPIG, ratio=LAM_RATIO, verbose=False):
    """CV over the data-driven grid on the union of `folds`; returns the core's
    result dict (fit, grid, curve, iteration counts ...)."""
    return T.cv_select(X, y, folds, num=num, ratio=ratio, verbose=verbose)


def _clusso_fit_predict(X_full, y, tr, te, lam, a0, b0):
    """One Mainfunction fit on tr; intercept-corrected test MSE and (alpha, beta)."""
    res = Mainfunction_albet(X_full[:, :, tr], y[tr], a0, b0, lam)
    a_, b_ = res['alpha'], res['bet']
    ptr = np.array([a_ @ X_full[:, :, i] @ b_ for i in tr])
    ic  = float(y[tr].mean() - ptr.mean())
    pte = np.array([a_ @ X_full[:, :, i] @ b_ for i in te]) + ic
    return float(np.mean((y[te] - pte) ** 2)), a_, b_


def clusso_select_and_fit(X_full, y, folds, train_idx, rng, lam_grid, n_cv_init=N_CV_INIT):
    """CLUSSO with multiple random restarts in BOTH the CV lambda-selection and
    the final fit. Its bilinear objective is non-convex, so a single random start
    can land in a degenerate optimum (all-zero beta -> R^2 ~ 0); taking the best of
    several restarts per fold stabilizes selection. Uses the same folds as TEPIG."""
    G_loc, q_loc = X_full.shape[0], X_full.shape[1]
    cv = []
    for lam in lam_grid:
        fmse = []
        for k in range(len(folds)):
            te = folds[k]
            tr = [i for f in folds for i in f if f is not folds[k]]
            best = np.inf
            for _ in range(n_cv_init):
                m, _, _ = _clusso_fit_predict(
                    X_full, y, tr, te, lam,
                    rng.dirichlet(np.ones(G_loc)), rng.uniform(-1, 1, q_loc))
                best = min(best, m)
            fmse.append(best)
        cv.append(float(np.mean(fmse)))
    best_lam = lam_grid[int(np.argmin(cv))]

    # final refit on the full training set, best of M_INIT restarts (by train MSE)
    best_tr, best_a, best_b = np.inf, np.ones(G_loc) / G_loc, np.zeros(q_loc)
    for _ in range(M_INIT):
        m, a_, b_ = _clusso_fit_predict(
            X_full, y, train_idx, train_idx, best_lam,
            rng.dirichlet(np.ones(G_loc)), rng.uniform(-1, 1, q_loc))
        if m < best_tr:
            best_tr, best_a, best_b = m, a_, b_
    ptr = np.array([best_a @ X_full[:, :, i] @ best_b for i in train_idx])
    ic  = float(y[train_idx].mean() - ptr.mean())
    return best_a, best_b, best_lam, float(np.min(cv)), ic, cv


def naive_lasso_fit(X_tr, y_tr, X_te, cv=5):
    """LassoCV on (n, q) profiles averaged over compartments and channels.
    `cv` is 5 (random) or a list of (train_pos, test_pos) pairs relative to X_tr
    (tepig.folds_to_positions) so the naive lasso shares TEPIG/CLUSSO's folds."""
    n_loc   = X_tr.shape[0]
    X_mu    = X_tr.mean(axis=0)
    X_c     = X_tr - X_mu
    X_scale = np.sqrt((X_c ** 2).mean(axis=0))
    X_scale = np.where(X_scale > 1e-10, X_scale, 1.0)
    X_std   = X_c / X_scale
    y_mu    = y_tr.mean(); y_c = y_tr - y_mu
    lam_max = float(np.max(np.abs(X_std.T @ y_c)) / n_loc)
    lambdas = np.exp(np.linspace(np.log(lam_max), np.log(lam_max * 1e-4), 100))
    model = LassoCV(cv=cv, alphas=lambdas / 2.0, fit_intercept=False, max_iter=10_000)
    model.fit(X_std, y_c)
    beta = model.coef_ / X_scale
    ic   = float(y_mu - X_mu @ beta)
    cvm  = float(model.mse_path_[list(model.alphas_).index(model.alpha_)].mean())
    return cvm, beta, ic + X_tr @ beta, ic + X_te @ beta


def r2(y_true, y_pred):
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / ss_tot if ss_tot > 0 else float('nan')


# ── Splits / folds ──────────────────────────────────────────────────────────
def plain_split(n, test_frac, seed):
    idx = np.random.default_rng(seed).permutation(n)
    n_test = int(round(n * test_frac))
    return np.sort(idx[n_test:]), np.sort(idx[:n_test])


def group_split(groups, test_frac, seed):
    """Hold out test_frac of the GROUPS (e.g. platemaps), not of the wells:
    the test wells' compounds are then entirely unseen during training."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    k = max(1, int(round(len(uniq) * test_frac)))
    te_g = np.random.default_rng(seed).choice(uniq, size=k, replace=False)
    m = np.isin(groups, te_g)
    return np.where(~m)[0], np.where(m)[0]


def plain_folds(train_idx, seed, k=5):
    return T.plain_folds(train_idx, seed, k)


def group_folds(train_idx, groups, seed, k=5):
    return T.group_folds(train_idx, groups, seed, k)


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


# ── Legacy LINCS-Pilot1 driver ──────────────────────────────────────────────
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
    mask  = cache['obs_dose_rank'] == dose
    X_all = cache['X'][:, :, :, mask]
    expr  = cache['expr'][mask]
    n = X_all.shape[3]
    y_raw, _ = build_outcome(gene, expr, cache['probes'], cache['sym2probe'])
    print(f"gene={gene}  dose_rank={dose}  n_subjects={n}  (G={G}, q={q}, S={S})")

    from stability_check import one_seed
    out = one_seed(X_all, y_raw, RANDOM_SEED)
    print(f"{'method':<8}{'test MSE':>10}{'test R2':>10}{'#sel':>7}")
    for m in ['TEPIG', 'clusso', 'naive']:
        print(f"{m:<8}{out[m][0]:>10.3f}{out[m][1]:>10.3f}{len(out[m][2]):>7}")
    outp = os.path.join(RES_DIR, f'gene_{gene}_dose{dose}.pkl')
    pickle.dump({'gene': gene, 'dose': dose, 'results': out, 'n': n, 'q': q, 'G': G, 'S': S,
                 'features': features, '_meta': T.run_metadata()}, open(outp, 'wb'))
    print(f"\nSaved: {outp}")


if __name__ == '__main__':
    main()

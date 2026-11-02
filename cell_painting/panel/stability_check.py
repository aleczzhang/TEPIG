"""
stability_check.py
------------------
one_seed(): fit TEPIG, CLUSSO and the naive lasso for ONE train/test split and
return test MSE / R^2 and the selected feature set of each, plus everything needed
to re-derive TEPIG's selection under any threshold rule WITHOUT refitting.

Split and folds
  * outer split: random 80/20 over wells, or -- when `groups` (platemap labels)
    are given -- 80/20 over PLATEMAPS, so every test compound is unseen in training
  * inner 5-fold CV for lambda: random over training wells, or grouped by platemap
    when `groups` are given (replicate plates of a platemap never straddle an inner
    train/validation boundary). The SAME folds are used by all three methods.
  * y is standardized with the training mean/sd (test R^2 is scale-free).

TEPIG selection (core/tepig.py): the reported set uses run_gene.SELECT_RULE /
SELECT_GROUP_TERM; out['detail']['TEPIG'] stores B_hat, gamma_j, column sds,
sigma_hat, n_train and every block statistic, so the rule is a post-hoc choice.

Usage (legacy LINCS):  python stability_check.py --gene GLRX --dose 6 --seeds 5
"""

import os
import sys
import argparse
import itertools
import warnings
import time

import numpy as np
import pickle
warnings.filterwarnings('ignore')


_HERE = os.path.dirname(os.path.abspath(__file__))
_CP = os.path.dirname(_HERE)                       # cell_painting/
_ROOT = os.path.dirname(_CP)                       # repository root
sys.path[:0] = [_ROOT, _CP, os.path.join(_CP, 'panel'), os.path.join(_CP, 'screen')]
import run_gene as R                                            # noqa: E402
from tepig import tepig as T                                               # noqa: E402
from compute_log import log_compute                             # noqa: E402

CACHE = os.path.join(_CP, 'cache', 'cp_lincs_tensor.pkl')
METHODS = ['TEPIG', 'clusso', 'naive']


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 1.0


def tepig_selected(detail, rule=None, group_term=None, which='min'):
    """Selected feature indices (in the fitted feature space) from a stored
    TEPIG detail dict under a rule -- the post-hoc path. which='min' uses the
    CV-minimum lambda (the reported fit), which='1se' the one-standard-error
    lambda (both are on the stored path)."""
    rule = rule or R.SELECT_RULE
    gt = R.SELECT_GROUP_TERM if group_term is None else group_term
    gs = detail['G'] * detail['S'] if gt else 0
    if which == 'min' or 'path_stats' not in detail:
        stats, sigma = detail['stats'], detail['sigma']
    else:
        k = detail['best_1se_idx']
        stats, sigma = detail['path_stats'][k], detail['path_sigma'][k]
    m = T.select_features(stats, sigma, detail['q'], detail['n_train'], rule, gs)
    return [int(j) for j in np.where(m)[0]]


def one_seed(X, y_raw, seed, log=True, screen=None, groups=None, methods=METHODS,
             partial=None, verbose=False):
    """Fit the methods with a given seed; return metrics + selected sets.

    screen : optional SUPERVISED pre-screen f(Xn_train, y_train) -> kept feature
             indices (preselect.py), applied to the training wells only.
    groups : per-well group labels (platemaps) -> grouped outer split + grouped
             inner folds.
    partial: path of a checkpoint pickle; finished methods are loaded from it and
             the file is updated after each method (preemptible-queue safety).
    Returns out[m] = (test_mse, test_r2, selected_indices) for each method, plus
    out['detail'][m] with the fitted objects' diagnostics and out['split']."""
    n = X.shape[3]
    if groups is not None:
        groups = np.asarray(groups)
        tr, te = R.group_split(groups, R.TEST_FRAC, seed)
    else:
        tr, te = R.plain_split(n, R.TEST_FRAC, seed)
    _t = time.time()
    keep = None
    if screen is not None:
        Xn = X.mean(axis=(0, 2)).T
        keep = np.asarray(screen(Xn[tr], np.asarray(y_raw, float)[tr]))
        X = X[:, keep, :, :]
        if log:
            log_compute('screen', time.time() - _t, method=screen.method,
                        q_out=len(keep), q_in=Xn.shape[1], n_train=len(tr), seed=seed)
        _t = time.time()
    G, q, S = X.shape[0], X.shape[1], X.shape[2]

    def _log(method):
        nonlocal _t
        if log:
            log_compute('fit_seed', time.time() - _t, method=method, q=q, n=n, seed=seed)
        _t = time.time()

    ymu, ysd = y_raw[tr].mean(), y_raw[tr].std()
    y = (y_raw - ymu) / (ysd if ysd > 0 else 1.0)
    folds = (T.group_folds(tr, groups, seed + 1) if groups is not None
             else T.plain_folds(tr, seed + 1))
    out = {'detail': {}, 'split': {'train_idx': tr.tolist(), 'test_idx': te.tolist(),
                                   'folds': [list(f) for f in folds],
                                   'grouped': groups is not None, 'seed': seed,
                                   'y_mu': float(ymu), 'y_sd': float(ysd)}}
    done = {}
    if partial and os.path.exists(partial):
        done = pickle.load(open(partial, 'rb'))
        for m in done:
            if m in methods:
                out[m] = done[m]['result']; out['detail'][m] = done[m]['detail']
                if verbose:
                    print(f'    {m}: loaded from checkpoint', flush=True)

    def _checkpoint(m):
        if partial:
            done[m] = {'result': out[m], 'detail': out['detail'][m]}
            tmp = partial + '.tmp'
            pickle.dump(done, open(tmp, 'wb')); os.replace(tmp, partial)

    # ── TEPIG ────────────────────────────────────────────────────────────────
    if 'TEPIG' in methods and 'TEPIG' not in out:
        res = R.tepig_select_and_fit(X, y, folds, verbose=verbose)
        f = res['fit']
        Xtr_, Xte_ = X[..., tr], X[..., te]
        ptr, pte = f.predict(Xtr_), f.predict(Xte_)
        sig = float(np.std(y[tr] - ptr, ddof=1))
        stats = f.block_stats(); stats.update(T.refit_stats(f, Xtr_, y[tr]))
        # every lambda on the path: block statistics, sigma_hat, train MSE, test
        # predictions -- so the 1-SE rule (or any lambda) is a post-hoc choice too
        path_stats, path_sigma, path_train_mse, path_pte = [], [], [], []
        for ff in res['path']:
            p_tr = ff.predict(Xtr_)
            st_ = ff.block_stats(); st_.update(T.refit_stats(ff, Xtr_, y[tr]))
            path_stats.append({k: (v.astype(np.float32) if isinstance(v, np.ndarray) else v)
                               for k, v in st_.items()})
            path_sigma.append(float(np.std(y[tr] - p_tr, ddof=1)))
            path_train_mse.append(float(np.mean((y[tr] - p_tr) ** 2)))
            path_pte.append(ff.predict(Xte_).astype(np.float32))
        detail = {'B': f.B.astype(np.float32), 'gamma': f.gamma, 'col_sd': f.col_sd.astype(np.float32),
                  'sigma': sig, 'n_train': len(tr), 'q': q, 'G': G, 'S': S,
                  'stats': {k: (v.astype(np.float64) if isinstance(v, np.ndarray) else v)
                            for k, v in stats.items()},
                  'lam': res['best_lam'], 'lams': res['lams'], 'cv_curve': res['cv_curve'],
                  'cv_se': res['cv_se'], 'cv_folds': res['cv_folds'],
                  'best_idx': res['best'], 'best_1se_idx': res['best_1se'],
                  'path_stats': path_stats, 'path_sigma': path_sigma,
                  'path_train_mse': path_train_mse, 'path_test_pred': np.array(path_pte),
                  'y_test': y[te].astype(np.float32),
                  'at_lower_bound': res['at_lower_bound'],
                  'at_upper_bound': res['at_upper_bound'],
                  'n_iter': f.n_iter, 'converged': f.converged, 'kkt': f.kkt,
                  'path_iters': [ff.n_iter for ff in res['path']],
                  'path_kkt': [ff.kkt for ff in res['path']],
                  'cv_iters': res['iters'],
                  'train_mse': float(np.mean((y[tr] - ptr) ** 2)),
                  'n_fitted': int((stats['unscaled'] > 0).sum()),
                  'rule': R.SELECT_RULE, 'group_term': R.SELECT_GROUP_TERM}
        sel = tepig_selected(detail)
        detail['n_selected'] = len(sel)
        out['TEPIG'] = (float(np.mean((y[te] - pte) ** 2)), R.r2(y[te], pte), sel)
        out['detail']['TEPIG'] = detail
        _log('TEPIG'); _checkpoint('TEPIG')

    # ── CLUSSO ───────────────────────────────────────────────────────────────
    if 'clusso' in methods and 'clusso' not in out:
        Xcl = X.mean(axis=2)
        rng = np.random.default_rng(seed + 2)
        a, b, lam, cvm, ic, curve = R.clusso_select_and_fit(Xcl, y, folds, tr, rng,
                                                            R.CLUSSO_LAM_GRID)
        pte = np.array([a @ Xcl[:, :, i] @ b for i in te]) + ic
        ptr = np.array([a @ Xcl[:, :, i] @ b for i in tr]) + ic
        out['clusso'] = (float(np.mean((y[te] - pte) ** 2)), R.r2(y[te], pte),
                         [j for j in range(q) if abs(b[j]) > 1e-6])
        out['detail']['clusso'] = {'alpha': a, 'beta': b, 'lam': float(lam), 'cv_curve': curve,
                                   'lams': list(R.CLUSSO_LAM_GRID), 'intercept': ic,
                                   'best_idx': int(np.argmin(curve)),
                                   'train_mse': float(np.mean((y[tr] - ptr) ** 2))}
        _log('clusso'); _checkpoint('clusso')

    # ── naive ────────────────────────────────────────────────────────────────
    if 'naive' in methods and 'naive' not in out:
        Xnv = X.mean(axis=(0, 2)).T
        cvm, bnv, ptr, pte = R.naive_lasso_fit(Xnv[tr], y[tr], Xnv[te],
                                               cv=T.folds_to_positions(folds, tr))
        out['naive'] = (float(np.mean((y[te] - pte) ** 2)), R.r2(y[te], pte),
                        [j for j in range(q) if abs(bnv[j]) > 1e-6])
        out['detail']['naive'] = {'beta': bnv, 'cv_mse': cvm,
                                  'train_mse': float(np.mean((y[tr] - ptr) ** 2))}
        _log('naive'); _checkpoint('naive')

    if keep is not None:                       # map back to original feature indices
        for m in methods:
            out[m] = (out[m][0], out[m][1], [int(keep[j]) for j in out[m][2]])
        out['keep'] = [int(j) for j in keep]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gene', default='GLRX')
    ap.add_argument('--dose', type=int, default=6)
    ap.add_argument('--seeds', type=int, default=5)
    args = ap.parse_args()

    c = pickle.load(open(CACHE, 'rb'))
    feats = c['features']
    m = c['obs_dose_rank'] == args.dose
    X = c['X'][:, :, :, m]
    y_raw, _ = R.build_outcome(args.gene, c['expr'][m], c['probes'], c['sym2probe'])

    seeds = [42 + 100 * i for i in range(args.seeds)]
    print(f'gene={args.gene}  dose={args.dose}  n={X.shape[3]}  q={len(feats)}  '
          f'seeds={seeds}\n')
    runs = {}
    for s in seeds:
        runs[s] = one_seed(X, y_raw, s)
        r = runs[s]
        print(f'  seed {s:>4}: ' + '  '.join(
            f'{mm} MSE={r[mm][0]:.3f} R2={r[mm][1]:+.3f} nsel={len(r[mm][2]):>2}'
            for mm in METHODS))
    print('\n' + '=' * 78)
    for mm in METHODS:
        mse = np.array([runs[s][mm][0] for s in seeds])
        rr = np.array([runs[s][mm][1] for s in seeds])
        js = [jaccard(runs[a][mm][2], runs[b][mm][2])
              for a, b in itertools.combinations(seeds, 2)]
        print(f'{mm:<8}{mse.mean():>13.3f} ± {mse.std():<6.3f}'
              f'{rr.mean():>13.3f} ± {rr.std():<6.3f}{np.mean(js):>15.2f}')


if __name__ == '__main__':
    main()

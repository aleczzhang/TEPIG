"""
stability_check.py
------------------
Sanity check requested at the 11 Jul meeting: with highly correlated predictors,
lasso-type selection is known to be unstable — it picks one representative from a
clump of near-duplicates, and which one it picks can flip between runs.

Refit all three methods on ONE dose across several random seeds (each seed changes
the train/test split, the CV folds, and CLUSSO's restarts) and report:

  * performance stability : test MSE / R^2 mean +/- sd across seeds
  * selection stability   : Jaccard overlap of the selected feature sets between
                            every pair of seeds (1.0 = identical, 0 = disjoint)

Expected pattern if correlation is driving instability: performance is stable,
selection is not.

Usage:  python stability_check.py --gene GLRX --dose 6 --seeds 5
"""

import os
import sys
import argparse
import itertools
import warnings

import numpy as np
import pickle
warnings.filterwarnings('ignore')

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, _HERE)

import run_gene as R                                            # noqa: E402

CACHE = os.path.join(_HERE, 'cache', 'cp_lincs_tensor.pkl')
METHODS = ['TEPIG', 'clusso', 'naive']


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 1.0


def one_seed(X, y_raw, seed):
    """Fit all three methods with a given seed; return metrics + selected sets."""
    n = X.shape[3]
    G, q, S = X.shape[0], X.shape[1], X.shape[2]
    tr, te = R.plain_split(n, R.TEST_FRAC, seed)
    ymu, ysd = y_raw[tr].mean(), y_raw[tr].std()
    y = (y_raw - ymu) / (ysd if ysd > 0 else 1.0)
    folds = R.plain_folds(tr, seed + 1)
    d = G * q * S
    out = {}

    # ── TEPIG ────────────────────────────────────────────────────────────────
    LAM = R.tepig_lambda_grid(X[:, :, :, tr], y[tr])
    cv = []
    for lam in LAM:
        fm = []
        for k in range(5):
            tek = folds[k]
            trk = [i for f in folds for i in f if f is not folds[k]]
            _, B = R.proxgrad_fit(X[:, :, :, trk], y[trk], lam)
            Xte = X[:, :, :, tek]
            Xc = (Xte - Xte.mean(axis=3, keepdims=True)).reshape(d, len(tek)).T
            yc = y[tek] - y[tek].mean()
            fm.append(float(np.mean((yc - Xc @ B.reshape(-1)) ** 2)))
        cv.append(np.mean(fm))
    lam = LAM[int(np.argmin(cv))]
    _, B = R.proxgrad_fit(X[:, :, :, tr], y[tr], lam)
    Xtrm = X[:, :, :, tr].mean(axis=3, keepdims=True)
    ytrm = float(np.mean(y[tr]))
    ptr = (X[:, :, :, tr] - Xtrm).reshape(d, len(tr)).T @ B.reshape(-1) + ytrm
    pte = (X[:, :, :, te] - Xtrm).reshape(d, len(te)).T @ B.reshape(-1) + ytrm
    sig = float(np.std(y[tr] - ptr, ddof=1))
    tau = sig * np.sqrt(2.0 * np.log(q) / len(tr))
    bn = np.sqrt((B ** 2).sum(axis=(0, 2)))
    out['TEPIG'] = (float(np.mean((y[te] - pte) ** 2)), R.r2(y[te], pte),
                    [j for j in range(q) if bn[j] > tau])

    # ── CLUSSO ───────────────────────────────────────────────────────────────
    Xcl = X.mean(axis=2)
    rng = np.random.default_rng(seed + 2)
    a, b, _, _, ic, _ = R.clusso_select_and_fit(Xcl, y, folds, tr, rng,
                                                R.CLUSSO_LAM_GRID)
    pte = np.array([a @ Xcl[:, :, i] @ b for i in te]) + ic
    out['clusso'] = (float(np.mean((y[te] - pte) ** 2)), R.r2(y[te], pte),
                     [j for j in range(q) if abs(b[j]) > 1e-6])

    # ── naive ────────────────────────────────────────────────────────────────
    Xnv = X.mean(axis=(0, 2)).T
    _, bnv, _, pte = R.naive_lasso_fit(Xnv[tr], y[tr], Xnv[te])
    out['naive'] = (float(np.mean((y[te] - pte) ** 2)), R.r2(y[te], pte),
                    [j for j in range(q) if abs(bnv[j]) > 1e-6])
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
            f'{m} MSE={r[m][0]:.3f} R2={r[m][1]:+.3f} nsel={len(r[m][2]):>2}'
            for m in METHODS))

    print('\n' + '=' * 78)
    print(f'{"method":<8}{"test MSE (mean±sd)":>22}{"test R² (mean±sd)":>22}'
          f'{"selection Jaccard":>20}')
    for m in METHODS:
        mse = np.array([runs[s][m][0] for s in seeds])
        r2 = np.array([runs[s][m][1] for s in seeds])
        js = [jaccard(runs[a][m][2], runs[b][m][2])
              for a, b in itertools.combinations(seeds, 2)]
        print(f'{m:<8}{mse.mean():>13.3f} ± {mse.std():<6.3f}'
              f'{r2.mean():>13.3f} ± {r2.std():<6.3f}'
              f'{np.mean(js):>15.2f}')
    print('\n  Jaccard = mean overlap of selected feature sets between seed pairs')
    print('  (1.00 = identical selections every run; 0.50 = half the picks change)')

    # which features are stable vs volatile for TEPIG
    cnt = {}
    for s in seeds:
        for j in runs[s]['TEPIG'][2]:
            cnt[j] = cnt.get(j, 0) + 1
    if cnt:
        print(f'\n  TEPIG: features selected in all {len(seeds)} seeds vs only some:')
        allseeds = [feats[j] for j, k in cnt.items() if k == len(seeds)]
        some = [(feats[j], k) for j, k in sorted(cnt.items(), key=lambda kv: -kv[1])
                if k < len(seeds)]
        print(f'    stable ({len(allseeds)}): ' + ', '.join(allseeds[:6]))
        print(f'    volatile ({len(some)}): ' +
              ', '.join(f'{f}({k}/{len(seeds)})' for f, k in some[:6]))


if __name__ == '__main__':
    main()

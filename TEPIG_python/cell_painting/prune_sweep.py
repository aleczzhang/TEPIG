"""
prune_sweep.py
--------------
Correlation pruning, per the 13 Jul direction: rather than aggregating features,
sequentially REMOVE features so that the maximum pairwise |r| among those that
remain falls below a threshold — keeping as many features as possible.

This is the same strategy used in the Random CLUSSO paper. The algorithm is the
standard greedy one (cf. caret::findCorrelation): repeatedly take the most
correlated remaining pair and drop whichever of the two has the higher average
absolute correlation with the other survivors. That targets the "hub" of each
correlated group, so few features are lost.

Pruning is unsupervised (it never looks at y), so it cannot leak the outcome.

For each threshold we report, across several random splits:
  * q retained            — how much information we gave up
  * selection stability   — Jaccard overlap of selected features between splits
  * performance           — test MSE / R^2 (to check pruning does not cost accuracy)

Usage:  python prune_sweep.py --gene GLRX --dose 6 --seeds 5
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

import run_gene as R                                     # noqa: E402
from stability_check import one_seed, jaccard            # noqa: E402

CACHE = os.path.join(_HERE, 'cache', 'cp_lincs_tensor.pkl')
METHODS = ['TEPIG', 'clusso', 'naive']
THRESHOLDS = [0.99, 0.95, 0.90, 0.85, 0.80]


def prune_correlated(Rabs, thresh):
    """Greedily drop features until max pairwise |r| <= thresh.
    At each step take the most-correlated surviving pair and remove whichever
    has the higher mean |r| to the other survivors. Returns kept indices."""
    R = Rabs.copy()
    np.fill_diagonal(R, 0.0)
    keep = list(range(R.shape[0]))
    while len(keep) > 1:
        sub = R[np.ix_(keep, keep)]
        mx = sub.max()
        if mx <= thresh:
            break
        i, j = np.unravel_index(sub.argmax(), sub.shape)
        # drop the more "central" of the pair (higher average correlation)
        drop = keep[i] if sub[i].mean() >= sub[j].mean() else keep[j]
        keep.remove(drop)
    return sorted(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gene', default='GLRX')
    ap.add_argument('--dose', type=int, default=6)
    ap.add_argument('--seeds', type=int, default=5)
    args = ap.parse_args()

    c = pickle.load(open(CACHE, 'rb'))
    feats = np.array(c['features'])
    m = c['obs_dose_rank'] == args.dose
    X_all = c['X'][:, :, :, m]
    y_raw, _ = R.build_outcome(args.gene, c['expr'][m], c['probes'], c['sym2probe'])
    q0 = X_all.shape[1]

    # feature-level correlation (compartment/channel-averaged), unsupervised
    Xn = X_all.mean(axis=(0, 2)).T
    Rabs = np.abs(np.corrcoef(Xn.T))

    seeds = [42 + 100 * i for i in range(args.seeds)]
    print(f'gene={args.gene}  dose={args.dose}  n={X_all.shape[3]}  '
          f'q(full)={q0}  seeds={seeds}\n')

    rows = []
    for thr in [None] + THRESHOLDS:
        if thr is None:
            keep = list(range(q0)); tag = 'none (all 68)'
        else:
            keep = prune_correlated(Rabs, thr)
            tag = f'|r| <= {thr:.2f}'
        X = X_all[:, keep, :, :]
        sel, mse, r2 = {m_: {} for m_ in METHODS}, {m_: [] for m_ in METHODS}, \
                       {m_: [] for m_ in METHODS}
        for s in seeds:
            out = one_seed(X, y_raw, s)
            for m_ in METHODS:
                mse[m_].append(out[m_][0]); r2[m_].append(out[m_][1])
                sel[m_][s] = set(out[m_][2])       # indices into the KEPT set
        line = {'thr': tag, 'q': len(keep)}
        for m_ in METHODS:
            js = [jaccard(sel[m_][a], sel[m_][b])
                  for a, b in itertools.combinations(seeds, 2)]
            line[m_] = (np.mean(js), np.mean(mse[m_]), np.mean(r2[m_]))
        rows.append(line)
        print(f'  {tag:<16} q={len(keep):>2}  ' + '  '.join(
            f'{m_}: J={line[m_][0]:.2f} R2={line[m_][2]:+.3f}' for m_ in METHODS))

    print('\n' + '=' * 92)
    print(f'{"prune threshold":<17}{"q kept":>7}' +
          ''.join(f'{m_+" J":>10}{m_+" R²":>10}' for m_ in METHODS))
    print('-' * 92)
    for r in rows:
        print(f'{r["thr"]:<17}{r["q"]:>7}' +
              ''.join(f'{r[m_][0]:>10.2f}{r[m_][2]:>10.3f}' for m_ in METHODS))
    print('\n  J = mean Jaccard overlap of selected features between split pairs')
    print('     (1.00 = identical selection every run; higher is more stable)')
    print('  Goal: raise J without losing R² or dropping too many features.')


if __name__ == '__main__':
    main()

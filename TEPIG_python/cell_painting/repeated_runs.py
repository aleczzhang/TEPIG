"""
repeated_runs.py
----------------
Reporting scheme agreed 13 Jul: because selected features vary across repeated
runs (feature correlation makes near-ties around the coefficient threshold), we
report, per setting:

  * average test MSE / R^2 over several successive runs (mean +/- sd), and
  * L1-normalised SELECTION FREQUENCIES -- among the features a method selects at
    least once, how often each is selected, normalised to sum to 1.

The coefficient threshold formulation is left exactly as-is (kept consistent with
the simulations); nothing about the estimator changes. This is a reporting layer.

Features are first pruned at |r| <= PRUNE so that near-duplicates do not split
each other's selection frequency.

Expectation to test (his hypothesis): highly correlated features of similar
relevance should end up with SIMILAR normalised frequencies, and genuinely
important features should be clearly DOMINANT.

Usage:  python repeated_runs.py --gene GLRX --dose 6 --runs 10
"""

import os
import sys
import argparse
import warnings
from collections import Counter

import numpy as np
import pickle
warnings.filterwarnings('ignore')

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'core'))
sys.path.insert(0, _HERE)

import run_gene as R                                  # noqa: E402
from stability_check import one_seed                  # noqa: E402
from prune_sweep import prune_correlated              # noqa: E402
import selection_metrics as SM                        # noqa: E402
import preselect as PS                                # noqa: E402
from compute_log import log_compute, peak_rss_gb      # noqa: E402

CACHE = os.path.join(_HERE, 'cache', 'cp_lincs_tensor.pkl')
METHODS = ['TEPIG', 'clusso', 'naive']
PRUNE = 0.95


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gene', default='GLRX')
    ap.add_argument('--dose', type=int, default=6)
    ap.add_argument('--runs', type=int, default=10)
    ap.add_argument('--cache', default=None,
                    help='cache filename in cache/ (default: the LINCS cache); '
                         'use cdrp_singlecell.pkl for the single-cell CDRP run')
    ap.add_argument('--select', default='prune', choices=list(PS.ALL),
                    help='pre-screening arm (see preselect.py): unsupervised prune / '
                         'varclust / pycytominer applied once to all wells; supervised '
                         'sis / dcsis applied inside each split to the training wells')
    ap.add_argument('--k', type=int, default=None,
                    help='features to keep (varclust: clusters; sis/dcsis: d, default '
                         'n_train/log n_train)')
    ap.add_argument('--thr', type=float, default=None,
                    help='cutoff |r| (prune: drop above; varclust: within-cluster '
                         'tightness). Defaults to 0.95 for prune.')
    args = ap.parse_args()
    import time
    t_start = time.time()

    cache_path = (os.path.join(_HERE, 'cache', args.cache) if args.cache else CACHE)
    c = pickle.load(open(cache_path, 'rb'))
    feats_all = np.array(c['features'])
    m = c['obs_dose_rank'] == args.dose
    X_all = c['X'][:, :, :, m]
    y_raw, _ = R.build_outcome(args.gene, c['expr'][m], c['probes'], c['sym2probe'])
    ok = np.isfinite(np.asarray(y_raw, float))   # wells with an L1000 outcome
    if not ok.all():
        print(f'({(~ok).sum()} of {len(ok)} wells have no L1000 outcome -> dropped)')
        X_all, y_raw = X_all[:, :, :, ok], np.asarray(y_raw, float)[ok]

    # pre-screening on the well-level (cluster/site-averaged) matrix
    Xn = X_all.mean(axis=(0, 2)).T
    Rabs = np.nan_to_num(np.abs(np.corrcoef(Xn.T)), nan=0.0)
    screen = None
    if PS.is_supervised(args.select):
        # supervised: keep the full feature space here; one_seed screens per split
        screen = PS.make_supervised_screen(args.select, args.k)
        keep, sel_desc = list(range(len(feats_all))), screen.desc
    else:
        keep, sel_desc = PS.unsupervised_select(args.select, Xn, feats_all, args.k, args.thr)
    X = X_all[:, keep, :, :]
    feats = feats_all[keep]
    Rk = Rabs[np.ix_(keep, keep)]

    seeds = [42 + 100 * i for i in range(args.runs)]
    print(f'gene={args.gene}  dose={args.dose}  n={X.shape[3]}  '
          f'q={len(keep)} ({sel_desc}, from {len(feats_all)})  '
          f'runs={args.runs}\n')

    mse = {m_: [] for m_ in METHODS}
    r2 = {m_: [] for m_ in METHODS}
    cnt = {m_: Counter() for m_ in METHODS}
    sets = {m_: [] for m_ in METHODS}          # selected index set per run
    screened = []
    for s in seeds:
        out = one_seed(X, y_raw, s, screen=screen)
        if screen is not None:
            screened.append(out['keep'])
        for m_ in METHODS:
            mse[m_].append(out[m_][0]); r2[m_].append(out[m_][1])
            sets[m_].append(list(out[m_][2]))
            for j in out[m_][2]:
                cnt[m_][j] += 1

    # ── performance: averages over successive runs ───────────────────────────
    print('=== average performance over %d runs ===' % args.runs)
    print(f'{"method":<8}{"test MSE (mean±sd)":>24}{"test R² (mean±sd)":>24}')
    for m_ in METHODS:
        a, b = np.array(mse[m_]), np.array(r2[m_])
        print(f'{m_:<8}{a.mean():>14.3f} ± {a.std():<7.3f}'
              f'{b.mean():>14.3f} ± {b.std():<7.3f}')

    # ── selection consistency (chance-corrected; replaces TPR/FPR on real data) ─
    print('\n=== selection consistency over %d runs (see selection_metrics.py) ===' % args.runs)
    print(SM.header())
    stab = {}
    for m_ in METHODS:
        stab[m_] = SM.summarize(sets[m_], len(keep))
        print(SM.format_row(m_, stab[m_]))
    if screened:
        stab['screen'] = SM.summarize(screened, len(keep))
        print(SM.format_row('screen', stab['screen']) + '   <- the pre-screen itself')
    print('  Nogueira Φ: 1 = same set every run, ~0 = no better than random picks of the '
          'same size; |pi>=.6| = size of the stable set (selection probability >= 0.6)')

    # ── L1-normalised selection frequencies ──────────────────────────────────
    for m_ in METHODS:
        tot = sum(cnt[m_].values())
        if tot == 0:
            print(f'\n=== {m_}: selected nothing in any run ===')
            continue
        print(f'\n=== {m_}: L1-normalised selection frequency '
              f'({len(cnt[m_])} features selected at least once) ===')
        print(f'  {"weight":>7}  {"picked":>7}   feature')
        for j, k in cnt[m_].most_common(10):
            print(f'  {k/tot:>7.3f}  {k:>3}/{args.runs:<3}   {feats[j]}')

    # ── his hypothesis: do correlated features get similar weights? ──────────
    tot = sum(cnt['TEPIG'].values())
    sel = [j for j, _ in cnt['TEPIG'].most_common()]
    if len(sel) >= 2:
        print('\n=== hypothesis check (TEPIG): correlated features -> similar weight? ===')
        top = sel[0]
        print(f'  most dominant: {feats[top]}  weight={cnt["TEPIG"][top]/tot:.3f}')
        print(f'  its correlation with the other selected features, vs their weight:')
        for j in sel[1:6]:
            print(f'    |r|={Rk[top, j]:.2f}  weight={cnt["TEPIG"][j]/tot:.3f}   {feats[j]}')

    # ── save for the figures ─────────────────────────────────────────────────
    tag = ('_' + os.path.splitext(args.cache)[0]) if args.cache else ''
    if args.select == 'varclust':
        tag += f'_vc{args.k}' if args.k else f'_vc{args.thr or 0.7}'
    elif args.select != 'prune':
        tag += f'_{args.select}' + (f'{args.k}' if args.k else '')
    out = os.path.join(_HERE, 'results',
                       f'repeated_{args.gene}_dose{args.dose}{tag}.pkl')
    freq = {m_: {feats[j]: k / max(sum(cnt[m_].values()), 1)
                 for j, k in cnt[m_].items()} for m_ in METHODS}
    picks = {m_: {feats[j]: k for j, k in cnt[m_].items()} for m_ in METHODS}
    pickle.dump({'gene': args.gene, 'dose': args.dose, 'runs': args.runs,
                 'q': len(keep), 'q_full': len(feats_all), 'prune': PRUNE,
                 'features': list(feats), 'corr': Rk,
                 'mse': mse, 'r2': r2, 'freq': freq, 'picks': picks,
                 'sets': sets, 'stability': stab, 'select': sel_desc,
                 'screened': screened},
                open(out, 'wb'))
    print(f'\nSaved: {out}')
    log_compute('repeated_runs_total', time.time() - t_start, gene=args.gene,
                q=len(keep), n=X.shape[3], runs=args.runs, select=sel_desc)
    print(f'wall {(time.time() - t_start) / 60:.1f} min, peak RSS {peak_rss_gb():.2f} GB')


if __name__ == '__main__':
    main()

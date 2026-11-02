"""
run_panel.py
------------
Run the TEPIG / CLUSSO / naive comparison over several outcome genes in ONE
long-lived process. --select picks the pre-screening arm (preselect.py):
varclust (--k clusters), pycytominer (community cut), sis / dcsis (supervised,
inside each split). Prints a per-gene mean+/-sd test R^2 line plus the Nogueira
stability of each method's selected set, and writes results/panel_<select>_<cache>.pkl
with, per gene and split, everything needed to re-derive TEPIG's selection under
any threshold rule (see core/tepig.py).

Preemptible-queue safety: with --seeds <one seed> (slurm array tasks) the output
pkl is skipped if it already exists, and each method's result is checkpointed to
results/partial/ as soon as it finishes, so a requeued task resumes.

Usage:  python run_panel.py --cache cdrp_4plates.pkl --select pycytominer --runs 10 CCNA2 ...
"""
import os, sys, pickle, argparse, warnings
import numpy as np
warnings.filterwarnings('ignore')
_HERE = os.path.dirname(os.path.abspath(__file__))
_CP = os.path.dirname(_HERE)                       # cell_painting/
_ROOT = os.path.dirname(_CP)                       # repository root
sys.path[:0] = [_ROOT, _CP, os.path.join(_CP, 'panel'), os.path.join(_CP, 'screen')]
import run_gene as R
import preselect as PS
from tepig import selection_metrics as SM
from tepig import tepig as T
from stability_check import one_seed, tepig_selected
from compute_log import log_compute, peak_rss_gb
import time

ap = argparse.ArgumentParser()
ap.add_argument('--select', default='varclust', choices=list(PS.ALL))
ap.add_argument('--k', type=int, default=None, help='varclust: clusters (default 262); sis/dcsis: d')
ap.add_argument('--thr', type=float, default=None)
ap.add_argument('--runs', type=int, default=10)
ap.add_argument('--cache', default='cdrp_singlecell.pkl')
ap.add_argument('--seeds', type=int, nargs='+', default=None,
                help='explicit split seeds (overrides --runs); one seed per task '
                     'in slurm arrays, output gets a _s<seed> suffix')
ap.add_argument('--split-by-plate', action='store_true',
                help='grouped 80/20 split: hold out 20%% of PLATES (platemaps), '
                     'so test compounds are unseen in training; inner CV folds are '
                     'grouped the same way')
ap.add_argument('--methods', nargs='+', default=['TEPIG', 'clusso', 'naive'])
ap.add_argument('--force', action='store_true', help='recompute even if the output exists')
ap.add_argument('genes', nargs='+')
a = ap.parse_args()

tag = '' if a.cache == 'cdrp_singlecell.pkl' else '_' + os.path.splitext(a.cache)[0]
sel = a.select + (str(a.k) if a.k else '')
if a.seeds and len(a.seeds) == 1:          # one array task; combine pkls later
    tag += f'_{a.genes[0]}_s{a.seeds[0]}'
out_pkl = os.path.join(_CP, 'results', f'panel_{sel}{tag}.pkl')
if os.path.exists(out_pkl) and not a.force:
    print(f'{out_pkl} exists; nothing to do (use --force to recompute)', flush=True)
    sys.exit(0)
os.makedirs(os.path.join(_CP, 'results', 'partial'), exist_ok=True)

c = pickle.load(open(os.path.join(_CP, 'cache', a.cache), 'rb'))
X_all = c['X']
Xn = X_all.mean(axis=(0, 2)).T
feats = np.array(c['features'])
screen = None
if PS.is_supervised(a.select):
    screen = PS.make_supervised_screen(a.select, a.k)
    keep, desc = list(range(len(feats))), screen.desc
else:
    if a.select == 'varclust' and not a.k and not a.thr:
        a.k = 262
    keep, desc = PS.unsupervised_select(a.select, Xn, feats, a.k, a.thr)
X = X_all[:, keep, :, :]
del X_all, Xn
seeds = a.seeds if a.seeds else [42 + 100 * i for i in range(a.runs)]
# group by PLATEMAP when the cache carries it (replicate plates share compounds,
# so all copies of a platemap must land on the same side of the split)
plates_all = (np.array(c.get('obs_platemap', c['obs_plate']))
              if a.split_by_plate else None)
METHODS = a.methods
print(f"panel: n={X.shape[3]} q={len(keep)} ({desc}) runs={len(seeds)} "
      f"split={'by-platemap' if a.split_by_plate else 'random-wells'} "
      f"TEPIG rule={R.SELECT_RULE}{'+GS' if R.SELECT_GROUP_TERM else ''}\n", flush=True)

results = {}
t_start = time.time()
for g in a.genes:
    t_gene = time.time()
    try:
        y, _ = R.build_outcome(g, c['expr'], c['probes'], c['sym2probe'])
    except SystemExit as e:                    # gene not in this cache's L1000 panel
        print(f"{g:>8}:  SKIPPED ({e})", flush=True)
        continue
    y = np.asarray(y, float)
    ok = np.isfinite(y)                     # wells whose compound has an L1000 profile
    if not ok.all():
        print(f"  ({(~ok).sum()} of {len(y)} wells have no L1000 outcome -> dropped)", flush=True)
    Xg, y = X[:, :, :, ok], y[ok]
    r2 = {m: [] for m in METHODS}
    mse = {m: [] for m in METHODS}
    sets = {m: [] for m in METHODS}
    details, splits = {m: [] for m in METHODS}, []
    for s in seeds:
        partial = os.path.join(_CP, 'results', 'partial',
                               f'panel_{sel}{"_" + os.path.splitext(a.cache)[0]}_{g}_s{s}.pkl')
        o = one_seed(Xg, y, s, screen=screen, methods=METHODS, partial=partial,
                     groups=plates_all[ok] if plates_all is not None else None,
                     verbose=bool(a.seeds and len(a.seeds) == 1))
        for m in METHODS:
            mse[m].append(o[m][0]); r2[m].append(o[m][1]); sets[m].append(list(o[m][2]))
            details[m].append(o['detail'][m])
        splits.append(o['split'])
    stab = {m: SM.summarize(sets[m], len(keep)) for m in METHODS}
    results[g] = {'r2': r2, 'mse': mse, 'sets': sets, 'stability': stab,
                  'detail': details, 'splits': splits,
                  'features': [str(f) for f in feats[keep]]}
    if 'TEPIG' in METHODS:                     # selection under the other rules, for the record
        alt = {}
        for which in ['min', '1se']:
            for rule in ['raw', 'unscaled', 'block_scaled', 'col_std', 'col_std_refit']:
                for gt in ([False] if rule == 'raw' else [False, True]):
                    key = rule + ('+GS' if gt else '') + ('' if which == 'min' else '@1se')
                    alt[key] = [tepig_selected(d, rule, gt, which) for d in details['TEPIG']]
        results[g]['tepig_sets_by_rule'] = alt
    log_compute('panel_gene', time.time() - t_gene, gene=g, q=len(keep), n=int(ok.sum()), runs=len(seeds))
    print(f"{g:>8}:  " + "   ".join(
        f"{m} R2={np.mean(r2[m]):+.3f}±{np.std(r2[m]):.3f} Φ={stab[m]['nogueira']:+.2f} "
        f"|sel|={stab[m]['size_mean']:.0f}" for m in METHODS), flush=True)
    if 'TEPIG' in METHODS:
        d0 = details['TEPIG']
        print("          TEPIG |J| by rule: " + ", ".join(
            f"{k}={np.mean([len(v) for v in alt[k]]):.0f}" for k in
            ['raw', 'block_scaled', 'col_std+GS', 'col_std_refit+GS', 'raw@1se', 'col_std_refit+GS@1se']))
        print(f"          TEPIG fitted(nonzero)={np.mean([d['n_fitted'] for d in d0]):.0f} "
              f"lambda idx={[d['best_idx'] for d in d0]} (1se {[d['best_1se_idx'] for d in d0]})"
              f"/{len(d0[0]['lams']) - 1} "
              f"iters={[d['n_iter'] for d in d0]} maxKKT={max(d['kkt'] for d in d0):.1e}",
              flush=True)

results['_meta'] = {'select': desc, 'q': len(keep), 'n': int(X.shape[3]),
                    'runs': len(seeds), 'seeds': seeds, 'methods': METHODS,
                    'split': 'by-platemap' if a.split_by_plate else 'random-wells',
                    'inner_cv': 'grouped-by-platemap' if a.split_by_plate else 'random-wells',
                    'cache': a.cache, 'tepig_rule': R.SELECT_RULE,
                    'tepig_group_term': R.SELECT_GROUP_TERM,
                    'tepig_grid': {'num': R.N_LAM_TEPIG, 'ratio': R.LAM_RATIO},
                    'clusso_grid': list(R.CLUSSO_LAM_GRID),
                    'clusso_restarts': {'cv': R.N_CV_INIT, 'final': R.M_INIT},
                    'run': T.run_metadata()}
tmp = out_pkl + '.tmp'
pickle.dump(results, open(tmp, 'wb')); os.replace(tmp, out_pkl)
log_compute('panel_total', time.time() - t_start, genes=len(a.genes), q=len(keep),
            n=int(ok.sum()), runs=len(seeds), cache=a.cache, select=desc)
print(f"\nsaved {os.path.relpath(out_pkl, _HERE)}   wall {(time.time()-t_start)/60:.1f} min, "
      f"peak RSS {peak_rss_gb():.2f} GB", flush=True)

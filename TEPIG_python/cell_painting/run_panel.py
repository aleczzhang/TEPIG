"""
run_panel.py
------------
Run the TEPIG / CLUSSO / naive comparison over several outcome genes in ONE
long-lived process (a bash for-loop spawning a python process per gene kept
getting reaped between genes). --select picks the pre-screening arm (preselect.py):
varclust (default, --k clusters), pycytominer (community cut), sis / dcsis
(supervised, inside each split). Prints a per-gene mean+/-sd test R^2 line plus
the Nogueira stability of each method's selected set, flushed as each gene
finishes, and writes results/panel_<select>_<cache>.pkl.

Usage:  python run_panel.py --cache cdrp_4plates.pkl --select pycytominer --runs 10 CCNA2 ...
"""
import os, sys, pickle, argparse, warnings
import numpy as np
warnings.filterwarnings('ignore')
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'core')); sys.path.insert(0, _HERE)
import run_gene as R
import preselect as PS
import selection_metrics as SM
from stability_check import one_seed
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
                     'so test compounds are unseen in training')
ap.add_argument('genes', nargs='+')
a = ap.parse_args()

c = pickle.load(open(os.path.join(_HERE, 'cache', a.cache), 'rb'))
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
seeds = a.seeds if a.seeds else [42 + 100 * i for i in range(a.runs)]
plates_all = np.array(c['obs_plate']) if a.split_by_plate else None
METHODS = ['TEPIG', 'clusso', 'naive']
print(f"panel: n={X.shape[3]} q={len(keep)} ({desc}) runs={a.runs}\n", flush=True)

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
    for s in seeds:
        o = one_seed(Xg, y, s, screen=screen,
                     groups=plates_all[ok] if plates_all is not None else None)
        for m in METHODS:
            mse[m].append(o[m][0]); r2[m].append(o[m][1]); sets[m].append(list(o[m][2]))
    stab = {m: SM.summarize(sets[m], len(keep)) for m in METHODS}
    results[g] = {'r2': r2, 'mse': mse, 'sets': sets, 'stability': stab,
                  'features': [str(f) for f in feats[keep]]}
    log_compute('panel_gene', time.time() - t_gene, gene=g, q=len(keep), n=int(ok.sum()), runs=a.runs)
    print(f"{g:>8}:  " + "   ".join(
        f"{m} R2={np.mean(r2[m]):+.3f}±{np.std(r2[m]):.3f} Φ={stab[m]['nogueira']:+.2f} "
        f"|sel|={stab[m]['size_mean']:.0f}" for m in METHODS),
        flush=True)

tag = '' if a.cache == 'cdrp_singlecell.pkl' else '_' + os.path.splitext(a.cache)[0]
sel = a.select + (str(a.k) if a.k else '')
if a.seeds and len(a.seeds) == 1:          # one array task; combine pkls later
    tag += f'_{a.genes[0]}_s{a.seeds[0]}'
out_pkl = os.path.join(_HERE, 'results', f'panel_{sel}{tag}.pkl')
results['_meta'] = {'select': desc, 'q': len(keep), 'n': int(X.shape[3]),
                    'runs': len(seeds), 'seeds': seeds,
                    'split': 'by-plate' if a.split_by_plate else 'random-wells',
                    'cache': a.cache}
pickle.dump(results, open(out_pkl, 'wb'))
log_compute('panel_total', time.time() - t_start, genes=len(a.genes), q=len(keep),
            n=int(ok.sum()), runs=a.runs, cache=a.cache, select=desc)
print(f"\nsaved {os.path.relpath(out_pkl, _HERE)}   wall {(time.time()-t_start)/60:.1f} min, "
      f"peak RSS {peak_rss_gb():.2f} GB", flush=True)

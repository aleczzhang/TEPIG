"""Merge ALL batch caches (cdrp_pmz_b* + cdrp_rep_b*, 213 plates / ~68k wells)
into cache/cdrp_213.pkl for the replicate-scale run.

Same cluster alignment as merge_pmz_caches.py (per-batch GMM labels matched to
the first cache by cluster-profile correlation), plus an `obs_platemap` list so
splits can group by PLATEMAP -- with replicates on disk, plates from the same
platemap share compounds and must never straddle a train/test split.

Memory-safe: the final float32 tensor (~12 GB) is preallocated and filled one
cache at a time. Run as a small slurm job (24-32 GB), not on a login node:
  sbatch --partition=scavenger --qos=scavenger --time=00:40:00 --mem=32G \
      --cpus-per-task=2 --output=logs/merge213_%j.out \
      --wrap "source ~/scratch.staraptor-lab/venv_tepig/bin/activate && \
              unset PYTHONPATH && cd \$HOME/scratch.staraptor-lab/TEPIG/TEPIG_python/cell_painting && \
              python merge_all_caches.py"
"""
import csv, glob, os, pickle
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
paths = (sorted(glob.glob(os.path.join(_HERE, 'cache', 'cdrp_pmz_b*.pkl'))) +
         sorted(glob.glob(os.path.join(_HERE, 'cache', 'cdrp_rep_b*.pkl'))))
print(f'{len(paths)} caches')

plate2pmap = {}
for r in csv.DictReader(open(os.path.join(_HERE, 'data',
                                          'platemap_l1000_coverage.csv'))):
    for p in r['plates'].split():
        plate2pmap[p] = r['platemap']

# pass 1: shared features + total wells
feat_sets, ns = [], []
for p in paths:
    c = pickle.load(open(p, 'rb'))
    feat_sets.append([str(f) for f in c['features']])
    ns.append(len(c['obs_plate']))
    del c
shared = [f for f in feat_sets[0] if all(f in set(s) for s in feat_sets[1:])]
N, Q = sum(ns), len(shared)
print(f'shared features: {Q}, total wells: {N}')

# pass 2: fill preallocated arrays, aligning clusters to the first cache
first = pickle.load(open(paths[0], 'rb'))
G, S = first['X'].shape[0], first['X'].shape[2]
E = first['expr'].shape[1]
X = np.empty((G, Q, S, N), np.float32)
expr = np.empty((N, E), np.float64)
plates, pmaps, ref, at = [], [], None, 0
del first
for pi, p in enumerate(paths):
    c = pickle.load(open(p, 'rb'))
    idx = {str(f): i for i, f in enumerate(c['features'])}
    cols = [idx[f] for f in shared]
    Xc = np.nan_to_num(c['X'][:, cols]).astype(np.float32)
    prof = Xc.mean(axis=(2, 3))
    if ref is None:
        ref = prof
    r = np.corrcoef(np.vstack([ref, prof]))[:G, G:]
    flip = (r[0, 1] + r[1, 0]) > (r[0, 0] + r[1, 1])
    if flip:
        Xc = Xc[::-1]
    n = Xc.shape[3]
    X[:, :, :, at:at + n] = Xc
    expr[at:at + n] = np.asarray(c['expr'], float)
    plates += list(c['obs_plate'])
    pmaps += [plate2pmap[str(pl)] for pl in c['obs_plate']]
    meta = c  # keep last cache's probes/sym2probe for the merged dict
    at += n
    print(f'{os.path.basename(p)}: n={n}' + (' FLIPPED' if flip else ''), flush=True)

merged = {'X': X, 'expr': expr, 'obs_plate': plates, 'obs_platemap': pmaps,
          'features': shared, 'probes': meta['probes'],
          'sym2probe': meta['sym2probe'], 'probe2sym': meta['probe2sym'],
          'compartments': meta.get('compartments')}
out = os.path.join(_HERE, 'cache', 'cdrp_213.pkl')
pickle.dump(merged, open(out, 'wb'), protocol=4)
print(f'saved {out}: X {X.shape}, {len(set(plates))} plates, '
      f'{len(set(pmaps))} platemaps')

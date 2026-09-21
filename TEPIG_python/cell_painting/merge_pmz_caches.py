"""Merge the 10 Zaratan batch caches (cdrp_pmz_b1..b10.pkl) into ONE
cache/cdrp_50pm_merged.pkl for run_panel.py.

Each batch was extracted with its own 2-cluster GMM (the 93 GiB scratch quota
forced batched extraction), so cluster labels are arbitrary per batch. We align
them to batch 1 by correlating the cluster-mean feature profiles (mean over
wells and sites, on the shared feature set) and flipping a batch's clusters
when the crossed match correlates better. The printed matrix shows the match
quality -- expect within-match r >> cross-match r. The clean fix (one global
GMM over all 50 plates) needs the storage allocation; this is the documented
interim.

Usage:  python merge_pmz_caches.py            (run where cache/ holds b1..b10)
"""
import os, pickle
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHES = [f'cdrp_pmz_b{i}.pkl' for i in range(1, 11)]

cs = [pickle.load(open(os.path.join(_HERE, 'cache', c), 'rb')) for c in CACHES]
sets = [{str(f) for f in c['features']} for c in cs]
shared = [f for f in map(str, cs[0]['features']) if all(f in s for s in sets[1:])]
print(f'shared features: {len(shared)}')

def cols(c):
    idx = {str(f): i for i, f in enumerate(c['features'])}
    return [idx[f] for f in shared]

def profile(X):                       # (G, q, S, n) -> (G, q) cluster-mean profile
    return np.nan_to_num(X).mean(axis=(2, 3))

ref = profile(cs[0]['X'][:, cols(cs[0])])
Xs, exprs, plates = [], [], []
for bi, c in enumerate(cs):
    X = np.nan_to_num(c['X'][:, cols(c)]).astype(np.float32)
    p = profile(X)
    r = np.corrcoef(np.vstack([ref, p]))[:2, 2:]      # ref clusters x batch clusters
    flip = (r[0, 1] + r[1, 0]) > (r[0, 0] + r[1, 1])
    print(f'b{bi + 1}: match r = [{r[0, 0]:+.2f} {r[1, 1]:+.2f}] '
          f'crossed [{r[0, 1]:+.2f} {r[1, 0]:+.2f}]' + ('  -> FLIPPED' if flip else ''))
    if flip:
        X = X[::-1]
    Xs.append(X)
    exprs.append(np.asarray(c['expr'], float))
    plates += list(c['obs_plate'])

merged = dict(cs[0])
merged['X'] = np.concatenate(Xs, axis=3)
merged['expr'] = np.vstack(exprs)
merged['obs_plate'] = plates
merged['features'] = shared
out = os.path.join(_HERE, 'cache', 'cdrp_50pm_merged.pkl')
pickle.dump(merged, open(out, 'wb'), protocol=4)
print(f"saved {out}: X {merged['X'].shape}, expr {merged['expr'].shape}, "
      f"{len(set(plates))} plates")

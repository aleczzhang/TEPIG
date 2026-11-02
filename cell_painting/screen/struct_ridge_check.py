"""Does the subpopulation x site structure carry outcome information beyond the
well average?  (Diagnostic quoted in the paper's "Why accuracy is insensitive" paragraph.)

For every landmark gene, fit a dense ridge regression on
  (a) the AVERAGED design  n x q      (mean over clusters and sites),
  (b) the STRUCTURED design n x (G q S) (the full TEPIG tensor, flattened),
with platemap-grouped K-fold cross-validation and the SAME folds for both designs.
Ridge alpha is chosen per gene either by GCV on the training fold (--alpha gcv) or
"in hindsight", i.e. the alpha that maximizes pooled held-out R^2 (--alpha oracle);
the oracle choice is what the paper reports and is the most favourable to (b).
Output: results/struct_ridge_<tag>.csv with per-gene r2_avg, r2_struct, gain.

Memory: the structured design has G*q*S columns; with --preselect pycytominer on the
213-plate cache that is about 68k x 8k doubles (4.4 GB) plus one SVD per fold, which fits
a 48 GB node.  Without preselection (43k columns) use a 4-plate cache or a big-memory node.

Usage:
    python struct_ridge_check.py --caches cdrp_all213.pkl --preselect pycytominer \
        --folds 5 --alpha oracle
"""
import argparse, os, pickle, time

import numpy as np

import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_CP = os.path.dirname(_HERE)                       # cell_painting/
_ROOT = os.path.dirname(_CP)                       # repository root
sys.path[:0] = [_ROOT, _CP, os.path.join(_CP, 'panel'), os.path.join(_CP, 'screen')]
from scan_genes import ALPHAS
from compute_log import log_compute

PANEL = ['HMGCS1', 'NSDHL', 'HMGCR', 'FDFT1', 'HERPUD1', 'XBP1', 'TMEM97',
         'CREB3L2', 'INSIG1', 'CRELD2']


def load(cache):
    c = pickle.load(open(os.path.join(_CP, 'cache', cache), 'rb'))
    X = np.nan_to_num(c['X'].astype(np.float32))            # (G, q, S, n)
    probes = list(c['probes'])
    genes = sorted(c['sym2probe'])
    Y = np.column_stack([c['expr'][:, probes.index(c['sym2probe'][g])]
                         for g in genes]).astype(np.float64)
    grp = np.array(list(c.get('obs_platemap', c['obs_plate'])))
    ok = np.isfinite(Y).all(axis=1)
    return X[..., ok], Y[ok], grp[ok], [str(f) for f in c['features']], genes


def grouped_folds(groups, k, seed=0):
    u = np.array(sorted(set(groups)))
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    return [np.isin(groups, u[i::k]) for i in range(k)]


def ridge_paths(Xtr, Xte, Ytr_c):
    """Held-out predictions (centered scale) and GCV for every alpha, all genes at once."""
    U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
    B = U.T @ Ytr_c
    A = Xte @ Vt.T
    n_tr = Xtr.shape[0]
    pred = np.empty((len(ALPHAS), Xte.shape[0], Ytr_c.shape[1]))
    gcv = np.empty((len(ALPHAS), Ytr_c.shape[1]))
    for ai, al in enumerate(ALPHAS):
        shrink = s ** 2 / (s ** 2 + al)
        rss = ((Ytr_c - U @ (shrink[:, None] * B)) ** 2).sum(axis=0)
        gcv[ai] = n_tr * rss / (n_tr - shrink.sum()) ** 2
        pred[ai] = A @ ((s / (s ** 2 + al))[:, None] * B)
    return pred, gcv


def evaluate(Xn, Y, folds, alpha_rule):
    """Pooled held-out R^2 per gene for one design."""
    n, Gn = Y.shape
    yhat = np.zeros((len(ALPHAS), n, Gn))
    gcv_pick = np.zeros((len(folds), Gn), int)
    for fi, te in enumerate(folds):
        tr = ~te
        mu, sd = Xn[tr].mean(axis=0), Xn[tr].std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        Xtr, Xte = (Xn[tr] - mu) / sd, (Xn[te] - mu) / sd
        ym = Y[tr].mean(axis=0)
        pred, gcv = ridge_paths(Xtr, Xte, Y[tr] - ym)
        yhat[:, te, :] = pred + ym
        gcv_pick[fi] = gcv.argmin(axis=0)
    ss = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ((Y[None] - yhat) ** 2).sum(axis=1) / ss          # (alphas, genes)
    if alpha_rule == 'oracle':
        return r2.max(axis=0)
    # GCV: alpha may differ per fold; recompute pooled R^2 with the fold-wise picks
    yg = np.zeros((n, Gn))
    for fi, te in enumerate(folds):
        yg[te] = yhat[gcv_pick[fi], :, :][:, te, :][np.arange(Gn), :, np.arange(Gn)].T
    return 1.0 - ((Y - yg) ** 2).sum(axis=0) / ss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--caches', nargs='+', default=['cdrp_4plates.pkl'])
    ap.add_argument('--preselect', default='none', choices=['none', 'pycytominer'])
    ap.add_argument('--folds', type=int, default=5, help='grouped folds (0 = leave-one-platemap-out)')
    ap.add_argument('--alpha', default='oracle', choices=['oracle', 'gcv'])
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    if len(a.caches) != 1:
        raise SystemExit('pass one (merged) cache; merge_all_caches.py builds it')

    X, Y, grp, feats, genes = load(a.caches[0])
    G, q, S, n = X.shape
    Xavg = X.mean(axis=(0, 2)).T.astype(np.float64)               # n x q
    if a.preselect != 'none':
        import preselect as PS
        keep, desc = PS.unsupervised_select(a.preselect, Xavg, feats)
        print(f'preselect: {desc} -> q={len(keep)}', flush=True)
        X, Xavg = X[:, keep], Xavg[:, keep]
        q = len(keep)
    Xstr = X.reshape(G * q * S, n).T.astype(np.float64)           # n x (G q S)
    del X
    folds = (grouped_folds(grp, a.folds, a.seed) if a.folds
             else [grp == u for u in sorted(set(grp))])
    print(f'n={n} wells, {len(set(grp))} platemaps, {len(folds)} grouped folds, '
          f'averaged q={q}, structured {G * q * S} columns, alpha rule={a.alpha}',
          flush=True)

    r2_avg = evaluate(Xavg, Y, folds, a.alpha)
    print(f'averaged design done ({time.time() - t0:.0f}s)', flush=True)
    r2_str = evaluate(Xstr, Y, folds, a.alpha)
    print(f'structured design done ({time.time() - t0:.0f}s)', flush=True)
    gain = r2_str - r2_avg

    tag = os.path.splitext(a.caches[0])[0] + ('_' + a.preselect if a.preselect != 'none' else '')
    out = os.path.join(_CP, 'results', f'struct_ridge_{tag}_{a.alpha}.csv')
    order = np.argsort(-gain)
    with open(out, 'w') as f:
        f.write('gene,r2_avg,r2_struct,gain\n')
        for i in order:
            f.write(f'{genes[i]},{r2_avg[i]:.4f},{r2_str[i]:.4f},{gain[i]:.4f}\n')
    big = [genes[i] for i in order if gain[i] > 0.02]
    print(f'\ngenes gaining > 0.02 R^2 from the structure: {len(big)} / {len(genes)}: '
          f'{", ".join(big)}')
    print('panel genes (avg -> struct, gain):')
    for g in PANEL:
        if g in genes:
            i = genes.index(g)
            print(f'  {g:>8}: {r2_avg[i]:.3f} -> {r2_str[i]:.3f}  ({gain[i]:+.3f})')
    log_compute('struct_ridge_check', time.time() - t0, genes=len(genes), n=n, q=q,
                folds=len(folds), alpha=a.alpha, preselect=a.preselect)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()

"""Platemap-grouped all-gene screen (professor's design, 10 Sep 2026).

Merges the two 4-plate caches (8 plates = 8 distinct platemaps = 8 disjoint
compound sets, 2,560 wells) and scores every landmark gene by LEAVE-ONE-
PLATEMAP-OUT cross-validation: train on 7 platemaps, test on the held-out one,
rotate. Each gene is ranked by its MEDIAN held-out R^2 across the 8 folds, so a
high score means "predicts wells of compounds the model never saw" -- the
compound-set dependence seen between the two 4-plate scans is built into the
metric instead of checked afterwards.

Models are the same cheap screens as scan_genes.py: --model ridge (one SVD per
fold, all genes at once) or --model lasso (per-gene LassoCV; run with
--preselect pycytominer). Here each plate IS one platemap, so grouping by
plate == grouping by platemap.

Usage:  python scan_platemap_cv.py [--model ridge|lasso] [--preselect none|pycytominer]
Writes results/gene_scan_pmcv_<model>.csv (gene, r2_median, r2_mean, r2_min, r2_max).
"""
import argparse, os, pickle, time

import numpy as np

from genes import CELL_CYCLE
from scan_genes import ridge_scan_split, lasso_scan_split
from compute_log import log_compute

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHES = ['cdrp_4plates.pkl', 'cdrp_confirm4.pkl']


def load_merged(caches=CACHES):
    cs = [pickle.load(open(os.path.join(_HERE, 'cache', c), 'rb')) for c in caches]
    if len(cs) == 1:
        c = cs[0]
        Xn = np.nan_to_num(c['X'].mean(axis=(0, 2)).T.astype(np.float64))
        probes = list(c['probes'])
        genes = sorted(c['sym2probe'])
        Y = np.column_stack([c['expr'][:, probes.index(c['sym2probe'][g])]
                             for g in genes]).astype(float)
        cc = [probes.index(c['sym2probe'][g]) for g in CELL_CYCLE if g in c['sym2probe']]
        Z = c['expr'][:, cc]
        Z = (Z - Z.mean(axis=0)) / np.where(Z.std(axis=0) > 0, Z.std(axis=0), 1.0)
        Y = np.column_stack([Y, Z.mean(axis=1)])
        ok = np.isfinite(Y).all(axis=1)
        return (Xn[ok], Y[ok], np.array(list(c['obs_plate']))[ok],
                [str(f) for f in c['features']], genes + ['CELLCYCLE'])
    shared = [f for f in map(str, cs[0]['features'])
              if f in {str(g) for g in cs[1]['features']}]
    Xs, Ys, plates = [], [], []
    genes = sorted(cs[0]['sym2probe'])
    for c in cs:
        idx = {str(f): i for i, f in enumerate(c['features'])}
        cols = [idx[f] for f in shared]
        Xs.append(np.nan_to_num(c['X'].mean(axis=(0, 2)).T[:, cols].astype(np.float64)))
        probes = list(c['probes'])
        Y = np.column_stack([c['expr'][:, probes.index(c['sym2probe'][g])]
                             for g in genes]).astype(float)
        cc = [probes.index(c['sym2probe'][g]) for g in CELL_CYCLE if g in c['sym2probe']]
        Z = c['expr'][:, cc]
        Z = (Z - Z.mean(axis=0)) / np.where(Z.std(axis=0) > 0, Z.std(axis=0), 1.0)
        Ys.append(np.column_stack([Y, Z.mean(axis=1)]))
        plates += list(c['obs_plate'])
    Xn, Y = np.vstack(Xs), np.vstack(Ys)
    ok = np.isfinite(Y).all(axis=1)
    return Xn[ok], Y[ok], np.array(plates)[ok], shared, genes + ['CELLCYCLE']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='ridge', choices=['ridge', 'lasso'])
    ap.add_argument('--preselect', default='none', choices=['none', 'pycytominer'])
    ap.add_argument('--caches', nargs='+', default=CACHES,
                    help='cache pkl(s) in cache/; e.g. --caches cdrp_50pm.pkl')
    a = ap.parse_args()
    t0 = time.time()

    Xn, Y, plates, feats, genes = load_merged(a.caches)
    if a.preselect != 'none':
        import preselect as PS
        keep, desc = PS.unsupervised_select(a.preselect, Xn, feats)
        Xn = Xn[:, keep]
        print(f'preselect: {desc} -> q={len(keep)}', flush=True)
    folds = sorted(set(plates))
    print(f'platemap-CV scan: {len(genes)} genes, n={len(Y)} wells, q={Xn.shape[1]}, '
          f'{len(folds)} leave-one-platemap-out folds', flush=True)

    scan = ridge_scan_split if a.model == 'ridge' else lasso_scan_split
    r2 = np.full((len(folds), len(genes)), np.nan)
    for fi, p in enumerate(folds):
        t = time.time()
        te = np.where(plates == p)[0]
        tr = np.where(plates != p)[0]
        mu, sd = Xn[tr].mean(axis=0), Xn[tr].std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        Xtr, Xte = (Xn[tr] - mu) / sd, (Xn[te] - mu) / sd
        ym = Y[tr].mean(axis=0)
        yhat = scan(Xtr, Xte, Y[tr] - ym, ym, Y[te])
        Yte = Y[te]
        ss = ((Yte - Yte.mean(axis=0)) ** 2).sum(axis=0)
        r2[fi] = 1.0 - ((Yte - yhat) ** 2).sum(axis=0) / np.where(ss > 0, ss, np.nan)
        np.save(os.path.join(_HERE, 'results', f'gene_scan_pmcv_{a.model}_partial.npy'), r2)
        print(f'  fold {fi + 1}/{len(folds)} (hold out {p}): {time.time() - t:.1f}s', flush=True)

    med = np.median(r2, axis=0)
    order = np.argsort(-med)
    out = os.path.join(_HERE, 'results', f'gene_scan_pmcv_{a.model}.csv')
    with open(out, 'w') as f:
        f.write('gene,r2_median,r2_mean,r2_min,r2_max\n')
        for i in order:
            f.write(f'{genes[i]},{med[i]:.4f},{r2[:, i].mean():.4f},'
                    f'{r2[:, i].min():.4f},{r2[:, i].max():.4f}\n')

    print(f'\nmedian held-out R2 over {len(genes)} genes: '
          f'>0.05: {(med > 0.05).sum()}, >0.10: {(med > 0.10).sum()}, '
          f'>0.15: {(med > 0.15).sum()}\ntop 20:')
    for i in order[:20]:
        print(f'  {genes[i]:>10}  median {med[i]:+.3f}  '
              f'[{r2[:, i].min():+.3f}, {r2[:, i].max():+.3f}]')
    log_compute('gene_scan_pmcv', time.time() - t0, genes=len(genes), n=len(Y),
                q=Xn.shape[1], folds=len(folds), model=a.model, preselect=a.preselect)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()

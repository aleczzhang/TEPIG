"""Honest all-gene ridge scan: which L1000 genes have morphology signal at all?

Stage-1 triage for the gene panel (see docs/arm_comparison.md discussion, 6 Sep):
run a cheap outcome-agnostic model over EVERY landmark gene with proper
train/test splits and report the full R^2 distribution -- no gene is selected,
so there is no winner's curse. The full TEPIG/CLUSSO/naive pipeline then runs
only on the top tier (confirmed on plates this scan did not use, once more
plates are extracted).

Model: ridge regression on the well-level matrix Xn = X.mean(G, S) with all raw
features (ridge is fine at q > n). Per split, alpha is chosen PER GENE by GCV on
the training fold only; test wells never touch model choice.

Speed trick: X is the same for every gene, so each split needs ONE SVD of the
standardized training matrix; all 977 genes' ridge fits and GCV curves are then
matrix products in that basis. Whole scan runs in minutes, not hours.

Caveat found on first run (6 Sep): dense ridge dilutes SPARSE signal -- CCNA2 and
CDKN1A score ~0 under ridge but 0.06-0.15 under the lasso-based pipeline. So the
scan has two models: --model ridge (dense signal, seconds) and --model lasso
(sparse signal, sklearn LassoCV per gene, ~1 h with --preselect pycytominer).
A gene "has signal" if EITHER scan finds it.

Second caveat (7 Sep): both of those see only the well-AVERAGED matrix, so a
gene whose signal lives in subpopulation x site differences -- exactly where
TEPIG should beat naive -- is under-ranked. --model ridge_struct runs the same
shared-SVD ridge on the unaveraged (G x q x S) block design (12 blocks, 42,948
columns); (struct - avg) R^2 per gene is a cheap "heterogeneity gain" screen.
NOTE: GCV alpha selection is unreliable at q=43k (most genes land far negative
regardless of the alpha grid); the honest read is the ORACLE-alpha comparison
(see 7 Sep session notes): even with the best alpha in hindsight, struct beats
avg for only 12/978 genes by >0.02 (RGS2 +0.08 the standout; HMGCS1, UGDH,
ATP1B1 next), and every current panel gene prefers the averaged design. So the
averaged screen is NOT materially biased against TEPIG-friendly genes on this
data -- consistent with the TEPIG-vs-naive tie.

Usage:  python scan_genes.py [--cache cdrp_4plates.pkl] [--splits 5]
                             [--model ridge|lasso] [--preselect none|pycytominer]
Writes results/gene_scan_<model>_<cache>.csv (gene, n, r2_mean, r2_sd), best first.
"""
import argparse, os, pickle, time

import numpy as np

import run_gene as R
from genes import CELL_CYCLE
from compute_log import log_compute

_HERE = os.path.dirname(os.path.abspath(__file__))
ALPHAS = np.logspace(1, 8, 22)


def ridge_scan_split(Xtr, Xte, Ytr_c, ym, Yte):
    """All genes at once via one SVD; per-gene alpha by GCV on the train fold."""
    U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
    B = U.T @ Ytr_c
    A = Xte @ Vt.T
    n_tr, G = Ytr_c.shape
    gcv = np.empty((len(ALPHAS), G))
    pred = np.empty((len(ALPHAS), len(Xte), G))
    for ai, al in enumerate(ALPHAS):
        shrink = s ** 2 / (s ** 2 + al)
        rss = ((Ytr_c - U @ (shrink[:, None] * B)) ** 2).sum(axis=0)
        gcv[ai] = n_tr * rss / (n_tr - shrink.sum()) ** 2
        pred[ai] = A @ ((s / (s ** 2 + al))[:, None] * B) + ym
    best = gcv.argmin(axis=0)
    return pred[best, :, np.arange(G)].T


def lasso_scan_split(Xtr, Xte, Ytr_c, ym, Yte):
    """Per-gene LassoCV (alpha chosen by 3-fold CV on the train fold only).

    eps=1e-2 stops the alpha path before the barely-regularized fits that never
    converge (they were ~all the compute); threadpool_limits(1) stops each loky
    worker from spawning a full BLAS thread pool (workers were thrashing cores).
    """
    from joblib import Parallel, delayed
    from sklearn.linear_model import LassoCV
    from threadpoolctl import threadpool_limits

    def one(j):
        with threadpool_limits(1):
            m = LassoCV(alphas=20, eps=1e-2, cv=3, max_iter=2000,
                        tol=1e-3, n_jobs=1).fit(Xtr, Ytr_c[:, j])
            return m.predict(Xte) + ym[j]

    n_jobs = int(os.environ.get('SCAN_NJOBS', 6))
    cols = Parallel(n_jobs=n_jobs)(delayed(one)(j) for j in range(Ytr_c.shape[1]))
    return np.column_stack(cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='cdrp_4plates.pkl')
    ap.add_argument('--splits', type=int, default=5)
    ap.add_argument('--model', default='ridge', choices=['ridge', 'ridge_struct', 'lasso'])
    ap.add_argument('--preselect', default='none', choices=['none', 'pycytominer'])
    a = ap.parse_args()
    t0 = time.time()

    c = pickle.load(open(os.path.join(_HERE, 'cache', a.cache), 'rb'))
    if a.model == 'ridge_struct':                # (n, G*q*S): keep cluster x site blocks
        if a.preselect != 'none':
            raise SystemExit('ridge_struct uses raw feature blocks; use --preselect none')
        Xn = np.nan_to_num(c['X'].transpose(3, 0, 1, 2)
                           .reshape(c['X'].shape[3], -1).astype(np.float64))
    else:
        Xn = np.nan_to_num(c['X'].mean(axis=(0, 2)).T.astype(np.float64))  # (n, q)
    expr, probes, sym2probe = c['expr'], list(c['probes']), c['sym2probe']
    if a.preselect != 'none':
        import preselect as PS
        keep, desc = PS.unsupervised_select(a.preselect, Xn,
                                            [str(f) for f in c['features']])
        Xn = Xn[:, keep]
        print(f'preselect: {desc} -> q={len(keep)}', flush=True)

    genes = sorted(sym2probe)
    Y = np.column_stack([expr[:, probes.index(sym2probe[g])] for g in genes]).astype(float)
    cc = [probes.index(sym2probe[g]) for g in CELL_CYCLE if g in sym2probe]
    Z = expr[:, cc]
    Z = (Z - Z.mean(axis=0)) / np.where(Z.std(axis=0) > 0, Z.std(axis=0), 1.0)
    Y = np.column_stack([Y, Z.mean(axis=1)])
    genes.append('CELLCYCLE')

    ok = np.isfinite(Y).all(axis=1)             # wells usable for every gene
    Xn, Y = Xn[ok], Y[ok]
    n, q = Xn.shape
    print(f'scan: {len(genes)} genes, n={n} wells (dropped {int((~ok).sum())}), '
          f'q={q}, {a.splits} splits', flush=True)

    scan = lasso_scan_split if a.model == 'lasso' else ridge_scan_split
    r2 = np.full((a.splits, len(genes)), np.nan)
    for si in range(a.splits):
        t = time.time()
        tr, te = R.plain_split(n, R.TEST_FRAC, 42 + 100 * si)
        mu, sd = Xn[tr].mean(axis=0), Xn[tr].std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        Xtr, Xte = (Xn[tr] - mu) / sd, (Xn[te] - mu) / sd
        ym = Y[tr].mean(axis=0)
        yhat = scan(Xtr, Xte, Y[tr] - ym, ym, Y[te])
        Yte = Y[te]
        ss_tot = ((Yte - Yte.mean(axis=0)) ** 2).sum(axis=0)
        r2[si] = 1.0 - ((Yte - yhat) ** 2).sum(axis=0) / np.where(ss_tot > 0, ss_tot, np.nan)
        np.save(os.path.join(_HERE, 'results',
                f'gene_scan_{a.model}_partial.npy'), r2)   # checkpoint per split
        print(f'  split {si + 1}/{a.splits}: {time.time() - t:.1f}s', flush=True)

    mean, sd_ = r2.mean(axis=0), r2.std(axis=0, ddof=1)
    order = np.argsort(-mean)
    tag = os.path.splitext(a.cache)[0]
    out = os.path.join(_HERE, 'results', f'gene_scan_{a.model}_{tag}.csv')
    with open(out, 'w') as f:
        f.write('gene,n,r2_mean,r2_sd\n')
        for i in order:
            f.write(f'{genes[i]},{n},{mean[i]:.4f},{sd_[i]:.4f}\n')

    print(f'\ndistribution over {len(genes)} genes: '
          f'median {np.median(mean):+.3f}, 90th pct {np.percentile(mean, 90):+.3f}, '
          f'max {mean.max():+.3f};  R2>0.05: {(mean > 0.05).sum()}, '
          f'R2>0.10: {(mean > 0.10).sum()}, R2>0.15: {(mean > 0.15).sum()}')
    print(f'\ntop 25:')
    for i in order[:25]:
        print(f'  {genes[i]:>10}  {mean[i]:+.3f} ± {sd_[i]:.3f}')
    log_compute('gene_scan', time.time() - t0, genes=len(genes), n=n, q=q,
                splits=a.splits, model=a.model, preselect=a.preselect)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()

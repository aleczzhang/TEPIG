"""
preselect.py
------------
Thin wrappers that plug EXISTING feature pre-screening methods (see
docs/feature_screening_survey.md) into repeated_runs.py / run_panel.py. Nothing
here is a new method; each entry names the package or paper it calls.

The arms are INDEPENDENT: each starts from the full raw morphology feature set in
the cache (cdrp_extract.py no longer prunes at the cell level; only nuisance
columns and constant features are removed) and reduces it its own way. Nothing
is chained.

Unsupervised (never see y; applied ONCE to the well-level matrix of all wells):
  prune        greedy |r| <= thr prune (prune_sweep.prune_correlated; caret::findCorrelation)
  varclust     ClustOfVar-style variable clustering, one real rep per cluster (varclust.py)
  pycytominer  the Cell Painting community standard: pycytominer.feature_select with
               variance_threshold + correlation_threshold (0.9) + drop_na + blocklist
               (Serrano et al.; Broad profiling-recipe defaults). The kept count falls
               out of the data.

Supervised (use y; applied INSIDE every train/test split to the training wells):
  sis          Sure Independence Screening (Fan & Lv 2008): top-d |Pearson corr(x_j, y)|
  dcsis        DC-SIS (Li, Zhong & Zhu 2012): top-d distance correlation, via the
               `dcor` package (Ramos-Carreno 2023)
  default d = floor(n_train / log n_train)  (Fan & Lv's rule)

Stability selection (Meinshausen & Buhlmann 2010) is NOT wired: the
scikit-learn-contrib package does not build on Python 3.14; use R `stabs` if wanted.
"""

import numpy as np

UNSUPERVISED = ('prune', 'varclust', 'pycytominer')
SUPERVISED = ('sis', 'dcsis')
ALL = UNSUPERVISED + SUPERVISED


def sis_d(n):
    """Fan & Lv (2008) screening size d = floor(n / log n)."""
    return int(max(1, np.floor(n / np.log(n))))


def is_supervised(method):
    return method in SUPERVISED


# ── unsupervised: once, on all wells ────────────────────────────────────────
def unsupervised_select(method, Xn, feats, k=None, thr=None):
    """Xn: (n_wells, q) well-level matrix; feats: names. -> (kept indices, description)."""
    if method == 'prune':
        from prune_sweep import prune_correlated
        Rabs = np.nan_to_num(np.abs(np.corrcoef(Xn.T)), nan=0.0)
        thr = thr or 0.95
        return sorted(prune_correlated(Rabs, thr)), f'prune |r|<={thr}'
    if method == 'varclust':
        import varclust as VC
        if k:
            return sorted(VC.select(Xn, k, rep='pc1')), f'varclust k={k}'
        thr = thr or 0.7
        return sorted(VC.select_tight(Xn, thr, rep='pc1')), f'varclust |r|>={thr}'
    if method == 'pycytominer':
        import pandas as pd
        from pycytominer import feature_select
        thr = thr or 0.9
        df = pd.DataFrame(np.asarray(Xn, dtype=float), columns=list(feats))
        out = feature_select(df, features='infer',
                             operation=['variance_threshold', 'correlation_threshold',
                                        'drop_na_columns', 'blocklist'],
                             corr_threshold=thr)
        pos = {f: i for i, f in enumerate(feats)}
        return sorted(pos[f] for f in out.columns), f'pycytominer |r|<={thr}+blocklist'
    raise ValueError(f'unknown unsupervised method {method!r}')


# ── supervised: closure applied inside each split on the training wells ─────
def make_supervised_screen(method, k=None):
    """-> f(Xn_tr, y_tr) returning kept indices (sorted). k=None -> n_train/log n_train."""
    def sis(Xn_tr, y_tr):
        d = k or sis_d(len(y_tr))
        X = np.asarray(Xn_tr, float); X = X - X.mean(0)
        sd = X.std(0); sd[sd == 0] = 1.0
        yc = y_tr - y_tr.mean()
        score = np.abs((X / sd).T @ yc)            # |corr| up to a constant
        return sorted(np.argsort(-score, kind='stable')[:d].tolist())

    def dcsis(Xn_tr, y_tr):
        import dcor
        d = k or sis_d(len(y_tr))
        X = np.asarray(Xn_tr, float)
        y = np.asarray(y_tr, float)
        score = dcor.rowwise(dcor.distance_correlation, X.T, np.tile(y, (X.shape[1], 1)))
        score = np.nan_to_num(np.asarray(score), nan=0.0)
        return sorted(np.argsort(-score, kind='stable')[:d].tolist())

    f = {'sis': sis, 'dcsis': dcsis}[method]
    f.method = method
    f.desc = f'{method} d={k or "n/log n"} (inside each split)'
    return f

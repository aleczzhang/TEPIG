"""
selection_metrics.py
--------------------
Selection-CONSISTENCY metrics for real data, where TPR / FPR are unavailable
(no ground-truth support). Given the feature sets selected over M repeated runs
(different train/test splits, CV folds, restarts), we report:

  jaccard   mean pairwise Jaccard overlap (what stability_check.py used). Simple,
            but NOT chance-corrected: a method that selects 90 % of all features
            every time gets a high Jaccard for free.
  kuncheva  Kuncheva (2007) consistency index, generalised to unequal set sizes:
            pairwise (|A&B| - E|A&B|) / (min(|A|,|B|) - E|A&B|) with
            E|A&B| = |A||B|/q the overlap expected by chance. 1 = identical,
            ~0 = no better than random, negative = worse than random.
  nogueira  Nogueira, Sechidis & Brown (2018, JMLR 19) stability estimator
            Phi = 1 - mean_f s_f^2 / [ (kbar/q)(1 - kbar/q) ],  s_f^2 the
            sample variance of feature f's 0/1 selection indicator across runs.
            Chance-corrected, handles variable set sizes, has a confidence
            interval, and is the current default in the feature-selection
            stability literature. <= 1; 1 = perfectly stable; ~0 = random.
            THIS is the recommended single number to replace TPR/FPR on real data.
  selection probabilities  per feature, pi_f = (# runs selecting f) / M -- the
            stability-selection quantity; features with pi_f >= 0.6..0.9 form the
            "stable set" (Meinshausen & Buhlmann 2010). Unlike the L1-normalised
            frequencies, pi_f is interpretable on its own (it is a probability).
"""

import itertools
import numpy as np


def _to_matrix(sets, q):
    """List of index lists -> (M, q) 0/1 matrix."""
    Z = np.zeros((len(sets), q), dtype=float)
    for i, s in enumerate(sets):
        Z[i, list(s)] = 1.0
    return Z


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 1.0


def mean_jaccard(sets):
    pairs = list(itertools.combinations(range(len(sets)), 2))
    return float(np.mean([jaccard(sets[i], sets[j]) for i, j in pairs])) if pairs else float('nan')


def kuncheva(sets, q):
    vals = []
    for A, B in itertools.combinations([set(s) for s in sets], 2):
        a, b, r = len(A), len(B), len(A & B)
        exp = a * b / q
        den = min(a, b) - exp
        if den > 1e-12:
            vals.append((r - exp) / den)
        elif a == 0 and b == 0:
            vals.append(1.0)
    return float(np.mean(vals)) if vals else float('nan')


def nogueira(sets, q):
    """Point estimate of Nogueira et al. (2018) stability Phi."""
    Z = _to_matrix(sets, q)
    M = Z.shape[0]
    if M < 2:
        return float('nan')
    p = Z.mean(axis=0)
    s2 = M / (M - 1) * p * (1 - p)               # unbiased per-feature variance
    kbar = Z.sum(axis=1).mean()
    den = (kbar / q) * (1 - kbar / q)
    if den <= 0:
        return float('nan')                       # all-or-nothing selections
    return float(1.0 - s2.mean() / den)


def nogueira_ci(sets, q, B=2000, alpha=0.05, seed=0):
    """Bootstrap CI over the M runs (resample runs with replacement)."""
    rng = np.random.default_rng(seed)
    M = len(sets)
    if M < 3:
        return (float('nan'), float('nan'))
    vals = []
    for _ in range(B):
        idx = rng.integers(0, M, M)
        v = nogueira([sets[i] for i in idx], q)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float('nan'), float('nan'))
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def selection_probabilities(sets, q):
    return _to_matrix(sets, q).mean(axis=0)


def stable_set(sets, q, pi=0.6):
    p = selection_probabilities(sets, q)
    return sorted(np.where(p >= pi)[0].tolist())


def summarize(sets, q, pis=(0.6, 0.8)):
    """All metrics in one dict."""
    sizes = [len(s) for s in sets]
    lo, hi = nogueira_ci(sets, q)
    out = {'runs': len(sets), 'q': q,
           'size_mean': float(np.mean(sizes)) if sizes else 0.0,
           'size_sd': float(np.std(sizes)) if sizes else 0.0,
           'jaccard': mean_jaccard(sets), 'kuncheva': kuncheva(sets, q),
           'nogueira': nogueira(sets, q), 'nogueira_ci': (lo, hi)}
    for pi in pis:
        out[f'stable_{pi}'] = stable_set(sets, q, pi)
    return out


def format_row(name, s):
    lo, hi = s['nogueira_ci']
    return (f"{name:<8}{s['size_mean']:>7.1f} ± {s['size_sd']:<5.1f}"
            f"{s['jaccard']:>9.2f}{s['kuncheva']:>10.2f}"
            f"{s['nogueira']:>10.2f} [{lo:+.2f},{hi:+.2f}]"
            + ''.join(f"{len(s[k]):>9}" for k in s if k.startswith('stable_')))


def header():
    return (f"{'method':<8}{'#selected':>14}{'Jaccard':>9}{'Kuncheva':>10}"
            f"{'Nogueira Φ [95% CI]':>24}{'|pi>=.6|':>9}{'|pi>=.8|':>9}")

"""
varclust.py
-----------
ClustOfVar-style feature selection (Chavent et al. 2012, JSS 50(13)), following
the professor's variant: hierarchically cluster the FEATURES by their correlation
structure, then keep ONE real representative feature per cluster (the medoid --
the feature most correlated, on average, with its cluster-mates) instead of
averaging or using a synthetic component. Keeps interpretability: every surviving
feature is an actual measured quantity.

Distance between two features:  d = 1 - r^2   (strong +/- correlation -> close)
Linkage: average.  Cut into k clusters; representative = within-cluster medoid.

This module is diagnostics-first: it exposes the merge structure so the "right"
number of clusters can be CHOSEN and defended, not guessed --
  * natural_cuts()   : the biggest jumps in merge height (natural cut levels),
                       reported as the correlation at which the next merge fuses.
  * describe_level() : for a chosen k -- cluster sizes, each representative, a few
                       members, and the NEXT merge that would happen (which two
                       groups, at what |r|).

select() returns the kept feature indices for a given k, to drop into the
pipeline in place of the greedy 0.95 prune.
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score


def corr_and_linkage(cols, method='complete'):
    """cols: (n_samples, n_features). Returns (|corr| matrix, linkage Z).

    Default linkage is 'complete': cutting the tree at height 1 - thr^2 then
    guarantees EVERY within-cluster pair has |r| >= thr (no chaining, no loose
    'junk-drawer' clusters). 'average' chains badly here; 'ward' is a balanced
    alternative but without the pairwise guarantee."""
    C = np.corrcoef(cols.T)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 1.0)
    Rabs = np.abs(C)
    d = 1.0 - C ** 2                      # squared-correlation distance in [0,1]
    np.fill_diagonal(d, 0.0)
    d = np.clip(d, 0.0, None)
    Z = linkage(squareform(d, checks=False), method=method)
    return Rabs, Z


def _height_to_r(h):
    """Merge height h = 1 - r^2  ->  |r| = sqrt(1 - h)."""
    return float(np.sqrt(max(0.0, 1.0 - h)))


def natural_cuts(Z, top=10):
    """Largest jumps in successive merge heights -> natural numbers of clusters.
    Returns rows (k, next_merge_|r|, gap): stop at k clusters; the NEXT merge
    would fuse two groups whose linkage corresponds to correlation next_merge_|r|
    (lower = the next merge joins more dissimilar groups = safer place to stop)."""
    h = Z[:, 2]
    n = len(h) + 1
    gaps = np.diff(h)
    idx = np.argsort(gaps)[::-1][:top]
    rows = []
    for i in sorted(idx):
        k = n - (i + 1)                   # clusters remaining AFTER merge i
        rows.append((k, _height_to_r(h[i + 1]), float(gaps[i])))
    return sorted(rows)                    # by k


def _medoid(members, Rabs):
    """Representative = member with the highest average |r| to the others."""
    if len(members) == 1:
        return members[0]
    sub = Rabs[np.ix_(members, members)]
    return members[int(np.argmax(sub.sum(axis=1)))]


def _pc1_rep(members, cols):
    """Representative = the real feature most correlated with the cluster's first
    principal component (ClustOfVar's criterion, but returning an actual feature
    rather than the synthetic component)."""
    if len(members) == 1:
        return members[0]
    z = cols[:, members].astype(float)
    z = z - z.mean(axis=0)
    sd = z.std(axis=0)
    sd[sd == 0] = 1.0
    z = z / sd
    # PC1 scores (unit-norm left singular vector); corr of each feature with it
    U, _, _ = np.linalg.svd(z, full_matrices=False)
    pc1 = U[:, 0]
    norm = np.sqrt((z ** 2).sum(axis=0))
    norm[norm == 0] = 1.0
    cors = np.abs(z.T @ pc1) / norm
    return members[int(np.argmax(cors))]


def _reps_from_labels(labels, Rabs, cols=None, rep='pc1'):
    reps = {}
    for c in np.unique(labels):
        members = np.where(labels == c)[0].tolist()
        if rep == 'pc1' and cols is not None:
            reps[c] = _pc1_rep(members, cols)
        else:
            reps[c] = _medoid(members, Rabs)
    return reps


def select(cols, k, method='complete', rep='pc1', return_all=False):
    """Cluster columns into exactly k groups; keep one representative each
    (rep='pc1' -> most correlated with cluster PC1; 'medoid' -> highest avg |r|).
    Returns kept indices; with return_all also (labels, reps, Rabs, Z)."""
    Rabs, Z = corr_and_linkage(cols, method)
    labels = fcluster(Z, k, criterion='maxclust')
    reps = _reps_from_labels(labels, Rabs, cols, rep)
    kept = sorted(reps.values())
    return (kept, labels, reps, Rabs, Z) if return_all else kept


def select_tight(cols, thr, method='complete', rep='pc1', return_all=False):
    """Keep one representative per group of features that are ALL mutually
    correlated at |r| >= thr (complete-linkage cut at height 1 - thr^2). The
    number of survivors falls out of the data. This is the cluster-based
    generalization of the greedy |r| <= thr prune, and the recommended mode."""
    Rabs, Z = corr_and_linkage(cols, method)
    labels = fcluster(Z, 1.0 - thr ** 2, criterion='distance')
    reps = _reps_from_labels(labels, Rabs, cols, rep)
    kept = sorted(reps.values())
    return (kept, labels, reps, Rabs, Z) if return_all else kept


def bootstrap_stability(cols, ks, B=50, method='complete', seed=0):
    """ClustOfVar-style stability selection for the number of clusters.

    Resample the ROWS (wells) with replacement B times; on each resample
    recompute the feature correlations and recluster; for each candidate number
    of clusters k, compare the bootstrap partition of the p features against the
    full-data partition via the adjusted Rand index. Return, per k, the mean and
    sd of the ARI over the B bootstraps (higher, flatter = more reproducible =
    a better-supported level). This never touches the outcome and does not care
    whether q>n, so it is valid even on a single plate.

    Returns dict: k -> (mean_ARI, sd_ARI, cutoff_|r|), where cutoff_|r| is the
    correlation level at which the full-data hierarchy yields k clusters."""
    n, p = cols.shape
    Rabs0, Z0 = corr_and_linkage(cols, method)
    base = {k: fcluster(Z0, k, criterion='maxclust') for k in ks}
    # |r| at which the full-data tree splits into k clusters (the next merge up)
    cutoff = {}
    for k in ks:
        h = Z0[p - k, 2] if k < p else 0.0        # height of the merge that would go k -> k-1
        cutoff[k] = _height_to_r(h)
    rng = np.random.default_rng(seed)
    ari = {k: [] for k in ks}
    for _ in range(B):
        idx = rng.integers(0, n, n)               # bootstrap resample of wells
        _, Zb = corr_and_linkage(cols[idx], method)
        for k in ks:
            lb = fcluster(Zb, k, criterion='maxclust')
            ari[k].append(adjusted_rand_score(base[k], lb))
    return {k: (float(np.mean(ari[k])), float(np.std(ari[k])), cutoff[k]) for k in ks}


def describe_level(labels, reps, Rabs, Z, feats, k, n_examples=4, show_clusters=8):
    """Human-readable dump of a k-cluster cut and the next merge that follows."""
    feats = np.asarray(feats)
    uniq = np.unique(labels)
    sizes = np.array([(labels == c).sum() for c in uniq])
    print(f"\n=== {k} clusters (from {len(feats)} features) ===")
    print(f"  cluster sizes: min={sizes.min()} median={int(np.median(sizes))} "
          f"max={sizes.max()}  ({int((sizes == 1).sum())} singletons)")

    # tightness: within-cluster minimum |r| to the representative
    order = np.argsort(-sizes)            # biggest clusters first
    print(f"  largest {show_clusters} clusters (representative + members):")
    for c in uniq[order][:show_clusters]:
        members = np.where(labels == c)[0]
        rep = reps[c]
        rmin = Rabs[rep, members].min()
        others = [feats[m] for m in members if m != rep][:n_examples]
        print(f"    [{len(members):>3}] rep: {feats[rep]}")
        print(f"          min |r| of rep to members = {rmin:.2f}")
        for o in others:
            print(f"          + {o}")
        if len(members) - 1 > n_examples:
            print(f"          + ... {len(members) - 1 - n_examples} more")

    # what the NEXT merge would do (k -> k-1)
    if k > 1:
        nxt = fcluster(Z, k - 1, criterion='maxclust')
        # the two level-k clusters that share a level-(k-1) label are the pair
        merged = None
        for c in np.unique(nxt):
            grp = np.unique(labels[nxt == c])
            if len(grp) == 2:
                merged = grp
                break
        if merged is not None:
            a, b = merged
            ra, rb = reps[a], reps[b]
            print(f"  NEXT merge (going {k} -> {k-1}) would fuse:")
            print(f"    '{feats[ra]}'  +  '{feats[rb]}'")
            print(f"    at |r| = {Rabs[ra, rb]:.2f} between their representatives")
            print(f"    (high |r| => you cut too late/early is fine; low |r| => "
                  f"this merge would force dissimilar groups together)")

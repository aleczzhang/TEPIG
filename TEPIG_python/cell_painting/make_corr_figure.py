"""
make_corr_figure.py
-------------------
Feature correlation structure for the LINCS Cell Painting predictors.

Motivation (from the 11 Jul meeting): lasso-type selection is unstable when
predictors are highly correlated — it picks one representative from a clump of
near-duplicates. This figure asks whether TEPIG's sparsity is explained by that:
are its selected features spread across distinct correlation clumps?

Produces: figures/fig4_correlation.png
  left  : q x q feature correlation matrix at one dose, seriated so correlated
          blocks sit together; features TEPIG selects are marked on the axis.
  right : size of each correlation clump, with how many TEPIG selects from it.
"""

import os
import glob
import pickle
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(_HERE, 'cache', 'cp_lincs_tensor.pkl')
RES = os.path.join(_HERE, 'results')
FIG = os.path.join(_HERE, 'figures')
os.makedirs(FIG, exist_ok=True)

DOSE = 6
R_THRESH = 0.8          # clump definition
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e3e2df'
ACCENT = '#2a78d6'

# diverging ramp: cool -> neutral grey midpoint -> warm  (never a hue at the middle)
DIV = LinearSegmentedColormap.from_list(
    'div', ['#123a68', '#2a78d6', '#ececeb', '#e34948', '#8d1f1f'])

plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': GRID, 'text.color': INK,
    'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})


def average_linkage(D, k_thresh):
    """Agglomerative average-linkage clustering on distance D; merge while the
    average linkage distance is below k_thresh. Returns cluster labels."""
    n = D.shape[0]
    clusters = {i: [i] for i in range(n)}
    while True:
        best, bi, bj = np.inf, None, None
        keys = list(clusters)
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                ia, ib = keys[a], keys[b]
                d = D[np.ix_(clusters[ia], clusters[ib])].mean()
                if d < best:
                    best, bi, bj = d, ia, ib
        if bi is None or best >= k_thresh:
            break
        clusters[bi] = clusters[bi] + clusters[bj]
        del clusters[bj]
    lab = np.zeros(n, dtype=int)
    for c, (_, members) in enumerate(sorted(clusters.items(),
                                            key=lambda kv: -len(kv[1]))):
        for m in members:
            lab[m] = c
    return lab


def main():
    c = pickle.load(open(CACHE, 'rb'))
    feats, X, dr = c['features'], c['X'], c['obs_dose_rank']
    Xn = X[:, :, :, dr == DOSE].mean(axis=(0, 2)).T      # (n, q)
    R = np.corrcoef(Xn.T)
    q = len(feats)

    # clumps via average linkage on correlation distance
    D = 1.0 - np.abs(R)
    lab = average_linkage(D, 1.0 - R_THRESH)
    sizes = Counter(lab)

    # seriate: order by clump (largest first), then by mean |r| within clump
    order = sorted(range(q), key=lambda j: (lab[j], -np.abs(R[j]).mean()))
    Rs = R[np.ix_(order, order)]
    fs = [feats[j] for j in order]
    ls = [lab[j] for j in order]

    # how often TEPIG selects each feature at this dose
    tep = Counter()
    for f in glob.glob(os.path.join(RES, f'gene_*_dose{DOSE}.pkl')):
        for ft in pickle.load(open(f, 'rb'))['results']['TEPIG']['selected']:
            tep[ft] += 1

    fig = plt.figure(figsize=(14.6, 8.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], wspace=0.28)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(Rs, cmap=DIV, vmin=-1, vmax=1, aspect='equal')

    # clump boundaries
    bounds = [i for i in range(1, q) if ls[i] != ls[i - 1]]
    for b in bounds:
        ax.axhline(b - .5, color='white', linewidth=1.6)
        ax.axvline(b - .5, color='white', linewidth=1.6)

    # mark TEPIG-selected features on the y axis
    picked = [i for i, f in enumerate(fs) if tep[f] > 0]
    ax.set_yticks(picked)
    ax.set_yticklabels([f'{fs[i]}  ({tep[fs[i]]})' for i in picked], fontsize=6.5)
    for t in ax.get_yticklabels():
        t.set_color(ACCENT)
    ax.set_xticks([])
    ax.set_title(f'Feature correlation matrix  ·  dose {DOSE}  ·  q={q}\n'
                 f'blocks = clumps at |r| > {R_THRESH};  '
                 f'labelled in blue = selected by TEPIG (n genes)',
                 fontsize=10.5, color=INK, loc='left', pad=10)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label('Pearson r', fontsize=9, color=MUTED)
    cb.outline.set_visible(False)

    # right: clump sizes vs TEPIG picks per clump
    ax2 = fig.add_subplot(gs[1])
    cl_ids = [cid for cid, _ in sorted(sizes.items(), key=lambda kv: -kv[1])]
    tot = [sizes[cid] for cid in cl_ids]
    hit = [sum(1 for j in range(q) if lab[j] == cid and tep[feats[j]] > 0)
           for cid in cl_ids]
    ypos = np.arange(len(cl_ids))
    ax2.barh(ypos, tot, color=GRID, height=0.68, label='features in clump', zorder=2)
    ax2.barh(ypos, hit, color=ACCENT, height=0.68, label='selected by TEPIG', zorder=3)
    for i, (t, h) in enumerate(zip(tot, hit)):
        ax2.text(t + 0.6, i, f'{h}/{t}', va='center', fontsize=8.5, color=INK)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([f'clump {i+1}' for i in range(len(cl_ids))], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel('# features', fontsize=9.5, color=MUTED)
    ax2.set_title('TEPIG draws from most clumps, but only a few members of each',
                  fontsize=10.5, color=INK, loc='left', pad=10)
    ax2.legend(frameon=False, fontsize=9, loc='lower right')
    ax2.grid(axis='x', color=GRID, linewidth=0.8, zorder=0)
    ax2.set_axisbelow(True)
    for s in ['top', 'right', 'left']:
        ax2.spines[s].set_visible(False)
    ax2.tick_params(length=0)

    fig.savefig(os.path.join(FIG, 'fig4_correlation.png'), dpi=200,
                bbox_inches='tight')
    print('  wrote fig4_correlation.png')

    # console summary
    iu = np.triu_indices(q, 1)
    r = np.abs(R[iu])
    print(f'\n  |r|>0.9: {(r>0.9).sum()}  |r|>0.8: {(r>0.8).sum()}  '
          f'of {len(r)} pairs;  median |r| = {np.median(r):.2f}')
    print(f'  clumps at |r|>{R_THRESH}: {len(sizes)}  sizes={sorted(sizes.values(), reverse=True)}')
    for cid, t, h in zip(cl_ids, tot, hit):
        print(f'    clump size {t:>2}: TEPIG selects {h}')


if __name__ == '__main__':
    main()

"""
make_figures.py
---------------
Build the summary figures for the LINCS Cell Painting analysis:

  fig1_performance.png  outcomes x methods heatmaps (test MSE and test R^2),
                        averaged over the 6 dose levels, plus method sparsity.
  fig2_features.png     which morphology features each method selects, per
                        outcome (selection frequency across the 6 doses).
  fig3_dose_robustness.png  test R^2 per dose level, per method.

Reads: cell_painting/results/gene_<GENE>_dose<d>.pkl  (66 files)
Writes: cell_painting/figures/*.png
"""

import os
import glob
import pickle
from collections import defaultdict, Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

_HERE = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(_HERE, 'results')
FIG_DIR = os.path.join(_HERE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Validated palette (dataviz reference instance): slots 1,2,3
COL = {'TEPIG': '#2a78d6', 'clusso': '#1baf7a', 'naive': '#eda100'}
METHODS = ['TEPIG', 'clusso', 'naive']
LABELS = {'TEPIG': 'TEPIG', 'clusso': 'CLUSSO', 'naive': 'naive'}
ORDER = ['GLRX', 'RAP1GAP', 'IGFBP3', 'ATF6', 'CDKN1A', 'DDIT4',
         'CCNA2', 'SMC4', 'TSC22D3', 'MELK', 'CELLCYCLE']

# single-hue sequential ramp (light -> dark), from the palette's blue
BLUES = LinearSegmentedColormap.from_list('seqblue',
                                          ['#eef4fc', '#2a78d6', '#123a68'])
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e3e2df'

plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': GRID, 'axes.linewidth': 0.8,
    'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})


def load():
    """-> data[gene][dose] = results dict"""
    data = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(RES_DIR, 'gene_*_dose*.pkl'))):
        d = pickle.load(open(f, 'rb'))
        data[d['gene']][d['dose']] = d['results']
    return data


def _heat(ax, M, title, fmt='{:.2f}', reverse=False):
    """M: (n_outcomes, n_methods). Darker = better in both panels."""
    disp = -M if reverse else M          # reverse=True for MSE (lower is better)
    im = ax.imshow(disp, cmap=BLUES, aspect='auto')
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([LABELS[m] for m in METHODS], fontsize=10)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels(ORDER, fontsize=9)
    ax.set_title(title, fontsize=11, color=INK, pad=8, loc='left')
    # direct value labels (relief rule: never color alone)
    lo, hi = disp.min(), disp.max()
    for i in range(M.shape[0]):
        best = np.nanargmin(M[i]) if reverse else np.nanargmax(M[i])
        for j in range(M.shape[1]):
            freq = (disp[i, j] - lo) / (hi - lo + 1e-9)
            c = 'white' if freq > 0.55 else INK
            ax.text(j, i, fmt.format(M[i, j]), ha='center', va='center',
                    color=c, fontsize=8.5,
                    fontweight='bold' if j == best else 'normal')
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(METHODS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(ORDER), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2)   # 2px surface gap
    ax.tick_params(which='both', length=0)
    return im


def fig1(data):
    mse = np.array([[np.mean([data[g][d][m]['test_mse'] for d in sorted(data[g])])
                     for m in METHODS] for g in ORDER])
    r2 = np.array([[np.mean([data[g][d][m]['test_r2'] for d in sorted(data[g])])
                    for m in METHODS] for g in ORDER])
    nsel = {m: [data[g][d][m]['n_selected'] for g in ORDER for d in sorted(data[g])]
            for m in METHODS}

    fig = plt.figure(figsize=(12.5, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.85], wspace=0.45)
    ax1, ax2, ax3 = (fig.add_subplot(gs[0]), fig.add_subplot(gs[1]),
                     fig.add_subplot(gs[2]))

    _heat(ax1, mse, 'Test MSE', fmt='{:.2f}', reverse=True)
    _heat(ax2, r2, 'Test R²', fmt='{:.2f}')

    # sparsity: features selected (of q=68)
    pos = np.arange(len(METHODS))
    meds = [np.median(nsel[m]) for m in METHODS]
    for i, m in enumerate(METHODS):
        ax3.bar(pos[i], meds[i], width=0.6, color=COL[m], zorder=3,
                edgecolor='white', linewidth=2)
        ax3.text(pos[i], meds[i] + 0.7, f'{meds[i]:.0f}', ha='center',
                 fontsize=10, color=INK, fontweight='bold')
    ax3.set_xticks(pos); ax3.set_xticklabels([LABELS[m] for m in METHODS])
    ax3.set_ylabel('median # features selected (of 68)', fontsize=9, color=MUTED)
    ax3.set_title('Features selected', fontsize=11, color=INK, pad=8, loc='left')
    ax3.grid(axis='y', color=GRID, linewidth=0.8, zorder=0)
    ax3.set_axisbelow(True)
    for s in ['top', 'right']:
        ax3.spines[s].set_visible(False)
    ax3.tick_params(length=0)

    fig.suptitle('TEPIG vs CLUSSO vs naive — LINCS Cell Painting '
                 '(11 outcomes, averaged over 6 dose levels)',
                 fontsize=12.5, color=INK, x=0.5, y=0.99)
    fig.savefig(os.path.join(FIG_DIR, 'fig1_performance.png'), dpi=200,
                bbox_inches='tight')
    print('  wrote fig1_performance.png')


def fig2(data):
    """Which features each method picks, per outcome (count over 6 doses)."""
    # features selected at least 6 times overall by any method
    tot = Counter()
    for g in ORDER:
        for d in data[g]:
            for m in METHODS:
                for f in data[g][d][m]['selected']:
                    tot[f] += 1
    feats = [f for f, c in tot.most_common() if c >= 12][:26]
    # order by TEPIG usage
    tep = Counter()
    for g in ORDER:
        for d in data[g]:
            for f in data[g][d]['TEPIG']['selected']:
                tep[f] += 1
    feats.sort(key=lambda f: -tep[f])

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 8.2), sharey=True)
    for ax, m in zip(axes, METHODS):
        M = np.array([[sum(1 for d in data[g] if f in data[g][d][m]['selected'])
                       for g in ORDER] for f in feats], dtype=float)
        im = ax.imshow(M, cmap=BLUES, aspect='auto', vmin=0, vmax=6)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels(ORDER, rotation=45, ha='right', fontsize=8.5)
        ax.set_title(f'{LABELS[m]}   (selected in {int(M.sum())} gene·dose cells)',
                     fontsize=11, color=INK, pad=8, loc='left')
        ax.set_xticks(np.arange(-.5, len(ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(feats), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1.5)
        ax.tick_params(which='both', length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_yticks(range(len(feats)))
    axes[0].set_yticklabels(feats, fontsize=8)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label('# of the 6 doses the feature was selected', fontsize=9, color=MUTED)
    cb.outline.set_visible(False)
    fig.suptitle('Which morphology features each method selects, per outcome '
                 '— TEPIG picks a sparse, consistent set',
                 fontsize=12.5, color=INK, x=0.5, y=0.97)
    fig.savefig(os.path.join(FIG_DIR, 'fig2_features.png'), dpi=200,
                bbox_inches='tight')
    print('  wrote fig2_features.png')


def fig3(data):
    """Test R^2 by dose level — robustness check."""
    doses = [1, 2, 3, 4, 5, 6]
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for m in METHODS:
        y = [np.mean([data[g][d][m]['test_r2'] for g in ORDER]) for d in doses]
        ax.plot(doses, y, marker='o', markersize=8, linewidth=2, color=COL[m],
                label=LABELS[m], markeredgecolor='white', markeredgewidth=2,
                zorder=3)
        ax.text(doses[-1] + 0.08, y[-1], LABELS[m], color=INK, fontsize=10,
                va='center', fontweight='bold' if m == 'TEPIG' else 'normal')
    ax.set_xlabel('dose level (1 = lowest, 6 = highest)', fontsize=10, color=MUTED)
    ax.set_ylabel('mean test R² across the 11 outcomes', fontsize=10, color=MUTED)
    ax.set_title('TEPIG leads at every dose level', fontsize=12, color=INK,
                 loc='left', pad=10)
    ax.set_xlim(0.8, 6.75)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(frameon=False, loc='upper left', fontsize=9.5)
    fig.savefig(os.path.join(FIG_DIR, 'fig3_dose_robustness.png'), dpi=200,
                bbox_inches='tight')
    print('  wrote fig3_dose_robustness.png')


if __name__ == '__main__':
    data = load()
    print(f'loaded {sum(len(v) for v in data.values())} results')
    fig1(data)
    fig2(data)
    fig3(data)
    print(f'figures -> {FIG_DIR}')

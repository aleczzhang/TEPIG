"""Forest plot of paired test-R^2 differences vs the naive lasso (Fig. delta in the paper).

Reads a combined panel pickle (combine_panel50.py output) and plots, per gene, the mean
over seeds of (method - naive) test R^2 for TEPIG and CLUSSO, plus the overall mean with a
t-based 95% CI over the gene-level means.

Usage:
    python make_forest_213.py [--combined results/panel213_combined.pkl]
                              [--out results/figures/delta_forest_213.png]
"""
import argparse, os, pickle

import numpy as np
from scipy.stats import t as tdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_CP = os.path.dirname(_HERE)                       # cell_painting/
_ROOT = os.path.dirname(_CP)                       # repository root
sys.path[:0] = [_ROOT, _CP, os.path.join(_CP, 'panel'), os.path.join(_CP, 'screen')]

GENES = ['HMGCS1', 'NSDHL', 'HMGCR', 'FDFT1', 'INSIG1', 'TMEM97',
         'HERPUD1', 'XBP1', 'CREB3L2', 'CRELD2']
SURF, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'
COL = {'TEPIG': '#2a78d6', 'CLUSSO': '#1baf7a'}


def ci(v):
    v = np.asarray(v, float)
    h = tdist.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--combined', default=os.path.join(_CP, 'results', 'panel213_combined.pkl'))
    ap.add_argument('--out', default=os.path.join(_CP, 'results', 'figures', 'delta_forest_213.png'))
    ap.add_argument('--title', default='Accuracy differences are indistinguishable from zero')
    a = ap.parse_args()

    d = pickle.load(open(a.combined, 'rb'))
    genes = [g for g in GENES if g in d]
    dtn, dcn = [], []
    for g in genes:
        r = d[g]['r2']; seeds = sorted(r['TEPIG'])
        dtn.append(np.mean([r['TEPIG'][s] - r['naive'][s] for s in seeds]))
        dcn.append(np.mean([r['clusso'][s] - r['naive'][s] for s in seeds]))

    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=200)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    for s_ in ['top', 'right']: ax.spines[s_].set_visible(False)
    for s_ in ['left', 'bottom']: ax.spines[s_].set_color('#d8d7d2')
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis='x', color='#eceae5', lw=0.8); ax.set_axisbelow(True)

    ys = np.arange(len(genes))
    ax.scatter(dtn, ys - 0.15, s=36, color=COL['TEPIG'])
    ax.scatter(dcn, ys + 0.15, s=36, color=COL['CLUSSO'])
    ax.text(0.02, 1.02, '● TEPIG − naive lasso', color=COL['TEPIG'], fontsize=9,
            transform=ax.transAxes)
    ax.text(0.52, 1.02, '● CLUSSO − naive lasso', color=COL['CLUSSO'], fontsize=9,
            transform=ax.transAxes)
    for v, dy, c in [(dtn, -0.15, COL['TEPIG']), (dcn, 0.15, COL['CLUSSO'])]:
        m, h = ci(v)
        ax.errorbar([m], [len(genes) + 0.5 + dy], xerr=[h], fmt='D', ms=7, color=c,
                    elinewidth=2.2, capsize=3.5, zorder=4)
        print(f'{"TEPIG" if dy < 0 else "CLUSSO"} - naive: mean {m:+.4f}, 95% CI '
              f'[{m - h:+.4f}, {m + h:+.4f}]')
    ax.axhline(len(genes) - 0.25, color='#d8d7d2', lw=0.8)
    ax.set_yticks(list(ys) + [len(genes) + 0.5], genes + ['MEAN (95% CI)'],
                  fontsize=9.5, color=INK)
    ax.invert_yaxis()
    ax.axvline(0, color='#b9b7b0', lw=0.9)
    ax.set_xlabel('difference in test R² vs naive lasso (paired splits)',
                  fontsize=9.5, color=INK2)
    ax.set_title(a.title, fontsize=11.5, color=INK, loc='left', pad=22)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, facecolor=SURF, bbox_inches='tight')
    print('wrote', a.out)


if __name__ == '__main__':
    main()

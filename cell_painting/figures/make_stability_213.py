"""Selection-stability bar chart (Nogueira Phi) for the panel (Fig. stab in the paper).

Bars are the mean of Phi over genes for each method; dots are the per-gene Phi values,
each computed over the five grouped splits with q = number of candidate features.

Usage:
    python make_stability_213.py [--combined results/panel213_combined.pkl]
                                 [--out results/figures/stability_bars_213.png]
"""
import argparse, os, pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_CP = os.path.dirname(_HERE)                       # cell_painting/
_ROOT = os.path.dirname(_CP)                       # repository root
sys.path[:0] = [_ROOT, _CP, os.path.join(_CP, 'panel'), os.path.join(_CP, 'screen')]
from tepig.selection_metrics import nogueira

GENES = ['HMGCS1', 'NSDHL', 'HMGCR', 'FDFT1', 'INSIG1', 'TMEM97',
         'HERPUD1', 'XBP1', 'CREB3L2', 'CRELD2']
METHODS = [('TEPIG', 'TEPIG', '#2a78d6'), ('clusso', 'CLUSSO', '#1baf7a'),
           ('naive', 'naive lasso', '#e8622a')]
SURF, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--combined', default=os.path.join(_CP, 'results', 'panel213_combined.pkl'))
    ap.add_argument('--out', default=os.path.join(_CP, 'results', 'figures', 'stability_bars_213.png'))
    ap.add_argument('--title', default='TEPIG selects the most reproducible feature sets')
    a = ap.parse_args()

    d = pickle.load(open(a.combined, 'rb'))
    genes = [g for g in GENES if g in d]
    phi = {m: [] for m, _, _ in METHODS}
    for g in genes:
        q = len(d[g]['features'])
        for m, _, _ in METHODS:
            sets = [set(v) for v in d[g]['sets'][m].values()]
            phi[m].append(nogueira(sets, q))
    for m, lab, _ in METHODS:
        print(f'{lab:>12}: mean Phi {np.mean(phi[m]):.3f}  '
              + ' '.join(f'{v:.2f}' for v in phi[m]))
    best = sum(np.argmax([phi[m][i] for m, _, _ in METHODS]) == 0 for i in range(len(genes)))
    print(f'TEPIG highest Phi on {best}/{len(genes)} genes')

    fig, ax = plt.subplots(figsize=(6.3, 4.5), dpi=200)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    for s_ in ['top', 'right', 'left']: ax.spines[s_].set_visible(False)
    ax.spines['bottom'].set_color('#d8d7d2')
    ax.tick_params(colors=INK2, labelsize=9.5, length=0)
    ax.grid(axis='y', color='#eceae5', lw=0.8); ax.set_axisbelow(True)
    rng = np.random.default_rng(0)
    for i, (m, lab, c) in enumerate(METHODS):
        mu = float(np.mean(phi[m]))
        ax.bar(i, mu, width=0.52, color=c, alpha=0.35, bottom=0, zorder=2)
        ax.plot([i - 0.26, i + 0.26], [mu, mu], color=c, lw=3, zorder=3)
        ax.text(i - 0.29, mu + 0.003, f'{mu:.2f}', ha='right', va='bottom',
                fontsize=11, fontweight='bold', color=INK)
        x = i + rng.uniform(-0.11, 0.11, len(phi[m]))
        ax.scatter(x, phi[m], s=24, color=c, edgecolor='white', lw=0.8, zorder=4)
    ax.set_xticks(range(len(METHODS)), [lab for _, lab, _ in METHODS], fontsize=10.5, color=INK)
    lo = min(min(v) for v in phi.values()); hi = max(max(v) for v in phi.values())
    ax.set_ylim(np.floor((lo - 0.05) * 20) / 20, np.ceil((hi + 0.03) * 20) / 20)
    ax.set_ylabel('selection stability (Φ)', fontsize=10, color=INK2)
    ax.set_title(a.title, fontsize=11.5, color=INK, loc='left', pad=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, facecolor=SURF, bbox_inches='tight')
    print('wrote', a.out)


if __name__ == '__main__':
    main()

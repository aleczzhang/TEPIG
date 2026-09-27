"""Advisor-requested follow-ups on the 50-platemap panel run (27 Sep 2026):

  A. dTN = TEPIG - naive and dCN = CLUSSO - naive per gene (paired on identical
     splits), with the overall mean and a 95% CI from the empirical mean/sd of
     the 10 gene-level means (no p-values, per advisor).
  B. Three-way Venn of STABLY selected features (pi >= 0.6 across the 5 splits)
     for HMGCS1 -- the gene all three methods predict best on average.
  C. Where the exclusive picks live: feature-group counts of TEPIG-only-stable
     vs naive-only-stable features for HMGCS1.

Reads results/panel50_combined.pkl; writes results/figures/delta_features.png
and results/hmgcs1_selection_freq.csv (per-feature frequencies, all methods).
"""
import os, pickle, re

import numpy as np
from scipy.stats import t as tdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
d = pickle.load(open(os.path.join(_HERE, 'results', 'panel50_combined.pkl'), 'rb'))
GENES = ['HMGCS1', 'NSDHL', 'HMGCR', 'FDFT1', 'INSIG1', 'TMEM97',
         'HERPUD1', 'XBP1', 'CREB3L2', 'CRELD2']
SURF, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'
COL = {'TEPIG': '#2a78d6', 'CLUSSO': '#1baf7a', 'naive': '#eb6834'}

dtn, dcn = [], []
for g in GENES:
    r = d[g]['r2']; seeds = sorted(r['TEPIG'])
    dtn.append(np.mean([r['TEPIG'][s] - r['naive'][s] for s in seeds]))
    dcn.append(np.mean([r['clusso'][s] - r['naive'][s] for s in seeds]))

def ci(v):
    v = np.asarray(v)
    h = tdist.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), h

g0 = 'HMGCS1'
feats = d[g0]['features']
freq = {}
for m in ['TEPIG', 'clusso', 'naive']:
    f = np.zeros(len(feats))
    for sel in d[g0]['sets'][m].values():
        f[list(sel)] += 1
    freq[m] = f / 5
with open(os.path.join(_HERE, 'results', 'hmgcs1_selection_freq.csv'), 'w') as fh:
    fh.write('feature,freq_TEPIG,freq_CLUSSO,freq_naive\n')
    for i, name in enumerate(feats):
        if max(freq['TEPIG'][i], freq['clusso'][i], freq['naive'][i]) > 0:
            fh.write(f"{name},{freq['TEPIG'][i]:.1f},{freq['clusso'][i]:.1f},"
                     f"{freq['naive'][i]:.1f}\n")

stb = {m: freq[m] >= 0.6 for m in freq}
T, C, N = stb['TEPIG'], stb['clusso'], stb['naive']
V = dict(T=(T & ~C & ~N).sum(), C=(~T & C & ~N).sum(), N=(~T & ~C & N).sum(),
         TC=(T & C & ~N).sum(), TN=(T & ~C & N).sum(), CN=(~T & C & N).sum(),
         TCN=(T & C & N).sum())

def group(name):
    ch = 'mito network' if 'mito_tubeness' in name else next(
        (c for c in ['DNA', 'ER', 'RNA', 'AGP', 'Mito'] if re.search(rf'_{c}(_|$)', name)),
        'shape')
    meas = name.split('_')[1]
    return f'{meas} · {ch}' if ch != 'shape' else meas

def counts(mask):
    out = {}
    for i in np.where(mask)[0]:
        out[group(feats[i])] = out.get(group(feats[i]), 0) + 1
    return out
tcnt, ncnt = counts(T & ~N), counts(N & ~T)
groups = sorted(set(tcnt) | set(ncnt),
                key=lambda k: -(tcnt.get(k, 0) + ncnt.get(k, 0)))[:12]

fig = plt.figure(figsize=(13, 4.9), dpi=200)
fig.patch.set_facecolor(SURF)
ax = fig.add_axes([0.075, 0.13, 0.26, 0.72])
bx = fig.add_axes([0.375, 0.06, 0.27, 0.80])
cx = fig.add_axes([0.75, 0.13, 0.23, 0.72])
for a in (ax, cx):
    a.set_facecolor(SURF)
    for s_ in ['top', 'right']:
        a.spines[s_].set_visible(False)
    for s_ in ['left', 'bottom']:
        a.spines[s_].set_color('#d8d7d2')
    a.tick_params(colors=INK2, labelsize=8.5)
    a.set_axisbelow(True)

# ── A: paired deltas vs naive, forest style ────────────────────────────────
ax.grid(axis='x', color='#eceae5', lw=0.8)
ys = np.arange(len(GENES))
ax.scatter(dtn, ys - 0.15, s=30, color=COL['TEPIG'])
ax.scatter(dcn, ys + 0.15, s=30, color=COL['CLUSSO'])
ax.text(0.02, 1.015, '● ΔTN = TEPIG − naive', color=COL['TEPIG'], fontsize=8,
        transform=ax.transAxes)
ax.text(0.55, 1.015, '● ΔCN = CLUSSO − naive', color=COL['CLUSSO'], fontsize=8,
        transform=ax.transAxes)
for v, dy, c in [(dtn, -0.15, COL['TEPIG']), (dcn, 0.15, COL['CLUSSO'])]:
    m, h = ci(v)
    ax.errorbar([m], [len(GENES) + 0.5 + dy], xerr=[h], fmt='D', ms=6, color=c,
                elinewidth=2, capsize=3, zorder=4)
ax.axhline(len(GENES) - 0.25, color='#d8d7d2', lw=0.8)
ax.set_yticks(list(ys) + [len(GENES) + 0.5], GENES + ['MEAN (95% CI)'],
              fontsize=8.5, color=INK)
ax.invert_yaxis()
ax.axvline(0, color='#b9b7b0', lw=0.9)
ax.set_xlabel('Δ test R² vs naive (paired splits)', fontsize=9, color=INK2)
ax.set_title('A · ΔR² vs naive, per gene + overall', fontsize=10, color=INK,
             loc='left', pad=20)

# ── B: Venn of stable selections for HMGCS1 ────────────────────────────────
bx.set_facecolor(SURF); bx.axis('off')
cent = {'T': (0.38, 0.62), 'C': (0.62, 0.62), 'N': (0.50, 0.40)}
for k, m in [('T', 'TEPIG'), ('C', 'CLUSSO'), ('N', 'naive')]:
    bx.add_patch(plt.Circle(cent[k], 0.26, fc=COL[m], alpha=0.28, ec=COL[m], lw=1.6))
lab = {'T': (0.24, 0.71), 'C': (0.76, 0.71), 'N': (0.50, 0.245),
       'TC': (0.50, 0.71), 'TN': (0.365, 0.455), 'CN': (0.635, 0.455),
       'TCN': (0.50, 0.545)}
for k, xy in lab.items():
    bx.text(*xy, str(V[k]), ha='center', va='center', fontsize=11, color=INK,
            fontweight='bold' if k == 'TCN' else 'normal')
bx.text(0.21, 0.90, 'TEPIG', color=COL['TEPIG'], fontsize=9.5, fontweight='bold')
bx.text(0.68, 0.90, 'CLUSSO', color=COL['CLUSSO'], fontsize=9.5, fontweight='bold')
bx.text(0.435, 0.075, 'naive', color=COL['naive'], fontsize=9.5, fontweight='bold')
bx.set_xlim(0, 1); bx.set_ylim(0, 1)
bx.set_title('B · HMGCS1: stably selected features (≥3/5 splits)',
             fontsize=10, color=INK, loc='left', pad=8)

# ── C: where the exclusive picks live ──────────────────────────────────────
cx.grid(axis='x', color='#eceae5', lw=0.8)
yc = np.arange(len(groups))
cx.barh(yc - 0.19, [tcnt.get(k, 0) for k in groups], height=0.36,
        color=COL['TEPIG'], label='TEPIG-only')
cx.barh(yc + 0.19, [ncnt.get(k, 0) for k in groups], height=0.36,
        color=COL['naive'], label='naive-only')
cx.set_yticks(yc, groups, fontsize=8, color=INK)
cx.invert_yaxis()
cx.set_xlabel('stable features exclusive to method', fontsize=9, color=INK2)
cx.legend(frameon=False, fontsize=7.5, loc='lower right', labelcolor=INK)
cx.set_title('C · HMGCS1: what each method picks alone', fontsize=10,
             color=INK, loc='left', pad=8)

fig.suptitle('Method differences on the 50-platemap panel run', fontsize=11.5,
             color=INK, x=0.075, ha='left', y=0.985)
out = os.path.join(_HERE, 'results', 'figures', 'delta_features.png')
fig.savefig(out, facecolor=SURF, bbox_inches='tight')
print('wrote', out)

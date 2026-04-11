"""
plot_real_data.py
-----------------
Publication-quality figures for poster: real data analysis results.

Figures produced:
  1. real_data_cvmse.png  — CV-MSE bar chart with ratio annotations
  2. real_data_coefs.png  — Lollipop plot of L1-normalized coefficients (3-panel)

Usage:
    python plot_real_data.py
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

_HERE   = os.path.dirname(os.path.abspath(__file__))
_BASE   = os.path.join(_HERE, '..', 'outputs')
OUT_RES = os.path.join(_BASE, 'results')
OUT_FIG = os.path.join(_BASE, 'figures')
os.makedirs(OUT_FIG, exist_ok=True)

# ── Load results ───────────────────────────────────────────────────────────────
with open(os.path.join(OUT_RES, 'real_data_results.pkl'), 'rb') as f:
    results = pickle.load(f)

tg = results['TEPIG']
cl = results['clusso']
nv = results['naive']

# ── Shared style ───────────────────────────────────────────────────────────────
COLORS = {
    'TEPIG':  'steelblue',
    'CLUSSO': 'mediumseagreen',
    'naive':  'darkorange',
}
BG_COLOR   = '#EBEBEB'
GRID_COLOR = 'white'
GRID_LW    = 0.8


def _ggplot_ax(ax):
    ax.set_facecolor(BG_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=GRID_LW, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


# ── Feature name shortener ─────────────────────────────────────────────────────
_ABBREV = {
    'Distance Transform': 'DT',
    'Standard Deviation': 'SD',
    ' By ':              ' — ',
    ' By':               '',
}

def _shorten(name):
    for long, short in _ABBREV.items():
        name = name.replace(long, short)
    return name.strip(' —')


# ── Figure 1: CV-MSE bar chart with ratio annotations ─────────────────────────
estimators  = ['TEPIG', 'CLUSSO', 'naive']
cv_mse_vals = [tg['cv_mse'], cl['cv_mse'], nv['cv_mse']]
bar_colors  = [COLORS[e] for e in estimators]
x_pos       = np.arange(len(estimators))
tepig_mse   = cv_mse_vals[0]

fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=False)
_ggplot_ax(ax)

bars = ax.bar(x_pos, cv_mse_vals, color=bar_colors, width=0.55,
              edgecolor='white', linewidth=0.8, zorder=3)

# Value labels above bars
for bar, val in zip(bars, cv_mse_vals):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 6,
            f'{val:.0f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Ratio annotations on CLUSSO and naive bars
for i in [1, 2]:
    ratio = cv_mse_vals[i] / tepig_mse
    bar   = bars[i]
    mid_y = bar.get_height() / 2
    ax.text(bar.get_x() + bar.get_width() / 2,
            mid_y,
            f'{ratio:.1f}×\nhigher',
            ha='center', va='center', fontsize=9.5,
            color='white', fontweight='bold',
            path_effects=[pe.withStroke(linewidth=1.5, foreground='#444444')])

# Dashed reference line at TEPIG level
ax.axhline(tepig_mse, color='steelblue', linewidth=1.4,
           linestyle='--', zorder=4, alpha=0.75)
ax.text(2.35, tepig_mse + 12, 'TEPIG baseline',
        color='steelblue', fontsize=8.5, va='bottom', ha='right')

ax.set_xticks(x_pos)
ax.set_xticklabels(estimators, fontsize=13)
ax.set_ylabel('5-fold CV-MSE', fontsize=12)
ax.set_title('Cross-Validated Prediction Error\n(Coimbra kidney dataset, bootstrapped n=300)',
             fontsize=12, pad=10)
ax.tick_params(labelsize=10)
ax.set_ylim(0, max(cv_mse_vals) * 1.25)

plt.subplots_adjust(top=0.88, bottom=0.09, left=0.16, right=0.97)
out1 = os.path.join(OUT_FIG, 'real_data_cvmse.png')
fig.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out1}")


# ── Figure 2: Lollipop plot — 3 panels (one per estimator) ────────────────────
est_data = [
    ('TEPIG',  tg['selected'], tg['beta_l1']),
    ('CLUSSO', cl['selected'], cl['beta_l1']),
    ('naive',  nv['selected'], nv['beta_l1']),
]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=False)

for ax, (est_name, feats, coefs) in zip(axes, est_data):
    _ggplot_ax(ax)
    color = COLORS[est_name]

    short_feats = [_shorten(f) for f in feats]
    y_pos       = np.arange(len(feats))

    # Stems (horizontal lines from 0 to coef)
    ax.hlines(y_pos, 0, coefs, color=color, linewidth=2.0, zorder=3)

    # Dots at coef values
    ax.scatter(coefs, y_pos, color=color, s=80, zorder=4, edgecolors='white',
               linewidth=0.8)

    # Value labels next to dots
    for y, coef in zip(y_pos, coefs):
        ha  = 'left' if coef >= 0 else 'right'
        pad = 0.02   if coef >= 0 else -0.02
        ax.text(coef + pad, y, f'{coef:+.3f}',
                ha=ha, va='center', fontsize=8.5, color='#333333')

    # Zero line
    ax.axvline(0, color='#777777', linewidth=1.0, zorder=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_feats, fontsize=9)
    ax.set_xlabel('L1-Normalized Coefficient', fontsize=10)
    ax.set_title(est_name, fontsize=13, fontweight='bold', color=color, pad=8)
    ax.tick_params(axis='x', labelsize=9)

    # Symmetric x-axis with breathing room
    max_abs = max(abs(c) for c in coefs) if len(coefs) > 0 else 1.0
    ax.set_xlim(-max_abs * 1.55, max_abs * 1.55)
    ax.set_ylim(-0.6, len(feats) - 0.4)

# Shared subtitle
fig.text(0.5, 0.01,
         'Positive = higher predicted eGFR  |  Negative = lower predicted eGFR',
         ha='center', fontsize=9.5, color='#555555', style='italic')

fig.suptitle('Selected Features and L1-Normalized Coefficients\n'
             '(Coimbra kidney dataset, bootstrapped n=300)',
             fontsize=12, y=1.01)

plt.subplots_adjust(top=0.88, bottom=0.12, left=0.18, right=0.97, wspace=0.55)
out2 = os.path.join(OUT_FIG, 'real_data_coefs.png')
fig.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out2}")

print("Done.")

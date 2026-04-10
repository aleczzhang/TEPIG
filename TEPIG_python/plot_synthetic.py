"""
plot_synthetic.py
-----------------
Generates line graphs from simulation_synthetic pkl results,
matching the CLUSSO paper figure style.

Produces 4 figures (one per metric), each with a 2x5 subplot grid:
  rows = sparsity (0.4, 0.8)
  cols = q (10, 50, 100, 150, 200)
  x-axis = n

Usage:
    python plot_synthetic.py
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_BASE    = os.path.join(os.path.dirname(__file__), '..', 'outputs')
OUT_DATA = os.path.join(_BASE, 'data')
OUT_FIG  = os.path.join(_BASE, 'figures')
os.makedirs(OUT_FIG, exist_ok=True)

N_VALUES = [300, 500, 700, 900, 1100, 1500, 2000]
Q_VALUES = [10, 50, 100, 150, 200]
S_VALUES = [0.4, 0.8]

ESTIMATORS = [
    ('tepig',  'TEPIG',   'steelblue',      '-',   'o', 5),
    ('clusso', 'CLUSSO',  'mediumseagreen', '-',   's', 5),
    ('naive',  'Naive',   'darkorange',     '--',  '^', 5),
    ('oracle', 'Oracle',  'mediumpurple',   '-.',  'D', 5),
]

METRICS = [
    ('tpr',  'TPR',  False),
    ('fpr',  'FPR',  False),
    ('l1',   'Bias', False),
    ('mse',  'MSE',  True),   # log scale
]

# ── Load all results ────────────────────────────────────────────────────────────
print("Loading pkl files...")
data = {}
missing = []
for sparsity in S_VALUES:
    sparsity_str = f"{int(sparsity * 10):02d}"
    for q in Q_VALUES:
        for n in N_VALUES:
            pkl_path = os.path.join(OUT_DATA,
                f'simulation_synthetic_n{n}_q{q}_s{sparsity_str}_results.pkl')
            if not os.path.exists(pkl_path):
                missing.append((sparsity, q, n))
                continue
            with open(pkl_path, 'rb') as f:
                d = pickle.load(f)
            data[(sparsity, q, n)] = d['summary']

if missing:
    print(f"  WARNING: {len(missing)} settings missing: {missing[:5]}{'...' if len(missing)>5 else ''}")
else:
    print(f"  Loaded {len(data)} settings.")

# ── Plot ────────────────────────────────────────────────────────────────────────
for metric_key, metric_label, use_log in METRICS:
    fig, axes = plt.subplots(
        nrows=2, ncols=5,
        figsize=(14, 5.5),
        constrained_layout=False,
    )

    for row, sparsity in enumerate(S_VALUES):
        for col, q in enumerate(Q_VALUES):
            ax = axes[row, col]

            # ggplot-style background
            ax.set_facecolor('#EBEBEB')
            ax.grid(True, color='white', linewidth=0.8, zorder=0)
            for spine in ax.spines.values():
                spine.set_visible(False)

            for est_key, est_label, color, ls, marker, ms in ESTIMATORS:
                xs, ys = [], []
                for n in N_VALUES:
                    if (sparsity, q, n) not in data:
                        continue
                    summary = data[(sparsity, q, n)]
                    if est_key not in summary:
                        continue
                    val = float(np.nanmean(summary[est_key][metric_key]))
                    xs.append(n)
                    ys.append(val)
                if xs:
                    ax.plot(xs, ys,
                            color=color, linestyle=ls,
                            marker=marker, markersize=ms,
                            linewidth=1.4, label=est_label, zorder=3)

            if use_log:
                ax.set_yscale('log')

            ax.set_title(f'q = {q}, $\\mathrm{{s}}_{{\\beta^*}}$ = {sparsity}', fontsize=8.5)
            ax.set_xlabel('n', fontsize=8)
            ax.set_ylabel(metric_label, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_xticks(N_VALUES)
            ax.set_xticklabels([str(n) for n in N_VALUES],
                               rotation=40, ha='right', fontsize=6.5)

    # Shared legend outside, bottom center
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center',
               ncol=len(ESTIMATORS),
               fontsize=9,
               frameon=True,
               fancybox=False,
               edgecolor='gray',
               bbox_to_anchor=(0.5, 0.0))

    plt.subplots_adjust(
        top=0.95, bottom=0.16,
        left=0.06, right=0.99,
        hspace=0.55, wspace=0.40,
    )

    out_path = os.path.join(OUT_FIG, f'simulation_synthetic_{metric_key}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")

print("Done.")

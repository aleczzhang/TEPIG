"""
plot_synthetic_full.py
----------------------
Generates comprehensive line graphs from simulation_synthetic pkl results.

Produces 4 figures (one per metric: FPR, TPR, L1 bias, MSE),
each with a 2x5 subplot grid:
  rows = sparsity (0.4, 0.8)
  cols = q (10, 50, 100, 150, 200)
  x-axis = n

Usage:
    python plot_synthetic_full.py [--folder threshold_cmp]
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_parser = argparse.ArgumentParser()
_parser.add_argument('--folder', default='threshold_cmp',
                     help='Subfolder under outputs/data/ to read from')
_args = _parser.parse_args()

_BASE    = os.path.join(os.path.dirname(__file__), '..', 'outputs')
OUT_DATA = os.path.join(_BASE, 'data', _args.folder)
OUT_FIG  = os.path.join(_BASE, 'figures', _args.folder + '_full')
os.makedirs(OUT_FIG, exist_ok=True)

N_VALUES = [300, 500, 700, 900, 1100, 1500, 2000]
Q_VALUES = [10, 50, 100, 150, 200]
S_VALUES = [0.4, 0.8]

ESTIMATORS = [
    ('tepig_norm_adapt', 'TEPIG',   'steelblue',      '-',   'o', 5),
    ('clusso',           'CLUSSO',  'mediumseagreen', '-',   's', 5),
    ('naive',            'Naive',   'darkorange',     '--',  '^', 5),
    ('oracle',           'Oracle',  'mediumpurple',   '-.',  'D', 5),
]

# (key, label, log_scale, aggregation)
METRICS = [
    ('fpr', 'FPR (mean)',        False, 'mean'),
    ('fpr', 'FPR (median)',      False, 'median'),
    ('tpr', 'TPR (mean)',        False, 'mean'),
    ('tpr', 'TPR (median)',      False, 'median'),
    ('l1',  'L1 Bias (mean)',    False, 'mean'),
    ('l1',  'L1 Bias (median)',  False, 'median'),
    ('mse', 'MSE (mean)',        True,  'mean'),
    ('mse', 'MSE (median)',      True,  'median'),
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
for metric_key, metric_label, use_log, agg_fn in METRICS:
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
                    fn  = np.nanmedian if agg_fn == 'median' else np.nanmean
                    val = float(fn(summary[est_key][metric_key]))
                    xs.append(n)
                    ys.append(val)
                if xs:
                    ax.plot(xs, ys,
                            color=color, linestyle=ls,
                            marker=marker, markersize=ms,
                            linewidth=1.4, label=est_label, zorder=3)

            if use_log:
                ax.set_yscale('log')
            elif metric_key in ('tpr', 'fpr'):
                ax.set_ylim([-0.05, 1.05])

            ax.set_title(f'q={q}, s={sparsity}', fontsize=8)
            ax.set_xlabel('n', fontsize=7.5)
            ax.set_ylabel(metric_label, fontsize=7.5)
            ax.tick_params(labelsize=6.5)
            ax.set_xticks(N_VALUES)
            ax.set_xticklabels([str(n) for n in N_VALUES],
                               rotation=40, ha='right', fontsize=6)

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
        top=0.93, bottom=0.18,
        left=0.05, right=0.99,
        hspace=0.60, wspace=0.38,
    )

    safe_label = metric_label.replace(' ', '_').replace('(', '').replace(')', '')
    out_path = os.path.join(OUT_FIG, f'simulation_synthetic_full_{safe_label}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")

print("Done.")

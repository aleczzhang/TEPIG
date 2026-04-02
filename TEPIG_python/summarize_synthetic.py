"""
summarize_synthetic.py
----------------------
Combines simulation_synthetic_n*_results.pkl files into a single summary table.

Usage:
    python summarize_synthetic.py
"""

import os
import pickle
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

N_VALUES    = [50, 100, 200, 300, 500]
ESTIMATORS  = ['tepig_grad', 'tepig', 'naive', 'oracle']
METRICS     = ['tpr', 'fpr', 'l1', 'mse']

rows = []
for n in N_VALUES:
    pkl_path = os.path.join(OUT_DIR, f'simulation_synthetic_n{n}_results.pkl')
    if not os.path.exists(pkl_path):
        print(f"  WARNING: missing {pkl_path}, skipping n={n}")
        continue
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    summary = data['summary']
    for est in ESTIMATORS:
        rows.append({
            'n':   n,
            'est': est,
            'tpr': float(np.nanmean(summary[est]['tpr'])),
            'fpr': float(np.nanmean(summary[est]['fpr'])),
            'l1':  float(np.nanmean(summary[est]['l1'])),
            'mse': float(np.nanmean(summary[est]['mse'])),
        })

# ── Print table ────────────────────────────────────────────────────────────────
header = f"{'n':>6}  {'Estimator':<14} {'TPR':>7} {'FPR':>7} {'L1 bias':>9} {'MSE':>9}"
divider = "-" * len(header)

lines = []
lines.append("=" * len(header))
lines.append("TEPIG SYNTHETIC SIMULATION — VARYING SAMPLE SIZE")
lines.append(f"q=10, G=2, S=2, n_nonzero=2, sparsity=0.8, B=200 reps")
lines.append("=" * len(header))
lines.append(header)

prev_n = None
for r in rows:
    if r['n'] != prev_n:
        lines.append(divider)
        prev_n = r['n']
    lines.append(
        f"  {r['n']:>4}  {r['est']:<14} {r['tpr']:>7.3f} {r['fpr']:>7.3f}"
        f" {r['l1']:>9.3f} {r['mse']:>9.3f}"
    )

lines.append("=" * len(header))

output = "\n".join(lines)
print(output)

out_path = os.path.join(OUT_DIR, 'simulation_synthetic_summary_all_n.txt')
with open(out_path, 'w') as f:
    f.write(output + "\n")
print(f"\nSaved to {out_path}")

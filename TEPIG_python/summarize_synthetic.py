"""
summarize_synthetic.py
----------------------
Combines all simulation_synthetic_n*_q*_s*_results.pkl files into a summary table.

Usage:
    python summarize_synthetic.py
"""

import os
import pickle
import numpy as np

_BASE    = os.path.join(os.path.dirname(__file__), '..', 'outputs')
OUT_DATA = os.path.join(_BASE, 'data')
OUT_SUMM = os.path.join(_BASE, 'summaries')

N_VALUES    = [300, 500, 700, 900, 1100, 1500, 2000]
Q_VALUES    = [10, 50, 100, 150, 200]
S_VALUES    = [0.4, 0.8]
ESTIMATORS  = ['tepig', 'clusso', 'naive', 'oracle']
METRICS     = ['tpr', 'fpr', 'l1', 'mse']

rows = []
missing = []
for sparsity in S_VALUES:
    sparsity_str = f"{int(sparsity * 10):02d}"
    for q in Q_VALUES:
        for n in N_VALUES:
            pkl_path = os.path.join(OUT_DATA,
                f'simulation_synthetic_n{n}_q{q}_s{sparsity_str}_results.pkl')
            if not os.path.exists(pkl_path):
                missing.append(f"n={n}, q={q}, sparsity={sparsity}")
                continue
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            summary = data['summary']
            for est in ESTIMATORS:
                if est not in summary:
                    continue
                rows.append({
                    'n': n, 'q': q, 'sparsity': sparsity, 'est': est,
                    'tpr': float(np.nanmean(summary[est]['tpr'])),
                    'fpr': float(np.nanmean(summary[est]['fpr'])),
                    'l1':  float(np.nanmean(summary[est]['l1'])),
                    'mse': float(np.nanmean(summary[est]['mse'])),
                })

if missing:
    print(f"WARNING: {len(missing)} settings missing:")
    for m in missing:
        print(f"  {m}")
    print()

# ── Print table ────────────────────────────────────────────────────────────────
header  = f"{'sparsity':>9} {'q':>5} {'n':>6}  {'Estimator':<14} {'TPR':>7} {'FPR':>7} {'L1 bias':>9} {'MSE':>9}"
divider = "-" * len(header)

lines = []
lines.append("=" * len(header))
lines.append("TEPIG SYNTHETIC SIMULATION — ALL SETTINGS")
lines.append("G=2, S=2, K=40 tubules/slide, B=200 reps")
lines.append("=" * len(header))
lines.append(header)

prev_key = None
for r in rows:
    key = (r['sparsity'], r['q'], r['n'])
    if key != prev_key:
        lines.append(divider)
        prev_key = key
    lines.append(
        f"  {r['sparsity']:>7.1f} {r['q']:>5} {r['n']:>6}  {r['est']:<14}"
        f" {r['tpr']:>7.3f} {r['fpr']:>7.3f} {r['l1']:>9.3f} {r['mse']:>9.3f}"
    )

lines.append("=" * len(header))

output = "\n".join(lines)
print(output)

out_path = os.path.join(OUT_SUMM, 'simulation_synthetic_summary_all.txt')
with open(out_path, 'w') as f:
    f.write(output + "\n")
print(f"\nSaved to {out_path}")

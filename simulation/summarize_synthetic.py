"""
summarize_synthetic.py
----------------------
Combines all simulation_synthetic_n*_q*_s*_results.pkl files of one run folder
into a summary table, with ONE statistic over replicates per file (medians by
default -- what the paper reports -- or means with --stat mean) and a solver /
grid diagnostic block.

Usage:
    python summarize_synthetic.py --folder rerun_2026_11 [--stat median|mean]
Writes outputs/summaries/<folder>/simulation_synthetic_summary_<folder>_<stat>.txt
and ..._<stat>.csv (one row per setting x estimator).
"""

import os
import pickle
import argparse
import numpy as np

_parser = argparse.ArgumentParser()
_parser.add_argument('--folder', default='rerun_2026_11',
                     help='Subfolder under outputs/data/ to read from')
_parser.add_argument('--stat', default='median', choices=['median', 'mean'])
_args = _parser.parse_args()

_BASE    = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')
OUT_DATA = os.path.join(_BASE, 'data', _args.folder)
OUT_SUMM = os.path.join(_BASE, 'summaries', _args.folder)
os.makedirs(OUT_SUMM, exist_ok=True)

N_VALUES    = [300, 500, 700, 900, 1100, 1500, 2000]
Q_VALUES    = [10, 50, 100, 150, 200]
S_VALUES    = [0.4, 0.8]
agg = np.nanmedian if _args.stat == 'median' else np.nanmean

rows, diag, missing, estimators, metrics = [], [], [], None, None
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
            if estimators is None:
                estimators = list(summary)
                metrics = list(summary[estimators[0]])
            for est in estimators:
                if est not in summary:
                    continue
                rows.append({'n': n, 'q': q, 'sparsity': sparsity, 'est': est,
                             **{m: float(agg(summary[est][m])) for m in metrics}})
            ex = data.get('extras')
            if ex:
                B = len(ex)
                diag.append(dict(
                    n=n, q=q, sparsity=sparsity, B=B,
                    tepig_lb=sum(e['tepig']['at_lower_bound'] for e in ex),
                    tepig_ub=sum(e['tepig']['at_upper_bound'] for e in ex),
                    tepig_unconverged=sum(not e['tepig']['converged'] for e in ex),
                    tepig_iter_med=float(np.median([e['tepig']['n_iter'] for e in ex])),
                    tepig_kkt_max=max(e['tepig']['kkt'] for e in ex),
                    clusso_lb=sum(e['clusso']['lam_idx'] == 0 for e in ex),
                    clusso_ub=sum(e['clusso']['lam_idx'] == e['clusso']['n_lam'] - 1 for e in ex)))

if missing:
    print(f"WARNING: {len(missing)} settings missing:")
    for m in missing:
        print(f"  {m}")
    print()
if not rows:
    raise SystemExit('no results found')

# ── Print table ────────────────────────────────────────────────────────────────
mcols = ' '.join(f'{m:>9}' for m in metrics)
header  = f"{'sparsity':>9} {'q':>5} {'n':>6}  {'Estimator':<22} {mcols}"
divider = "-" * len(header)
lines = ["=" * len(header),
         f"TEPIG SYNTHETIC SIMULATION — ALL SETTINGS ({_args.folder}), {_args.stat.upper()} over replicates",
         "G=2, S=2, K=40 objects/section", "=" * len(header), header]
prev_key = None
for r in rows:
    key = (r['sparsity'], r['q'], r['n'])
    if key != prev_key:
        lines.append(divider)
        prev_key = key
    lines.append(f"  {r['sparsity']:>7.1f} {r['q']:>5} {r['n']:>6}  {r['est']:<22} "
                 + ' '.join(f"{r[m]:>9.3f}" for m in metrics))
lines.append("=" * len(header))
if diag:
    lines += ["", "Solver / grid diagnostics per setting (counts out of B replicates):",
              f"{'sparsity':>9} {'q':>5} {'n':>6} {'B':>4} {'TEPIG lam@lo':>13} {'lam@hi':>7} "
              f"{'unconv':>7} {'iter_med':>9} {'kkt_max':>9} {'CLUSSO lam@lo':>14} {'lam@hi':>7}"]
    for d in diag:
        lines.append(f"{d['sparsity']:>9.1f} {d['q']:>5} {d['n']:>6} {d['B']:>4} {d['tepig_lb']:>13} "
                     f"{d['tepig_ub']:>7} {d['tepig_unconverged']:>7} {d['tepig_iter_med']:>9.0f} "
                     f"{d['tepig_kkt_max']:>9.1e} {d['clusso_lb']:>14} {d['clusso_ub']:>7}")
output = "\n".join(lines)
print(output)

base = os.path.join(OUT_SUMM, f'simulation_synthetic_summary_{_args.folder}_{_args.stat}')
with open(base + '.txt', 'w') as f:
    f.write(output + "\n")
with open(base + '.csv', 'w') as f:
    f.write('sparsity,q,n,est,' + ','.join(metrics) + '\n')
    for r in rows:
        f.write(f"{r['sparsity']},{r['q']},{r['n']},{r['est']}," +
                ','.join(f"{r[m]:.5f}" for m in metrics) + '\n')
print(f"\nSaved to {base}.txt and .csv")

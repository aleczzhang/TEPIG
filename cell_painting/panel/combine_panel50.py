"""Combine the per-task pkls from run_panel50/213.slurm into the final table.

Each task ran one (gene, seed): TEPIG/CLUSSO/naive on a 40/10 platemap split.
This script merges them per gene, then prints:
  * per-gene R^2 mean+/-sd per method (5 grouped splits) + Nogueira stability
  * paired TEPIG - naive differences (pooled Wilcoxon + gene-level t-test + CI)
  * the cholesterol-program vs ER-stress-program comparison
  * TEPIG's selected-set size and stability under EVERY threshold rule (post hoc,
    from the stored block statistics), and the solver diagnostics (lambda at a
    grid bound? iterations? KKT residual?) so the paper's convergence claims
    come from logs
and writes results/<prefix>_combined.pkl + results/<prefix>_table.csv.

Usage:  python combine_panel50.py --tag cdrp_213 [--rule col_std+GS]
"""
import argparse, glob, os, pickle, re

import numpy as np
from scipy.stats import ttest_1samp, wilcoxon, t as tdist

import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_CP = os.path.dirname(_HERE)                       # cell_painting/
_ROOT = os.path.dirname(_CP)                       # repository root
sys.path[:0] = [_ROOT, _CP, os.path.join(_CP, 'panel'), os.path.join(_CP, 'screen')]
from tepig import selection_metrics as SM
from stability_check import tepig_selected

GENES = ['HMGCS1', 'NSDHL', 'HMGCR', 'FDFT1', 'HERPUD1',
         'XBP1', 'TMEM97', 'CREB3L2', 'INSIG1', 'CRELD2']
PROGRAM = {'HMGCS1': 'cholesterol', 'NSDHL': 'cholesterol', 'HMGCR': 'cholesterol',
           'FDFT1': 'cholesterol', 'INSIG1': 'cholesterol', 'TMEM97': 'cholesterol',
           'HERPUD1': 'ER-stress', 'XBP1': 'ER-stress', 'CREB3L2': 'ER-stress',
           'CRELD2': 'ER-stress'}
METHODS = ['TEPIG', 'clusso', 'naive']
RULES = ['raw', 'unscaled', 'unscaled+GS', 'block_scaled', 'block_scaled+GS',
         'col_std', 'col_std+GS', 'col_std_refit', 'col_std_refit+GS',
         'raw@1se', 'block_scaled@1se', 'col_std+GS@1se', 'col_std_refit+GS@1se']

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='cdrp_50pm_merged',
                help="cache tag in the task filenames, e.g. cdrp_213")
ap.add_argument('--out', default=None,
                help="output prefix (default: panel50 / panel213 by tag)")
ap.add_argument('--rule', default=None,
                help="TEPIG selection rule for the main table, e.g. col_std+GS "
                     "(default: the rule the run was made with)")
a = ap.parse_args()
PREFIX = a.out or ('panel213' if '213' in a.tag else 'panel50')


def _rule(name):
    """'col_std_refit+GS@1se' -> ('col_std_refit', True, '1se')"""
    which = '1se' if name.endswith('@1se') else 'min'
    name = name.replace('@1se', '')
    return (name.replace('+GS', ''), name.endswith('+GS'), which)


merged, meta = {}, None
for f in sorted(glob.glob(os.path.join(
        _CP, 'results', f'panel_pycytominer_{a.tag}_*_s*.pkl'))):
    m = re.search(rf'{a.tag}_(\w+)_s(\d+)\.pkl$', f)
    g, seed = m.group(1), int(m.group(2))
    r = pickle.load(open(f, 'rb'))
    if g not in r:
        continue
    meta = r.get('_meta', meta)
    METHODS = [m_ for m_ in METHODS if m_ in r[g]['r2']]      # tolerate partial runs
    d = merged.setdefault(g, {'r2': {m_: {} for m_ in METHODS},
                              'mse': {m_: {} for m_ in METHODS},
                              'sets': {m_: {} for m_ in METHODS},
                              'detail': {m_: {} for m_ in METHODS},
                              'features': r[g]['features']})
    for m_ in METHODS:
        d['r2'][m_][seed] = r[g]['r2'][m_][0]
        d['mse'][m_][seed] = r[g]['mse'][m_][0]
        d['sets'][m_][seed] = r[g]['sets'][m_][0]
        if 'detail' in r[g]:
            d['detail'][m_][seed] = r[g]['detail'][m_][0]

print(f"genes found: {len(merged)}; seeds per gene: "
      f"{sorted({len(d['r2']['TEPIG']) for d in merged.values()})}")
if meta:
    print(f"run meta: {meta.get('split')} split, inner CV {meta.get('inner_cv')}, "
          f"TEPIG rule {meta.get('tepig_rule')}{'+GS' if meta.get('tepig_group_term') else ''}, "
          f"git {meta.get('run', {}).get('git')}")

# TEPIG sets under the requested rule (post hoc, from the stored block statistics)
if a.rule:
    rule, gt, which = _rule(a.rule)
    for g, d in merged.items():
        for s, det in d['detail']['TEPIG'].items():
            d['sets']['TEPIG'][s] = tepig_selected(det, rule, gt, which)
            if which == '1se':                       # R^2 of the 1-SE fit (stored test preds)
                k = det['best_1se_idx']
                d['r2'][m_ := 'TEPIG'][s] = 1 - np.mean((det['y_test'] - det['path_test_pred'][k]) ** 2) / np.var(det['y_test'])
    print(f"TEPIG sets recomputed under rule {a.rule}")

rows, diffs = [], {}
print(f"\n{'gene':>8} {'prog':>12} | " +
      ' | '.join(f'{m_:^22}' for m_ in METHODS) + ' |  TEPIG-naive')
for g in GENES:
    if g not in merged:
        continue
    d = merged[g]
    seeds = sorted(d['r2']['TEPIG'])
    line = f'{g:>8} {PROGRAM[g]:>12} | '
    stats = {}
    for m_ in METHODS:
        v = np.array([d['r2'][m_][s] for s in seeds])
        sets = [d['sets'][m_][s] for s in seeds]
        phi = SM.summarize(sets, len(d['features']))
        stats[m_] = (v.mean(), v.std(), phi['nogueira'], phi['size_mean'])
        line += (f'{v.mean():+.3f}±{v.std():.3f} Φ{phi["nogueira"]:+.2f} '
                 f'|{phi["size_mean"]:>3.0f}| | ')
    dv = (np.array([d['r2']['TEPIG'][s] - d['r2']['naive'][s] for s in seeds])
          if 'naive' in d['r2'] and 'TEPIG' in d['r2'] else np.array([np.nan]))
    diffs[g] = dv
    line += f'{dv.mean():+.3f} ({int((dv > 0).sum())}/{len(dv)})'
    print(line)
    rows.append([g, PROGRAM[g]] + [x for m_ in METHODS for x in stats[m_]])

pooled = np.concatenate(list(diffs.values()))
gm = np.array([d.mean() for d in diffs.values()])
h = tdist.ppf(0.975, len(gm) - 1) * gm.std(ddof=1) / np.sqrt(len(gm))
print(f'\nTEPIG vs naive: pooled mean {pooled.mean():+.4f}, '
      f'wins {int((pooled > 0).sum())}/{len(pooled)}, '
      f'Wilcoxon p={wilcoxon(pooled).pvalue:.3f}; '
      f'gene-level mean {gm.mean():+.4f} 95% CI [{gm.mean() - h:+.4f}, {gm.mean() + h:+.4f}], '
      f't-test p={ttest_1samp(gm, 0).pvalue:.3f}')
for prog in ['cholesterol', 'ER-stress']:
    gs = [g for g in diffs if PROGRAM[g] == prog]
    tep = np.array([[merged[g]['r2']['TEPIG'][s] for s in sorted(merged[g]['r2']['TEPIG'])]
                    for g in gs])
    nai = np.array([[merged[g]['r2']['naive'][s] for s in sorted(merged[g]['r2']['naive'])]
                    for g in gs])
    print(f'  {prog:>12}: TEPIG {tep.mean():+.3f}  naive {nai.mean():+.3f}  '
          f'({len(gs)} genes)')

# ── TEPIG selection under every rule + solver diagnostics ───────────────────
have_detail = all(merged[g]['detail']['TEPIG'] for g in merged)
if have_detail:
    print(f"\nTEPIG selected-set size (mean over genes of mean over splits) and Φ by rule:")
    print(f"  {'rule':<16}{'|J|':>8}{'Φ':>8}")
    for rn in RULES:
        rule, gt, which = _rule(rn)
        sizes, phis = [], []
        for g in GENES:
            if g not in merged:
                continue
            det = merged[g]['detail']['TEPIG']
            sets = [tepig_selected(det[s], rule, gt, which) for s in sorted(det)]
            ph = SM.summarize(sets, len(merged[g]['features']))
            sizes.append(ph['size_mean']); phis.append(ph['nogueira'])
        print(f"  {rn:<16}{np.mean(sizes):>8.0f}{np.nanmean(phis):>8.2f}")
    lb = ub = tot = 0; iters, kkt = [], []
    for g in merged:
        for s, det in merged[g]['detail']['TEPIG'].items():
            tot += 1; lb += det['at_lower_bound']; ub += det['at_upper_bound']
            iters.append(det['n_iter']); kkt.append(det['kkt'])
    print(f"\nTEPIG solver: lambda at grid lower bound in {lb}/{tot} fits, upper bound {ub}/{tot}; "
          f"final-fit iterations median {np.median(iters):.0f} max {max(iters)}; "
          f"KKT residual max {max(kkt):.1e}")
    cl = [det['best_idx'] for g in merged for det in merged[g]['detail'].get('clusso', {}).values()]
    if cl:
        ncl = len(next(det for g in merged for det in merged[g]['detail']['clusso'].values())['lams'])
        print(f"CLUSSO lambda index histogram (0 = largest of {ncl}): "
              f"{np.bincount(cl, minlength=ncl).tolist()}")

with open(os.path.join(_CP, 'results', f'{PREFIX}_table.csv'), 'w') as f:
    f.write('gene,program,' + ','.join(f'{m_}_{c}' for m_ in METHODS
            for c in ['r2_mean', 'r2_sd', 'phi', 'nsel']) + '\n')
    for r_ in rows:
        f.write(','.join(str(x) if isinstance(x, str) else f'{x:.4f}'
                         for x in r_) + '\n')
merged['_meta'] = meta
pickle.dump(merged, open(os.path.join(_CP, 'results', f'{PREFIX}_combined.pkl'), 'wb'))
print(f'\nwrote results/{PREFIX}_table.csv and results/{PREFIX}_combined.pkl')

"""Combine the 50 per-task pkls from run_panel50.slurm into the final table.

Each task ran one (gene, seed): TEPIG/CLUSSO/naive on a 40/10 platemap split
of cache/cdrp_50pm_merged.pkl. This script merges them per gene, then prints:
  * per-gene R^2 mean+/-sd per method (5 grouped splits) + Nogueira stability
  * paired TEPIG - naive differences (pooled Wilcoxon + gene-level t-test)
  * the cholesterol-program vs ER-stress-program comparison
and writes results/panel50_combined.pkl + results/panel50_table.csv.

Usage:  python combine_panel50.py
"""
import glob, os, pickle, re

import numpy as np
from scipy.stats import ttest_1samp, wilcoxon

import selection_metrics as SM

_HERE = os.path.dirname(os.path.abspath(__file__))
GENES = ['HMGCS1', 'NSDHL', 'HMGCR', 'FDFT1', 'HERPUD1',
         'XBP1', 'TMEM97', 'CREB3L2', 'INSIG1', 'CRELD2']
PROGRAM = {'HMGCS1': 'cholesterol', 'NSDHL': 'cholesterol', 'HMGCR': 'cholesterol',
           'FDFT1': 'cholesterol', 'INSIG1': 'cholesterol', 'TMEM97': 'cholesterol',
           'HERPUD1': 'ER-stress', 'XBP1': 'ER-stress', 'CREB3L2': 'ER-stress',
           'CRELD2': 'ER-stress'}
METHODS = ['TEPIG', 'clusso', 'naive']

merged = {}
for f in sorted(glob.glob(os.path.join(
        _HERE, 'results', 'panel_pycytominer_cdrp_50pm_merged_*_s*.pkl'))):
    m = re.search(r'merged_(\w+)_s(\d+)\.pkl$', f)
    g, seed = m.group(1), int(m.group(2))
    r = pickle.load(open(f, 'rb'))
    if g not in r:
        continue
    d = merged.setdefault(g, {'r2': {m_: {} for m_ in METHODS},
                              'sets': {m_: {} for m_ in METHODS},
                              'features': r[g]['features']})
    for m_ in METHODS:
        d['r2'][m_][seed] = r[g]['r2'][m_][0]
        d['sets'][m_][seed] = r[g]['sets'][m_][0]

print(f"genes found: {len(merged)}; seeds per gene: "
      f"{sorted({len(d['r2']['TEPIG']) for d in merged.values()})}")

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
    dv = np.array([d['r2']['TEPIG'][s] - d['r2']['naive'][s] for s in seeds])
    diffs[g] = dv
    line += f'{dv.mean():+.3f} ({int((dv > 0).sum())}/{len(dv)})'
    print(line)
    rows.append([g, PROGRAM[g]] + [x for m_ in METHODS for x in stats[m_]])

pooled = np.concatenate(list(diffs.values()))
gm = np.array([d.mean() for d in diffs.values()])
print(f'\nTEPIG vs naive: pooled mean {pooled.mean():+.4f}, '
      f'wins {int((pooled > 0).sum())}/{len(pooled)}, '
      f'Wilcoxon p={wilcoxon(pooled).pvalue:.3f}; '
      f'gene-level t-test p={ttest_1samp(gm, 0).pvalue:.3f}')
for prog in ['cholesterol', 'ER-stress']:
    gs = [g for g in diffs if PROGRAM[g] == prog]
    tep = np.array([[merged[g]['r2']['TEPIG'][s] for s in sorted(merged[g]['r2']['TEPIG'])]
                    for g in gs])
    nai = np.array([[merged[g]['r2']['naive'][s] for s in sorted(merged[g]['r2']['naive'])]
                    for g in gs])
    print(f'  {prog:>12}: TEPIG {tep.mean():+.3f}  naive {nai.mean():+.3f}  '
          f'({len(gs)} genes)')

with open(os.path.join(_HERE, 'results', 'panel50_table.csv'), 'w') as f:
    f.write('gene,program,' + ','.join(f'{m_}_{c}' for m_ in METHODS
            for c in ['r2_mean', 'r2_sd', 'phi', 'nsel']) + '\n')
    for r_ in rows:
        f.write(','.join(str(x) if isinstance(x, str) else f'{x:.4f}'
                         for x in r_) + '\n')
pickle.dump(merged, open(os.path.join(_HERE, 'results', 'panel50_combined.pkl'), 'wb'))
print('\nwrote results/panel50_table.csv and results/panel50_combined.pkl')

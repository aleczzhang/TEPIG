"""Simulation figures for the paper, ALL from one run folder, one statistic over
replicates and one TEPIG variant.

Usage:
    python simulation/plot_figures.py [--folder rerun_2026_11] [--stat median]
                                    [--est tepig] [--metrics TPR FPR L1bias MSE MSEtest]
Reads  outputs/summaries/<folder>/simulation_synthetic_summary_<folder>_<stat>.csv
       (summarize_synthetic.py output)
Writes outputs/figures/<folder>/sim_<est>_<METRIC>_<stat>.png
"""
import argparse, csv, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--folder', default='rerun_2026_11')
ap.add_argument('--stat', default='median', choices=['median', 'mean'])
ap.add_argument('--est', default='tepig', help='TEPIG variant key in the summary')
ap.add_argument('--metrics', nargs='+', default=['TPR', 'FPR', 'L1bias', 'MSE', 'MSEtest'])
a = ap.parse_args()
SRC = os.path.join(ROOT, 'outputs', 'summaries', a.folder,
                   f'simulation_synthetic_summary_{a.folder}_{a.stat}.csv')
OUT = os.path.join(ROOT, 'outputs', 'figures', a.folder)
os.makedirs(OUT, exist_ok=True)
COL = {'TPR': 'tpr', 'FPR': 'fpr', 'L1bias': 'l1', 'MSE': 'mse', 'MSEtest': 'mse_test'}

rows = []
for r in csv.DictReader(open(SRC)):
    rows.append(dict(sp=float(r['sparsity']), q=int(r['q']), n=int(r['n']), est=r['est'],
                     **{k: float(r[v]) for k, v in COL.items() if v in r}))

SERIES = [(a.est, 'TEPIG', '#2a78d6', 'o', '-'),
          ('clusso', 'CLUSSO', '#1baf7a', 's', '-'),
          ('naive', 'Naive', '#eb6834', '^', '--'),
          ('oracle', 'Oracle', '#4a3aa7', 'D', '-.')]
QS = sorted({r['q'] for r in rows}); SPS = sorted({r['sp'] for r in rows})
NS = sorted({r['n'] for r in rows})
SURF, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'
LABEL = {'TPR': 'TPR', 'FPR': 'FPR', 'L1bias': 'L1 bias', 'MSE': 'training MSE',
         'MSEtest': 'test MSE'}

for metric in a.metrics:
    if metric not in rows[0]:
        print(f'skip {metric}: not in summary'); continue
    logy = metric in ('MSE', 'MSEtest')
    fig, axes = plt.subplots(len(SPS), len(QS), figsize=(3.1 * len(QS), 2.9 * len(SPS)),
                             dpi=200, squeeze=False)
    fig.patch.set_facecolor(SURF)
    for i, sp in enumerate(SPS):
        for j, q in enumerate(QS):
            ax = axes[i][j]; ax.set_facecolor(SURF)
            for s_ in ['top', 'right']: ax.spines[s_].set_visible(False)
            for s_ in ['left', 'bottom']: ax.spines[s_].set_color('#d8d7d2')
            ax.grid(color='#eceae5', lw=0.7); ax.set_axisbelow(True)
            ax.tick_params(colors=INK2, labelsize=7)
            for est, lab, col, mk, ls in SERIES:
                ys = [next((r[metric] for r in rows if r['sp'] == sp and r['q'] == q
                            and r['n'] == n and r['est'] == est), float('nan')) for n in NS]
                ax.plot(NS, ys, ls, marker=mk, ms=4, lw=1.4, color=col, label=lab)
            if logy: ax.set_yscale('log')
            elif metric in ('TPR', 'FPR'): ax.set_ylim(-0.03, 1.03)
            ax.set_title(f'q = {q}, $s_{{\\beta^*}}$ = {sp}', fontsize=8.5, color=INK)
            ax.set_xticks(NS); ax.set_xticklabels(NS, rotation=45, fontsize=6.5)
            if j == 0: ax.set_ylabel(f'{LABEL[metric]} ({a.stat})', fontsize=8, color=INK2)
            if i == len(SPS) - 1: ax.set_xlabel('n', fontsize=8, color=INK2)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(OUT, f'sim_{a.est}_{metric}_{a.stat}.png')
    fig.savefig(out, facecolor=SURF, bbox_inches='tight'); plt.close(fig)
    print('wrote', out)

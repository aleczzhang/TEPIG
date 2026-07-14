"""
make_selection_figures.py
-------------------------
Figures for the agreed reporting scheme (13 Jul): averages over repeated runs,
plus L1-normalised selection frequencies.

  fig5_selection_frequency.png
      The headline. Left: L1-normalised selection weight for every feature a
      method ever selects, sorted. TEPIG concentrates on a few features with a
      clear leader; CLUSSO and naive spread thin weight over nearly everything.
      Right: TEPIG's named features, with how many of the 10 runs picked each.

  fig6_threshold_pileup.png
      Why any single run is unstable: the block norms bunch up right at the
      coefficient threshold, so a new split flips a batch of features across it.

  fig7_prune_sweep.png
      Why |r| <= 0.95: it removes half the features at no cost in R^2, and
      pruning harder starts destroying signal.

Reads: results/repeated_<GENE>_dose<D>.pkl  (from repeated_runs.py)
"""

import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(_HERE, 'results')
FIG = os.path.join(_HERE, 'figures')
os.makedirs(FIG, exist_ok=True)

GENE, DOSE = 'GLRX', 6
COL = {'TEPIG': '#2a78d6', 'clusso': '#1baf7a', 'naive': '#eda100'}
LAB = {'TEPIG': 'TEPIG', 'clusso': 'CLUSSO', 'naive': 'naive'}
METHODS = ['TEPIG', 'clusso', 'naive']
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e3e2df'

plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': GRID, 'axes.linewidth': 0.8,
    'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})

# Prune sweep results (from prune_sweep.py, GLRX dose 6, 5 seeds)
SWEEP = {
    'thr':     ['none', '0.99', '0.95', '0.90', '0.85', '0.80'],
    'q':       [68, 59, 34, 26, 22, 17],
    'TEPIG_J': [0.29, 0.27, 0.39, 0.37, 0.34, 0.26],
    'naive_J': [0.53, 0.51, 0.73, 0.74, 0.81, 0.56],
    'TEPIG_R2':[0.406, 0.410, 0.410, 0.401, 0.386, 0.365],
}


def _clean(ax, keep=('left', 'bottom')):
    for s in ['top', 'right', 'left', 'bottom']:
        ax.spines[s].set_visible(s in keep)
    ax.tick_params(length=0)


def fig5(d):
    """One plot per setting: grouped bars, every feature selected by at least one
    method, showing all three methods' selection frequencies side by side.

    Two panels because the two normalisations answer different questions:
      left  — L1-normalised weight (share of a method's OWN selection budget).
              Comparable within a method; TEPIG's bars run taller because it
              concentrates its weight on fewer features. That concentration is
              the finding.
      right — raw pick rate (runs out of N). Directly comparable ACROSS methods.
    """
    runs = d['runs']
    # union of features selected by >= 1 method, ordered by TEPIG's weight then total
    union = sorted(set().union(*[set(d['freq'][m]) for m in METHODS]),
                   key=lambda f: (-d['freq']['TEPIG'].get(f, 0.0),
                                  -sum(d['freq'][m].get(f, 0.0) for m in METHODS)))
    ypos = np.arange(len(union))
    h = 0.26

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(14.6, max(7.0, 0.30 * len(union))), sharey=True,
        gridspec_kw={'wspace': 0.08})

    for k, m in enumerate(METHODS):
        off = (1 - k) * h                       # TEPIG top, CLUSSO mid, naive low
        w = [d['freq'][m].get(f, 0.0) for f in union]
        p = [d['picks'][m].get(f, 0) for f in union]
        ax.barh(ypos + off, w, height=h, color=COL[m], label=LAB[m],
                edgecolor='white', linewidth=1.2, zorder=3)
        ax2.barh(ypos + off, p, height=h, color=COL[m], label=LAB[m],
                 edgecolor='white', linewidth=1.2, zorder=3)

    ax.set_yticks(ypos)
    ax.set_yticklabels(union, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('L1-normalised selection frequency\n(share of that method’s own '
                  'selection weight)', fontsize=9.5, color=MUTED)
    ax.set_title('Relative weight within each method', fontsize=11.5, color=INK,
                 loc='left', pad=10)
    ax.legend(frameon=False, fontsize=9.5, loc='lower right')
    ax.grid(axis='x', color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    _clean(ax, keep=('bottom',))

    ax2.set_xlabel(f'runs (of {runs}) in which the feature was selected\n'
                   '(directly comparable across methods)',
                   fontsize=9.5, color=MUTED)
    ax2.set_title('Raw pick rate', fontsize=11.5, color=INK, loc='left', pad=10)
    ax2.set_xlim(0, runs + 0.5)
    ax2.set_xticks(range(0, runs + 1, 2))
    ax2.grid(axis='x', color=GRID, linewidth=0.8, zorder=0)
    ax2.set_axisbelow(True)
    _clean(ax2, keep=('bottom',))

    r2 = {m: np.mean(d['r2'][m]) for m in METHODS}
    sd = {m: np.std(d['r2'][m]) for m in METHODS}
    fig.suptitle(
        f'Selection frequency by feature and method  ·  {d["gene"]}, dose {d["dose"]}  ·  '
        f'{runs} repeated runs  ·  q={d["q"]} (pruned at |r|≤{d["prune"]})\n'
        f'test R²:  TEPIG {r2["TEPIG"]:.3f}±{sd["TEPIG"]:.3f}   '
        f'naive {r2["naive"]:.3f}±{sd["naive"]:.3f}   '
        f'CLUSSO {r2["clusso"]:.3f}±{sd["clusso"]:.3f}   —   '
        f'TEPIG selects {len(d["freq"]["TEPIG"])} distinct features, '
        f'CLUSSO {len(d["freq"]["clusso"])}, naive {len(d["freq"]["naive"])}',
        fontsize=11, color=INK, y=1.0)
    fig.savefig(os.path.join(FIG, 'fig5_selection_frequency.png'), dpi=200,
                bbox_inches='tight')
    print('  wrote fig5_selection_frequency.png')


def fig6():
    """Block norms piled at the cutoff — why a single run is unstable."""
    import sys
    sys.path.insert(0, os.path.join(_HERE, '..')); sys.path.insert(0, _HERE)
    import run_gene as R
    c = pickle.load(open(os.path.join(_HERE, 'cache', 'cp_lincs_tensor.pkl'), 'rb'))
    m = c['obs_dose_rank'] == DOSE
    X = c['X'][:, :, :, m]
    y_raw, _ = R.build_outcome(GENE, c['expr'][m], c['probes'], c['sym2probe'])

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6), sharex=True)
    for ax, seed in zip(axes, [42, 142, 442]):
        tr, _ = R.plain_split(X.shape[3], 0.2, seed)
        ymu, ysd = y_raw[tr].mean(), y_raw[tr].std()
        y = (y_raw - ymu) / ysd
        lam = R.tepig_lambda_grid(X[:, :, :, tr], y[tr])[6]
        _, B = R.proxgrad_fit(X[:, :, :, tr], y[tr], lam)
        G, q, S = B.shape
        d_ = G * q * S
        Xm = X[:, :, :, tr].mean(axis=3, keepdims=True)
        pred = (X[:, :, :, tr] - Xm).reshape(d_, len(tr)).T @ B.reshape(-1) + y[tr].mean()
        sig = float(np.std(y[tr] - pred, ddof=1))
        tau = sig * np.sqrt(2 * np.log(q) / len(tr))
        bn = np.sort(np.sqrt((B ** 2).sum(axis=(0, 2))))[::-1][:16]
        sel = bn > tau
        ax.scatter(bn, np.arange(len(bn)), s=52,
                   color=[COL['TEPIG'] if s_ else '#c9ced6' for s_ in sel],
                   edgecolor='white', linewidth=1.2, zorder=3)
        ax.axvline(tau, color='#e34948', linewidth=2, zorder=2)
        # label the cutoff at the BOTTOM of the panel (x in data coords, y in
        # axes fraction) so it cannot collide with the title above
        ax.text(tau, 0.02, f'  cutoff τ={tau:.3f}', color='#e34948', fontsize=9,
                va='bottom', ha='left', transform=ax.get_xaxis_transform(),
                zorder=4)
        ax.set_title(f'split {seed}  →  {int(sel.sum())} selected',
                     fontsize=11, color=INK, loc='left', pad=8)
        ax.invert_yaxis()
        ax.set_yticks([])
        ax.set_xlabel('feature score (block norm)', fontsize=9.5, color=MUTED)
        ax.grid(axis='x', color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        _clean(ax, keep=('bottom',))
    fig.suptitle('Why any one run is unstable: the feature scores pile up right at the cutoff '
                 '(blue = selected). Only the leftmost feature clears it every time.',
                 fontsize=11.5, color=INK, y=1.02)
    fig.savefig(os.path.join(FIG, 'fig6_threshold_pileup.png'), dpi=200,
                bbox_inches='tight')
    print('  wrote fig6_threshold_pileup.png')


def fig7():
    x = np.arange(len(SWEEP['thr']))
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.plot(x, SWEEP['TEPIG_J'], marker='o', markersize=8, linewidth=2,
            color=COL['TEPIG'], label='TEPIG — selection stability (Jaccard)',
            markeredgecolor='white', markeredgewidth=1.6, zorder=3)
    ax.plot(x, SWEEP['naive_J'], marker='o', markersize=8, linewidth=2,
            color=COL['naive'], label='naive — selection stability (Jaccard)',
            markeredgecolor='white', markeredgewidth=1.6, zorder=3)
    ax.plot(x, SWEEP['TEPIG_R2'], marker='s', markersize=7, linewidth=2,
            color=INK, linestyle='--', label='TEPIG — test R²',
            markeredgecolor='white', markeredgewidth=1.4, zorder=3)
    ax.axvspan(1.6, 2.4, color=COL['TEPIG'], alpha=0.08, zorder=0)
    ax.text(2, 0.86, 'chosen: |r| ≤ 0.95\nhalf the features removed,\nno loss of R²',
            ha='center', fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t}\nq={q}' for t, q in zip(SWEEP['thr'], SWEEP['q'])],
                       fontsize=9.5)
    ax.set_xlabel('correlation pruning threshold', fontsize=10, color=MUTED)
    ax.set_ylim(0.2, 0.95)
    ax.set_title('Pruning near-duplicates: |r| ≤ 0.95 is the sweet spot',
                 fontsize=12, color=INK, loc='left', pad=10)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5, loc='lower left')
    _clean(ax)
    fig.savefig(os.path.join(FIG, 'fig7_prune_sweep.png'), dpi=200,
                bbox_inches='tight')
    print('  wrote fig7_prune_sweep.png')


if __name__ == '__main__':
    p = os.path.join(RES, f'repeated_{GENE}_dose{DOSE}.pkl')
    d = pickle.load(open(p, 'rb'))
    fig5(d)
    fig6()
    fig7()
    print(f'figures -> {FIG}')

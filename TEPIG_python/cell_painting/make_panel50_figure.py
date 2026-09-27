"""Summary figure for the 50-platemap panel run (results/panel50_table.csv).

Two panels:
  A. per-gene test R^2 (mean +/- sd over 5 grouped 40/10 platemap splits) for
     TEPIG / CLUSSO / naive, grouped by biological program.
  B. accuracy vs number of selected features, TEPIG vs naive per gene --
     the parsimony story (same R^2, ~half the features).

Reads results/panel50_table.csv when present (the Zaratan output of
combine_panel50.py); otherwise uses the embedded copy of those numbers
(21 Sep 2026 run). Writes results/figures/panel50_summary.png.
"""
import csv, os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
# gene: (program, (TEPIG r2, sd, nsel), (clusso r2, sd, nsel), (naive r2, sd, nsel))
EMBED = {
    'HMGCS1':  ('cholesterol', (.205, .008, 132), (.190, .017, 148), (.214, .016, 132)),
    'NSDHL':   ('cholesterol', (.098, .042, 128), (.089, .041, 125), (.103, .039, 125)),
    'HMGCR':   ('cholesterol', (.099, .033, 112), (.092, .031, 113), (.103, .027, 78)),
    'FDFT1':   ('cholesterol', (.086, .036, 95),  (.080, .054, 141), (.087, .036, 107)),
    'INSIG1':  ('cholesterol', (.048, .023, 36),  (.032, .023, 136), (.045, .020, 72)),
    'TMEM97':  ('cholesterol', (.031, .018, 57),  (.016, .018, 101), (.035, .015, 69)),
    'HERPUD1': ('ER stress',   (.199, .038, 63),  (.186, .033, 137), (.210, .033, 184)),
    'XBP1':    ('ER stress',   (.129, .040, 76),  (.124, .053, 127), (.135, .042, 118)),
    'CREB3L2': ('ER stress',   (.038, .008, 24),  (.042, .015, 72),  (.044, .013, 102)),
    'CRELD2':  ('ER stress',   (.040, .020, 40),  (.042, .029, 92),  (.044, .028, 78)),
}
csv_path = os.path.join(_HERE, 'results', 'panel50_table.csv')
data = {}
if os.path.exists(csv_path):
    for r in csv.DictReader(open(csv_path)):
        data[r['gene']] = (
            'cholesterol' if r['program'] == 'cholesterol' else 'ER stress',
            *[(float(r[f'{m}_r2_mean']), float(r[f'{m}_r2_sd']),
               float(r[f'{m}_nsel'])) for m in ['TEPIG', 'clusso', 'naive']])
else:
    data = EMBED

SURF, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'
COL = {'TEPIG': '#2a78d6', 'CLUSSO': '#1baf7a', 'naive': '#eb6834'}

order = [g for prog in ['cholesterol', 'ER stress']
         for g in sorted((g for g in data if data[g][0] == prog),
                         key=lambda g: -data[g][1][0])]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 5.2), dpi=200)
fig.patch.set_facecolor(SURF)
for a in (ax, bx):
    a.set_facecolor(SURF)
    for s in ['top', 'right']:
        a.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        a.spines[s].set_color('#d8d7d2')
    a.tick_params(colors=INK2, labelsize=9)
    a.grid(axis='x', color='#eceae5', lw=0.8)
    a.set_axisbelow(True)

# ── A: per-gene R^2 dot plot ────────────────────────────────────────────────
ys = range(len(order))
for dy, (m, key) in zip((0.22, 0.0, -0.22),
                        [('TEPIG', 1), ('CLUSSO', 2), ('naive', 3)]):
    ax.errorbar([data[g][key][0] for g in order], [y + dy for y in ys],
                xerr=[data[g][key][1] for g in order], fmt='o', ms=5.5,
                color=COL[m], ecolor=COL[m], elinewidth=1.4, capsize=0,
                label=m, zorder=3)
ax.set_yticks(list(ys), order, fontsize=9.5, color=INK)
ax.invert_yaxis()
n_chol = sum(1 for g in order if data[g][0] == 'cholesterol')
ax.axhline(n_chol - 0.5, color='#d8d7d2', lw=0.8)
ax.text(-0.19, (n_chol - 1) / 2, 'CHOLESTEROL\nSYNTHESIS', ha='center', va='center',
        fontsize=7.5, color=INK2, rotation=90, transform=ax.get_yaxis_transform())
ax.text(-0.19, n_chol + (len(order) - n_chol - 1) / 2, 'ER\nSTRESS', ha='center',
        va='center', fontsize=7.5, color=INK2, rotation=90,
        transform=ax.get_yaxis_transform())
ax.axvline(0, color='#b9b7b0', lw=0.8)
ax.set_xlabel('test R² on held-out platemaps (mean ± sd, 5 splits)',
              fontsize=9.5, color=INK2)
ax.legend(frameon=False, fontsize=9, loc='lower right', labelcolor=INK)
ax.set_title('A · All 10 screened genes confirm on unseen compound sets',
             fontsize=10.5, color=INK, loc='left', pad=10)

# ── B: feature-count ratio vs naive (professor's redesign, 26 Sep) ────────
bx.grid(axis='x', which='both', color='#eceae5', lw=0.8)
for dy, (m, key) in zip((0.16, -0.16), [('TEPIG', 1), ('CLUSSO', 2)]):
    bx.scatter([data[g][key][2] / data[g][3][2] for g in order],
               [y + dy for y in ys], s=42, color=COL[m], zorder=3, label=m)
bx.axvline(1.0, color='#b9b7b0', lw=1.0, ls='--', zorder=1)
bx.set_xscale('log')
bx.set_xticks([0.25, 0.5, 1.0, 2.0], ['0.25×', '0.5×', '1×', '2×'])
bx.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
bx.set_xlim(0.18, 2.4)
bx.set_yticks(list(ys), order, fontsize=9.5, color=INK)
bx.invert_yaxis()
bx.axhline(n_chol - 0.5, color='#d8d7d2', lw=0.8)
bx.text(-0.19, (n_chol - 1) / 2, 'CHOLESTEROL\nSYNTHESIS', ha='center', va='center',
        fontsize=7.5, color=INK2, rotation=90, transform=bx.get_yaxis_transform())
bx.text(-0.19, n_chol + (len(order) - n_chol - 1) / 2, 'ER\nSTRESS', ha='center',
        va='center', fontsize=7.5, color=INK2, rotation=90,
        transform=bx.get_yaxis_transform())
bx.set_xlabel('features selected relative to naive lasso (log scale)',
              fontsize=9.5, color=INK2)
bx.legend(frameon=False, fontsize=9, loc='upper left', labelcolor=INK)
bx.set_title('B · Features selected, as a ratio to naive lasso',
             fontsize=10.5, color=INK, loc='left', pad=10)

fig.suptitle('Ten-gene panel on 50 CDRP platemaps (15,748 wells): '
             'grouped 40/10 platemap splits, pycytominer features',
             fontsize=11.5, color=INK, x=0.01, ha='left', y=1.0)
fig.text(0.01, 0.945, 'Genes chosen by an all-978-gene screen (median leave-one-'
         'platemap-out R², permutation-null threshold); TEPIG vs naive: '
         'ΔR² = −0.005, p = 0.002 — equal accuracy; fewer features for most genes (up to 4× fewer for ER-stress).',
         fontsize=8.5, color=INK2)
fig.tight_layout(rect=[0, 0, 1, 0.90])
out = os.path.join(_HERE, 'results', 'figures', 'panel50_summary.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, facecolor=SURF, bbox_inches='tight')
print('wrote', out)

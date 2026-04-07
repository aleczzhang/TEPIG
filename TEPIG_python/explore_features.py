"""
Step 1 of TEPIG simulation setup:
  - Build naive-averaged feature matrix (one row per subject, one value per
    feature = mean across ALL that subject's tubules across ALL their slides)
  - Drop highly correlated features (|corr| > 0.95) using greedy removal (remove features with most high correlations pairs first)
  - Display full correlation heatmap of remaining features
  - Identify candidate informative features for beta_star:
    features whose max absolute correlation with any other feature is low,
    satisfying the irrepresentable condition so lasso can recover them

Use Donors_included_after_biopsy_QCed, excluded folder has biopsis that failed 
quality control (eg. poor tissue quality, imaging artifacts, etc.). Excluded them
because would cause distortion in feature correlation structure.

Drop compartment_id: row identifier assigned by the image analysis software, not a biological measurement.
In Medulla: used to filter to cortical tubules only (In Medulla == 0) before dropping — matches
professor's CT_GRANULAR averaging which uses cortical tubules only. Verified: cortex-only averages
match Renal_Data.csv within 0.5%; all-tubule averages differ by ~8.6%.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from utils import load_tubule_data, build_naive_average, prune_correlated_features

# ── Config ────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
BASE        = os.path.join(_HERE, '..', 'Object_level_data', 'Donors_included_after_biopsy_QCed')
CORR_THRESH = 0.95   # greedily remove one feature from each pair above this
CAND_THRESH = 0.80   # candidate beta_star features: max |corr| with any other < this
N_TOP       = 10     # show pairwise correlations among the N most independent features
DROP_COLS   = ['compartment_id', 'In Medulla']
_BASE    = os.path.join(_HERE, '..', 'outputs')
OUT_REF  = os.path.join(_BASE, 'reference')
OUT_SUMM = os.path.join(_BASE, 'summaries')

# ── Setup output directory ─────────────────────────────────────────────────────
os.makedirs(OUT_REF, exist_ok=True)
os.makedirs(OUT_SUMM, exist_ok=True)

# ── Load all tubule data and compute naive average per subject ─────────────────
print("Loading tubule data from all included donors...")
slide_data, subject_dfs = load_tubule_data(BASE, drop_cols=DROP_COLS)
print(f"  Loaded {len(slide_data)} slides across {len(subject_dfs)} subjects")

X_avg = build_naive_average(subject_dfs)
print(f"  Naive-averaged matrix shape: {X_avg.shape}  (subjects x features)\n")

# ── Greedy removal of highly correlated features ───────────────────────────────
print(f"Dropping features with pairwise |corr| > {CORR_THRESH} (greedy removal)...")
remaining, dropped = prune_correlated_features(X_avg, corr_thresh=CORR_THRESH)
print(f"  Dropped {len(dropped)} features, {len(remaining)} remaining")
print(f"  Dropped: {dropped}\n")

X_clean  = X_avg[remaining]
# Recompute signed correlation (not absolute) for the heatmap
corr_mat = X_clean.corr()

# ── Plot full correlation heatmap ──────────────────────────────────────────────
print("Plotting correlation heatmap of remaining features...")
fig, ax = plt.subplots(figsize=(18, 16))

# imshow renders the correlation matrix as a colour grid.
# vmin/vmax set the colour scale range; cmap='RdBu_r' maps -1 to red, +1 to blue.
im = ax.imshow(corr_mat.values, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
plt.colorbar(im, ax=ax, shrink=0.8, label='Pearson correlation')

ax.set_xticks(range(len(remaining)))
ax.set_yticks(range(len(remaining)))
ax.set_xticklabels(remaining, rotation=90, fontsize=7)
ax.set_yticklabels(remaining, fontsize=7)
ax.set_title(
    f'Naive-averaged feature correlation matrix ({len(remaining)} features after pruning)',
    fontsize=12
)
plt.tight_layout()
heatmap_path = os.path.join(OUT_REF, 'correlation_heatmap.png')
plt.savefig(heatmap_path, dpi=150)
plt.close()

# ── Compute candidate features for beta_star ───────────────────────────────────
# A good candidate for a nonzero beta_star entry has LOW correlation with all
# other features. This is the key requirement of the irrepresentable condition:
# the design matrix columns for informative features must not be too similar
# to columns for uninformative features.
corr_abs = corr_mat.abs()

# For each feature, find its highest absolute correlation with any OTHER feature.
# np.eye creates an identity matrix; ~np.eye gives True everywhere off-diagonal.
max_corr_with_others = corr_abs.where(
    ~np.eye(len(remaining), dtype=bool)
).max()

candidates  = max_corr_with_others[max_corr_with_others < CAND_THRESH].sort_values()
cand_corr   = corr_mat.loc[candidates.index, candidates.index]
top_n_feats = max_corr_with_others.sort_values().head(N_TOP).index.tolist()
top_corr    = corr_mat.loc[top_n_feats, top_n_feats]

# ── Write all results to a text file ──────────────────────────────────────────
results_path = os.path.join(OUT_SUMM, 'feature_selection_results.txt')
with open(results_path, 'w') as f:

    f.write("=" * 80 + "\n")
    f.write("TEPIG FEATURE SELECTION RESULTS\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Subjects loaded  : {len(subject_dfs)}\n")
    f.write(f"Slides loaded    : {len(slide_data)}\n")
    f.write(f"Features (raw)   : {X_avg.shape[1]}\n")
    f.write(f"Pruning threshold: |corr| > {CORR_THRESH}\n\n")

    f.write("-" * 80 + "\n")
    f.write(f"DROPPED FEATURES ({len(dropped)}):\n")
    f.write("-" * 80 + "\n")
    for feat in dropped:
        f.write(f"  {feat}\n")
    f.write(f"\nFeatures remaining: {len(remaining)}\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write(f"ALL {len(remaining)} FEATURES RANKED BY MAX |CORR| WITH OTHERS\n")
    f.write("(low = more independent; better candidate for nonzero beta_star)\n")
    f.write("=" * 80 + "\n")
    f.write(f"{'Rank':<6} {'Feature':<55} {'Max |corr|':>10}\n")
    f.write("-" * 74 + "\n")
    for rank, (feat, val) in enumerate(max_corr_with_others.sort_values().items(), 1):
        marker = " <-- candidate" if feat in candidates.index else ""
        f.write(f"  {rank:<4} {feat:<53} {val:.3f}{marker}\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write(f"CANDIDATE FEATURES FOR BETA_STAR (max |corr| with any other < {CAND_THRESH})\n")
    f.write("=" * 80 + "\n")
    f.write(f"{'Feature':<55} {'Max |corr|':>10}\n")
    f.write("-" * 68 + "\n")
    for feat, val in candidates.items():
        f.write(f"  {feat:<53} {val:.3f}\n")
    f.write("\nPairwise correlations among candidates:\n")
    f.write(cand_corr.round(3).to_string() + "\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write(f"PAIRWISE CORRELATIONS AMONG TOP-{N_TOP} MOST INDEPENDENT FEATURES\n")
    f.write("Pick 3-5 with low values both here and in the ranked list above.\n")
    f.write("=" * 80 + "\n")
    f.write(top_corr.round(3).to_string() + "\n")

# Also save the full correlation matrix as CSV for inspection
corr_csv_path = os.path.join(OUT_REF, 'correlation_matrix.csv')
corr_mat.round(4).to_csv(corr_csv_path)

print(f"Saved: {heatmap_path}")
print(f"Saved: {results_path}")
print(f"Saved: {corr_csv_path}")

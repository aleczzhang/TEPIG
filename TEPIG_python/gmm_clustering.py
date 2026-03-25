"""
Step 2 of TEPIG simulation setup:
  - Load all tubule-level data from included donors
  - Drop highly correlated features using same greedy pruning as explore_features.py
  - Pool all tubules across all subjects and slides into one matrix
  - Fit a 2-component Gaussian Mixture Model (GMM) on the pooled tubules
  - Assign every tubule a cluster label (1 or 2)
  - For each subject x slide, compute:
      * Cluster proportions (weights w1, w2 = 1 - w1)
      * Mean feature vector per cluster (unweighted average)
      * Weighted cluster average: row g = w_g * mean(features in cluster g)
  - Save results to outputs/ for use by the simulation script

Why pool all tubules before clustering?
  GMM is fit on all tubules from all subjects at once so the cluster definitions
  are shared/consistent across subjects. If we clustered per-subject, cluster 1
  in subject A might mean something different than cluster 1 in subject B.

Why G=2 clusters?
  Confirmed by professor. In kidney pathology, tubules broadly split into two
  populations: relatively normal tubules and atrophic/injured tubules.

Outputs saved to outputs/ (relative to repo root):
  - gmm_model.pkl            : the fitted GaussianMixture object
  - cluster_results.pkl      : per-subject, per-slide clustering results and tensors
  - remaining_features.txt   : the 53 features kept after correlation pruning
  - gmm_summary.txt          : human-readable summary of clustering results
"""

import os
import pickle
import numpy as np
from collections import defaultdict
from sklearn.mixture import GaussianMixture
from utils import load_tubule_data, build_naive_average, prune_correlated_features, get_subject

# ── Config ─────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
BASE        = os.path.join(_HERE, '..', 'Object_level_data', 'Donors_included_after_biopsy_QCed')
OUT_DIR     = os.path.join(_HERE, '..', 'outputs')
CORR_THRESH = 0.95   # same pruning threshold as explore_features.py
DROP_COLS   = ['compartment_id', 'In Medulla']
G           = 2      # number of clusters (confirmed by professor)
RANDOM_SEED = 42     # for reproducibility

os.makedirs(OUT_DIR, exist_ok=True)

def get_slide_num(folder_name):
    """
    Extract slide number from folder name (the number before 'PAS').
    e.g. 'H19-00319_1_PAS' -> '1'
    """
    parts = folder_name.replace('_', ' ').split()
    return parts[-2]

# ── Step 1: Load all tubule data ───────────────────────────────────────────────
print("Loading tubule data...")
slide_data, subject_dfs = load_tubule_data(BASE, drop_cols=DROP_COLS)
for d, df in slide_data.items():
    print(f"  {d}: {len(df)} tubules")

print(f"\n  Total slides: {len(slide_data)}")
print(f"  Total subjects: {len(subject_dfs)}")

# ── Step 2: Greedy correlation pruning (same as explore_features.py) ──────────
print(f"\nPruning features with |corr| > {CORR_THRESH}...")
X_avg = build_naive_average(subject_dfs)
remaining, _ = prune_correlated_features(X_avg, corr_thresh=CORR_THRESH)
print(f"  Features remaining after pruning: {len(remaining)}")

# Save the feature list so the simulation script uses the exact same set
feat_list_path = os.path.join(OUT_DIR, 'remaining_features.txt')
with open(feat_list_path, 'w') as f:
    for feat in remaining:
        f.write(feat + '\n')
print(f"  Saved feature list: {feat_list_path}")

# Apply pruning to all slides
for d in slide_data:
    slide_data[d] = slide_data[d][remaining]

# ── Step 3: Pool all tubules and fit GMM ───────────────────────────────────────
# Stack all tubule rows from all slides into one large matrix.
# We track which slide each row came from so we can split labels back later.
print(f"\nPooling all tubules for GMM fitting...")

all_rows    = []   # list of feature arrays
slide_index = []   # parallel list: which slide_dir each row belongs to

for d, df in slide_data.items():
    all_rows.append(df.values)
    slide_index.extend([d] * len(df))

# np.vstack stacks a list of 2D arrays vertically (row-wise).
X_all       = np.vstack(all_rows)
slide_index = np.array(slide_index)
print(f"  Pooled matrix shape: {X_all.shape}  (tubules x features)")

print(f"\nFitting GMM with G={G} components...")
# GaussianMixture fits a mixture of G multivariate Gaussian distributions via
# the Expectation-Maximisation (EM) algorithm.
# covariance_type='full': each cluster gets its own full covariance matrix
#   (most flexible; matches R's Mclust default).
# n_init=10: run EM from 10 different random starting points and keep the best
#   result (guards against converging to a poor local optimum).
# max_iter=1000: maximum number of EM iterations per initialisation.
# random_state: seeds the random number generator for reproducibility.
gmm = GaussianMixture(
    n_components=G,
    covariance_type='full',
    n_init=10,
    max_iter=1000,
    random_state=RANDOM_SEED
)
gmm.fit(X_all)

# gmm.predict assigns each tubule to its most likely cluster (hard assignment).
# Labels are 0-indexed (0 or 1); we add 1 to make them 1-indexed (1 or 2).
labels_all = gmm.predict(X_all) + 1
print(f"  GMM converged: {gmm.converged_}")
print(f"  Cluster sizes: "
      f"cluster 1 = {(labels_all==1).sum()} tubules, "
      f"cluster 2 = {(labels_all==2).sum()} tubules")

# Save the fitted GMM model so it can be reused without re-fitting
gmm_path = os.path.join(OUT_DIR, 'gmm_model.pkl')
with open(gmm_path, 'wb') as f:
    # pickle serialises a Python object to binary so it can be saved and
    # reloaded exactly later.
    pickle.dump(gmm, f)
print(f"  Saved GMM model: {gmm_path}")

# ── Step 4: Build per-subject, per-slide cluster results ──────────────────────
# For each subject, for each of their slides:
#   1. Extract the cluster labels for that slide's tubules
#   2. Compute cluster proportions (weights)
#   3. Compute mean feature vector per cluster
#   4. Compute weighted cluster average: row g = w_g * mean_g
# Then stack across slides to get the X_i tensor of shape (G, q, S)
# where S = number of slides for that subject.
print("\nBuilding per-subject tensors...")

# Map subject -> list of their slide_dirs (sorted for consistency)
subject_slides = defaultdict(list)
for d in sorted(slide_data.keys()):
    subject_slides[get_subject(d)].append(d)

cluster_results = {}

for subj, slides in subject_slides.items():
    subj_result = {
        'slides'           : slides,
        'n_tubules'        : [],    # number of tubules per slide
        'labels'           : [],    # cluster labels per slide (array of 1s and 2s)
        'weights'          : [],    # [w1, w2] per slide
        'cluster_avgs'     : [],    # unweighted: list of (G, q) arrays
        'weighted_avgs'    : [],    # weighted:   list of (G, q) arrays
    }

    for d in slides:
        # Find which rows in X_all belong to this slide
        slide_mask  = slide_index == d
        slide_labels = labels_all[slide_mask]
        slide_feats  = X_all[slide_mask]
        n_tubules    = slide_feats.shape[0]

        # Cluster proportions: fraction of tubules in each cluster
        w1 = (slide_labels == 1).sum() / n_tubules
        w2 = 1.0 - w1

        # Mean feature vector for each cluster (unweighted average within cluster)
        # np.mean with axis=0 averages across rows (tubules), giving a (q,) vector.
        mean1 = slide_feats[slide_labels == 1].mean(axis=0)
        mean2 = slide_feats[slide_labels == 2].mean(axis=0)

        # Weighted cluster average: scale each cluster's mean by its proportion.
        # This is the X_i representation used in CLUSSO / TEPIG:
        # row g of X_i = w_g * mean(features in cluster g)
        # Shape: (G, q) = (2, 53)
        cluster_avg  = np.vstack([mean1,      mean2     ])   # unweighted (G, q)
        weighted_avg = np.vstack([w1 * mean1, w2 * mean2])   # weighted   (G, q)

        subj_result['n_tubules'].append(n_tubules)
        subj_result['labels'].append(slide_labels)
        subj_result['weights'].append([w1, w2])
        subj_result['cluster_avgs'].append(cluster_avg)
        subj_result['weighted_avgs'].append(weighted_avg)

    # If the subject has only 1 slide, create a noisy copy as slide 2.
    # This ensures all X tensors have S=2 for consistent tensor dimensions.
    # Noise is additive Gaussian with variance=1 (confirmed by professor).
    if len(subj_result['weighted_avgs']) == 1:
        rng  = np.random.default_rng(abs(hash(subj)) % (2**32))
        noisy_copy = (subj_result['weighted_avgs'][0]
                      + rng.normal(0, 1.0, subj_result['weighted_avgs'][0].shape))
        subj_result['weighted_avgs'].append(noisy_copy)

    # Stack slides into a tensor of shape (G, q, S=2)
    # np.stack with axis=2 stacks a list of (G, q) matrices along a new 3rd axis.
    subj_result['X_tensor'] = np.stack(subj_result['weighted_avgs'], axis=2)
    subj_result['n_slides']  = 2

    cluster_results[subj] = subj_result

# ── Step 5: Save cluster results ───────────────────────────────────────────────
results_path = os.path.join(OUT_DIR, 'cluster_results.pkl')
with open(results_path, 'wb') as f:
    pickle.dump(cluster_results, f)
print(f"  Saved cluster results: {results_path}")

# ── Step 6: Summary ───────────────────────────────────────────────────────────
n_slides_dist = defaultdict(int)
for subj, res in cluster_results.items():
    n_slides_dist[res['n_slides']] += 1

summary_path = os.path.join(OUT_DIR, 'gmm_summary.txt')
with open(summary_path, 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("GMM CLUSTERING SUMMARY\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Total subjects        : {len(cluster_results)}\n")
    f.write(f"Total slides          : {len(slide_data)}\n")
    f.write(f"Total tubules         : {len(X_all)}\n")
    f.write(f"Features after pruning: {len(remaining)}\n")
    f.write(f"GMM components (G)    : {G}\n")
    f.write(f"GMM converged         : {gmm.converged_}\n\n")
    f.write(f"Cluster sizes (all tubules pooled):\n")
    f.write(f"  Cluster 1: {(labels_all==1).sum()} tubules "
            f"({(labels_all==1).mean()*100:.1f}%)\n")
    f.write(f"  Cluster 2: {(labels_all==2).sum()} tubules "
            f"({(labels_all==2).mean()*100:.1f}%)\n\n")
    f.write(f"Slides per subject:\n")
    for n, count in sorted(n_slides_dist.items()):
        f.write(f"  {n} slide(s): {count} subjects\n")
    f.write(f"\nNote: subjects with 1 slide will get a noisy copy\n")
    f.write(f"for slide 2 in the simulation (noise level TBD by professor).\n\n")
    f.write("Per-subject tensor shapes (G x q x S):\n")
    for subj, res in sorted(cluster_results.items()):
        f.write(f"  {subj}: {res['X_tensor'].shape}\n")

print(f"  Saved summary: {summary_path}")
print("\nDone.")

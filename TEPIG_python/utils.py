"""
utils.py
--------
Shared utility functions for the TEPIG simulation pipeline.

Used by:
  - explore_features.py  : load_tubule_data, prune_correlated_features, get_subject
  - gmm_clustering.py    : load_tubule_data, prune_correlated_features, get_subject
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict


def get_subject(folder_name):
    """
    Extract subject ID from a slide folder name.

    Slide folders follow patterns like 'H19-00319_1_PAS' or 'H20-00927 1 PAS'.
    This function normalises underscores to spaces, then drops the trailing
    slide number and 'PAS' token to return just the subject ID.

    Examples:
        'H19-00319_1_PAS'  ->  'H19-00319'
        'H20-00927 1 PAS'  ->  'H20-00927'
        'H22-16058 PAS'    ->  'H22-16058'

    Used by: explore_features.py, gmm_clustering.py
    """
    parts = folder_name.replace('_', ' ').split()
    return ' '.join(parts[:-2])


def load_tubule_data(base, drop_cols=None):
    """
    Load tubule-level feature data from all included donor slide folders.

    For each slide folder inside `base` that contains a
    'final_combined_features_tubules.xlsx' file, this function:
      1. Reads the Excel file into a DataFrame (one row = one tubule)
      2. Drops any columns listed in `drop_cols`
      3. Stores the result keyed by slide folder name
      4. Groups slides by subject for computing per-subject averages

    Parameters
    ----------
    base      : str  - path to the Donors_included_after_biopsy_QCed folder
    drop_cols : list - column names to drop (e.g. non-feature metadata columns)

    Returns
    -------
    slide_data   : dict  {slide_dir -> DataFrame}  one DataFrame per slide
    subject_dfs  : dict  {subject_id -> list of DataFrames}  grouped by subject

    Used by: explore_features.py, gmm_clustering.py
    """
    if drop_cols is None:
        drop_cols = []

    # List all slide subfolders (folders whose names contain 'PAS')
    slide_dirs = sorted([
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and 'PAS' in d
    ])

    slide_data  = {}                  # slide_dir -> DataFrame
    subject_dfs = defaultdict(list)   # subject_id -> list of DataFrames

    for d in slide_dirs:
        fpath = os.path.join(base, d, 'final_combined_features_tubules.xlsx')
        if os.path.exists(fpath):
            # pd.read_excel reads an Excel file into a DataFrame.
            # engine='openpyxl' specifies the library used to parse .xlsx files.
            df = pd.read_excel(fpath, engine='openpyxl')
            df = df.drop(columns=[c for c in drop_cols if c in df.columns])
            slide_data[d] = df
            subject_dfs[get_subject(d)].append(df)

    return slide_data, subject_dfs


def build_naive_average(subject_dfs):
    """
    Compute the naive-averaged feature matrix: one row per subject,
    each value is the mean of that feature across ALL of the subject's
    tubules (across all their slides, ignoring cluster or slide structure).

    Parameters
    ----------
    subject_dfs : dict  {subject_id -> list of DataFrames}

    Returns
    -------
    X_avg : pd.DataFrame  shape (n_subjects, n_features)

    Used by: explore_features.py, gmm_clustering.py
    """
    X_avg_rows = {}
    for subj, dfs in subject_dfs.items():
        # pd.concat stacks all slide DataFrames vertically into one,
        # then .mean() averages across all tubule rows column-wise.
        combined = pd.concat(dfs, ignore_index=True)
        X_avg_rows[subj] = combined.mean()

    # pd.DataFrame(dict).T makes each dict key a row index.
    return pd.DataFrame(X_avg_rows).T


def prune_correlated_features(X_avg, corr_thresh=0.95):
    """
    Greedily remove features with pairwise |correlation| above corr_thresh.

    The irrepresentable condition for lasso requires that informative features
    are not too similar to uninformative ones. Features with |corr| > 0.95
    violate this and cause the sparse group lasso to become unstable.

    Algorithm:
      1. Compute the absolute Pearson correlation matrix of X_avg columns.
      2. Find all pairs with |corr| > corr_thresh.
      3. Remove the feature that appears in the most such pairs.
      4. Repeat until no pairs remain.

    Parameters
    ----------
    X_avg       : pd.DataFrame  shape (n_subjects, n_features)
    corr_thresh : float         threshold above which a pair is flagged

    Returns
    -------
    remaining : list of str  - feature names that survive pruning
    dropped   : list of str  - feature names that were removed

    Used by: explore_features.py, gmm_clustering.py
    """
    # DataFrame.corr() computes pairwise Pearson correlation of all columns.
    corr_full = X_avg.corr().abs()
    remaining = list(X_avg.columns)
    dropped   = []

    while True:
        sub_corr = corr_full.loc[remaining, remaining]
        # np.triu with k=1: upper triangle excluding diagonal, so each pair
        # is counted once and self-correlations (always 1.0) are excluded.
        upper = sub_corr.where(np.triu(np.ones(sub_corr.shape), k=1).astype(bool))
        bad   = upper > corr_thresh
        if not bad.any().any():
            break
        # Count how many bad pairs each feature is involved in (row + column).
        # Remove the feature with the most, as this eliminates the most pairs
        # in a single step.
        counts = bad.sum(axis=1) + bad.sum(axis=0)
        worst  = counts.idxmax()
        dropped.append(worst)
        remaining.remove(worst)

    return remaining, dropped

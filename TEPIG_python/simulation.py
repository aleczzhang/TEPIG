"""
simulation.py
-------------
Step 3 of the TEPIG pipeline. Runs B=200 simulation repetitions using real
tubule data as the basis for X_i tensors, generates synthetic outcomes, and
compares three estimators.

True generating model (rank-1 tensor):
    y_i = <X_i, B_true> + eps_i,   eps_i ~ N(0, SIGMA_SQ)
    B_true[g,j,s] = alpha_true[g] * beta_star[j] * gamma_true[s]

    X_i  : (G, q, S) weighted cluster average tensor for subject i
    alpha : true cluster weights    (G,)  -- cluster 2 weighted 2x
    beta  : true feature coefs      (q,)  -- 3 nonzero at value 15, rest 0
    gamma : true slide weights      (S,)  -- slide 1 weighted 2x

Three estimators
----------------
TEPIG       : alternating 3-mode structured lasso on full (G, q, S) tensor
Naive CLUSSO: average X_i across slides -> (G, q), run existing CLUSSO solver
Oracle TEPIG: fix true alpha and gamma, lasso only on beta (reduced 1D problem)

Metrics (per repetition, averaged over B)
-----------------------------------------
TPR : fraction of nonzero beta_star entries correctly identified as nonzero
FPR : fraction of zero beta_star entries incorrectly identified as nonzero
L1  : ||beta_hat - beta_star||_1  (L1 estimation bias)
MSE : mean squared prediction error on training subjects

Outputs saved to outputs/
    simulation_results.pkl  : raw per-repetition metrics + config
    simulation_summary.txt  : human-readable table
"""

import os
import sys
import pickle
import numpy as np
from joblib import Parallel, delayed

# Allow importing CLUSSO_python utilities from sibling directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CLUSSO_python'))
from Mainfunction_albet import Mainfunction_albet, _glmnet_lasso
from SLasso_MSE import lambda_CV_mse

# ── Config ─────────────────────────────────────────────────────────────────────
OUT_DIR       = os.path.join(os.path.dirname(__file__), '..', 'outputs')

# Beta_star: three most independent features (from explore_features.py output),
# each assigned value 15 so the signal is large enough to detect with lasso.
BETA_VAL      = 15.0
BETA_FEATURES = [
    'Standard Deviation Red Nuclei',
    'Average TBM Thickness',
    'Total Object Aspect Ratio',
]

# True cluster and slide weights (unnormalized; normalised internally below).
# Cluster 2 is given double weight so the two clusters are not symmetric.
# Slide 1 is given double weight so naive averaging (equal weights) loses info.
ALPHA_TRUE    = np.array([1.0, 2.0])
GAMMA_TRUE    = np.array([2.0, 1.0])

SIGMA_SQ      = 1.0     # outcome noise variance  (confirmed by professor)
B_SIMS        = 200     # simulation repetitions
LAM_GRID      = np.logspace(-4, 0, 15)   # lambda candidates for CV
M_INIT        = 3       # random initialisations per final TEPIG fit
MAX_ITER      = 50      # max alternating iterations in tepig_fit
TOL           = 1e-2    # convergence tolerance
N_JOBS        = -1      # joblib: -1 uses all available cores
RANDOM_SEED   = 42

# ── Load pre-computed cluster results ──────────────────────────────────────────
print("Loading cluster results...")
with open(os.path.join(OUT_DIR, 'cluster_results.pkl'), 'rb') as f:
    cluster_results = pickle.load(f)

with open(os.path.join(OUT_DIR, 'remaining_features.txt')) as f:
    features = [line.strip() for line in f]

subjects = sorted(cluster_results.keys())
n        = len(subjects)
q        = len(features)
S        = cluster_results[subjects[0]]['n_slides']
G        = 2

print(f"  n={n} subjects, q={q} features, G={G} clusters, S={S} slides")

# Stack X_i tensors into one 4D array: shape (G, q, S, n)
X_all = np.stack(
    [cluster_results[s]['X_tensor'] for s in subjects],
    axis=3
)

# ── Define beta_star ───────────────────────────────────────────────────────────
beta_star = np.zeros(q)
for feat in BETA_FEATURES:
    if feat in features:
        beta_star[features.index(feat)] = BETA_VAL
    else:
        print(f"  WARNING: '{feat}' not found in remaining features — check name")

nonzero_idx = np.where(beta_star != 0)[0]
print(f"  beta_star: {len(nonzero_idx)} nonzero at value {BETA_VAL}")
print(f"  Nonzero features: {[features[i] for i in nonzero_idx]}")

# L1-normalise true alpha and gamma for generating y_true.
# The scale is absorbed into beta_star (value=15), so we only need the direction.
alpha_norm = ALPHA_TRUE / np.sum(np.abs(ALPHA_TRUE))   # [1/3, 2/3]
gamma_norm = GAMMA_TRUE / np.sum(np.abs(GAMMA_TRUE))   # [2/3, 1/3]

# True signal for all subjects: y_true[i] = sum_{g,j,s} alpha[g]*beta[j]*gamma[s]*X[g,j,s,i]
y_true = np.einsum('gjsn,g,j,s->n', X_all, alpha_norm, beta_star, gamma_norm)

# ── Joint W: collapse cluster and slide dimensions ─────────────────────────────
# Combine alpha (G,) and gamma (S,) into a single weight vector W of shape (G*S,).
# This reduces the 3-mode problem back to the 2-mode CLUSSO structure:
#   y_i = W' X_collapsed[:,:,i] beta
# where X_collapsed is (G*S, q, n) = (4, 53, n).
#
# X_all is (G, q, S, n). To get (G*S, q, n):
#   transpose to (G, S, q, n) then reshape to (G*S, q, n)
# Row g*S+s of X_collapsed[:,: ,i] = X_all[g,:,s,i]
# i.e. the 4 rows are: (cluster1,slide1), (cluster1,slide2), (cluster2,slide1), (cluster2,slide2)
X_collapsed = X_all.transpose(0, 2, 1, 3).reshape(G * S, q, n)   # (4, 53, 116)


# ── TEPIG: alternating 3-mode structured lasso ────────────────────────────────

def _normalize_vec(v):
    """L1-normalise with positive-sign convention (largest |entry| is positive)."""
    if np.max(np.abs(v)) > 0:
        sig = np.sign(v[np.argmax(np.abs(v))])
        return sig * v / np.sum(np.abs(v))
    return v


def tepig_fit(X, y, lam, alpha_init, beta_init, gamma_init):
    """
    Fit the TEPIG rank-1 tensor model by alternating structured lasso.

    Model: y_i = sum_{g,j,s} alpha[g] * beta[j] * gamma[s] * X[g,j,s,i]

    Alternates three steps:
      1. Fix alpha, gamma -> contract X -> lasso on beta
      2. Fix beta,  gamma -> contract X -> OLS   on alpha
      3. Fix alpha, beta  -> contract X -> OLS   on gamma

    Parameters
    ----------
    X           : (G, q, S, n) array
    y           : (n,) response
    lam         : regularisation parameter (glmnet scale)
    alpha_init  : (G,)  starting cluster weights
    beta_init   : (q,)  starting feature coefficients
    gamma_init  : (S,)  starting slide weights

    Returns
    -------
    alpha_hat : (G,)
    beta_hat  : (q,)
    gamma_hat : (S,)
    """
    G, q, S, n = X.shape
    alpha = np.asarray(alpha_init, dtype=float).copy()
    beta  = np.asarray(beta_init,  dtype=float).copy()
    gamma = np.asarray(gamma_init, dtype=float).copy()

    alpha    = _normalize_vec(alpha)
    nr_alpha = max(0.0, float(np.sum(np.abs(alpha))))

    for _ in range(MAX_ITER):
        alpha0, beta0, gamma0 = alpha.copy(), beta.copy(), gamma.copy()

        # Step 1: fix alpha, gamma -> update beta via lasso
        # Z[j, i] = sum_g sum_s alpha[g] * gamma[s] * X[g,j,s,i]
        Z = np.einsum('gjsn,g,s->jn', X, alpha, gamma)        # (q, n)
        eff_lam = float(lam) * np.sum(np.abs(alpha)) * np.sum(np.abs(gamma))
        beta = _glmnet_lasso(Z.T, y, eff_lam)                 # (q,)

        if np.max(np.abs(beta)) > 0:
            beta = _normalize_vec(beta)

        # Step 2: fix beta, gamma -> update alpha via OLS (with intercept)
        if np.max(np.abs(beta)) > 0:
            W = np.einsum('gjsn,j,s->gn', X, beta, gamma)     # (G, n)
            W_int = np.column_stack([np.ones(n), W.T])         # (n, G+1)
            coeffs, _, _, _ = np.linalg.lstsq(W_int, y, rcond=None)
            alpha = coeffs[1:]                                  # drop intercept

        nr_alpha = max(0.0, float(np.sum(np.abs(alpha))))
        if np.max(np.abs(alpha)) > 0:
            alpha = _normalize_vec(alpha)

        # Step 3: fix alpha, beta -> update gamma via OLS (with intercept)
        if np.max(np.abs(alpha)) > 0 and np.max(np.abs(beta)) > 0:
            V = np.einsum('gjsn,g,j->sn', X, alpha, beta)     # (S, n)
            V_int = np.column_stack([np.ones(n), V.T])         # (n, S+1)
            coeffs, _, _, _ = np.linalg.lstsq(V_int, y, rcond=None)
            gamma = coeffs[1:]                                  # drop intercept

        # Convergence: sum of L2 changes across all three vectors
        dif = (np.linalg.norm(alpha - alpha0) +
               np.linalg.norm(beta  - beta0 ) +
               np.linalg.norm(gamma - gamma0))
        if dif < TOL:
            break

    # Zero out numerically tiny entries; restore true alpha scale
    alpha[np.abs(alpha) < 0.001] = 0.0
    beta[np.abs(beta)   < 0.001] = 0.0
    gamma[np.abs(gamma) < 0.001] = 0.0
    alpha = nr_alpha * alpha

    return alpha, beta, gamma


def _make_folds(n, rng):
    """Partition n subject indices into 5 folds (mirrors SLasso_MSE.CV_make_folds)."""
    idx  = rng.permutation(n).tolist()
    size = n // 5
    folds = [sorted(idx[k * size:(k + 1) * size]) for k in range(4)]
    folds.append(sorted(idx[4 * size:]))    # fold 5 gets remainder
    return folds


def tepig_select_and_fit(X, y, rng):
    """
    Select lambda by 5-fold CV (1 uniform init per fold), then re-fit on all
    data with M_INIT random starts; return the best-MSE result.
    """
    G, q, S, n = X.shape
    all_idx = list(range(n))

    # Lambda selection via CV
    cv_mse = []
    for lam in LAM_GRID:
        folds    = _make_folds(n, rng)
        fold_mse = []
        for k in range(5):
            te = folds[k]
            tr = [i for i in all_idx if i not in te]
            a0 = np.ones(G) / G
            b0 = np.ones(q) / q
            g0 = np.ones(S) / S
            a, b, g = tepig_fit(X[:, :, :, tr], y[tr], lam, a0, b0, g0)
            y_pred = np.einsum('gjsn,g,j,s->n', X[:, :, :, te], a, b, g)
            fold_mse.append(float(np.mean((y[te] - y_pred) ** 2)))
        cv_mse.append(float(np.mean(fold_mse)))

    best_lam = LAM_GRID[int(np.argmin(cv_mse))]

    # Final fit: M_INIT random starts, keep lowest training MSE
    best_mse = np.inf
    best_res = (np.ones(G) / G, np.zeros(q), np.ones(S) / S)
    for _ in range(M_INIT):
        a0 = rng.dirichlet(np.ones(G))
        b0 = rng.uniform(-1, 1, q)
        g0 = rng.dirichlet(np.ones(S))
        a, b, g = tepig_fit(X, y, best_lam, a0, b0, g0)
        y_pred  = np.einsum('gjsn,g,j,s->n', X, a, b, g)
        mse     = float(np.mean((y - y_pred) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_res = (a, b, g)

    return best_res   # (alpha_hat, beta_hat, gamma_hat)


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(beta_hat, y, y_pred):
    """
    Compute TPR, FPR, L1 bias, MSE for a fitted beta_hat.

    TPR : among true nonzero entries of beta_star, fraction estimated nonzero
    FPR : among true zero    entries of beta_star, fraction estimated nonzero
    L1  : sum |beta_hat_norm - beta_star_norm|  (both L1-normalised)
          Matches R code: CLUSSO_Functions_Project1_6_16_23.R lines 177, 320
          beta_star /= sum(|beta_star|) and beta_hat /= sum(|beta_hat|) before comparison
    MSE : mean (y_i - y_hat_i)^2  using original-scale predictions (R code line 333)
    """
    nz     = beta_star != 0          # true nonzero mask
    z      = ~nz                     # true zero mask
    hat_nz = np.abs(beta_hat) > 1e-6 # estimated nonzero (after lasso thresholding)

    tpr = float(hat_nz[nz].mean()) if nz.sum() > 0 else float('nan')
    fpr = float(hat_nz[z].mean())  if z.sum()  > 0 else float('nan')

    # L1 bias: normalise both to unit L1 norm before comparing (matches R code)
    beta_star_norm = beta_star / np.sum(np.abs(beta_star))
    beta_hat_norm  = (beta_hat / np.sum(np.abs(beta_hat))
                      if np.sum(np.abs(beta_hat)) > 0 else beta_hat.copy())
    l1  = float(np.sum(np.abs(beta_hat_norm - beta_star_norm)))

    mse = float(np.mean((y - y_pred) ** 2))
    return tpr, fpr, l1, mse


# ── Single simulation repetition ──────────────────────────────────────────────

def run_one_sim(seed):
    """
    One simulation repetition.

    1. Draw noise eps ~ N(0, SIGMA_SQ) and form y = y_true + eps.
    2. Fit TEPIG, naive CLUSSO, oracle TEPIG on (X_all, y).
    3. Return metrics for each estimator.
    """
    rng = np.random.default_rng(seed)
    y   = y_true + rng.normal(0, np.sqrt(SIGMA_SQ), n)

    results = {}

    # ── TEPIG ─────────────────────────────────────────────────────────────────
    a_hat, b_hat, g_hat = tepig_select_and_fit(X_all, y, rng)
    y_pred_tepig = np.einsum('gjsn,g,j,s->n', X_all, a_hat, b_hat, g_hat)
    results['tepig'] = compute_metrics(b_hat, y, y_pred_tepig)

    # ── Naive CLUSSO ──────────────────────────────────────────────────────────
    # Average X over the slide dimension to collapse (G, q, S, n) -> (G, q, n).
    # This is equivalent to treating all slides as a single average tissue section.
    X_naive = X_all.mean(axis=2)   # (G, q, n)

    cv_mse_naive = [
        lambda_CV_mse(X_naive, y, np.ones(G) / G, np.ones(q) / q, lam)
        for lam in LAM_GRID
    ]
    best_lam_naive = LAM_GRID[int(np.argmin(cv_mse_naive))]

    best_mse_naive = np.inf
    best_b_naive   = np.zeros(q)
    for _ in range(M_INIT):
        a0  = rng.dirichlet(np.ones(G))
        b0  = rng.uniform(-1, 1, q)
        res = Mainfunction_albet(X_naive, y, a0, b0, best_lam_naive)
        a_n, b_n = res['alpha'], res['bet']
        # Prediction: y_hat_i = alpha' X_naive[:,:,i] beta
        y_pred_n = np.array([a_n @ X_naive[:, :, i] @ b_n for i in range(n)])
        mse_n    = float(np.mean((y - y_pred_n) ** 2))
        if mse_n < best_mse_naive:
            best_mse_naive = mse_n
            best_b_naive   = b_n
            best_a_naive   = a_n

    y_pred_naive = np.array([best_a_naive @ X_naive[:, :, i] @ best_b_naive
                             for i in range(n)])
    results['naive'] = compute_metrics(best_b_naive, y, y_pred_naive)

    # ── Joint W TEPIG ─────────────────────────────────────────────────────────
    # Collapse cluster and slide dimensions into one weight vector W of shape (G*S=4,).
    # X_collapsed is (G*S, q, n) = (4, 53, n) — same structure as CLUSSO but p=4.
    # This lets us reuse Mainfunction_albet directly, alternating between W and beta.
    GS = G * S
    cv_mse_joint = [
        lambda_CV_mse(X_collapsed, y, np.ones(GS) / GS, np.ones(q) / q, lam)
        for lam in LAM_GRID
    ]
    best_lam_joint = LAM_GRID[int(np.argmin(cv_mse_joint))]

    best_mse_joint = np.inf
    best_b_joint   = np.zeros(q)
    best_w_joint   = np.ones(GS) / GS
    for _ in range(M_INIT):
        w0  = rng.dirichlet(np.ones(GS))
        b0  = rng.uniform(-1, 1, q)
        res = Mainfunction_albet(X_collapsed, y, w0, b0, best_lam_joint)
        w_j, b_j = res['alpha'], res['bet']
        y_pred_j = np.array([w_j @ X_collapsed[:, :, i] @ b_j for i in range(n)])
        mse_j    = float(np.mean((y - y_pred_j) ** 2))
        if mse_j < best_mse_joint:
            best_mse_joint = mse_j
            best_b_joint   = b_j
            best_w_joint   = w_j

    y_pred_joint = np.array([best_w_joint @ X_collapsed[:, :, i] @ best_b_joint
                             for i in range(n)])
    results['tepig_joint'] = compute_metrics(best_b_joint, y, y_pred_joint)

    # ── Oracle TEPIG ──────────────────────────────────────────────────────────
    # Fix true alpha_norm and gamma_norm; only estimate beta via lasso.
    # This is the best achievable estimator given knowledge of the true structure.
    # Contracted design matrix: Z[i,j] = sum_g sum_s alpha[g]*gamma[s]*X[g,j,s,i]
    Z_oracle = np.einsum('gjsn,g,s->nj', X_all, alpha_norm, gamma_norm)  # (n, q)

    folds = _make_folds(n, rng)
    all_idx = list(range(n))
    cv_mse_oracle = []
    for lam in LAM_GRID:
        fold_mse = []
        for k in range(5):
            te = folds[k]
            tr = [i for i in all_idx if i not in te]
            b_o    = _glmnet_lasso(Z_oracle[tr], y[tr], lam)
            y_pred = Z_oracle[te] @ b_o
            fold_mse.append(float(np.mean((y[te] - y_pred) ** 2)))
        cv_mse_oracle.append(float(np.mean(fold_mse)))

    best_lam_oracle = LAM_GRID[int(np.argmin(cv_mse_oracle))]
    b_oracle  = _glmnet_lasso(Z_oracle, y, best_lam_oracle)
    y_pred_oracle = Z_oracle @ b_oracle
    results['oracle'] = compute_metrics(b_oracle, y, y_pred_oracle)

    return results


# ── Main: run B_SIMS repetitions in parallel ──────────────────────────────────
if __name__ == '__main__':
    print(f"\nRunning {B_SIMS} simulation repetitions (n_jobs={N_JOBS})...")

    # Generate reproducible seeds for each repetition
    seed_seq  = np.random.SeedSequence(RANDOM_SEED)
    seed_ints = [int(s.generate_state(1)[0]) for s in seed_seq.spawn(B_SIMS)]

    # joblib parallelises over repetitions; each worker gets its own RNG seed
    all_results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(run_one_sim)(s) for s in seed_ints
    )

    # ── Collect and summarise ──────────────────────────────────────────────────
    print("\nSummarising results...")
    estimators = ['tepig', 'tepig_joint', 'naive', 'oracle']
    metric_keys = ['tpr', 'fpr', 'l1', 'mse']

    summary = {est: {m: [] for m in metric_keys} for est in estimators}
    for rep in all_results:
        for est in estimators:
            tpr, fpr, l1, mse = rep[est]
            summary[est]['tpr'].append(tpr)
            summary[est]['fpr'].append(fpr)
            summary[est]['l1'].append(l1)
            summary[est]['mse'].append(mse)

    # Save raw results for further analysis
    results_path = os.path.join(OUT_DIR, 'simulation_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump({
            'summary': summary,
            'config': {
                'B': B_SIMS, 'n': n, 'q': q, 'G': G, 'S': S,
                'BETA_VAL': BETA_VAL, 'BETA_FEATURES': BETA_FEATURES,
                'ALPHA_TRUE': ALPHA_TRUE.tolist(),
                'GAMMA_TRUE': GAMMA_TRUE.tolist(),
                'SIGMA_SQ': SIGMA_SQ,
            }
        }, f)

    # Write human-readable summary
    summary_path = os.path.join(OUT_DIR, 'simulation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("TEPIG SIMULATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"B={B_SIMS} repetitions | n={n} subjects | q={q} features "
                f"| G={G} clusters | S={S} slides\n")
        f.write(f"beta_star: {len(nonzero_idx)} nonzero features at value "
                f"{BETA_VAL}\n")
        f.write(f"Nonzero: {', '.join(features[i] for i in nonzero_idx)}\n")
        f.write(f"alpha_true (normalised): {alpha_norm.tolist()}\n")
        f.write(f"gamma_true (normalised): {gamma_norm.tolist()}\n\n")
        f.write(f"{'Estimator':<16} {'TPR':>8} {'FPR':>8} "
                f"{'L1 bias':>10} {'MSE':>10}\n")
        f.write("-" * 56 + "\n")
        for est in estimators:
            tpr = float(np.nanmean(summary[est]['tpr']))
            fpr = float(np.nanmean(summary[est]['fpr']))
            l1  = float(np.nanmean(summary[est]['l1']))
            mse = float(np.nanmean(summary[est]['mse']))
            f.write(f"  {est:<14} {tpr:>8.3f} {fpr:>8.3f} "
                    f"{l1:>10.3f} {mse:>10.3f}\n")

    print(f"  Saved: {results_path}")
    print(f"  Saved: {summary_path}")
    print("\nDone.")

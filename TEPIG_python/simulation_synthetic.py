"""
simulation_synthetic.py
-----------------------
Full simulation study with synthetic data (CLUSSO Section 4.1 style).

Usage:
    python simulation_synthetic.py --n 300 --q 10 --sparsity 0.8

Data generation (with individual tubules, matching CLUSSO paper):
  1. Draw latent population-level cluster means once per rep (G, q, S):
       Slide 1: cluster 1 ~ N(2,1),  cluster 2 ~ N(5,3)
       Slide 2: cluster 1 ~ N(4,2),  cluster 2 ~ N(8,1)
  2. For each subject i, for each slide s:
       Draw weight w_{i,s} ~ Uniform{0.2,...,0.8}
       Draw K tubules per cluster by adding N(0, SIGMA_R) noise to latent means
       Compute cluster weighted averages -> (G, q) per slide
  3. Stack S=2 slides -> X_tepig_i in R^(G=2, q, S=2)  [for TEPIG]
     Pool tubules from both slides per cluster -> X_clusso_i in R^(G=2, q)  [for CLUSSO]

True model (full tensor, no low-rank):
    y_i = intercept + <B_true, X_tepig_i> + eps_i,   eps_i ~ N(0, SIGMA)
    B_true[:, j, :] = B_TRUE_BLOCK  for j in nonzero_features (randomly sampled each rep)

Estimators:
    tepig         : proximal gradient + group lasso on full (G, q, S, n) tensor  [main method]
    tepig_lowrank : alternating rank-1 structured lasso on (G, q, S, n) tensor   [reference only]
    clusso        : Mainfunction_albet on (G=2, q, n) mega-slide (pool both slides' tubules)
    naive         : average TEPIG slides -> (G=2, q, n), run Mainfunction_albet
    oracle        : group lasso on true population-level X_true with known nonzero features (lam=1e-6)

Lambda grid (fixed, same for ALL estimators):
    {0.25, 0.50, 0.75, ..., 2.50, 3.00, 3.50, 4.00, 4.50, 5.00}
    (union of CLUSSO paper grid {0.5,...,5} and our divided-by-2 grid {0.25,...,2.5})
"""

import os
import sys
import argparse
import pickle
import warnings
import numpy as np
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')

from sklearn.linear_model import LassoCV
from Mainfunction_albet import Mainfunction_albet, _glmnet_lasso
from SLasso_MSE import lambda_CV_mse

_BASE    = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')
OUT_DATA = os.path.join(_BASE, 'data', 'threshold_cmp')
OUT_SUMM = os.path.join(_BASE, 'results')
os.makedirs(OUT_DATA, exist_ok=True)
os.makedirs(OUT_SUMM, exist_ok=True)

# ── Parse args ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--n',        type=int,   required=True, help='Number of subjects')
parser.add_argument('--q',        type=int,   default=10,    help='Number of features')
parser.add_argument('--sparsity', type=float, default=0.8,   help='Proportion of zero features')
args = parser.parse_args()

N        = args.n
Q        = args.q
SPARSITY = args.sparsity
N_NZ     = max(1, round(Q * (1.0 - SPARSITY)))  # number of nonzero features

# ── Config (fixed across all settings) ────────────────────────────────────────
G          = 2      # number of clusters
S          = 2      # number of slides
K          = 40     # mean tubules per slide per subject (CLUSSO paper: mu_M=40)
SIGMA_SQ_M = 5      # variance of tubule count (CLUSSO paper: sigma_sq_m=5)
SIGMA      = 1.0    # outcome noise std
SIGMA_R    = 1.0    # tubule measurement noise std (CLUSSO paper: sigma_R=1)

# Discrete uniform range for tubule count, matching CLUSSO paper exactly:
#   H = floor(sqrt(12 * sigma_sq_m + 1)) = 7 (odd, no adjustment needed)
#   K_is ~ Uniform{K - (H-1)/2, ..., K + (H-1)/2} = Uniform{37, ..., 43}
_H    = int(np.sqrt(12 * SIGMA_SQ_M + 1))
if _H % 2 == 0:
    _H -= 1
K_LO  = K - (_H - 1) // 2   # 37
K_HI  = K + (_H - 1) // 2   # 43
B_SIMS  = 200    # simulation repetitions
N_JOBS  = -1     # joblib: -1 uses all available cores
RANDOM_SEED = 42

MAX_ITER = 50
TOL      = 1e-2
M_INIT   = 3

# Fixed lambda grid — same for ALL estimators across ALL settings
LAM_GRID = np.array([0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75,
                     2.00, 2.25, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00])

# True B block: (G=2, S=2), rank-2 (det != 0)
B_TRUE_BLOCK = np.array([[1.0, 2.0],
                          [3.0, 4.0]])

sparsity_str = f"{int(SPARSITY * 10):02d}"   # "08" for 0.8, "04" for 0.4

print(f"simulation_synthetic.py | n={N}, q={Q}, sparsity={SPARSITY}, n_nonzero={N_NZ}")
print(f"  G={G}, S={S}, K={K} tubules/slide (discrete uniform {{{K_LO}..{K_HI}}})")
print(f"  B_TRUE_BLOCK det={float(np.linalg.det(B_TRUE_BLOCK)):.3f}  (rank-2)")
print(f"  Lambda grid: {LAM_GRID[0]:.2f} to {LAM_GRID[-1]:.2f}  ({len(LAM_GRID)} values)")
print(f"  Running {B_SIMS} reps with n_jobs={N_JOBS}...")


# ── Helpers ────────────────────────────────────────────────────────────────────

def compute_metrics(beta_hat, y, y_pred, beta_star):
    """TPR, FPR, L1 bias, MSE. beta_star passed per-rep since nonzero_idx varies."""
    nz     = beta_star != 0
    z      = ~nz
    hat_nz = np.abs(beta_hat) > 1e-6

    tpr = float(hat_nz[nz].mean()) if nz.sum() > 0 else float('nan')
    fpr = float(hat_nz[z].mean())  if z.sum()  > 0 else float('nan')

    bsn = beta_star / np.sum(np.abs(beta_star))
    bhn = (beta_hat / np.sum(np.abs(beta_hat))
           if np.sum(np.abs(beta_hat)) > 0 else beta_hat.copy())
    l1  = float(np.sum(np.abs(bhn - bsn)))
    mse = float(np.mean((y - y_pred) ** 2))
    return tpr, fpr, l1, mse


def _normalize_vec(v):
    if np.max(np.abs(v)) > 0:
        sig = np.sign(v[np.argmax(np.abs(v))])
        return sig * v / np.sum(np.abs(v))
    return v


def _make_folds(n, rng):
    idx  = rng.permutation(n).tolist()
    size = n // 5
    folds = [sorted(idx[k * size:(k + 1) * size]) for k in range(4)]
    folds.append(sorted(idx[4 * size:]))
    return folds


# ── Data generation with individual tubules ────────────────────────────────────

def generate_data(rng, n, q):
    """
    Generate synthetic data following CLUSSO Section 4.1 with individual tubules.

    Returns
    -------
    X_tepig  : (G, q, S, n)  — per-slide cluster weighted averages for TEPIG
    X_true   : (G, q, S, n)  — true population-level weighted averages (no tubule noise)
    X_clusso : (G, q, n)     — mega-slide cluster weighted averages for CLUSSO
    nonzero_idx : (N_NZ,)    — randomly sampled nonzero feature indices
    B_true   : (G, q, S)     — true coefficient tensor
    beta_star: (q,)          — Frobenius norm of B_true[:,j,:] per feature
    """
    weight_choices = np.arange(0.2, 0.9, 0.1)

    # Step 1: Draw latent population-level cluster means (one per cluster per slide)
    # Shape: (G, q, S)
    mu = np.zeros((G, q, S))
    mu[0, :, 0] = rng.normal(2.0, 1.0,          q)  # cluster 1, slide 1
    mu[1, :, 0] = rng.normal(5.0, np.sqrt(3.0), q)  # cluster 2, slide 1
    mu[0, :, 1] = rng.normal(4.0, np.sqrt(2.0), q)  # cluster 1, slide 2
    mu[1, :, 1] = rng.normal(8.0, 1.0,          q)  # cluster 2, slide 2

    # Step 2: Randomly sample nonzero feature indices for this rep
    nonzero_idx = rng.choice(q, size=N_NZ, replace=False)
    B_true = np.zeros((G, q, S))
    for j in nonzero_idx:
        B_true[:, j, :] = B_TRUE_BLOCK
    beta_star = np.array([float(B_true[:, j, :].mean()) for j in range(q)])

    # Step 3: Generate per-subject data
    X_tepig  = np.zeros((G, q, S, n))
    X_true   = np.zeros((G, q, S, n))  # true population-level weighted averages
    X_clusso = np.zeros((G, q, n))

    for i in range(n):
        # Store tubules per cluster across slides for CLUSSO pooling
        clusso_tubules  = {g: [] for g in range(G)}   # list of tubule arrays
        clusso_n_tubs   = {g: 0  for g in range(G)}   # total tubule count per cluster

        for s in range(S):
            w = rng.choice(weight_choices)
            weights = np.array([w, 1.0 - w])

            # Variable tubule count per subject per slide: K_is ~ Uniform{K_LO, ..., K_HI}
            # Matches CLUSSO paper (sigma_sq_m=5 -> range 37..43)
            K_is = int(rng.integers(K_LO, K_HI + 1))

            for g in range(G):
                # Number of tubules for this cluster in this slide
                n_tub = max(1, round(K_is * weights[g]))

                # Draw tubules: each is latent mean + Gaussian noise
                tubules = rng.normal(mu[g, :, s], SIGMA_R, size=(n_tub, q))

                # TEPIG: weighted cluster average (estimated from noisy tubules)
                X_tepig[g, :, s, i] = weights[g] * tubules.mean(axis=0)

                # Oracle: true population-level weighted average (no tubule noise)
                X_true[g, :, s, i] = weights[g] * mu[g, :, s]

                # Accumulate for CLUSSO mega-slide
                clusso_tubules[g].append(tubules)
                clusso_n_tubs[g] += n_tub

        # CLUSSO: pool tubules from both slides per cluster
        total_tubs = sum(clusso_n_tubs[g] for g in range(G))
        for g in range(G):
            all_tubs_g = np.vstack(clusso_tubules[g])   # (n_tubs_g_total, q)
            overall_w_g = clusso_n_tubs[g] / total_tubs
            X_clusso[g, :, i] = overall_w_g * all_tubs_g.mean(axis=0)

    return X_tepig, X_true, X_clusso, nonzero_idx, B_true, beta_star


# ── proxgrad_fit ───────────────────────────────────────────────────────────────

def proxgrad_fit(X, y, lam, max_iter=2000, tol=1e-6):
    G, q, S, n_tr = X.shape
    d = G * q * S

    gnorms = np.array([
        float(np.sqrt(np.mean(np.sum(X[:, j, :, :] ** 2, axis=(0, 1)))))
        for j in range(q)
    ])
    gnorms = np.where(gnorms > 1e-10, gnorms, 1.0)
    Xs     = np.clip(X / gnorms[np.newaxis, :, np.newaxis, np.newaxis], -1e3, 1e3)
    X_flat = Xs.reshape(d, n_tr).T

    sigma_max = np.linalg.svd(X_flat, compute_uv=False)[0]
    L         = float(sigma_max ** 2) / n_tr
    eta       = 1.0 / L
    threshold = eta * lam

    x         = np.zeros(d)
    z         = np.zeros(d)
    t         = 1.0
    intercept = float(np.mean(y))

    for _ in range(max_iter):
        x_old = x.copy()
        if not np.isfinite(z).all():
            break
        pred_z    = X_flat @ z
        intercept = float(np.mean(y - pred_z))
        residuals = y - intercept - pred_z
        grad      = -(1.0 / n_tr) * (X_flat.T @ residuals)
        if not np.isfinite(grad).all():
            break

        v = (z - eta * grad).reshape(G, q, S)
        for j in range(q):
            block = v[:, j, :]
            norm  = float(np.linalg.norm(block))
            if norm > threshold:
                v[:, j, :] = (1.0 - threshold / norm) * block
            else:
                v[:, j, :] = 0.0
        x = v.reshape(-1)

        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z     = x + ((t - 1.0) / t_new) * (x - x_old)
        t     = t_new

        if np.max(np.abs(x - x_old)) < tol:
            break

    B_std       = x.reshape(G, q, S)
    B           = B_std / gnorms[np.newaxis, :, np.newaxis]
    X_flat_orig = X.reshape(d, n_tr).T
    intercept   = float(np.mean(y - X_flat_orig @ B.reshape(-1)))
    return intercept, B


# ── tepig_fit ──────────────────────────────────────────────────────────────────

def tepig_lowrank_fit(X, y, lam, alpha_init, beta_init, gamma_init):
    G, q, S, n = X.shape
    alpha    = _normalize_vec(np.asarray(alpha_init, dtype=float).copy())
    beta     = np.asarray(beta_init,  dtype=float).copy()
    gamma    = np.asarray(gamma_init, dtype=float).copy()
    nr_alpha = max(0.0, float(np.sum(np.abs(alpha))))

    for _ in range(MAX_ITER):
        alpha0, beta0, gamma0 = alpha.copy(), beta.copy(), gamma.copy()

        Z       = np.einsum('gjsn,g,s->jn', X, alpha, gamma)
        eff_lam = float(lam) * np.sum(np.abs(alpha)) * np.sum(np.abs(gamma))
        beta    = _glmnet_lasso(Z.T, y, eff_lam)
        if np.max(np.abs(beta)) > 0:
            beta = _normalize_vec(beta)

        if np.max(np.abs(beta)) > 0:
            W     = np.einsum('gjsn,j,s->gn', X, beta, gamma)
            W_int = np.column_stack([np.ones(n), W.T])
            coeffs, _, _, _ = np.linalg.lstsq(W_int, y, rcond=None)
            alpha = coeffs[1:]

        nr_alpha = max(0.0, float(np.sum(np.abs(alpha))))
        if np.max(np.abs(alpha)) > 0:
            alpha = _normalize_vec(alpha)

        if np.max(np.abs(alpha)) > 0 and np.max(np.abs(beta)) > 0:
            V     = np.einsum('gjsn,g,j->sn', X, alpha, beta)
            V_int = np.column_stack([np.ones(n), V.T])
            coeffs, _, _, _ = np.linalg.lstsq(V_int, y, rcond=None)
            gamma = coeffs[1:]

        dif = (np.linalg.norm(alpha - alpha0) +
               np.linalg.norm(beta  - beta0)  +
               np.linalg.norm(gamma - gamma0))
        if dif < TOL:
            break

    alpha[np.abs(alpha) < 0.001] = 0.0
    beta[np.abs(beta)   < 0.001] = 0.0
    gamma[np.abs(gamma) < 0.001] = 0.0
    return nr_alpha * alpha, beta, gamma


# ── clusso_fit: Mainfunction_albet on (G, q, n) matrix ────────────────────────

def clusso_select_and_fit(X_mat, y, rng, folds, all_idx):
    """
    Select lambda by 5-fold CV then refit with M_INIT random starts.
    X_mat : (G, q, n)
    """
    n = X_mat.shape[2]
    q = X_mat.shape[1]

    cv_mse = [
        lambda_CV_mse(X_mat, y, np.ones(G) / G, np.ones(q) / q, lam)
        for lam in LAM_GRID
    ]
    best_lam = LAM_GRID[int(np.argmin(cv_mse))]

    best_mse = np.inf
    best_b   = np.zeros(q)
    best_a   = np.ones(G) / G
    for _ in range(M_INIT):
        a0  = rng.dirichlet(np.ones(G))
        b0  = rng.uniform(-1, 1, q)
        res = Mainfunction_albet(X_mat, y, a0, b0, best_lam)
        a_, b_ = res['alpha'], res['bet']
        yp = np.array([a_ @ X_mat[:, :, i] @ b_ for i in range(n)])
        mse = float(np.mean((y - yp) ** 2))
        if mse < best_mse:
            best_mse = mse; best_b = b_; best_a = a_

    y_pred = np.array([best_a @ X_mat[:, :, i] @ best_b for i in range(n)])
    return best_b, y_pred


# ── naive_lasso_fit: standard lasso on (n, q) averaged design matrix ──────────

def naive_lasso_fit(X_flat, y):
    """
    X_flat : (n, q) — X averaged over G and S (one vector per subject).
    Matches CLUSSO paper naive approach: standardize X, center y, LassoCV
    with 100-point glmnet-style lambda path, back-transform coefficients.
    No alpha term. Returns beta (q,), y_pred (n,).
    """
    n = X_flat.shape[0]
    # Standardize X (match glmnet standardize=TRUE)
    X_mu    = X_flat.mean(axis=0)
    X_c     = X_flat - X_mu
    X_scale = np.sqrt((X_c ** 2).mean(axis=0))
    X_scale = np.where(X_scale > 1e-10, X_scale, 1.0)
    X_std   = X_c / X_scale
    # Center y
    y_mu = y.mean(); y_c = y - y_mu
    # 100-point glmnet-style lambda path
    lambda_max = float(np.max(np.abs(X_std.T @ y_c)) / n)
    lambdas_R  = np.exp(np.linspace(np.log(lambda_max),
                                    np.log(lambda_max * 1e-4), 100))
    model = LassoCV(cv=5, alphas=lambdas_R / 2.0,
                    fit_intercept=False, max_iter=100_000)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(X_std, y_c)
    # Back-transform to original scale
    beta      = model.coef_ / X_scale
    intercept = float(y_mu - X_mu @ beta)
    y_pred    = intercept + X_flat @ beta
    return beta, y_pred


# ── Single simulation rep ──────────────────────────────────────────────────────

def run_one_sim(seed):
    rng = np.random.default_rng(seed)
    n   = N
    q   = Q

    # Generate data
    X_tepig, X_true, X_clusso, nonzero_idx, B_true, beta_star = generate_data(rng, n, q)

    # Outcome from estimated tensor model (uses noisy X_tepig, matching CLUSSO simulation design)
    # Oracle still uses X_true — but y is generated from X_tepig so the regression is
    # correctly specified for TEPIG (avoids perfect collinearity through shared weights)
    y_true = np.einsum('gjs,gjsn->n', B_true, X_tepig)
    y      = y_true + rng.normal(0, SIGMA, n)

    results  = {}
    all_idx  = list(range(n))
    folds    = _make_folds(n, rng)

    # ── tepig: fit once, apply multiple post-estimation thresholds ───────────
    d_full = G * q * S
    cv_mse = []
    for lam in LAM_GRID:
        fold_mse = []
        for k in range(5):
            te = folds[k]; tr = [i for i in all_idx if i not in te]
            ic, B  = proxgrad_fit(X_tepig[:, :, :, tr], y[tr], lam)
            ypred  = ic + X_tepig[:, :, :, te].reshape(d_full, len(te)).T @ B.reshape(-1)
            fold_mse.append(float(np.mean((y[te] - ypred) ** 2)))
        cv_mse.append(float(np.mean(fold_mse)))

    best_lam  = LAM_GRID[int(np.argmin(cv_mse))]
    ic_f, B_f = proxgrad_fit(X_tepig, y, best_lam)
    beta_raw  = B_f.mean(axis=(0, 2))          # raw beta, shape (q,)
    y_pred_f  = ic_f + X_tepig.reshape(d_full, n).T @ B_f.reshape(-1)

    # Block Frobenius norms — shape (q,). Used for norm-based selection variants.
    block_norms = np.sqrt(np.sum(B_f ** 2, axis=(0, 2)))

    # Adaptive threshold: sigma_hat * sqrt(2 * log(q) / n)
    sigma_hat = float(np.std(y - y_pred_f, ddof=1))
    tau_adapt = sigma_hat * np.sqrt(2.0 * np.log(q) / n)

    # Mean selection: threshold on |mean(B[:,j,:])|
    for thr, key in [(0.001, 'tepig_raw_001'), (0.01, 'tepig_raw_010'), (0.02, 'tepig_raw_020')]:
        b = beta_raw.copy()
        b[np.abs(b) < thr] = 0.0
        results[key] = compute_metrics(b, y, y_pred_f, beta_star)

    beta_adapt = beta_raw.copy()
    beta_adapt[np.abs(beta_adapt) < tau_adapt] = 0.0
    results['tepig_raw_adapt'] = compute_metrics(beta_adapt, y, y_pred_f, beta_star)

    # Norm selection: threshold on ||B[:,j,:]||_F, report mean as coefficient value
    for thr, key in [(0.001, 'tepig_norm_001'), (0.01, 'tepig_norm_010'), (0.02, 'tepig_norm_020')]:
        b = beta_raw.copy()
        b[block_norms < thr] = 0.0
        results[key] = compute_metrics(b, y, y_pred_f, beta_star)

    beta_norm_adapt = beta_raw.copy()
    beta_norm_adapt[block_norms < tau_adapt] = 0.0
    results['tepig_norm_adapt'] = compute_metrics(beta_norm_adapt, y, y_pred_f, beta_star)

    # ── tepig_lowrank (reference only — not included in summary/plots) ────────
    cv_mse_t = []
    for lam in LAM_GRID:
        fold_mse = []
        for k in range(5):
            te = folds[k]; tr = [i for i in all_idx if i not in te]
            a0 = np.ones(G) / G; b0 = np.ones(q) / q; g0 = np.ones(S) / S
            a, b, g = tepig_lowrank_fit(X_tepig[:, :, :, tr], y[tr], lam, a0, b0, g0)
            yp = np.einsum('gjsn,g,j,s->n', X_tepig[:, :, :, te], a, b, g)
            fold_mse.append(float(np.mean((y[te] - yp) ** 2)))
        cv_mse_t.append(float(np.mean(fold_mse)))

    best_lam_t = LAM_GRID[int(np.argmin(cv_mse_t))]
    best_mse_t = np.inf
    best_res_t = (np.ones(G) / G, np.zeros(q), np.ones(S) / S)
    for _ in range(M_INIT):
        a0 = rng.dirichlet(np.ones(G))
        b0 = rng.uniform(-1, 1, q)
        g0 = rng.dirichlet(np.ones(S))
        a, b, g = tepig_lowrank_fit(X_tepig, y, best_lam_t, a0, b0, g0)
        yp  = np.einsum('gjsn,g,j,s->n', X_tepig, a, b, g)
        mse = float(np.mean((y - yp) ** 2))
        if mse < best_mse_t:
            best_mse_t = mse; best_res_t = (a, b, g)

    a_hat, b_hat, g_hat = best_res_t
    y_pred_t = np.einsum('gjsn,g,j,s->n', X_tepig, a_hat, b_hat, g_hat)
    results['tepig_lowrank'] = compute_metrics(b_hat, y, y_pred_t, beta_star)

    # ── clusso: mega-slide (pool tubules from both slides) ────────────────────
    b_clusso, y_pred_clusso = clusso_select_and_fit(X_clusso, y, rng, folds, all_idx)
    results['clusso'] = compute_metrics(b_clusso, y, y_pred_clusso, beta_star)

    # ── naive: average over G and S → (n, q), fit standard lasso ─────────────
    X_naive_flat = X_tepig.mean(axis=(0, 2)).T   # (n, q)
    b_naive, y_pred_naive = naive_lasso_fit(X_naive_flat, y)
    results['naive'] = compute_metrics(b_naive, y, y_pred_naive, beta_star)

    # ── oracle: group lasso on true population-level X, knowing true nonzero features
    # Uses X_true (no tubule sampling noise) — Full Information oracle analog
    n_nz          = len(nonzero_idx)
    X_oracle_true = X_true[:, nonzero_idx, :, :]   # (G, n_nz, S, n)
    ic_or, B_or   = proxgrad_fit(X_oracle_true, y, lam=1e-6)
    d_or          = G * n_nz * S
    y_pred_oracle = ic_or + X_oracle_true.reshape(d_or, n).T @ B_or.reshape(-1)
    beta_hat_oracle = np.zeros(q)
    for k, j in enumerate(nonzero_idx):
        beta_hat_oracle[j] = float(B_or[:, k, :].mean())
    results['oracle'] = compute_metrics(beta_hat_oracle, y, y_pred_oracle, beta_star)

    return results


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    seed_seq  = np.random.SeedSequence(RANDOM_SEED)
    seed_ints = [int(s.generate_state(1)[0]) for s in seed_seq.spawn(B_SIMS)]

    all_results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(run_one_sim)(s) for s in seed_ints
    )

    estimators  = ['tepig_raw_001', 'tepig_raw_010', 'tepig_raw_020', 'tepig_raw_adapt',
                    'tepig_norm_001', 'tepig_norm_010', 'tepig_norm_020', 'tepig_norm_adapt',
                    'tepig_lowrank', 'clusso', 'naive', 'oracle']
    metric_keys = ['tpr', 'fpr', 'l1', 'mse']

    summary = {est: {m: [] for m in metric_keys} for est in estimators}
    for rep in all_results:
        for est in estimators:
            tpr, fpr, l1, mse = rep[est]
            summary[est]['tpr'].append(tpr)
            summary[est]['fpr'].append(fpr)
            summary[est]['l1'].append(l1)
            summary[est]['mse'].append(mse)

    # Save pickle
    pkl_path = os.path.join(OUT_DATA,
        f'simulation_synthetic_n{N}_q{Q}_s{sparsity_str}_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'summary': summary,
            'config': {
                'B': B_SIMS, 'N': N, 'Q': Q, 'G': G, 'S': S,
                'N_NZ': N_NZ, 'SPARSITY': SPARSITY, 'K': K,
                'B_TRUE_BLOCK': B_TRUE_BLOCK.tolist(),
                'LAM_GRID': LAM_GRID.tolist(),
                'SIGMA': SIGMA, 'SIGMA_R': SIGMA_R,
            }
        }, f)

    # Save summary text
    txt_path = os.path.join(OUT_SUMM,
        f'simulation_synthetic_n{N}_q{Q}_s{sparsity_str}_summary.txt')
    with open(txt_path, 'w') as f:
        f.write("=" * 65 + "\n")
        f.write("TEPIG SYNTHETIC SIMULATION SUMMARY\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"B={B_SIMS} reps | n={N} | q={Q} | G={G} | S={S} | K={K}\n")
        f.write(f"n_nonzero={N_NZ} | sparsity={SPARSITY} | sigma={SIGMA} | sigma_R={SIGMA_R}\n")
        f.write(f"B_TRUE_BLOCK: {B_TRUE_BLOCK.tolist()}\n\n")
        f.write(f"{'Estimator':<18} {'TPR':>8} {'FPR':>8} {'L1 bias':>10} {'MSE':>10}\n")
        f.write("-" * 58 + "\n")
        for est in estimators:
            tpr = float(np.nanmean(summary[est]['tpr']))
            fpr = float(np.nanmean(summary[est]['fpr']))
            l1  = float(np.nanmean(summary[est]['l1']))
            mse = float(np.nanmean(summary[est]['mse']))
            f.write(f"  {est:<16} {tpr:>8.3f} {fpr:>8.3f} {l1:>10.3f} {mse:>10.3f}\n")

    print(f"\nSaved: {pkl_path}")
    print(f"Saved: {txt_path}")
    print(f"\n{'Estimator':<18} {'TPR':>8} {'FPR':>8} {'L1 bias':>10} {'MSE':>10}")
    print("-" * 58)
    for est in estimators:
        tpr = float(np.nanmean(summary[est]['tpr']))
        fpr = float(np.nanmean(summary[est]['fpr']))
        l1  = float(np.nanmean(summary[est]['l1']))
        mse = float(np.nanmean(summary[est]['mse']))
        print(f"  {est:<16} {tpr:>8.3f} {fpr:>8.3f} {l1:>10.3f} {mse:>10.3f}")

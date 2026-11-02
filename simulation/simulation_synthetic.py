"""
simulation_synthetic.py
-----------------------
Full simulation study with synthetic data (CLUSSO Section 4.1 style, extended to
S = 2 sections and a full-rank coefficient block). One call = one (n, q, sparsity)
setting x B replicates, run in parallel with joblib.

Usage:
    python simulation_synthetic.py --n 300 --q 10 --sparsity 0.8 [--reps 200]
                                   [--folder rerun_2026_11] [--clusso-grid extended]

Data generation (per replicate, per subject i; labels known):
  1. A random set of round(q (1 - sparsity)) active features; every active block is
     B_TRUE_BLOCK = [[1, 2], [3, 4]] (rank 2), inactive blocks are zero.
  2. Latent cluster means per subject and feature:
       section 1: cluster 1 ~ N(2, var 1), cluster 2 ~ N(5, var 3)
       section 2: cluster 1 ~ N(4, var 2), cluster 2 ~ N(8, var 1)
  3. Per section: weight w ~ Uniform{0.2,...,0.8}, K ~ Uniform{37..43} objects,
     max(1, round(K w_g)) objects per cluster ~ N(mu, SIGMA_R^2 I).
  4. Designs: TEPIG = proportion-weighted means of the noisy objects (G, q, S);
     oracle = true w * mu; CLUSSO = objects of both sections pooled per cluster
     (G, q); naive = TEPIG design averaged over g and s (q,).
  5. y_i = <B_true, X_true_i> + eps_i, eps ~ N(0, SIGMA^2).

Estimators (all tuned by the SAME seeded 5-fold split of the n subjects):
    tepig   : core/tepig.py (FISTA group lasso), lambda by CV on a data-driven
              geometric grid (lambda_max -> LAM_RATIO * lambda_max, N_LAM points);
              selection reported under EVERY threshold rule (see core/tepig.py):
                tepig_raw              nonzero blocks at the CV lambda
                tepig_col_std[_gs]     ||sd o B||_F   > sigma_hat sqrt((GS? + 2 log q)/n)
                tepig_block_scaled[_gs]gamma ||B||_F  > ...
                tepig_unscaled         ||B||_F (original units) > ...   [pre-Nov-2026 'norm' rule]
                tepig_mean_unscaled    |mean_gs B|  (original units) > ... [pre-Nov-2026 'raw_adapt' rule]
                tepig_col_std_refit[_gs]  OLS refit on the CV support, standardised columns,
                                       block norm > ... (debiased; the calibration the
                                       universal threshold assumes)
              each also at the one-standard-error lambda (tepig1se_<rule>);
              'tepig' is an alias of the default rule (DEFAULT_RULE below).
    clusso  : Mainfunction_albet on the pooled (G, q, n) design, lambda by CV
              (lambda_CV_mse with the shared folds), M_INIT random restarts.
              Grid: --clusso-grid paper = {0.5, ..., 5} U {0.25, ...} (15 points, the
              pre-Nov-2026 grid) or extended (17 points, 0.1 .. 20, contains it).
    naive   : LassoCV (100-point glmnet path) on the averaged design, shared folds.
    oracle  : group lasso (lam = 1e-6) on the TRUE tensor restricted to the true support.

Metrics per replicate (stored individually; summarize_synthetic.py takes medians
and means): TPR, FPR, sign-aligned L1 bias of the L1-normalised marginal
coefficient (mean over g, s of B), TRAINING MSE and TEST MSE on N_TEST fresh
subjects from the same DGP. Every TEPIG block statistic, sigma_hat, the chosen
lambda index, iteration counts and KKT residuals are stored too.
"""

import os
import sys
import argparse
import pickle
import time
import warnings
import numpy as np
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')

from sklearn.linear_model import LassoCV
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tepig.clusso.Mainfunction_albet import Mainfunction_albet
from tepig.clusso.SLasso_MSE import lambda_CV_mse
from tepig import tepig as T

_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')

# ── Parse args ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--n',        type=int,   required=True, help='Number of subjects')
parser.add_argument('--q',        type=int,   default=10,    help='Number of features')
parser.add_argument('--sparsity', type=float, default=0.8,   help='Proportion of zero features')
parser.add_argument('--reps',     type=int,   default=200)
parser.add_argument('--n-jobs',   type=int,   default=-1)
parser.add_argument('--folder',   default='rerun_2026_11',
                    help='outputs/data/<folder>/ for the per-setting pkl')
parser.add_argument('--clusso-grid', default='extended', choices=['paper', 'extended'])
parser.add_argument('--n-test',   type=int,   default=1000, help='fresh subjects for test MSE')
args = parser.parse_args()

N        = args.n
Q        = args.q
SPARSITY = args.sparsity
N_NZ     = max(1, round(Q * (1.0 - SPARSITY)))  # number of nonzero features
OUT_DATA = os.path.join(_BASE, 'data', args.folder)
OUT_SUMM = os.path.join(_BASE, 'results', args.folder)
os.makedirs(OUT_DATA, exist_ok=True)
os.makedirs(OUT_SUMM, exist_ok=True)

# ── Config (fixed across all settings) ────────────────────────────────────────
G          = 2      # number of clusters
S          = 2      # number of sections
K          = 40     # mean objects per section per subject (CLUSSO paper: mu_M=40)
SIGMA_SQ_M = 5      # variance of object count (CLUSSO paper: sigma_sq_m=5)
SIGMA      = 1.0    # outcome noise sd
SIGMA_R    = 1.0    # object measurement noise sd (CLUSSO paper: sigma_R=1)
_H    = int(np.sqrt(12 * SIGMA_SQ_M + 1))
if _H % 2 == 0:
    _H -= 1
K_LO  = K - (_H - 1) // 2   # 37
K_HI  = K + (_H - 1) // 2   # 43
B_SIMS  = args.reps
N_JOBS  = args.n_jobs
RANDOM_SEED = 42
N_TEST = args.n_test

M_INIT   = 3                # CLUSSO random restarts for the final fit
N_LAM    = 20               # TEPIG grid points
LAM_RATIO = 1e-5            # TEPIG grid: lambda_max -> LAM_RATIO * lambda_max (CV optimum
                            # sits at 1e-4..2e-3 lambda_max in this DGP; 1e-3 was binding)
TOL, MAX_ITER = 1e-6, 10000
DEFAULT_RULE = ('col_std_refit', True, 'min')   # (statistic, G*S term, lambda choice) -> 'tepig'

LAM_GRID_CLUSSO_PAPER = np.array([0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75,
                                  2.00, 2.25, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00])
LAM_GRID_CLUSSO_EXT = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5,
                                3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0])
LAM_GRID_CLUSSO = (LAM_GRID_CLUSSO_PAPER if args.clusso_grid == 'paper'
                   else LAM_GRID_CLUSSO_EXT)

B_TRUE_BLOCK = np.array([[1.0, 2.0],
                         [3.0, 4.0]])

sparsity_str = f"{int(SPARSITY * 10):02d}"   # "08" for 0.8, "04" for 0.4
RULES = [('raw', False), ('col_std', False), ('col_std', True),
         ('block_scaled', False), ('block_scaled', True),
         ('unscaled', False), ('mean_unscaled', False),
         ('col_std_refit', False), ('col_std_refit', True)]
LAM_CHOICES = ['min', '1se']           # CV-minimum lambda and the one-standard-error lambda


def rule_key(rule, gs, which='min'):
    return ('tepig_' if which == 'min' else 'tepig1se_') + rule + ('_gs' if gs else '')


ESTIMATORS = (['tepig'] + [rule_key(r, g, w) for w in LAM_CHOICES for r, g in RULES]
              + ['clusso', 'naive', 'oracle'])
METRIC_KEYS = ['tpr', 'fpr', 'l1', 'mse', 'mse_test']

if __name__ == '__main__':
    print(f"simulation_synthetic.py | n={N}, q={Q}, sparsity={SPARSITY}, n_nonzero={N_NZ}, "
          f"reps={B_SIMS}")
    print(f"  G={G}, S={S}, K={K} objects/section (discrete uniform {{{K_LO}..{K_HI}}})")
    print(f"  TEPIG grid: {N_LAM} pts lambda_max -> {LAM_RATIO} lambda_max; "
          f"CLUSSO grid ({args.clusso_grid}): {LAM_GRID_CLUSSO[0]} .. {LAM_GRID_CLUSSO[-1]} "
          f"({len(LAM_GRID_CLUSSO)} pts); naive: LassoCV 100-pt path; all share seeded folds")


# ── Helpers ────────────────────────────────────────────────────────────────────
def compute_metrics(selected, beta_hat, y, y_pred, y_test, y_pred_test, truth_mask, beta_star):
    """TPR/FPR from the selected mask; sign-aligned L1 bias of the L1-normalised
    marginal coefficient restricted to the selected set; train and test MSE."""
    z = ~truth_mask
    tpr = float(selected[truth_mask].mean()) if truth_mask.sum() > 0 else float('nan')
    fpr = float(selected[z].mean()) if z.sum() > 0 else float('nan')
    b = np.where(selected, beta_hat, 0.0)
    bsn = beta_star / np.sum(np.abs(beta_star))
    if np.sum(np.abs(b)) > 0:
        bhn = b / np.sum(np.abs(b))
        if float(bhn @ bsn) < 0:                     # sign-aligned
            bhn = -bhn
    else:
        bhn = b
    l1 = float(np.sum(np.abs(bhn - bsn)))
    mse = float(np.mean((y - y_pred) ** 2))
    mse_test = float(np.mean((y_test - y_pred_test) ** 2))
    return dict(tpr=tpr, fpr=fpr, l1=l1, mse=mse, mse_test=mse_test)


# ── Data generation with individual objects ───────────────────────────────────
def generate_data(rng, n, q, nonzero_idx=None):
    """
    Returns X_tepig (G,q,S,n), X_true (G,q,S,n), X_clusso (G,q,n), nonzero_idx,
    B_true (G,q,S), beta_star (q,) = mean over (g,s) of B_true.
    True cluster means mu_i are drawn INDEPENDENTLY per subject (CLUSSO Sec 4.1).
    """
    weight_choices = np.arange(0.2, 0.9, 0.1)
    if nonzero_idx is None:
        nonzero_idx = rng.choice(q, size=N_NZ, replace=False)
    B_true = np.zeros((G, q, S))
    for j in nonzero_idx:
        B_true[:, j, :] = B_TRUE_BLOCK
    beta_star = B_true.mean(axis=(0, 2))

    X_tepig  = np.zeros((G, q, S, n))
    X_true   = np.zeros((G, q, S, n))
    X_clusso = np.zeros((G, q, n))
    for i in range(n):
        mu_i = np.zeros((G, q, S))
        mu_i[0, :, 0] = rng.normal(2.0, 1.0,          q)
        mu_i[1, :, 0] = rng.normal(5.0, np.sqrt(3.0), q)
        mu_i[0, :, 1] = rng.normal(4.0, np.sqrt(2.0), q)
        mu_i[1, :, 1] = rng.normal(8.0, 1.0,          q)
        clusso_objs = {g: [] for g in range(G)}
        clusso_cnt = {g: 0 for g in range(G)}
        for s in range(S):
            w = rng.choice(weight_choices)
            weights = np.array([w, 1.0 - w])
            K_is = int(rng.integers(K_LO, K_HI + 1))
            for g in range(G):
                n_obj = max(1, round(K_is * weights[g]))
                objs = rng.normal(mu_i[g, :, s], SIGMA_R, size=(n_obj, q))
                X_tepig[g, :, s, i] = weights[g] * objs.mean(axis=0)
                X_true[g, :, s, i] = weights[g] * mu_i[g, :, s]
                clusso_objs[g].append(objs)
                clusso_cnt[g] += n_obj
        total = sum(clusso_cnt.values())
        for g in range(G):
            allg = np.vstack(clusso_objs[g])
            X_clusso[g, :, i] = (clusso_cnt[g] / total) * allg.mean(axis=0)
    return X_tepig, X_true, X_clusso, nonzero_idx, B_true, beta_star


# ── CLUSSO ─────────────────────────────────────────────────────────────────────
def clusso_select_and_fit(X_mat, y, rng, folds):
    """lambda by 5-fold CV (shared seeded folds, deterministic init as in the R
    code) then refit with M_INIT random starts, keeping the best training MSE."""
    n = X_mat.shape[2]
    q = X_mat.shape[1]
    cv_mse = [lambda_CV_mse(X_mat, y, np.ones(G) / G, np.ones(q) / q, lam, folds=folds)
              for lam in LAM_GRID_CLUSSO]
    best_i = int(np.argmin(cv_mse))
    best_lam = LAM_GRID_CLUSSO[best_i]
    best_mse, best_b, best_a = np.inf, np.zeros(q), np.ones(G) / G
    for _ in range(M_INIT):
        a0 = rng.dirichlet(np.ones(G))
        b0 = rng.uniform(-1, 1, q)
        res = Mainfunction_albet(X_mat, y, a0, b0, best_lam)
        a_, b_ = res['alpha'], res['bet']
        yp_raw = np.array([a_ @ X_mat[:, :, i] @ b_ for i in range(n)])
        yp = yp_raw + (np.mean(y) - np.mean(yp_raw))
        mse = float(np.mean((y - yp) ** 2))
        if mse < best_mse:
            best_mse = mse; best_b = b_; best_a = a_
    y_pred_raw = np.array([best_a @ X_mat[:, :, i] @ best_b for i in range(n)])
    ic = float(np.mean(y) - np.mean(y_pred_raw))
    return best_a, best_b, ic, y_pred_raw + ic, best_i


# ── naive ──────────────────────────────────────────────────────────────────────
def naive_lasso_fit(X_flat, y, cv):
    """Standardize X, center y, LassoCV with a 100-point glmnet-style path on the
    shared folds (cv = list of (train_pos, test_pos)); back-transform."""
    n = X_flat.shape[0]
    X_mu = X_flat.mean(axis=0)
    X_c = X_flat - X_mu
    X_scale = np.sqrt((X_c ** 2).mean(axis=0))
    X_scale = np.where(X_scale > 1e-10, X_scale, 1.0)
    X_std = X_c / X_scale
    y_mu = y.mean(); y_c = y - y_mu
    lambda_max = float(np.max(np.abs(X_std.T @ y_c)) / n)
    lambdas_R = np.exp(np.linspace(np.log(lambda_max), np.log(lambda_max * 1e-4), 100))
    model = LassoCV(cv=cv, alphas=lambdas_R / 2.0, fit_intercept=False, max_iter=100_000)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(X_std, y_c)
    beta = model.coef_ / X_scale
    intercept = float(y_mu - X_mu @ beta)
    return beta, intercept


# ── Single simulation rep ──────────────────────────────────────────────────────
def run_one_sim(seed):
    rng = np.random.default_rng(seed)
    n, q = N, Q
    X_tepig, X_true, X_clusso, nonzero_idx, B_true, beta_star = generate_data(rng, n, q)
    y = np.einsum('gjs,gjsn->n', B_true, X_true) + rng.normal(0, SIGMA, n)
    # fresh test subjects from the same DGP (same true support)
    Xt_te, Xtrue_te, Xcl_te, _, _, _ = generate_data(rng, N_TEST, q, nonzero_idx)
    y_te = np.einsum('gjs,gjsn->n', B_true, Xtrue_te) + rng.normal(0, SIGMA, N_TEST)
    truth = np.zeros(q, bool); truth[nonzero_idx] = True

    folds = T.plain_folds(np.arange(n), int(rng.integers(0, 2**31 - 1)))
    cv_pos = T.folds_to_positions(folds, np.arange(n))
    results, extra = {}, {}

    # ── TEPIG ─────────────────────────────────────────────────────────────────
    res = T.cv_select(X_tepig, y, folds, num=N_LAM, ratio=LAM_RATIO,
                      max_iter=MAX_ITER, tol=TOL)
    extra['tepig'] = dict(lam_idx=res['best'], lam=res['best_lam'], n_lam=len(res['lams']),
                          lam_idx_1se=res['best_1se'], cv_curve=res['cv_curve'].astype(np.float32),
                          cv_se=res['cv_se'].astype(np.float32),
                          path_iters=[ff.n_iter for ff in res['path']],
                          at_lower_bound=res['at_lower_bound'],
                          at_upper_bound=res['at_upper_bound'], truth=nonzero_idx)
    for which in LAM_CHOICES:
        f = res['fit'] if which == 'min' else res['fit_1se']
        y_pred = f.predict(X_tepig)
        y_pred_te = f.predict(Xt_te)
        sigma_hat = float(np.std(y - y_pred, ddof=1))
        stats = f.block_stats(); stats.update(T.refit_stats(f, X_tepig, y))
        beta_raw = f.B.mean(axis=(0, 2))
        for rule, gs in RULES:
            sel = T.select_features(stats, sigma_hat, q, n, rule, G * S if gs else 0)
            results[rule_key(rule, gs, which)] = compute_metrics(
                sel, beta_raw, y, y_pred, y_te, y_pred_te, truth, beta_star)
        extra['tepig'][which] = dict(
            stats={k: (v.astype(np.float32) if isinstance(v, np.ndarray) else v)
                   for k, v in stats.items()},
            sigma_hat=sigma_hat, beta_raw=beta_raw.astype(np.float32),
            n_iter=f.n_iter, converged=f.converged, kkt=f.kkt)
    results['tepig'] = results[rule_key(*DEFAULT_RULE)]
    extra['tepig'].update(n_iter=extra['tepig']['min']['n_iter'],
                          converged=extra['tepig']['min']['converged'],
                          kkt=extra['tepig']['min']['kkt'])

    # ── CLUSSO ────────────────────────────────────────────────────────────────
    a_cl, b_cl, ic_cl, y_pred_cl, li = clusso_select_and_fit(X_clusso, y, rng, folds)
    y_pred_cl_te = np.array([a_cl @ Xcl_te[:, :, i] @ b_cl for i in range(N_TEST)]) + ic_cl
    results['clusso'] = compute_metrics(np.abs(b_cl) > 1e-6, b_cl, y, y_pred_cl,
                                        y_te, y_pred_cl_te, truth, beta_star)
    extra['clusso'] = dict(lam_idx=li, n_lam=len(LAM_GRID_CLUSSO), alpha=a_cl)

    # ── naive ─────────────────────────────────────────────────────────────────
    Xn = X_tepig.mean(axis=(0, 2)).T
    b_nv, ic_nv = naive_lasso_fit(Xn, y, cv_pos)
    results['naive'] = compute_metrics(np.abs(b_nv) > 1e-6, b_nv, y, ic_nv + Xn @ b_nv,
                                       y_te, ic_nv + Xt_te.mean(axis=(0, 2)).T @ b_nv,
                                       truth, beta_star)

    # ── oracle: true tensor, true support, negligible penalty ─────────────────
    Xo = X_true[:, nonzero_idx, :, :]
    fo = T.fit_one(Xo, y, 1e-6, max_iter=MAX_ITER, tol=TOL)
    beta_or = np.zeros(q)
    beta_or[nonzero_idx] = fo.B.mean(axis=(0, 2))
    results['oracle'] = compute_metrics(truth.copy(), beta_or, y, fo.predict(Xo),
                                        y_te, fo.predict(Xtrue_te[:, nonzero_idx, :, :]),
                                        truth, beta_star)
    return results, extra


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    t0 = time.time()
    seed_seq = np.random.SeedSequence(RANDOM_SEED)
    seed_ints = [int(s.generate_state(1)[0]) for s in seed_seq.spawn(B_SIMS)]
    all_out = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(run_one_sim)(s) for s in seed_ints)

    summary = {est: {m: [] for m in METRIC_KEYS} for est in ESTIMATORS}
    for rep, _ in all_out:
        for est in ESTIMATORS:
            for m in METRIC_KEYS:
                summary[est][m].append(rep[est][m])
    extras = [e for _, e in all_out]

    pkl_path = os.path.join(OUT_DATA, f'simulation_synthetic_n{N}_q{Q}_s{sparsity_str}_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'summary': summary, 'extras': extras, 'seeds': seed_ints,
            'config': {
                'B': B_SIMS, 'N': N, 'Q': Q, 'G': G, 'S': S, 'N_NZ': N_NZ,
                'SPARSITY': SPARSITY, 'K': K, 'K_range': [K_LO, K_HI], 'N_TEST': N_TEST,
                'B_TRUE_BLOCK': B_TRUE_BLOCK.tolist(), 'SIGMA': SIGMA, 'SIGMA_R': SIGMA_R,
                'TEPIG_GRID': {'num': N_LAM, 'ratio': LAM_RATIO}, 'TOL': TOL,
                'MAX_ITER': MAX_ITER, 'DEFAULT_RULE': DEFAULT_RULE,
                'CLUSSO_GRID': LAM_GRID_CLUSSO.tolist(), 'CLUSSO_M_INIT': M_INIT,
                'RANDOM_SEED': RANDOM_SEED, 'run': T.run_metadata(),
            }
        }, f)

    lb = sum(e['tepig']['at_lower_bound'] for e in extras)
    ub = sum(e['tepig']['at_upper_bound'] for e in extras)
    cl = np.bincount([e['clusso']['lam_idx'] for e in extras], minlength=len(LAM_GRID_CLUSSO))
    it = [e['tepig']['n_iter'] for e in extras]; kk = [e['tepig']['kkt'] for e in extras]
    lines = [f"B={B_SIMS} reps | n={N} | q={Q} | G={G} | S={S} | K={K} | "
             f"n_nonzero={N_NZ} | sparsity={SPARSITY} | sigma={SIGMA} | sigma_R={SIGMA_R}",
             f"TEPIG lambda at grid lower bound {lb}/{B_SIMS}, upper bound {ub}/{B_SIMS}; "
             f"final-fit iterations median {np.median(it):.0f} max {max(it)}; max KKT {max(kk):.1e}",
             f"CLUSSO lambda index counts (grid {LAM_GRID_CLUSSO[0]}..{LAM_GRID_CLUSSO[-1]}): {cl.tolist()}",
             f"wall {time.time() - t0:.0f}s", "",
             f"{'Estimator':<22} {'TPR':>8} {'FPR':>8} {'L1 bias':>10} {'MSE':>10} {'test MSE':>10}   (medians | means)"]
    for est in ESTIMATORS:
        med = [float(np.nanmedian(summary[est][m])) for m in METRIC_KEYS]
        mean = [float(np.nanmean(summary[est][m])) for m in METRIC_KEYS]
        lines.append(f"  {est:<20} " + ' '.join(f'{v:>8.3f}' if i < 3 else f'{v:>10.3f}' for i, v in enumerate(med))
                     + '   | ' + ' '.join(f'{v:.3f}' for v in mean))
    txt_path = os.path.join(OUT_SUMM, f'simulation_synthetic_n{N}_q{Q}_s{sparsity_str}_summary.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f"\nSaved: {pkl_path}\nSaved: {txt_path}")

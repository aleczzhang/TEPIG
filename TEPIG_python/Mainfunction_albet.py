# Date: 7/20/23
# Main function for fitting structured lasso model

import warnings
import numpy as np
from sklearn.linear_model import Lasso

from K_prdu import K_prdu
from mat_vec_prd import mat_vec_prd


def _glmnet_lasso(X, y, lam):
    """
    Lasso fit equivalent to R's glmnet(X, y, alpha=1, lambda=lam).

    Matches R's glmnet defaults:
      intercept=TRUE  : centers y and each column of X before fitting
      standardize=TRUE: scales X columns by population std  sqrt(mean((x-mean)^2))

    Objective correction
    --------------------
    R glmnet  minimises  (1/n)   * ||y - Xb||^2  +  lam * ||b||_1
    sklearn   minimises  (1/2n)  * ||y - Xb||^2  +  alpha * ||b||_1
    Match via:  alpha_sklearn = lam / 2

    Returns coefficients on the ORIGINAL (un-standardized) scale,
    matching R's fit$beta[, 1].
    """
    n = X.shape[0]

    # Center y  (glmnet intercept=TRUE)
    y_c = y - y.mean()

    # Center and scale X  (glmnet standardize=TRUE, population std)
    X_mu    = X.mean(axis=0)
    X_c     = X - X_mu
    X_scale = np.sqrt((X_c ** 2).mean(axis=0))          # sqrt(sum((x-mu)^2) / n)
    X_scale = np.where(X_scale > 1e-10, X_scale, 1.0)  # guard zero-variance columns
    X_std   = X_c / X_scale

    # Fit on standardized data with corrected alpha
    sk_alpha = max(float(lam) / 2.0, 1e-10)
    model = Lasso(alpha=sk_alpha, fit_intercept=False, max_iter=100_000)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(X_std, y_c)

    # Back-transform to original scale  (matches R's fit$beta[, 1])
    return model.coef_ / X_scale


def Mainfunction_albet(X_tr, Y_tr, alpha, bet, lam):
    """
    Estimate alpha and beta by alternating structured lasso.

    Iteratively:
      1. Fix alpha, solve for beta via lasso on  (alpha' X_i)
      2. Fix beta,  solve for alpha via OLS on   (X_i beta)
    until convergence (||beta x alpha - beta0 x alpha0|| < 0.01) or 50 iterations.

    Parameters
    ----------
    X_tr  : (p, q, n) array  -  X_tr[:,:,k] is the k-th subject's matrix
    Y_tr  : (n,)  array
    alpha : (p,)  non-zero initial vector
    bet   : (q,)  non-zero initial vector
    lam   : fixed regularisation parameter (R glmnet scale)

    Returns
    -------
    dict {'alpha': (p,), 'bet': (q,)}
    """
    p, q, n = X_tr.shape
    Y_tr  = np.asarray(Y_tr,  dtype=float).flatten()
    alpha = np.asarray(alpha, dtype=float).flatten()
    bet   = np.asarray(bet,   dtype=float).flatten()

    # L1-normalise alpha with sign convention (largest absolute value is positive)
    if np.max(np.abs(alpha)) > 0:
        sig_al = np.sign(alpha[np.argmax(np.abs(alpha))])
        alpha  = sig_al * alpha / np.sum(np.abs(alpha))

    dif      = 100.0
    nrun     = 0
    nr_alpha = 0.0

    while (dif > 0.01 and nrun < 50
           and np.max(np.abs(alpha)) > 0
           and np.max(np.abs(bet))   > 0):

        nrun  += 1
        bet0   = bet.copy()
        alpha0 = alpha.copy()

        # ── Step 1: fix alpha, update beta via lasso ──────────────────────────
        # Equivalent to R: glmnet(X_alpha, Y_tr, lambda = lam * sum(abs(alpha)))
        # After L1-normalisation sum(abs(alpha)) == 1, so effective lambda == lam
        X_alpha   = mat_vec_prd(X_tr, alpha, 'vec_mat')          # (n, q)
        lasso_lam = float(lam) * np.sum(np.abs(alpha))
        bet       = _glmnet_lasso(X_alpha, Y_tr, lasso_lam)      # (q,)

        if np.max(np.abs(bet)) > 0:
            sig_bet = np.sign(bet[np.argmax(np.abs(bet))])
            bet     = sig_bet * bet / np.sum(np.abs(bet))

        # ── Step 2: fix beta, update alpha via OLS ────────────────────────────
        # Equivalent to R: lm(Y_tr ~ X_bet)  (includes intercept; we drop it)
        if np.max(np.abs(bet)) > 0:
            X_bet      = mat_vec_prd(X_tr, bet, 'mat_vec')       # (n, p)
            X_with_int = np.column_stack([np.ones(n), X_bet])
            coeffs, _, _, _ = np.linalg.lstsq(X_with_int, Y_tr, rcond=None)
            alpha = coeffs[1:]                                    # drop intercept

        nr_alpha = max(0.0, float(np.sum(np.abs(alpha))))

        if np.max(np.abs(alpha)) > 0:
            sig_al = np.sign(alpha[np.argmax(np.abs(alpha))])
            alpha  = sig_al * alpha / np.sum(np.abs(alpha))

        dif = float(np.linalg.norm(K_prdu(bet, alpha) - K_prdu(bet0, alpha0)))

    # Zero out very small coefficients (after L1-normalisation)
    alpha[np.abs(alpha) < 0.001] = 0.0
    bet[np.abs(bet)     < 0.001] = 0.0
    alpha = nr_alpha * alpha      # restore original scale

    return {'alpha': alpha, 'bet': bet}

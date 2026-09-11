# Author: Jeremy Rubin
# Date: 7/20/23
# Functions to compute MSE for structured lasso across a grid of regularisation
# parameters with cross-validation

import numpy as np
from Mainfunction_albet import Mainfunction_albet


def slasso_mse(X_train, Y_train, X_test, Y_test, alpha_init, beta_init, lam):
    """
    Fit structured lasso on training data; return MSE on test data.

    X_train, X_test : (p, q, n_train / n_test) arrays
    Y_train, Y_test : (n,) arrays
    """
    results   = Mainfunction_albet(X_train, Y_train, alpha_init, beta_init, lam)
    beta_hat  = results['bet']
    alpha_hat = results['alpha']

    Y_test  = np.asarray(Y_test, dtype=float).flatten()
    Y_test  = Y_test - Y_test.mean()          # mean-centre (scale=FALSE in R)

    X_mean  = X_test.mean(axis=2)             # (p, q)  mean over subjects
    n_test  = X_test.shape[2]

    SSE = 0.0
    for i in range(n_test):
        Xi  = X_test[:, :, i] - X_mean
        SSE += (Y_test[i] - alpha_hat @ Xi @ beta_hat) ** 2

    return SSE / n_test


def CV_make_folds(n_train):
    """
    Partition n_train indices into 5 non-overlapping folds.
    Returns a list of 5 lists of integer indices (mirrors R's setdiff logic).
    """
    size      = n_train // 5
    available = list(range(n_train))
    folds     = []

    for _ in range(4):
        fold      = sorted(np.random.choice(available, size=size, replace=False).tolist())
        folds.append(fold)
        available = sorted(set(available) - set(fold))

    folds.append(available)   # fold 5 gets all remaining indices
    return folds


def lambda_CV_mse(X_train, Y_train, alpha_init, beta_init, lam):
    """
    5-fold cross-validated MSE for a given regularisation value lam.
    """
    folds       = CV_make_folds(X_train.shape[2])
    all_indices = list(range(X_train.shape[2]))
    mse_vec     = []

    for k in range(5):
        test_idx  = folds[k]
        train_idx = sorted(set(all_indices) - set(test_idx))
        mse       = slasso_mse(
            X_train[:, :, train_idx], Y_train[train_idx],
            X_train[:, :, test_idx],  Y_train[test_idx],
            alpha_init, beta_init, lam
        )
        mse_vec.append(mse)

    return float(np.mean(mse_vec))

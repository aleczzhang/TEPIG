# Date: 7/20/23
# Helper function for doing structured lasso fit

import numpy as np


def mat_vec_prd(X_tr, alph, order):
    """
    Matrix-vector products for structured lasso.

    X_tr  : (p, q, n) array
    alph  : 1-D coefficient vector
    order : 'vec_mat'  ->  alpha' @ X[:,:,i]  for each i  ->  returns (n, q)
            'mat_vec'  ->  X[:,:,i] @ beta    for each i  ->  returns (n, p)
    """
    p, q, n = X_tr.shape
    alph = np.asarray(alph).flatten()

    if order == 'vec_mat':
        return np.array([alph @ X_tr[:, :, i] for i in range(n)])   # (n, q)
    elif order == 'mat_vec':
        return np.array([X_tr[:, :, i] @ alph for i in range(n)])   # (n, p)
    else:
        raise ValueError("order must be 'vec_mat' or 'mat_vec'")

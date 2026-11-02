# Date: 7/20/23
# Another helper function for doing structured lasso fit

import numpy as np


def K_prdu(A, B):
    """Kronecker product of two vectors A and B, returned as a 1-D array."""
    return np.kron(np.asarray(A).flatten(), np.asarray(B).flatten())

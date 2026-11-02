"""Assertion tests for core/tepig.py (run from the repo root: python tests/test_tepig_core.py, or pytest).

1. The Gram-form FISTA reaches the same minimizer as the pre-Nov-2026 direct-form
   solver (run_gene.proxgrad_fit) when both use uncentered block scaling.
2. KKT residual of the returned solution is tiny.
3. lambda_max zeroes every block; slightly below it, at least one block is active.
4. The three unit-invariant selection statistics ARE unit-invariant (rescaling a
   feature's columns leaves them unchanged), the unscaled one is not.
5. group_folds never splits a group and covers the training set exactly once.
6. CV with warm starts selects an interior lambda and the fitted B predicts.
"""
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
from tepig import tepig as T                                               # noqa: E402


def _old_proxgrad_fit(X, y, lam, max_iter=20000, tol=1e-9):
    """Verbatim pre-Nov-2026 solver (run_gene.py / simulation_synthetic.py)."""
    G, q, S, n_tr = X.shape
    d = G * q * S
    gnorms = np.array([float(np.sqrt(np.mean(np.sum(X[:, j, :, :] ** 2, axis=(0, 1)))))
                       for j in range(q)])
    gnorms = np.where(gnorms > 1e-10, gnorms, 1.0)
    Xs = X / gnorms[np.newaxis, :, np.newaxis, np.newaxis]
    X_flat = Xs.reshape(d, n_tr).T
    sigma_max = np.linalg.svd(X_flat, compute_uv=False)[0]
    eta = 1.0 / (float(sigma_max ** 2) / n_tr)
    threshold = eta * lam
    x, z, t = np.zeros(d), np.zeros(d), 1.0
    for _ in range(max_iter):
        x_old = x.copy()
        pred_z = X_flat @ z
        ic = float(np.mean(y - pred_z))
        grad = -(1.0 / n_tr) * (X_flat.T @ (y - ic - pred_z))
        v = (z - eta * grad).reshape(G, q, S)
        for j in range(q):
            blk = v[:, j, :]; nrm = float(np.linalg.norm(blk))
            v[:, j, :] = (1.0 - threshold / nrm) * blk if nrm > threshold else 0.0
        x = v.reshape(-1)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z = x + ((t - 1.0) / t_new) * (x - x_old); t = t_new
        if np.max(np.abs(x - x_old)) < tol:
            break
    B = x.reshape(G, q, S) / gnorms[np.newaxis, :, np.newaxis]
    ic = float(np.mean(y - X.reshape(d, n_tr).T @ B.reshape(-1)))
    return ic, B


def make_data(seed=0, n=200, q=12, G=2, S=3, s_true=4, snr=1.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(G, q, S, n)) * rng.uniform(0.5, 3, size=(1, q, 1, 1)) + \
        rng.uniform(-2, 2, size=(1, q, 1, 1))
    B = np.zeros((G, q, S))
    act = rng.choice(q, s_true, replace=False)
    B[:, act, :] = rng.normal(size=(G, s_true, S))
    y = 1.5 + np.einsum('gjs,gjsn->n', B, X) + snr * rng.normal(size=n)
    return X, y, B, act


def test_equivalence_with_old_solver():
    X, y, _, _ = make_data()
    for lam in [0.3, 0.1, 0.03]:
        ic_old, B_old = _old_proxgrad_fit(X, y, lam)
        f = T.fit_one(X, y, lam, center_scale=False, tol=1e-10, max_iter=50000)
        assert np.allclose(B_old, f.B, atol=1e-5, rtol=1e-4), (lam, np.abs(B_old - f.B).max())
        assert abs(ic_old - f.intercept) < 1e-4, (ic_old, f.intercept)
        assert f.kkt < 1e-5, f.kkt
    print('  equivalence with the old solver: OK')


def test_lambda_max():
    X, y, _, _ = make_data(seed=1)
    lmax = T.lambda_max(X, y)
    f0 = T.fit_one(X, y, lmax * 1.0001)
    assert (f0.B == 0).all()
    f1 = T.fit_one(X, y, lmax * 0.9)
    assert (np.abs(f1.B).sum(axis=(0, 2)) > 0).any()
    print('  lambda_max: OK')


def test_unit_invariance():
    X, y, _, _ = make_data(seed=2)
    f = T.fit_one(X, y, 0.05)
    st = f.block_stats()
    X2 = X.copy(); X2[:, 3, :, :] *= 1000.0; X2[:, 5, :, :] *= 1e-3       # change units
    f2 = T.fit_one(X2, y, 0.05)
    st2 = f2.block_stats()
    for k in ['block_scaled', 'col_std', 'mean_scaled']:
        assert np.allclose(st[k], st2[k], rtol=1e-4, atol=1e-7), (k, st[k], st2[k])
    assert not np.allclose(st['unscaled'], st2['unscaled'], rtol=1e-2)
    # predictions unchanged by the unit change
    assert np.allclose(f.predict(X), f2.predict(X2), atol=1e-6)
    print('  unit invariance of block_scaled / col_std (and NOT of unscaled): OK')


def test_group_folds():
    rng = np.random.default_rng(3)
    groups = np.array([f'pm{i}' for i in rng.integers(0, 20, 500)])
    train_idx = np.arange(0, 400)
    folds = T.group_folds(train_idx, groups, seed=7, k=5)
    allidx = sorted(i for f in folds for i in f)
    assert allidx == list(range(400))
    seen = {}
    for k, f in enumerate(folds):
        for i in f:
            g = groups[i]
            assert seen.setdefault(g, k) == k, 'group split across folds'
    pos = T.folds_to_positions(folds, train_idx)
    assert all(len(tr) + len(te) == 400 for tr, te in pos)
    print('  group_folds: OK')


def test_cv_and_threshold():
    X, y, Btrue, act = make_data(seed=4, n=400, q=20, snr=0.5)
    folds = T.plain_folds(np.arange(400), seed=1)
    res = T.cv_select(X, y, folds, num=10)
    assert not res['at_lower_bound'] and not res['at_upper_bound'], res['best']
    f = res['fit']
    r2 = 1 - np.mean((y - f.predict(X)) ** 2) / np.var(y)
    assert r2 > 0.9, r2
    sig = T.residual_sd(f, X, y)
    st = f.block_stats()
    sel = T.select_features(st, sig, 20, 400, 'col_std', group_size=6)
    assert set(np.where(sel)[0]) == set(act), (np.where(sel)[0], act)
    for rule in ['block_scaled', 'col_std', 'mean_scaled']:        # no false negatives
        assert set(act) <= set(np.where(T.select_features(st, sig, 20, 400, rule))[0])
    assert all(ff.converged for ff in res['path']), [ff.n_iter for ff in res['path']]
    print('  cv_select + adaptive threshold recovers the true support: OK')


if __name__ == '__main__':
    test_equivalence_with_old_solver()
    test_lambda_max()
    test_unit_invariance()
    test_group_folds()
    test_cv_and_threshold()
    print('all core tests passed')

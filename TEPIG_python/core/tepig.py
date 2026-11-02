"""
tepig.py
--------
THE single implementation of the TEPIG estimator (feature-wise group lasso on the
(G, q, S, n) tensor, solved by FISTA), its lambda path, its cross-validation and
its adaptive selection threshold. Every driver (simulation, Cell Painting, Coimbra)
imports from here; there are no other copies of the solver.

Objective (paper eq. tepig, stated with the 1/(2n) constant):

    (mu_hat, B_hat) = argmin  (1/2n) sum_i (y_i - mu - <B, X_i>_F)^2
                              + lam * sum_j || B[:, j, :] ||_F

Implementation notes
  * The intercept is profiled out: minimizing over mu is the same as fitting
    centered y on centered X. So the solver works on X_c = X - mean_i X and
    y_c = y - mean(y) with no intercept; mu_hat = mean(y) - <B_hat, mean_i X>.
  * Each feature's (G, S) block of columns is scaled by gamma_j (the CENTERED
    block RMS, gamma_j^2 = (1/n) sum_i ||X_c[:, j, :, i]||_F^2) so that lam has the
    same meaning for every feature; B_hat is unscaled on return. With
    `center_scale=False` the pre-Nov-2026 uncentered RMS is used instead.
  * Because n >> d in every application (d = G q S), the smooth part is handled
    through the d x d Gram matrix Q = X_s' X_s / n and the vector b = X_s' y_c / n,
    computed ONCE per training set and shared by every lambda on the path:
    grad(x) = Q x - b. One FISTA iteration is then O(d^2) instead of O(n d).
    The iterates are mathematically identical to the direct form.
  * The Lipschitz constant L = lambda_max(Q) is computed by Lanczos (scipy eigsh)
    and inflated by 0.1 % so the step 1/L is always a valid (slightly
    conservative) step.
  * The lambda path is solved from large to small lambda with warm starts, as in
    glmnet. Convergence: max |x_new - x_old| < tol * max(1, max|x|) or max_iter.
    The returned fit records the iteration count and the KKT residual so the
    paper's convergence claims can be made from logs, not assumed.

Adaptive selection threshold (paper eq. thresh), applied AFTER the fit:
    a feature is selected iff  stat_j > tau,
    tau = sigma_hat * sqrt((group_term + 2 log q) / n_train),
where sigma_hat is the training residual sd. `stat_j` must be a unit-invariant
block norm. Three statistics are available (all stored per fit so that the
choice can be made post hoc without refitting):
    'block_scaled' : gamma_j * ||B_hat[:, j, :]||_F   = the block norm in the
                     solver's own coordinates (the coordinates the universal
                     threshold is calibrated for; what the paper's gamma implies)
    'col_std'      : || sd_j o B_hat[:, j, :] ||_F, the block norm when every
                     column is expressed in units of its own standard deviation
    'unscaled'     : ||B_hat[:, j, :]||_F in original units (NOT unit-invariant;
                     this is what the pre-Nov-2026 code did -- kept only for
                     comparison)
and `group_term` is 0 (paper's sqrt(2 log q / n)) or G*S (Lounici-style
group-size term). See select_features().
"""

import os
import subprocess
import sys
import time
import platform

import numpy as np

__all__ = ['TepigFit', 'fit_path', 'fit_one', 'lambda_grid', 'lambda_max',
           'cv_select', 'group_folds', 'plain_folds', 'select_features',
           'selection_threshold', 'block_stats', 'refit_stats', 'residual_sd',
           'predict', 'run_metadata']


# ── design preparation ──────────────────────────────────────────────────────
def _prepare(X, y, center_scale=True):
    """X: (G, q, S, n), y: (n,).  Returns a dict with the centered, block-scaled
    flat design (n, d), its Gram matrix, the right-hand side and the scalings."""
    X = np.asarray(X)
    G, q, S, n = X.shape
    d = G * q * S
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if y.shape[0] != n:
        raise ValueError(f'y has {y.shape[0]} entries, X has n={n}')
    Xmean = X.mean(axis=3, dtype=np.float64)                        # (G, q, S)
    Xc = X.astype(np.float64) - Xmean[..., None]                    # centered
    if center_scale:
        gamma = np.sqrt((Xc ** 2).sum(axis=(0, 2)).mean(axis=1))    # (q,)
    else:
        gamma = np.sqrt((X.astype(np.float64) ** 2).sum(axis=(0, 2)).mean(axis=1))
    gamma = np.where(gamma > 1e-10, gamma, 1.0)
    col_sd = np.sqrt((Xc ** 2).mean(axis=3))                        # (G, q, S)
    Xc /= gamma[None, :, None, None]
    X_flat = np.ascontiguousarray(Xc.reshape(d, n).T)               # (n, d)
    del Xc
    ymean = float(y.mean())
    yc = y - ymean
    Q = (X_flat.T @ X_flat) / n                                     # (d, d)
    b = (X_flat.T @ yc) / n
    return dict(G=G, q=q, S=S, n=n, d=d, X_flat=X_flat, Q=Q, b=b, yc=yc,
                ymean=ymean, Xmean=Xmean, gamma=gamma, col_sd=col_sd)


def _lipschitz(Q):
    """Largest eigenvalue of the (symmetric PSD) Gram matrix, inflated by 0.1 %
    so that 1/L is a safe FISTA step."""
    d = Q.shape[0]
    if d <= 400:
        lmax = float(np.linalg.eigvalsh(Q)[-1])
    else:
        try:
            from scipy.sparse.linalg import eigsh
            lmax = float(eigsh(Q, k=1, which='LA', tol=1e-7,
                               return_eigenvectors=False)[0])
        except Exception:                                   # pragma: no cover
            rng = np.random.default_rng(0)
            v = rng.standard_normal(d); v /= np.linalg.norm(v)
            lmax = 0.0
            for _ in range(1000):
                w = Q @ v
                nw = float(np.linalg.norm(w))
                if nw == 0:
                    break
                v = w / nw
                if abs(nw - lmax) < 1e-10 * max(nw, 1.0):
                    lmax = nw; break
                lmax = nw
    return max(lmax, 1e-12) * 1.001


# ── FISTA on the Gram form ──────────────────────────────────────────────────
def _block_prox(v, G, q, S, thr):
    """prox of thr * sum_j ||v_(j)||: block soft-thresholding, in place on a copy."""
    V = v.reshape(G, q, S)
    nrm = np.sqrt((V ** 2).sum(axis=(0, 2)))                        # (q,)
    scale = np.where(nrm > thr, 1.0 - thr / np.where(nrm > 0, nrm, 1.0), 0.0)
    return (V * scale[None, :, None]).reshape(-1)


RESTART = True      # adaptive (gradient-scheme) restart of the momentum, O'Donoghue & Candes 2015


def _fista(Q, b, lam, eta, G, q, S, x0=None, max_iter=10000, tol=1e-6, restart=None):
    """Minimize (1/2) x'Qx - b'x + lam * sum_j ||x_(j)|| by FISTA with step eta.
    With restart=True (default: module flag RESTART) the Nesterov momentum is reset
    whenever the step and the momentum point in opposite directions
    ((z - x_new) . (x_new - x) > 0), which removes the oscillation FISTA shows on
    ill-conditioned designs without changing the minimizer (the objective is convex).
    Returns x, n_iter, converged."""
    if restart is None:
        restart = RESTART
    d = Q.shape[0]
    x = np.zeros(d) if x0 is None else np.array(x0, dtype=np.float64)
    z = x.copy()
    t = 1.0
    thr = eta * lam
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        grad = Q @ z - b
        x_new = _block_prox(z - eta * grad, G, q, S, thr)
        step = x_new - x
        if restart and float((z - x_new) @ step) > 0.0:
            t = 1.0                                   # restart momentum
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z = x_new + ((t - 1.0) / t_new) * step
        delta = float(np.max(np.abs(step))) if d else 0.0
        x, t = x_new, t_new
        if delta < tol * max(1.0, float(np.max(np.abs(x)))):
            converged = True
            break
    return x, it, converged


def _kkt_residual(Q, b, x, lam, G, q, S):
    """Max violation of the group-lasso optimality conditions at x:
    active blocks: ||grad_j + lam x_j/||x_j|| ||; inactive: max(0, ||grad_j|| - lam)."""
    grad = (Q @ x - b).reshape(G, q, S)
    X3 = x.reshape(G, q, S)
    xn = np.sqrt((X3 ** 2).sum(axis=(0, 2)))
    gn = np.sqrt((grad ** 2).sum(axis=(0, 2)))
    act = xn > 0
    viol = np.zeros(q)
    if act.any():
        r = grad[:, act, :] + lam * X3[:, act, :] / xn[act][None, :, None]
        viol[act] = np.sqrt((r ** 2).sum(axis=(0, 2)))
    viol[~act] = np.maximum(0.0, gn[~act] - lam)
    return float(viol.max()) if q else 0.0


class TepigFit:
    """Result of one TEPIG fit at one lambda. Attributes:
    B (G,q,S) unscaled coefficient tensor; intercept; lam; gamma (q,) block scales;
    col_sd (G,q,S) training column sds; Xmean (G,q,S); ymean; x (d,) scaled
    coefficients; n_iter; converged; kkt; n_train."""

    def __init__(self, prep, x, lam, n_iter, converged, kkt):
        G, q, S = prep['G'], prep['q'], prep['S']
        self.G, self.q, self.S, self.n_train = G, q, S, prep['n']
        self.x = x
        self.lam = float(lam)
        self.gamma = prep['gamma']
        self.col_sd = prep['col_sd']
        self.Xmean = prep['Xmean']
        self.ymean = prep['ymean']
        self.B = x.reshape(G, q, S) / self.gamma[None, :, None]
        self.intercept = self.ymean - float((self.Xmean * self.B).sum())
        self.n_iter, self.converged, self.kkt = int(n_iter), bool(converged), float(kkt)

    # convenience ---------------------------------------------------------
    def predict(self, X):
        return predict(self, X)

    def block_stats(self):
        return block_stats(self.B, self.gamma, self.col_sd)

    def summary(self):
        return dict(lam=self.lam, n_iter=self.n_iter, converged=self.converged,
                    kkt=self.kkt, n_fitted=int((self.block_stats()['unscaled'] > 0).sum()))


def predict(fit, X):
    """mu_hat + <B_hat, X_i>_F for every subject in X (G, q, S, m)."""
    X = np.asarray(X)
    d = fit.G * fit.q * fit.S
    return fit.intercept + X.reshape(d, X.shape[3]).T.astype(np.float64) @ fit.B.reshape(-1)


def fit_path(X, y, lams, max_iter=10000, tol=1e-6, center_scale=True, prep=None,
             warm=True):
    """Fit TEPIG at every lambda in `lams` (any order; solved from largest to
    smallest with warm starts). Returns a list of TepigFit in the ORDER GIVEN."""
    lams = np.asarray(lams, dtype=np.float64).reshape(-1)
    p = prep if prep is not None else _prepare(X, y, center_scale)
    G, q, S = p['G'], p['q'], p['S']
    L = _lipschitz(p['Q'])
    eta = 1.0 / L
    order = np.argsort(-lams)
    fits = [None] * len(lams)
    x = None
    for k in order:
        lam = float(lams[k])
        x, it, conv = _fista(p['Q'], p['b'], lam, eta, G, q, S,
                             x0=x if warm else None, max_iter=max_iter, tol=tol)
        kkt = _kkt_residual(p['Q'], p['b'], x, lam, G, q, S)
        fits[k] = TepigFit(p, x.copy(), lam, it, conv, kkt)
    return fits


def fit_one(X, y, lam, **kw):
    return fit_path(X, y, [lam], **kw)[0]


# ── lambda grid ─────────────────────────────────────────────────────────────
def lambda_max(X, y, center_scale=True, prep=None):
    """Smallest lambda at which every block is zero: max_j ||b_(j)||_2 where
    b = X_s' y_c / n (the gradient of the loss at B = 0)."""
    p = prep if prep is not None else _prepare(X, y, center_scale)
    bn = np.sqrt((p['b'].reshape(p['G'], p['q'], p['S']) ** 2).sum(axis=(0, 2)))
    return float(bn.max())


def lambda_grid(X, y, num=12, ratio=1e-3, center_scale=True, prep=None):
    """Geometric grid from lambda_max down to ratio * lambda_max (descending)."""
    lmax = lambda_max(X, y, center_scale, prep)
    return lmax * np.geomspace(1.0, ratio, num)


# ── folds ───────────────────────────────────────────────────────────────────
def plain_folds(train_idx, seed, k=5):
    """k random folds over the given indices (lists of sorted global indices)."""
    idx = np.random.default_rng(seed).permutation(np.asarray(train_idx))
    return [sorted(int(i) for i in g) for g in np.array_split(idx, k)]


def group_folds(train_idx, groups, seed, k=5):
    """k folds over `train_idx` that never split a GROUP (e.g. a platemap and
    all of its replicate plates): the unique groups present in train_idx are
    shuffled with `seed` and dealt into k chunks."""
    train_idx = np.asarray(train_idx)
    g = np.asarray(groups)[train_idx]
    uniq = np.array(sorted(set(g.tolist())), dtype=object)
    uniq = np.random.default_rng(seed).permutation(uniq)
    if len(uniq) < k:                       # fewer groups than folds: one group per fold
        k = len(uniq)
    folds = []
    for chunk in np.array_split(uniq, k):
        m = np.isin(g, chunk)
        folds.append(sorted(int(i) for i in train_idx[m]))
    return folds


def folds_to_positions(folds, train_idx):
    """Map folds of GLOBAL indices to (train_pos, test_pos) pairs relative to
    train_idx -- the format sklearn's cv= argument wants."""
    pos = {int(i): k for k, i in enumerate(train_idx)}
    out = []
    for f in folds:
        te = np.array([pos[int(i)] for i in f])
        tr = np.array([pos[int(i)] for g in folds if g is not f for i in g])
        out.append((tr, te))
    return out


# ── cross-validation ────────────────────────────────────────────────────────
def cv_select(X, y, folds, lams=None, num=12, ratio=1e-3, max_iter=10000,
              tol=1e-6, center_scale=True, verbose=False):
    """Five-fold (or k-fold) CV over a lambda path. `folds` are lists of global
    indices into X's last axis (the TRAINING set only; their union is the
    training set). Held-out MSE uses the training-fold intercept and centering
    (the same prediction rule as the final model). Returns a dict with the
    grid, the CV curve, the argmin and the final fit on the union of folds."""
    train_idx = np.array(sorted(int(i) for f in folds for i in f))
    X = np.asarray(X); y = np.asarray(y, dtype=np.float64)
    Xtr, ytr = X[..., train_idx], y[train_idx]
    prep_full = _prepare(Xtr, ytr, center_scale)
    if lams is None:
        lams = lambda_grid(None, None, num, ratio, prep=prep_full)
    lams = np.asarray(lams, dtype=np.float64)
    cv = np.zeros((len(folds), len(lams)))
    iters = np.zeros((len(folds), len(lams)), int)
    t0 = time.time()
    for k, te in enumerate(folds):
        tr = sorted(int(i) for f in folds if f is not te for i in f)
        fits = fit_path(X[..., tr], y[tr], lams, max_iter, tol, center_scale)
        Xte = X[..., te]
        for j, f in enumerate(fits):
            cv[k, j] = float(np.mean((y[te] - f.predict(Xte)) ** 2))
            iters[k, j] = f.n_iter
        if verbose:
            print(f'    fold {k + 1}/{len(folds)}: {time.time() - t0:.0f}s', flush=True)
    curve = cv.mean(axis=0)
    se = cv.std(axis=0, ddof=1) / np.sqrt(len(folds)) if len(folds) > 1 else np.zeros_like(curve)
    best = int(np.argmin(curve))
    # one-standard-error rule: the LARGEST lambda whose CV error is within one SE of the minimum
    ok = np.where(curve <= curve[best] + se[best])[0]
    best_1se = int(ok[np.argmax(lams[ok])])
    fits = fit_path(None, None, lams, max_iter, tol, center_scale, prep=prep_full)
    return dict(lams=lams, cv_curve=curve, cv_se=se, cv_folds=cv, best=best,
                best_lam=float(lams[best]), best_1se=best_1se, best_1se_lam=float(lams[best_1se]),
                fit=fits[best], fit_1se=fits[best_1se], path=fits, iters=iters,
                train_idx=train_idx,
                at_lower_bound=bool(best == len(lams) - 1),
                at_upper_bound=bool(best == 0))


# ── selection ───────────────────────────────────────────────────────────────
def block_stats(B, gamma, col_sd):
    """Per-feature block norms under every coordinate convention (q,) each."""
    B = np.asarray(B)
    return {
        'unscaled': np.sqrt((B ** 2).sum(axis=(0, 2))),
        'block_scaled': gamma * np.sqrt((B ** 2).sum(axis=(0, 2))),
        'col_std': np.sqrt(((col_sd * B) ** 2).sum(axis=(0, 2))),
        'mean_scaled': np.abs(gamma * B.mean(axis=(0, 2))),
        'mean_unscaled': np.abs(B.mean(axis=(0, 2))),
    }


def refit_stats(fit, X, y):
    """Debiased block statistic: ordinary least squares of y on the STANDARDISED
    columns (centred, unit sd) of the blocks that are nonzero in `fit`, then the
    block norms of the refitted coefficients ('col_std_refit'). The universal
    threshold is calibrated for unshrunk coefficients, so this is the statistic to
    compare with tau when the lasso's own shrinkage would hide real effects.
    Returns dict(col_std_refit=(q,), sigma_refit=float, n_active=int,
    underdetermined=bool). When the active design has >= n columns the refit is
    the minimum-norm least-squares solution and `underdetermined` is True."""
    X = np.asarray(X); y = np.asarray(y, dtype=np.float64)
    G, q, S, n = X.shape
    act = np.where(np.sqrt((fit.B ** 2).sum(axis=(0, 2))) > 0)[0]
    out = {'col_std_refit': np.zeros(q), 'sigma_refit': float('nan'),
           'n_active': int(len(act)), 'underdetermined': False}
    if len(act) == 0:
        return out
    Xa = X[:, act, :, :].astype(np.float64)                        # (G, a, S, n)
    mu = Xa.mean(axis=3, keepdims=True)
    sd = np.sqrt(((Xa - mu) ** 2).mean(axis=3, keepdims=True))
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = ((Xa - mu) / sd).reshape(G * len(act) * S, n).T              # (n, dA) standardised
    yc = y - y.mean()
    coef, _, rank, _ = np.linalg.lstsq(Z, yc, rcond=None)
    out['underdetermined'] = bool(Z.shape[1] >= n or rank < Z.shape[1])
    C = coef.reshape(G, len(act), S)
    out['col_std_refit'][act] = np.sqrt((C ** 2).sum(axis=(0, 2)))
    dof = max(n - Z.shape[1] - 1, 1)
    out['sigma_refit'] = float(np.sqrt(((yc - Z @ coef) ** 2).sum() / dof))
    return out


def selection_threshold(sigma, q, n_train, group_size=0):
    """tau = sigma * sqrt((group_size + 2 log q) / n_train)."""
    return float(sigma) * np.sqrt((group_size + 2.0 * np.log(max(q, 2))) / n_train)


def select_features(stats, sigma, q, n_train, rule='block_scaled', group_size=0):
    """Boolean mask of selected features under one rule. `stats` is the dict
    from block_stats(). rule in {'block_scaled','col_std','unscaled',
    'mean_scaled','mean_unscaled','raw'} ('raw' = nonzero blocks, no threshold)."""
    if rule == 'raw':
        return stats['unscaled'] > 0
    tau = selection_threshold(sigma, q, n_train, group_size)
    return stats[rule] > tau


def residual_sd(fit, X, y):
    """Training residual standard deviation (ddof=1): the sigma_hat of eq. thresh."""
    r = np.asarray(y, dtype=np.float64) - fit.predict(X)
    return float(np.std(r, ddof=1))


# ── provenance ──────────────────────────────────────────────────────────────
def run_metadata():
    """Git hash, python/numpy/sklearn versions, host -- stored in every output."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        h = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=here, capture_output=True,
                           text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'],
                               cwd=here, capture_output=True, text=True, timeout=5).stdout
        h = h + ('+dirty' if dirty.strip() else '')
    except Exception:
        h = 'unknown'
    vers = {'python': platform.python_version(), 'numpy': np.__version__}
    try:
        import sklearn; vers['sklearn'] = sklearn.__version__
    except Exception:
        pass
    try:
        import scipy; vers['scipy'] = scipy.__version__
    except Exception:
        pass
    return dict(git=h, versions=vers, host=platform.node(), argv=sys.argv,
                time=time.strftime('%Y-%m-%dT%H:%M:%S'))

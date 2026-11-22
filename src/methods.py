"""From-scratch implementation of mean-variance portfolio optimization.

One function per numbered step of the Iteration 2 algorithm ("The Algorithm as Math"),
using NumPy arrays and matrix products only -- no numpy.linalg, SciPy or scikit-learn.

    Step (1)  mu_hat = (1/T) R^T 1                          my_sample_mean
    Step (2)  R_c = R - 1 mu_hat^T                          my_center
    Step (3)  Sigma_hat = R_c^T R_c / (T - 1)               my_sample_cov
              (1)-(3) together                              my_estimate_mu_sigma
    Step (4)  solve Sigma x = 1 and Sigma y = mu            my_gauss_solve
    Step (5)  a, b, c, d                                    my_frontier_constants
    Step (6)  w* = x / a                                    my_min_variance
    Step (7)  2x2 system -> w(r), sigma^2(r)                my_frontier_weights, my_frontier_variance
    Extra     annualize, portfolio return / variance        my_annualize, my_portfolio_stats

Symbols match the report: R (T x N returns), m = mu_hat, S = Sigma_hat, w = weights,
a = 1'S^{-1}1, b = mu'S^{-1}1, c = mu'S^{-1}mu, d = ac - b^2.

Exact arithmetic: every function uses only +, -, *, / and comparisons, so it also
works on NumPy arrays of dtype=object holding fractions.Fraction values. That lets the
SAME functions run the small worked example exactly, the simulation and the real data.
"""
from fractions import Fraction

import numpy as np


# ---- helpers -----------------------------------------------------------------
def _as_array(A):
    """Keep Fraction (object) arrays exact; convert everything else to float."""
    A = np.asarray(A)
    return A.copy() if A.dtype == object else A.astype(float)


def _ones(n, like):
    """Vector of n ones, exact (Fraction) if `like` is an object array, else float."""
    if np.asarray(like).dtype == object:
        return np.array([Fraction(1)] * n, dtype=object)
    return np.ones(n)


# ---- Step (1): sample mean ---------------------------------------------------
# mu_hat = (1/T) R^T 1_T        R in R^{T x N}  ->  mu_hat in R^N        cost ~ NT
def my_sample_mean(R):
    R = _as_array(R)
    T = R.shape[0]
    return (R.T @ _ones(T, R)) / T


# ---- Step (2): centre the data -----------------------------------------------
# R_c = R - 1_T mu_hat^T        (subtract each column's mean)            cost ~ NT
def my_center(R, mu_hat):
    R = _as_array(R)
    return R - np.outer(_ones(R.shape[0], R), mu_hat)


# ---- Step (3): sample covariance ---------------------------------------------
# Sigma_hat = R_c^T R_c / (T - 1)      (unbiased; Sigma_hat in R^{N x N}) cost ~ 2 N^2 T
def my_sample_cov(R_c):
    T = R_c.shape[0]
    return (R_c.T @ R_c) / (T - 1)


# ---- Steps (1)-(3) together ----------------------------------------------------
# returns (mu_hat, Sigma_hat)
def my_estimate_mu_sigma(R):
    R = _as_array(R)
    m = my_sample_mean(R)
    return m, my_sample_cov(my_center(R, m))


# ---- Step (4): Gaussian elimination with partial pivoting ---------------------
# Solves A X = B for X.  A is n x n; B is a vector (n,) or a matrix (n, k), so several
# right-hand sides (e.g. 1 and mu) share ONE elimination, as in the algorithm.
#   Forward elimination on [A | B]:  for column k, swap up the row with the largest
#   |entry| (partial pivoting), then  row_i <- row_i - (A_ik / A_kk) row_k  for i > k.
#   Back substitution:  x_i = (b_i - sum_{j>i} U_ij x_j) / U_ii,  i = n-1, ..., 0.
# Cost ~ (2/3) n^3 for the elimination + n^2 per right-hand side.
# verbose=True prints the augmented matrix after every row operation (worked example).
def my_gauss_solve(A, B, verbose=False):
    A, B = _as_array(A), _as_array(B)
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError(f"A must be square, got {A.shape}")
    vector_rhs = B.ndim == 1
    B = B.reshape(n, -1)
    M = np.concatenate([A, B], axis=1)              # augmented matrix [A | B]
    if A.dtype == object or B.dtype == object:
        M = M.astype(object)

    def show(msg):
        if verbose:
            print(msg)
            for row in M:
                print("   [" + "  ".join(f"{str(v):>10}" for v in row[:n]) + "  |  "
                      + "  ".join(f"{str(v):>10}" for v in row[n:]) + " ]")

    show("Start: [A | B]")
    for k in range(n):
        # partial pivoting: largest |M_ik| for i >= k
        p = k + int(np.argmax([abs(v) for v in M[k:, k]]))
        if M[p, k] == 0:
            raise ValueError(f"matrix is singular: no non-zero pivot in column {k}")
        if p != k:
            M[[k, p]] = M[[p, k]]
            show(f"Swap R{k + 1} <-> R{p + 1}")
        for i in range(k + 1, n):
            f = M[i, k] / M[k, k]
            if f != 0:
                M[i] = M[i] - f * M[k]
                show(f"R{i + 1} <- R{i + 1} - ({f}) R{k + 1}")

    # back substitution on the upper-triangular system
    X = M[:, n:].copy()
    for i in range(n - 1, -1, -1):
        X[i] = (M[i, n:] - M[i, i + 1:n] @ X[i + 1:]) / M[i, i]
    return X[:, 0] if vector_rhs else X


# ---- Step (5): frontier constants ----------------------------------------------
# x = S^{-1} 1,  y = S^{-1} mu   (one elimination, two right-hand sides)
# a = 1^T x,  b = mu^T x,  c = mu^T y,  d = a c - b^2
def my_frontier_constants(m, S):
    m = _as_array(m)
    one = _ones(len(m), m)
    XY = my_gauss_solve(S, np.column_stack([one, m]))
    x, y = XY[:, 0], XY[:, 1]
    a, b, c = one @ x, m @ x, m @ y
    return a, b, c, a * c - b * b


# ---- Step (6): global minimum-variance portfolio (Theorem 3) --------------------
# w* = S^{-1} 1 / (1^T S^{-1} 1) = x / a          sigma^2_min = 1 / a
def my_min_variance(S):
    S = _as_array(S)
    x = my_gauss_solve(S, _ones(len(S), S))
    return x / x.sum()                       # x.sum() = 1^T x = a


# ---- Step (7): target-return portfolio (Theorem 4) and the frontier (Theorem 5) --
# [c b; b a] [l1; l2] = [r; 1]   ->   w(r) = S^{-1}(l1 mu + l2 1) = l1 y + l2 x
def my_frontier_weights(m, S, r):
    m = _as_array(m)
    one = _ones(len(m), m)
    XY = my_gauss_solve(S, np.column_stack([one, m]))
    x, y = XY[:, 0], XY[:, 1]
    a, b, c = one @ x, m @ x, m @ y
    K = np.array([[c, b], [b, a]], dtype=XY.dtype)
    rhs = np.array([r, one[0]], dtype=XY.dtype)
    l1, l2 = my_gauss_solve(K, rhs)
    return l1 * y + l2 * x


# sigma^2(r) = (a r^2 - 2 b r + c) / d      (closed form, Theorem 5)
def my_frontier_variance(m, S, r):
    a, b, c, d = my_frontier_constants(m, S)
    r = np.asarray(r) if np.asarray(r).dtype == object else np.asarray(r, float)
    return (a * r * r - 2 * b * r + c) / d


# ---- extras --------------------------------------------------------------------
# portfolio return w^T mu and variance w^T S w   (Proposition 1)
def my_portfolio_stats(w, m, S):
    return w @ m, w @ S @ w


# annualize daily estimates: mu_ann = k mu,  Sigma_ann = k Sigma   (k = 252 trading days)
def my_annualize(m, S, k=252):
    return m * k, S * k

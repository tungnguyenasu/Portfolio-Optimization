"""Core mean-variance portfolio math, written from scratch with numpy.

Every function maps to one derivation / math question in the report:
    portfolio_stats      -> Q1  (return w'mu, risk w'Sigma w)
    sample_moments       -> mu_hat, Sigma_hat (Definitions section)
    is_psd               -> Q2  (Sigma is positive semi-definite)
    min_variance_weights -> Q3  (w* = Sigma^{-1} 1 / 1' Sigma^{-1} 1)
    frontier_constants   -> Q4/Q5 (a, b, c and the 2x2 multiplier system)
    frontier_weights     -> Q4  (solve the 2x2 system, then w = Sigma^{-1}(l1 mu + l2 1))
    frontier_variance    -> Q5  (sigma^2(r) = (a r^2 - 2 b r + c) / (a c - b^2))

No library routine that solves the whole method is used; only np.linalg.solve
for linear systems and np.linalg.eigvalsh for the PSD check.
"""
import numpy as np


def sample_moments(R):
    """R: (T, N) returns, one row per period, one column per asset.

    Returns mu_hat (N,) and the unbiased sample covariance Sigma_hat (N, N),
        Sigma_hat = Rc' Rc / (T - 1),   Rc = R - 1 mu_hat'.
    """
    R = np.asarray(R, dtype=float)
    T = R.shape[0]
    mu_hat = R.mean(axis=0)
    Rc = R - mu_hat
    Sigma_hat = Rc.T @ Rc / (T - 1)
    return mu_hat, Sigma_hat


def portfolio_stats(w, mu, Sigma):
    """Expected return w'mu and variance w'Sigma w of portfolio w (Q1)."""
    return float(w @ mu), float(w @ Sigma @ w)


def is_psd(Sigma, tol=1e-12):
    """True if every eigenvalue of the symmetric matrix Sigma is >= -tol (Q2)."""
    lam = np.linalg.eigvalsh(Sigma)
    return bool(lam.min() >= -tol), lam


def min_variance_weights(Sigma):
    """Global minimum-variance portfolio (Q3): w* = Sigma^{-1} 1 / (1' Sigma^{-1} 1)."""
    one = np.ones(Sigma.shape[0])
    Si1 = np.linalg.solve(Sigma, one)
    return Si1 / (one @ Si1)


def frontier_constants(mu, Sigma):
    """a = 1'S^{-1}1, b = mu'S^{-1}1, c = mu'S^{-1}mu, d = ac - b^2 (Q4, Q5)."""
    one = np.ones(len(mu))
    Si1 = np.linalg.solve(Sigma, one)
    Simu = np.linalg.solve(Sigma, mu)
    a, b, c = one @ Si1, mu @ Si1, mu @ Simu
    return a, b, c, a * c - b ** 2


def frontier_weights(mu, Sigma, r):
    """Minimum-variance weights with target return r (Q4).

    Lagrange conditions give w = Sigma^{-1}(l1 mu + l2 1); plugging into the two
    constraints gives the 2x2 system  [c b; b a] [l1; l2] = [r; 1].
    """
    a, b, c, _ = frontier_constants(mu, Sigma)
    l1, l2 = np.linalg.solve(np.array([[c, b], [b, a]]), np.array([r, 1.0]))
    one = np.ones(len(mu))
    return np.linalg.solve(Sigma, l1 * mu + l2 * one), (l1, l2)


def frontier_variance(mu, Sigma, r):
    """Closed-form frontier variance (Q5): (a r^2 - 2 b r + c) / (a c - b^2)."""
    a, b, c, d = frontier_constants(mu, Sigma)
    r = np.asarray(r, dtype=float)
    return (a * r ** 2 - 2 * b * r + c) / d

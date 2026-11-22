"""Simulation with known truth for Project 10 - Portfolio Optimization.

The model (unchanged from Iteration 2): four assets with returns
    r_t ~ iid N(mu, Sigma),   Sigma = diag(vol) Corr diag(vol),
with mu, vol and Corr chosen below, so the true answer is known.

Iteration 3 entry point:
    R, mu, Sigma = simulate_returns(T, seed)

Running this file directly reproduces the Iteration 2 simulation check (Q1-Q5)
with exactly the same numbers as before (seed 0 for T = 2000, seed 1 for T = 60).

Usage (from the project root):  python src/simulate.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from portfolio import (sample_moments, portfolio_stats, is_psd, min_variance_weights,
                       frontier_constants, frontier_weights, frontier_variance)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")

# ---- True model (parameters WE choose) -------------------------------------
mu = np.array([0.05, 0.08, 0.12, 0.07])
vol = np.array([0.10, 0.15, 0.25, 0.12])
Corr = np.array([[1, .3, .2, .1],
                 [.3, 1, .4, .2],
                 [.2, .4, 1, .3],
                 [.1, .2, .3, 1]])
Sigma = np.outer(vol, vol) * Corr
T = 2000
N = len(mu)


def simulate_returns(T, seed=0):
    """Draw T periods of returns for the 4-asset model with known mu and Sigma.

    r_t ~ iid N(mu, Sigma),  t = 1, ..., T.

    Parameters
    ----------
    T    : number of periods (rows of R).
    seed : int seed for np.random.default_rng, or an existing Generator
           (np.random.default_rng returns a Generator unchanged, so a caller can
           keep drawing from the same stream afterwards).

    Returns
    -------
    R     : (T, 4) simulated returns, one row per period, one column per asset.
    mu    : (4,)   true expected returns.
    Sigma : (4, 4) true covariance matrix.
    """
    rng = np.random.default_rng(seed)
    R = rng.multivariate_normal(mu, Sigma, size=T)
    return R, mu.copy(), Sigma.copy()


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def main():
    rng = np.random.default_rng(0)
    R, _, _ = simulate_returns(T, rng)      # same stream as Iteration 2
    m, S = sample_moments(R)

    section("Estimates vs truth (T = 2000)")
    print("mu true :", mu)
    print("mu hat  :", m.round(4))
    print("max |Sigma_hat - Sigma| =", np.abs(S - Sigma).max().round(5))

    # Q1: portfolio return and variance
    section("Q1: E[w'R] = w'mu and Var(w'R) = w'Sigma w")
    w_min = min_variance_weights(S)
    ret, var = portfolio_stats(w_min, m, S)
    port = R @ w_min
    print(f"w'mu_hat = {ret:.6f}   sample mean of R w = {port.mean():.6f}")
    print(f"w'S w    = {var:.8f}   sample var of R w  = {port.var(ddof=1):.8f}")

    # Q2: PSD
    section("Q2: Sigma_hat is positive semi-definite")
    ok, lam = is_psd(S)
    print("eigenvalues of Sigma_hat:", lam.round(6), "  all >= 0:", ok)
    z = rng.normal(size=(10000, N))
    print("min over 10,000 random z of z'S z =", np.einsum("ij,jk,ik->i", z, S, z).min())

    # Q3: global minimum variance
    section("Q3: w* = Sigma^{-1} 1 / 1'Sigma^{-1} 1")
    w_true = min_variance_weights(Sigma)
    print("w* (true Sigma)   :", w_true.round(4), " sum =", w_true.sum())
    print("w* (Sigma_hat)    :", w_min.round(4), " sum =", w_min.sum())
    W = rng.dirichlet(np.ones(N), size=20000)          # random long-only weights
    W = np.vstack([W, rng.normal(size=(20000, N))])
    W = W / W.sum(axis=1, keepdims=True)               # any weights summing to 1
    rand_var = np.einsum("ij,jk,ik->i", W, S, W)
    print(f"min variance among 40,000 random portfolios = {rand_var.min():.8f}")
    print(f"variance of w*                               = {var:.8f}")
    print("any random portfolio beats w*? ", bool((rand_var < var - 1e-12).any()))

    # Q4 + Q5: frontier
    section("Q4/Q5: 2x2 multiplier system and the parabola")
    a, b, c, d = frontier_constants(m, S)
    print(f"a = {a:.4f}, b = {b:.4f}, c = {c:.6f}, ac - b^2 = {d:.6f}")
    print(f"{'r':>6} {'w S w (solve)':>15} {'parabola':>12} {'|diff|':>10} {'w.mu-r':>10} {'sum w-1':>10}")
    for r in np.linspace(0.04, 0.14, 6):
        w, _ = frontier_weights(m, S, r)
        v1 = w @ S @ w
        v2 = frontier_variance(m, S, r)
        print(f"{r:6.3f} {v1:15.8f} {v2:12.8f} {abs(v1 - v2):10.2e} "
              f"{w @ m - r:10.2e} {w.sum() - 1:10.2e}")
    r_min = b / a
    print(f"vertex of parabola r = b/a = {r_min:.5f}, w*'mu = {ret:.5f}  (should match)")

    # Estimation error: T = 2000 vs T = 60
    section("Estimation error: weights from T=2000 vs T=60 periods")
    R60, _, _ = simulate_returns(60, seed=1)
    m60, S60 = sample_moments(R60)
    w60 = min_variance_weights(S60)
    print("w* true   :", w_true.round(4))
    print("w* T=2000 :", w_min.round(4), " error =", np.linalg.norm(w_min - w_true).round(4))
    print("w* T=60   :", w60.round(4), " error =", np.linalg.norm(w60 - w_true).round(4))

    # Figure: frontier, true vs estimated
    os.makedirs(FIG_DIR, exist_ok=True)
    rs = np.linspace(0.03, 0.15, 200)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(np.sqrt(frontier_variance(mu, Sigma, rs)), rs, label="True frontier")
    ax.plot(np.sqrt(frontier_variance(m, S, rs)), rs, "--", label="Estimated, T=2000")
    ax.plot(np.sqrt(frontier_variance(m60, S60, rs)), rs, ":", label="Estimated, T=60")
    rv = np.einsum("ij,jk,ik->i", W[:3000], Sigma, W[:3000])
    ax.scatter(np.sqrt(rv), W[:3000] @ mu, s=2, alpha=0.2, c="gray", label="Random portfolios")
    ax.scatter(np.sqrt(vol ** 2), mu, c="k", marker="x", label="Single assets")
    ax.set_xlim(0, 0.35); ax.set_ylim(0.03, 0.15)
    ax.set_xlabel(r"Risk $\sigma = \sqrt{w^\top \Sigma w}$")
    ax.set_ylabel(r"Return $r = w^\top \mu$")
    ax.set_title("Simulation: efficient frontier")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "sim_frontier.png"), dpi=150)
    print(f"\nSaved {os.path.join(FIG_DIR, 'sim_frontier.png')}")


if __name__ == "__main__":
    main()

"""Iteration 3, Step 6: check assumptions on simulation and real data."""

import argparse
import numpy as np
from scipy.stats import jarque_bera, kurtosis, skew

from load_data import TICKERS
from methods import my_estimate_mu_sigma, my_frontier_constants, my_min_variance
from run_real import clean_returns
from simulate import simulate_returns

WINDOW = 252
STEP = 21


def check_assumptions(name, R, assets):
    T, N = R.shape
    mu, Sigma = my_estimate_mu_sigma(R)
    print(f"\n=== {name} (T={T}, N={N}) ===")

    # A1: rolling estimates of mean and covariance, same window and stride.
    mean_drift, cov_drift = [], []
    se = np.sqrt(np.diag(Sigma) / WINDOW)
    starts = list(range(0, T - WINDOW + 1, STEP))
    if starts[-1] != T - WINDOW:
        starts.append(T - WINDOW)
    for start in starts:
        m, S = my_estimate_mu_sigma(R[start:start + WINDOW])
        mean_drift.append(np.max(np.abs(m - mu) / se))
        cov_drift.append(np.linalg.norm(S - Sigma, "fro") /
                         np.linalg.norm(Sigma, "fro"))
    print(f"A1: windows={len(starts)}, length={WINDOW}, stride={STEP} "
          "(plus final window)")
    print(f"  max component mean deviation / SE: "
          f"median={np.median(mean_drift):.4f}, max={max(mean_drift):.4f}")
    print(f"  relative covariance Frobenius deviation: "
          f"median={np.median(cov_drift):.4f}, max={max(cov_drift):.4f}")

    # A2 and A3: positive definiteness, condition number and rank.
    eigenvalues = np.linalg.eigvalsh(Sigma)
    rank = np.linalg.matrix_rank(Sigma)
    print(f"A2: lambda_min={eigenvalues.min():.8g}, "
          f"lambda_max={eigenvalues.max():.8g}, cond={np.linalg.cond(Sigma):.4f}")
    print(f"A3: T>N = {T>N}; rank={rank}/{N}")

    # A4: d is strictly positive when the two constraints are independent.
    a, b, c, d = my_frontier_constants(mu, Sigma)
    print(f"A4: a={a:.8g}, b={b:.8g}, c={c:.8g}, d={d:.8g}, d>0={d>0}")

    # A5: marginal normality diagnostics; not a proof of joint normality.
    print(f"A5: Jarque-Bera marginal tests (Bonferroni threshold={0.05/N:.5g})")
    print(f"{'Asset':8s} {'Skew':>10s} {'Excess kurt.':>13s} {'JB p-value':>12s}")
    rejected = 0
    for i, asset in enumerate(assets):
        v = R[:, i]
        p = jarque_bera(v).pvalue
        rejected += int(p < 0.05/N)
        print(f"{asset:8s} {skew(v, bias=False):10.4f} "
              f"{kurtosis(v, fisher=True, bias=False):13.4f} {p:12.4g}")
    print(f"  rejected: {rejected}/{N}")

    # A6: negative weights are allowed by the model, not empirical evidence
    # that trading constraints or costs are absent.
    w = my_min_variance(Sigma)
    print("A6: negative GMV weights:", {asset: round(float(wi), 8)
                                      for asset, wi in zip(assets, w) if wi < 0})
    print(f"  sum={w.sum():.12f}, gross exposure={np.abs(w).sum():.8f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    R_sim, _, _ = simulate_returns(2000, seed=0)
    check_assumptions("Simulation", R_sim, [f"Asset {i+1}" for i in range(R_sim.shape[1])])
    check_assumptions("Real data", clean_returns(args.refresh), TICKERS)

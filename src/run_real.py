"""Iteration 3, Step 5: real-data verification; optional Step 7 frontier.

Usage: python src/run_real.py [--refresh] [--plot]
"""
import argparse
from pathlib import Path

import numpy as np

from load_data import download_prices, TICKERS, START, TRADING_DAYS
from methods import (
    my_estimate_mu_sigma,
    my_annualize,
    my_min_variance,
    my_portfolio_stats,
    my_frontier_weights,
    my_frontier_constants,
    my_frontier_variance,
)
from verify import report

END = "2026-11-13"
TOL = 1e-12
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"


def clean_returns(refresh=False):
    prices = download_prices(refresh).loc[:, TICKERS].sort_index()
    
    if prices.index.has_duplicates:
        raise ValueError("Duplicate dates in prices; review the raw dataset.")
    
    total = len(prices)
    prices = prices.loc[START:END]
    in_window = len(prices)
    
    print("Price rows outside the original window:", total - in_window)
    print("Price rows in window:", in_window)
    print("Missing prices by asset:\n" + prices.isna().sum().to_string())

    # Same complete-case alignment and simple returns used in Iteration 2.
    prices = prices.dropna()
    print("Rows removed for missing prices:", in_window - len(prices))
    
    if len(prices) < 2 or not np.isfinite(prices.to_numpy()).all():
        raise ValueError("Insufficient or invalid adjusted closing prices.")
    
    if (prices <= 0).any().any():
        raise ValueError("Non-positive adjusted closing prices.")
    
    R = prices.pct_change(fill_method=None).dropna()
    
    print("Rows removed when computing returns:", len(prices) - len(R))
    print("Return rows:", len(R), "; assets:", R.shape[1])
    print("Return window:", R.index.min().date(), "to", R.index.max().date())
    print("Missing returns:", int(R.isna().sum().sum()))
    print("Transform: simple daily returns in decimals; no train/test split.")
    
    if R.index.min().strftime("%Y-%m-%d") != "2020-10-01" or R.index.max().strftime("%Y-%m-%d") != END:
        raise ValueError("Return dates do not match the Iteration 2 window.")
    
    if len(R) != 1538:
        raise ValueError(f"Expected 1538 return rows from Iteration 2; found {len(R)}.")
    
    return R.to_numpy(dtype=float)


def plot_frontier(mu, Sigma, w):
    """Plot the efficient branch using the annualized sample moments."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a, b, c, d = my_frontier_constants(mu, Sigma)
    r_min = b / a
    var_min = 1 / a
    targets = np.linspace(r_min, mu.max(), 150)
    frontier_var = my_frontier_variance(mu, Sigma, targets)
    # Independently evaluate the quadratic form at every plotted target.
    direct_var = []
    for r in targets:
        wr = my_frontier_weights(mu, Sigma, r)
        assert abs(wr.sum() - 1) <= TOL and abs(wr @ mu - r) <= TOL
        direct_var.append(wr @ Sigma @ wr)
    abs_err, _ = report("frontier variance vs direct", frontier_var, direct_var)
    assert abs_err <= TOL
    assert abs(w @ mu - r_min) <= TOL and abs(w @ Sigma @ w - var_min) <= TOL

    print("\nAnnualized covariance matrix (asset order above):")
    print(np.array2string(Sigma, precision=6, suppress_small=True, max_line_width=140))
    print(f"Annualized frontier constants: a={a:.10f}, b={b:.10f}, "
          f"c={c:.10f}, d={d:.10f}")
    print(f"GMV vertex: return={r_min:.10f}, volatility={np.sqrt(var_min):.10f}")
    print("Sample efficient frontier points (annual return, volatility):")
    for r in [r_min, 0.30, 0.40, 0.50, 0.60, float(mu.max())]:
        print(f"  {r:.10f}  {np.sqrt(my_frontier_variance(mu, Sigma, r)):.10f}")

    FIG_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(100 * np.sqrt(frontier_var), 100 * targets,
            label="Efficient frontier", color="#1764ab", linewidth=2.2)
    ax.scatter(100 * np.sqrt(np.diag(Sigma)), 100 * mu,
               color="#657584", label="Individual assets", s=30, zorder=3)
    
    for ticker, vol, ret in zip(TICKERS, np.sqrt(np.diag(Sigma)), mu):
        ax.annotate(ticker, (100 * vol, 100 * ret),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
        
    ax.scatter(100 * np.sqrt(var_min), 100 * r_min, color="#c04b25",
               marker="*", s=180, label="Global minimum variance", zorder=4)
    ax.set_xlabel("Annualized volatility (%)")
    ax.set_ylabel("Annualized expected return (%)")
    ax.set_title("Efficient frontier: nine technology stocks")
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = FIG_DIR / "real_efficient_frontier.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--plot", action="store_true", help="generate the Step 7 efficient frontier")
    args = parser.parse_args()
    R = clean_returns(args.refresh)

    # From-scratch estimates and weights (Iteration 2 algorithm).
    mu, Sigma = my_estimate_mu_sigma(R)
    mu_ann, Sigma_ann = my_annualize(mu, Sigma, TRADING_DAYS)
    w = my_min_variance(Sigma_ann)
    expected_return, variance = my_portfolio_stats(w, mu_ann, Sigma_ann)

    # Independent NumPy references: library functions are checks only.
    mu_lib = R.mean(axis=0)
    Sigma_lib = np.cov(R, rowvar=False, ddof=1)
    mu_ann_lib = TRADING_DAYS * mu_lib
    Sigma_ann_lib = TRADING_DAYS * Sigma_lib
    one = np.ones(R.shape[1])
    x_lib = np.linalg.solve(Sigma_ann_lib, one)
    w_lib = x_lib / x_lib.sum()

    print("\nAnnualized estimates and minimum-variance weights:")
    
    for ticker, m, v, wi in zip(TICKERS, mu_ann, np.diag(Sigma_ann), w):
        print(f"{ticker:5s} mean={m:.8f}  vol={np.sqrt(v):.8f}  weight={wi:+.8f}")
    
    print(f"Expected annual return: {expected_return:.10f}")
    print(f"Annual variance: {variance:.10f}")
    print(f"Annual volatility: {np.sqrt(variance):.10f}")
    print(f"Budget residual: {w.sum()-1:+.2e}")

    print("\nVerification:")
    
    checks = [
        report("daily mean vs numpy", mu, mu_lib),
        report("daily covariance vs numpy", Sigma, Sigma_lib),
        report("annual mean vs numpy", mu_ann, mu_ann_lib),
        report("annual covariance vs numpy", Sigma_ann, Sigma_ann_lib),
        report("GMV weights vs numpy.solve", w, w_lib),
    ]

    # Independent full KKT system to verify target-return portfolio weights.
    r = float(mu_ann.mean())
    wr = my_frontier_weights(mu_ann, Sigma_ann, r)
    n = len(w)
    
    K = np.zeros((n + 2, n + 2))
    K[:n, :n] = Sigma_ann_lib
    K[:n, n], K[:n, n+1] = -mu_ann_lib, -one
    K[n, :n], K[n+1, :n] = mu_ann_lib, one
    
    wr_lib = np.linalg.solve(K, np.r_[np.zeros(n), r, 1.0])[:n]
    checks.append(report("target weights vs numpy KKT", wr, wr_lib))
    
    print(f"Target r: {r:.10f}; budget residual: {wr.sum()-1:+.2e}; "
          f"return residual: {wr @ mu_ann-r:+.2e}")
    
    assert all(a <= TOL for a, _ in checks), "Library comparison failed"
    assert abs(w.sum() - 1) <= TOL
    assert abs(wr.sum() - 1) <= TOL and abs(wr @ mu_ann - r) <= TOL
    
    print("All real-data verification checks passed.")
    if args.plot:
        plot_frontier(mu_ann, Sigma_ann, w)


if __name__ == "__main__":
    main()

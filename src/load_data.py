"""Load and describe the real dataset: daily returns of 9 large US tech stocks.

Tickers: AAPL NVDA GOOGL META AMZN AMD PLTR MU INTC
Source:  Yahoo Finance via the `yfinance` package (adjusted close prices, so
         dividends and splits are accounted for).
Window:  2020-09-30 (PLTR's first trading day) to the latest available day, so all
         9 assets are observed on the same dates.

One row = one trading day t; one column = one stock i;
one entry R[t, i] = (p[t, i] - p[t-1, i]) / p[t-1, i], the simple daily return.

Usage (from the project root):
    python src/load_data.py             # download (or use cached CSV), describe, plot
    python src/load_data.py --refresh   # force a new download
    python src/load_data.py --end 2026-10-31
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from portfolio import sample_moments, is_psd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
PROC_DIR = os.path.join(ROOT, "data", "processed")
FIG_DIR = os.path.join(ROOT, "figures")

TICKERS = ["AAPL", "NVDA", "GOOGL", "META", "AMZN", "AMD", "PLTR", "MU", "INTC"]
START = "2020-09-30"
PRICE_CSV = os.path.join(RAW_DIR, "prices_adjclose.csv")
TRADING_DAYS = 252


def download_prices(refresh=False):
    """Adjusted close prices, one column per ticker. Cached in data/raw/."""
    if os.path.exists(PRICE_CSV) and not refresh:
        return pd.read_csv(PRICE_CSV, index_col=0, parse_dates=True)
    import yfinance as yf  # only used to fetch the raw data
    os.makedirs(RAW_DIR, exist_ok=True)
    px = yf.download(TICKERS, start=START, auto_adjust=True, progress=False)["Close"]
    px = px[TICKERS]
    px.index.name = "date"
    px.to_csv(PRICE_CSV)
    return px


def load_returns(end=None, refresh=False):
    """Return (prices, returns). Returns are simple daily returns as decimals."""
    px = download_prices(refresh)
    if end:
        px = px.loc[:end]
    print("Missing prices per ticker before alignment:\n" + px.isna().sum().to_string())
    px = px.dropna()                   # keep only days where all 9 trade
    R = px.pct_change().dropna()       # R_t = (p_t - p_{t-1}) / p_{t-1}
    return px, R


def describe(px, R):
    print("\n=== Daily returns, 9 tech stocks ===")
    print(f"Rows (trading days T): {R.shape[0]}   Columns (assets N): {R.shape[1]}")
    print(f"Period: {R.index.min():%Y-%m-%d} to {R.index.max():%Y-%m-%d}")
    print("\nVariable types:\n" + R.dtypes.to_string())
    print("\nMissing returns per column:\n" + R.isna().sum().to_string())

    stats = pd.DataFrame({
        "mean_daily": R.mean(),
        "std_daily": R.std(),
        "ann_mean": R.mean() * TRADING_DAYS,
        "ann_vol": R.std() * np.sqrt(TRADING_DAYS),
        "min": R.min(),
        "max": R.max(),
        "skew": R.skew(),
        "kurtosis": R.kurt(),          # excess kurtosis; 0 for a normal
    })
    print("\nSummary statistics:\n" + stats.round(4).to_string())

    mu_hat, S_hat = sample_moments(R.values)
    ok, lam = is_psd(S_hat)
    print(f"\nSample covariance PSD? {ok}; eigenvalues in [{lam.min():.2e}, {lam.max():.2e}]")
    print(f"Condition number of Sigma_hat: {lam.max() / lam.min():.1f}")
    return stats


def plots(px, R):
    os.makedirs(FIG_DIR, exist_ok=True)

    # 1) Growth of $1 (log scale)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    (px / px.iloc[0]).plot(ax=ax, logy=True, lw=1)
    ax.set_title(f"Growth of $1 since {px.index[0]:%Y-%m-%d} (log scale)")
    ax.set_ylabel("Wealth"); ax.set_xlabel("")
    ax.legend(ncol=5, fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "data_cumulative.png"), dpi=150)
    plt.close(fig)

    # 2) Annualized risk vs return, and correlation heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    m, s = R.mean() * TRADING_DAYS, R.std() * np.sqrt(TRADING_DAYS)
    ax1.scatter(s, m)
    for name in R.columns:
        ax1.annotate(name, (s[name], m[name]), fontsize=8, xytext=(3, 3),
                     textcoords="offset points")
    ax1.set_xlabel("Annualized std. dev."); ax1.set_ylabel("Annualized mean return")
    ax1.set_title("Risk vs. return by stock")
    im = ax2.imshow(R.corr().values, vmin=0, vmax=1, cmap="viridis")
    ax2.set_xticks(range(R.shape[1]), R.columns, rotation=90, fontsize=8)
    ax2.set_yticks(range(R.shape[1]), R.columns, fontsize=8)
    ax2.set_title("Correlation of daily returns")
    fig.colorbar(im, ax=ax2, fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "data_risk_return_corr.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved figures to {FIG_DIR}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--end", default=None, help="last date, e.g. 2026-10-31")
    p.add_argument("--refresh", action="store_true", help="re-download prices")
    args = p.parse_args()
    prices, returns = load_returns(args.end, args.refresh)
    describe(prices, returns)
    plots(prices, returns)
    os.makedirs(PROC_DIR, exist_ok=True)
    returns.to_csv(os.path.join(PROC_DIR, "returns_daily.csv"))

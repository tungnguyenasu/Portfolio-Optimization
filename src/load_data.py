"""Load and describe the real dataset: Kenneth French 10 Industry Portfolios (monthly).

Source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
File:   10_Industry_Portfolios_CSV.zip

One row = one month t; one column = one industry portfolio i;
one entry R[t, i] = that portfolio's simple return in month t (as a decimal).

Usage (from the project root):
    python src/load_data.py                 # download if needed, describe, save plots
    python src/load_data.py --start 1990-01 # restrict the sample period
"""
import argparse
import io
import os
import urllib.request
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from portfolio import sample_moments, is_psd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
FIG_DIR = os.path.join(ROOT, "figures")
URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
       "10_Industry_Portfolios_CSV.zip")
ZIP_PATH = os.path.join(RAW_DIR, "10_Industry_Portfolios_CSV.zip")


def download(force=False):
    """Download the zip into data/raw/ (skipped if it is already there)."""
    os.makedirs(RAW_DIR, exist_ok=True)
    if os.path.exists(ZIP_PATH) and not force:
        return ZIP_PATH
    print(f"Downloading {URL}")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(ZIP_PATH, "wb") as f:
        f.write(resp.read())
    return ZIP_PATH


def _read_raw_text():
    """Return the CSV text, from the zip or from a CSV placed by hand in data/raw/."""
    csvs = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(".csv")] \
        if os.path.isdir(RAW_DIR) else []
    if csvs:
        with open(os.path.join(RAW_DIR, csvs[0]), encoding="latin-1") as f:
            return f.read()
    with zipfile.ZipFile(download()) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        return z.read(name).decode("latin-1")


def load_returns(start=None, end=None):
    """Parse the FIRST table in the file (value-weighted monthly returns).

    The file has a text header, then a line ',NoDur,Durbl,...' followed by rows
    'YYYYMM, r1, ..., r10' (percent) until a blank line. -99.99 / -999 mark missing.
    Returns a DataFrame indexed by month with returns as decimals.
    """
    lines = _read_raw_text().splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.strip().startswith(",") and "NoDur" in l)
    rows = []
    for l in lines[hdr + 1:]:
        if not l.strip() or not l.strip()[0].isdigit():
            break
        rows.append(l)
    cols = ["date"] + [c.strip() for c in lines[hdr].split(",")[1:]]
    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None, names=cols)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m")
    df = df.set_index("date").replace([-99.99, -999], np.nan) / 100.0
    if start:
        df = df.loc[start:]
    if end:
        df = df.loc[:end]
    return df


def describe(df):
    print("=== 10 Industry Portfolios, value-weighted monthly returns ===")
    print(f"Rows (months): {df.shape[0]}   Columns (assets): {df.shape[1]}")
    print(f"Period: {df.index.min():%Y-%m} to {df.index.max():%Y-%m}")
    print("\nVariable types:\n" + df.dtypes.to_string())
    print("\nMissing values per column:\n" + df.isna().sum().to_string())
    print("\nSummary statistics (monthly, decimal):")
    print(df.describe().T[["mean", "std", "min", "max"]].round(4).to_string())

    mu_hat, S_hat = sample_moments(df.dropna().values)
    ok, lam = is_psd(S_hat)
    print(f"\nSample covariance PSD? {ok}; eigenvalues in [{lam.min():.2e}, {lam.max():.2e}]")
    print(f"Condition number of Sigma_hat: {lam.max() / lam.min():.1f}")


def plots(df):
    os.makedirs(FIG_DIR, exist_ok=True)

    # 1) Cumulative growth of $1 in each industry
    fig, ax = plt.subplots(figsize=(8, 4.5))
    (1 + df.fillna(0)).cumprod().plot(ax=ax, logy=True, lw=1)
    ax.set_title("Growth of $1, 10 industry portfolios (log scale)")
    ax.set_ylabel("Wealth"); ax.set_xlabel("")
    ax.legend(ncol=5, fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "data_cumulative.png"), dpi=150)
    plt.close(fig)

    # 2) Mean vs. standard deviation, plus correlation heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    m, s = df.mean() * 12, df.std() * np.sqrt(12)
    ax1.scatter(s, m)
    for name in df.columns:
        ax1.annotate(name, (s[name], m[name]), fontsize=8, xytext=(3, 3),
                     textcoords="offset points")
    ax1.set_xlabel("Annualized std. dev."); ax1.set_ylabel("Annualized mean return")
    ax1.set_title("Risk vs. return by industry")
    im = ax2.imshow(df.corr().values, vmin=0, vmax=1, cmap="viridis")
    ax2.set_xticks(range(df.shape[1]), df.columns, rotation=90, fontsize=8)
    ax2.set_yticks(range(df.shape[1]), df.columns, fontsize=8)
    ax2.set_title("Correlation matrix")
    fig.colorbar(im, ax=ax2, fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "data_risk_return_corr.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved figures to {FIG_DIR}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default=None, help="e.g. 1990-01")
    p.add_argument("--end", default=None, help="e.g. 2024-12")
    args = p.parse_args()
    data = load_returns(args.start, args.end)
    describe(data)
    plots(data)
    os.makedirs(os.path.join(ROOT, "data", "processed"), exist_ok=True)
    data.to_csv(os.path.join(ROOT, "data", "processed", "industry10_monthly.csv"))

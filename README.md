# DS5020 Final Project — Portfolio Optimization (Project 10)

Mean–variance portfolio optimization built from first principles: portfolio risk
as a quadratic form, the global minimum-variance portfolio via Lagrange multipliers,
the target-return 2×2 multiplier system, and the efficient frontier as a parabola.

## Repository layout

```
src/        Python code (numpy, pandas, matplotlib, scipy.stats only)
  portfolio.py   core math: sample moments, w*, 2x2 system, frontier (from scratch)
  load_data.py   download 9 stock prices (yfinance), describe returns, save figures
  simulate.py    simulation with known truth; checks Q1–Q5 numerically
data/       raw/ (downloaded, NOT in Git) and processed/ (generated)
figures/    plots produced by the scripts
docs/       report.tex (Overleaf source), compiled PDF, project_plan.xlsx
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

**Daily returns of 9 tech stocks** — AAPL, NVDA, GOOGL, META, AMZN, AMD, PLTR, MU, INTC —
from Yahoo Finance (adjusted close, via `yfinance`), 2020-09-30 (PLTR's first trading
day) to today. One row = one trading day; one column = one stock; one entry =
that stock's simple daily return `(p_t - p_{t-1}) / p_{t-1}`.

`python src/load_data.py` downloads the prices into `data/raw/prices_adjclose.csv`
on first run and reuses that file afterwards (`--refresh` to re-download).
`yfinance` is used only to fetch the raw prices; all analysis uses numpy/pandas.

SpaceX (SPCX) was left out: it IPO'd on 2026-06-12, so it has too short a history.

Raw files are ignored by Git (see `.gitignore`).

## Run

```bash
python src/simulate.py                  # simulation check (seed 0) -> figures/sim_frontier.png
python src/load_data.py                 # describe real data -> figures/data_*.png
```

## Report

`docs/report.tex` is the Iteration 2 report. Upload it to Overleaf together with the
three PNGs from `figures/` (put them in a `figures/` folder in Overleaf). Every
derivation has an *Owner* / *Checked by* line. Use `\by{rule}` to name the rule at each step.

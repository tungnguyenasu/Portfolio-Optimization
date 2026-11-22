# DS5020 Final Project — Portfolio Optimization (Project 10)

Mean–variance portfolio optimization built from first principles: portfolio risk
as a quadratic form, the global minimum-variance portfolio via Lagrange multipliers,
the target-return 2×2 multiplier system, and the efficient frontier as a parabola.

## Repository layout

```
src/        Python code (numpy, pandas, matplotlib, scipy.stats only)
  methods.py           From-scratch implementation of all seven algorithm steps
  small_example.py     Exact four-period, two-asset worked example (steps 1–7)
  simulate.py          simulation with known truth; checks Q1–Q5 numerically
  check_simulation.py  Truth/library comparisons, sample-size study, stress tests
  run_real.py          Real-data cleaning and verification; optional efficient frontier
  check_assumptions.py Assumption checks on simulated and real data
  verify.py            Shared report() helper from the Iteration 3 handout
  load_data.py         download 9 stock prices (yfinance), describe returns, save figures
  portfolio.py         core math: sample moments, w*, 2x2 system, frontier (from scratch)
data/
  raw/                 Cached Yahoo Finance prices
  processed/           Generated returns from load_data.py
figures/               plots produced by the scripts
docs/
  report.tex           Iteration 2 Overleaf source
  report_it3.tex       Iteration 3 Overleaf source
  project_plan.xlsx    Project task plan
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Iteration 3: reproduce the results

```bash
python src/small_example.py
python src/check_simulation.py
python src/run_real.py
python src/check_assumptions.py
python src/run_real.py --plot
```

- **Small example:** prints each intermediate matrix and vector using exact
  fractions, checks the budget and stationarity conditions, and compares with
  a separate scalar solution and NumPy.
- **Simulation:** uses the original four-asset model (T=2000, seed 0), verifies
  against known truth and library solves, examines sample sizes 50, 500, 2000,
  and 5000 with 200 replications each, and tests a short sample and a duplicate
  asset. Generates `figures/sim_error_vs_T.png`.
- **Real data:** cleans the original fixed date window, estimates and annualizes
  the moments, and checks minimum-variance and target-return weights against NumPy.
- **Assumptions:** `check_assumptions.py` checks all six assumptions on both datasets.
- **Efficient frontier:** `run_real.py --plot` generates `figures/real_efficient_frontier.png` for Project 10, Question 5.

The shared `report()` helper prints maximum absolute differences and relative
errors. Real-data library errors must be at most 1e-12; simulation library errors
are also at rounding level.

## Data and reproducibility

Nine technology stocks: AAPL, NVDA, GOOGL, META, AMZN, AMD, PLTR, MU, INTC.
Yahoo Finance adjusted closing prices are aligned over **2020-09-30 through
2026-11-13**, producing 1,538 daily returns from 2020-10-01 through 2026-11-13.
Each row is one trading day; each column is one stock. Returns are decimals,
computed as `(p_t - p_(t-1)) / p_(t-1)`. Annualization uses 252 trading days.
No train/test split is needed for in-sample optimization.

The first run of `run_real.py` downloads and caches prices in
`data/raw/prices_adjclose.csv` if missing; later runs reuse the cache. It filters
to the original window and checks the expected sample size and dates.
Use `python src/run_real.py --refresh` only if a fresh Yahoo Finance download is
needed; vendor price adjustments may change historical estimates slightly.
An internet connection is needed for the initial download.

For the original descriptive analysis:

```bash
python src/load_data.py
python src/simulate.py
```

## Report

`docs/report.tex` is the Iteration 2 report. Upload it to Overleaf together with the PNGs from `figures/`. 
`docs/report_it3.tex` is the Iteration 3 report.
Every derivation has an *Owner* / *Checked by* line. Use `\by{rule}` to name the rule at each step.

# DS5020 Final Project — Portfolio Optimization (Project 10)

Mean–variance portfolio optimization built from first principles: portfolio risk
as a quadratic form, the global minimum-variance portfolio via Lagrange multipliers,
the target-return 2×2 multiplier system, and the efficient frontier as a parabola.

## Repository layout

```
src/        Python code (numpy, pandas, matplotlib, scipy.stats only)
  portfolio.py   core math: sample moments, w*, 2x2 system, frontier (from scratch)
  load_data.py   download + describe the real dataset, save data figures
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

**Kenneth French Data Library — 10 Industry Portfolios, monthly.**
One row = one month; one column = one industry portfolio; one entry = that
portfolio's return that month.

`python src/load_data.py` downloads it automatically into `data/raw/`. If the
download fails, get it by hand:

1. Go to <https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html>
2. Under *Industry Portfolios*, download **10 Industry Portfolios** (CSV).
3. Put the `.zip` (or the extracted `.CSV`) in `data/raw/`.

Raw files are ignored by Git (see `.gitignore`).

## Run

```bash
python src/simulate.py                  # simulation check (seed 0) -> figures/sim_frontier.png
python src/load_data.py --start 1990-01 # describe real data -> figures/data_*.png
```

## Report

`docs/report.tex` is the Iteration 2 report. Upload it to Overleaf together with the
three PNGs from `figures/` (put them in a `figures/` folder in Overleaf). Every
derivation has an *Owner* / *Checked by* line. Use `\by{rule}` to name the rule at each step.

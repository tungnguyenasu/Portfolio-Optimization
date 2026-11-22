"""Iteration 3 - Simulation test: recover the known truth.

Runs the from-scratch functions in methods.py on the Iteration 2 simulation
(4 assets, mu and Sigma known) and makes the two comparisons the handout asks for:

  (A) with a LIBRARY (numpy.linalg.solve on the SAME Sigma_hat):
      tests that our code computes the formula correctly -> rounding level.
  (B) with the TRUTH (w* from the true Sigma):
      tests that the METHOD recovers the truth -> sampling-error level,
      shrinking like 1/sqrt(T).

Then two stress tests that break an assumption on purpose:
  (C1) T = 60 periods          -> Sigma_hat is noisy, weights unstable (A3 / Q4).
  (C2) an exact duplicate asset -> Sigma_hat singular, w* not unique   (A2).

Requires from methods.py (Step 1):
    my_estimate_mu_sigma(R)      -> m (N,), S (N, N)      sample mean, covariance /(T-1)
    my_min_variance(S)           -> w (N,)                 S^{-1}1 / 1'S^{-1}1, own Gauss. elim.
    my_frontier_weights(m, S, r) -> w (N,)                 target-return weights (Thm 4)

numpy.linalg is used ONLY to produce reference answers for checking.

Usage (from the project root):  python src/check_simulation.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from methods import my_estimate_mu_sigma, my_min_variance, my_frontier_weights
from simulate import simulate_returns
from verify import report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")

SEED = 0              # main run, same as Iteration 2
SEED_T60 = 1          # Iteration 2 used seed 1 for the T = 60 run; kept for consistency
T_MAIN = 2000         # Iteration 2 sample size
T_GRID = [50, 500, 2000, 5000]
N_REPS = 200          # replications used to estimate standard errors
ONE = np.ones(4)


def section(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


# ---- reference answers (library, check only) ---------------------------------
def lib_min_variance(S):
    """w = S^{-1} 1 / (1' S^{-1} 1) using numpy.linalg.solve."""
    x = np.linalg.solve(S, np.ones(len(S)))
    return x / x.sum()


def lib_frontier_weights(m, S, r):
    """Solve the full KKT system of Theorem 4 with numpy, independent of our 2x2 route:
        [ S   -m  -1 ] [w ]   [0]
        [ m'   0   0 ] [l1] = [r]
        [ 1'   0   0 ] [l2]   [1]
    """
    N = len(m)
    K = np.zeros((N + 2, N + 2))
    K[:N, :N] = S
    K[:N, N], K[:N, N + 1] = -m, -1
    K[N, :N], K[N + 1, :N] = m, 1
    rhs = np.r_[np.zeros(N), r, 1.0]
    return np.linalg.solve(K, rhs)[:N]


# ---- (A) library comparison --------------------------------------------------
def part_a():
    section(f"(A) Library comparison  (T = {T_MAIN}, seed = {SEED})")
    R, mu, Sigma = simulate_returns(T_MAIN, SEED)
    m, S = my_estimate_mu_sigma(R)

    report("mu_hat vs numpy mean", m, R.mean(axis=0))
    report("Sigma_hat vs numpy cov", S, np.cov(R.T))      # np.cov also divides by T-1

    w = my_min_variance(S)
    report("w_hat vs numpy.linalg.solve", w, lib_min_variance(S))
    print(f"{'1^T w - 1':32s} {w.sum() - 1:+.2e}")

    worst = 0.0
    for r in np.linspace(0.04, 0.14, 6):
        wr = my_frontier_weights(m, S, r)
        worst = max(worst, np.abs(wr - lib_frontier_weights(m, S, r)).max())
    print(f"{'frontier w(r) vs numpy KKT':32s} max|diff| = {worst:.2e}   (6 targets r)")

    lam = np.linalg.eigvalsh(S)
    print(f"\nlambda_min(S) = {lam.min():.3e} > 0 : {lam.min() > 0}"
          f"     cond(S) = {np.linalg.cond(S):.2f}")
    return R, mu, Sigma, m, S, w


# ---- (B) truth comparison ----------------------------------------------------
def part_b(mu, Sigma, m, S, w):
    section(f"(B) Truth comparison  (T = {T_MAIN}, seed = {SEED})")
    w_true = lib_min_variance(Sigma)

    # mu_hat: the standard error has a closed form, SE(mu_hat_i) = sqrt(Sigma_ii / T)
    se_mu = np.sqrt(np.diag(Sigma) / T_MAIN)
    z_mu = (m - mu) / se_mu
    print("mu true        :", mu)
    print("mu hat         :", m.round(4))
    print("(mu_hat-mu)/SE :", z_mu.round(2), "  all |z| < 3:", bool(np.all(np.abs(z_mu) < 3)))

    print("\nw* true        :", w_true.round(4))
    print("w hat          :", w.round(4))
    report("w_hat vs true w*", w, w_true)
    print(f"{'||w_hat - w*||_2':32s} {np.linalg.norm(w - w_true):.4f}")
    return w_true


def replicate(T, w_true, seeds):
    """Run the whole pipeline on independent data sets; return all w_hat (reps, N)."""
    W = np.empty((len(seeds), 4))
    for k, s in enumerate(seeds):
        R, _, _ = simulate_returns(T, s)
        W[k] = my_min_variance(my_estimate_mu_sigma(R)[1])
    return W


def part_b_vs_T(w_true):
    section(f"(B) Error vs T   ({N_REPS} replications per T, seeds 1000..{999 + N_REPS})")
    # w_hat has no simple closed-form SE, so estimate it by repetition:
    # SE_i(T) = standard deviation of w_hat_i over independent simulated data sets.
    seeds = range(1000, 1000 + N_REPS)
    print(f"{'T':>6} {'||w_hat-w*|| (seed 0)':>22} {'RMSE over reps':>15} "
          f"{'max |err|/SE':>13} {'within 3 SE':>12}")
    rows = []
    for T in T_GRID:
        W = replicate(T, w_true, seeds)
        se = W.std(axis=0, ddof=1)
        rmse = np.sqrt(np.mean(np.sum((W - w_true) ** 2, axis=1)))
        R, _, _ = simulate_returns(T, SEED)
        w0 = my_min_variance(my_estimate_mu_sigma(R)[1])
        z = np.abs(w0 - w_true) / se
        rows.append((T, np.linalg.norm(w0 - w_true), rmse, z.max()))
        print(f"{T:6d} {rows[-1][1]:22.4f} {rmse:15.4f} {z.max():13.2f} {str(z.max() < 3):>12}")

    Ts = np.array([r[0] for r in rows], float)
    rmses = np.array([r[2] for r in rows])
    slope, intercept = np.polyfit(np.log(Ts), np.log(rmses), 1)
    print(f"\nlog-log slope of RMSE vs T = {slope:.3f}   (theory: -0.5, the 1/sqrt(T) rate)")

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.loglog(Ts, rmses, "o-", label=f"RMSE over {N_REPS} replications")
    ax.loglog(Ts, [r[1] for r in rows], "s--", alpha=0.7, label="single run (seed 0)")
    ax.loglog(Ts, np.exp(intercept) * Ts ** -0.5, ":", c="gray", label=r"slope $-1/2$ reference")
    ax.set_xlabel("Sample size $T$")
    ax.set_ylabel(r"$\|\hat{w} - w^*\|_2$")
    ax.set_title(f"Error against the truth (fitted slope {slope:.2f})")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "sim_error_vs_T.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
    return rows, slope


# ---- (C) stress tests --------------------------------------------------------
def part_c1(w_true, w_2000):
    section(f"(C1) Stress test: short sample, T = 60 (seed = {SEED_T60})")
    R60, _, _ = simulate_returns(60, SEED_T60)
    m60, S60 = my_estimate_mu_sigma(R60)
    w60 = my_min_variance(S60)
    report("w_hat (T=60) vs true w*", w60, w_true)
    report("w_hat (T=2000) vs true w*", w_2000, w_true)
    e60, e2000 = np.linalg.norm(w60 - w_true), np.linalg.norm(w_2000 - w_true)
    print(f"||error||: T=60 -> {e60:.4f}, T=2000 -> {e2000:.4f}  (ratio {e60 / e2000:.1f}x;"
          f" 1/sqrt(T) predicts {np.sqrt(2000 / 60):.1f}x)")
    print(f"cond(S_hat): T=60 -> {np.linalg.cond(S60):.2f}, T=2000 -> "
          f"{np.linalg.cond(my_estimate_mu_sigma(simulate_returns(2000, SEED)[0])[1]):.2f}")
    return e60


def part_c2():
    section("(C2) Stress test: exact duplicate asset (breaks A2: Sigma positive definite)")
    R, _, _ = simulate_returns(T_MAIN, SEED)
    R5 = np.column_stack([R, R[:, 0]])                 # asset 5 = copy of asset 1
    _, S5 = my_estimate_mu_sigma(R5)
    lam = np.linalg.eigvalsh(S5)
    cond5 = np.linalg.cond(S5)
    print(f"eigenvalues of S_hat (5 assets): {lam}")
    print(f"lambda_min = {lam.min():.2e}  (0 in exact arithmetic),  cond(S_hat) = {cond5:.2e}")
    print("Why: column 5 of R equals column 1, so R_c e = 0 for e = e_1 - e_5, hence "
          "S_hat e = 0 and S_hat is singular.")

    # Non-uniqueness: moving weight between the two identical assets changes nothing.
    w_a = np.r_[lib_min_variance(my_estimate_mu_sigma(R)[1]), 0.0]   # all in asset 1
    w_b = w_a.copy(); w_b[0], w_b[4] = w_a[0] / 2, w_a[0] / 2        # split with its copy
    print(f"\nTwo DIFFERENT budget-feasible portfolios with the SAME variance:")
    print(f"  w_a = {w_a.round(4)}  var = {w_a @ S5 @ w_a:.10f}")
    print(f"  w_b = {w_b.round(4)}  var = {w_b @ S5 @ w_b:.10f}")
    print("so the minimiser is not unique and the formula S^{-1}1/(1'S^{-1}1) breaks down.")

    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            w5 = my_min_variance(S5)
        if not np.all(np.isfinite(w5)):
            print(f"\nmy_min_variance on the singular S_hat returned {w5}")
            print("  (elimination reached a zero pivot: 0/0 = NaN, no unique solution)")
        else:
            print(f"\nmy_min_variance on the singular S_hat returned {w5.round(3)}")
            print("  (meaningless: elimination divided by a pivot that is only rounding noise)")
    except (ZeroDivisionError, np.linalg.LinAlgError, ValueError, FloatingPointError) as err:
        print(f"\nmy_min_variance on the singular S_hat failed: {type(err).__name__}: {err}")
    return lam.min(), cond5


def main():
    R, mu, Sigma, m, S, w = part_a()
    w_true = part_b(mu, Sigma, m, S, w)
    part_b_vs_T(w_true)
    part_c1(w_true, w)
    part_c2()


if __name__ == "__main__":
    main()

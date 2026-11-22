"""Iteration 3: exact two-asset worked example using the shared methods."""

from fractions import Fraction

import numpy as np
import methods


def main():
    R = np.array([[Fraction(0), Fraction(0)],
                  [Fraction(0), Fraction(1, 5)],
                  [Fraction(1, 5), Fraction(1, 5)],
                  [Fraction(1, 5), Fraction(1, 5)]], dtype=object)
    T, N = R.shape
    one_T = np.array([Fraction(1)] * T, dtype=object)
    one = np.array([Fraction(1)] * N, dtype=object)

    print("Tiny exact example: four periods, two assets")
    print("R =", R)
    print("T =", T, "; N =", N)
    print("one_T =", one_T, "; one =", one)

    # Step 1: mu_hat = R.T @ one_T / T.
    print("\nStep 1: Sample mean")
    print("R.T @ one_T =", R.T @ one_T)
    mu = methods.my_sample_mean(R)
    print("mu_hat =", mu)

    # Step 2: R_c = R - one_T @ mu_hat.T (outer product).
    print("\nStep 2: Center the data")
    print("one_T mu_hat.T =", np.outer(one_T, mu))
    R_c = methods.my_center(R, mu)
    print("R_c =", R_c)

    # Step 3: Sigma_hat = R_c.T @ R_c / (T - 1).
    print("\nStep 3: Sample covariance")
    print("R_c.T @ R_c =", R_c.T @ R_c)
    Sigma = methods.my_sample_cov(R_c)
    print("Sigma_hat =", Sigma)
    det = Sigma[0, 0] * Sigma[1, 1] - Sigma[0, 1] * Sigma[1, 0]
    print("Positive definite: leading minor =", Sigma[0, 0], "; det =", det)

    # Step 4: solve Sigma x = one and Sigma y = mu together.
    print("\nStep 4: Gaussian elimination with partial pivoting")
    XY = methods.my_gauss_solve(Sigma, np.column_stack([one, mu]), verbose=True)
    x, y = XY[:, 0], XY[:, 1]
    print("x =", x, "; y =", y)
    print("Sigma x =", Sigma @ x, "; Sigma y =", Sigma @ y)

    # Step 5: compute the frontier constants.
    print("\nStep 5: Frontier constants")
    a, b, c, d = methods.my_frontier_constants(mu, Sigma)
    print("a =", a, "; b =", b, "; c =", c, "; d =", d)

    # Step 6: minimum-variance portfolio and risk.
    print("\nStep 6: Minimum-variance portfolio")
    w_star = methods.my_min_variance(Sigma)
    mean, variance = methods.my_portfolio_stats(w_star, mu, Sigma)
    print("w* =", w_star, "; expected return =", mean)
    print("Sigma w* =", Sigma @ w_star)
    print("variance =", variance, "; 1/a =", 1 / a)

    # Step 7: solve the target-return multiplier system.
    print("\nStep 7: Target-return portfolio")
    r = Fraction(1, 5)
    K = np.array([[c, b], [b, a]], dtype=object)
    rhs = np.array([r, Fraction(1)], dtype=object)
    l1, l2 = methods.my_gauss_solve(K, rhs, verbose=True)
    w_k = methods.my_frontier_weights(mu, Sigma, r)
    target_mean, target_var = methods.my_portfolio_stats(w_k, mu, Sigma)
    frontier_var = methods.my_frontier_variance(mu, Sigma, r)
    print("r =", r, "; lambda1 =", l1, "; lambda2 =", l2)
    print("lambda1 y =", l1 * y, "; lambda2 x =", l2 * x)
    print("w(r) =", w_k, "; return =", target_mean)
    print("variance =", target_var, "; frontier variance =", frontier_var)
    print("Sigma w(r) =", Sigma @ w_k)
    print("lambda1 mu + lambda2 one =", l1 * mu + l2 * one)

    # Second-way checks: exact substitution and independent library solve.
    print("\nVerification")
    residual = 2 * (Sigma @ w_star) - (2 / a) * one
    print("budget =", one @ w_star, "; stationarity residual =", residual)

    # Independent two-asset formula from minimizing the scalar quadratic.
    denominator = Sigma[0, 0] + Sigma[1, 1] - 2 * Sigma[0, 1]
    exact_w1 = (Sigma[1, 1] - Sigma[0, 1]) / denominator
    exact = np.array([exact_w1, 1 - exact_w1], dtype=object)
    print("max exact weight difference =", max(abs(w_star - exact)))
    library = np.linalg.solve(Sigma.astype(float), one.astype(float))
    library /= library.sum()
    error = np.max(np.abs(w_star.astype(float) - library))
    print("numpy weights =", library, "; max difference =", error)
    
    assert np.array_equal(Sigma @ x, one) and np.array_equal(Sigma @ y, mu)
    assert np.array_equal(w_star, exact) and np.array_equal(w_star, x / a)
    assert one @ w_star == 1 and all(v == 0 for v in residual)
    assert variance == 1 / a and error <= 1e-12
    assert one @ w_k == 1 and target_mean == r
    assert np.array_equal(w_k, l1 * y + l2 * x)
    assert np.array_equal(Sigma @ w_k, l1 * mu + l2 * one)
    assert target_var == frontier_var == l1 * r + l2

    print("All checks passed.")


if __name__ == "__main__":
    main()

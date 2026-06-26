"""Finite-Sum Objective -- concept 01-finite-sum-objective of the paper learning map.

Witness that the global objective f(w) is literally the average of the per-example
losses f_i(w), i.e. f(w) = (1/n) sum_i f_i(w) (Eq. 0.1 of the deep dive).
Runnable code analog of concepts/01-finite-sum-objective.py.
"""
import numpy as np


def main():
    np.random.seed(0)

    # ---- n=12 toy 1-D regression examples (x_i, y_i) ----
    n = 12
    xs = np.linspace(-2.0, 2.0, n)          # inputs x_i
    true_w = 1.7                            # ground-truth slope (intercept-free)
    noise = 0.3 * np.random.randn(n)        # small label noise
    ys = true_w * xs + noise                # labels y_i

    # Per-example squared-error loss f_i(w) = (w*x_i - y_i)^2 = l(x_i, y_i; w).
    def f_i(w, i):
        return (w * xs[i] - ys[i]) ** 2

    # Global objective f(w) = (1/n) sum_i f_i(w).
    def f(w):
        return float(np.mean([f_i(w, i) for i in range(n)]))

    print("Finite-Sum Objective: f(w) = (1/n) * sum_{i=1}^n f_i(w)")
    print("n = %d toy 1-D regression examples; f_i = squared error." % n)
    print("-" * 60)

    for w in (0.0, 1.7):
        per_example = np.array([f_i(w, i) for i in range(n)])
        f_val = f(w)                                  # via the f() helper
        manual_mean = float(per_example.mean())       # explicit average
        manual_sum = float(per_example.sum() / n)     # (1/n)*sum form

        print("w = %.3f" % w)
        print("  per-example losses f_i(w):")
        print("   ", np.array2string(per_example, precision=4,
                                     floatmode="fixed", separator=", "))
        print("  f(w) via mean of f_i      = %.6f" % f_val)
        print("  f(w) via (1/n)*sum f_i    = %.6f" % manual_sum)
        print("  numpy .mean() of f_i      = %.6f" % manual_mean)
        # f is LITERALLY the average: all three agree to machine precision.
        assert abs(f_val - manual_mean) < 1e-12
        assert abs(f_val - manual_sum) < 1e-12
        print("  -> f(w) IS the average of the f_i(w)  [verified, diff < 1e-12]")
        print("-" * 60)

    # Sanity: the minimizer of this finite sum sits near true_w (least squares).
    grid = np.linspace(0.0, 3.0, 3001)
    losses = np.array([f(w) for w in grid])
    w_star = grid[int(np.argmin(losses))]
    print("argmin_w f(w) on grid = %.3f  (true slope = %.2f)" % (w_star, true_w))


if __name__ == "__main__":
    main()

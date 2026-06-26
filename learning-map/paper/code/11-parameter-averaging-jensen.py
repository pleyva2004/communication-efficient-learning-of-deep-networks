"""Parameter Averaging & Jensen -- concept 11-parameter-averaging-jensen of the paper learning map.

On a convex quadratic f, sample several client models w^k and verify numerically the
Jensen chain f(sum_k beta_k w^k) <= sum_k beta_k f(w^k) <= max_k f(w^k) (Eq. 7.1):
why averaging is safe in the convex case (and thus why non-convex needs shared init).
Runnable code analog of concepts/11-parameter-averaging-jensen.md.
"""
import numpy as np


def main():
    np.random.seed(0)

    # A convex quadratic global objective f(w) = 0.5 (w-c)^T A (w-c) + const.
    # A is SPD => f is convex (Hessian = A >= 0). d = parameter dimension.
    d = 4
    M = np.random.randn(d, d)
    A = M @ M.T + d * np.eye(d)          # symmetric positive definite
    c = np.random.randn(d)               # minimizer location

    def f(w):
        r = w - c
        return 0.5 * float(r @ A @ r)

    # K client models w^k, scattered around the optimum (the FedAvg parents).
    K = 5
    W = c + 1.5 * np.random.randn(K, d)  # rows are the w^k

    # Mixture weights beta_k = n_k / m_t: a probability vector (beta_k >= 0, sum = 1).
    n_k = np.random.randint(50, 500, size=K).astype(float)
    beta = n_k / n_k.sum()

    # LHS: loss of the weighted-averaged model (what the server broadcasts).
    w_bar = beta @ W                      # sum_k beta_k w^k
    lhs = f(w_bar)

    # MIDDLE: weighted average of the per-client losses.
    fk = np.array([f(W[k]) for k in range(K)])
    mid = float(beta @ fk)

    # RHS: the worst client's loss.
    rhs = float(fk.max())

    print("Jensen chain on a convex quadratic f (Eq. 7.1), K=%d clients, d=%d:" % (K, d))
    print("  beta (mixture weights, sum=%.3f): %s" % (beta.sum(), np.round(beta, 3)))
    print("  per-client losses f(w^k):        %s" % np.round(fk, 4))
    print("  LHS  f(avg model)            = %.6f" % lhs)
    print("  MID  sum_k beta_k f(w^k)     = %.6f" % mid)
    print("  RHS  max_k f(w^k)            = %.6f" % rhs)

    eps = 1e-9
    jensen_ok = lhs <= mid + eps
    max_ok = mid <= rhs + eps
    print("  Jensen   LHS <= MID ? %s   (gap MID-LHS = %.6f >= 0)" % (jensen_ok, mid - lhs))
    print("  Bound    MID <= RHS ? %s   (slack RHS-MID = %.6f >= 0)" % (max_ok, rhs - mid))

    assert jensen_ok, "Jensen inequality violated -- f should be convex"
    assert max_ok, "weighted mean exceeds the max -- impossible for a convex combination"
    print("VERDICT: PASS -- averaging the convex parents is no worse than the worst parent.")


if __name__ == "__main__":
    main()

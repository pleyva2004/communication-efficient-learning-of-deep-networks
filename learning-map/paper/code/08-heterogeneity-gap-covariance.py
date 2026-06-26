"""Heterogeneity Gap = eta^2 Cov_k(H_k, g_k) -- concept 08-heterogeneity-gap-covariance of the paper learning map.

Numerically witnesses Eq. (3.5): for tau=2 local steps the FedAvg-vs-centralized gap
equals eta^2 * weighted-Cov_k(H_k, g_k). With quadratic F_k the Hessian H_k is constant,
so the O(eta^3) remainder vanishes and the covariance prediction matches the measured gap.
Runnable code analog of concepts/08-heterogeneity-gap-covariance.py.
"""
import numpy as np


def main():
    np.random.seed(0)
    d, K = 3, 4                      # parameter dim, number of clients
    eta = 1e-2                       # learning rate
    n_k = np.array([10.0, 20.0, 30.0, 40.0])   # examples per client
    alpha = n_k / n_k.sum()          # mixture weights n_k / n

    # Per-client quadratics  F_k(w) = 1/2 w^T H_k w - b_k^T w  =>  grad g_k(w) = H_k w - b_k.
    # H_k constant (quadratic) so Taylor of (3.5) is exact through O(eta^2); O(eta^3) = 0.
    Hs, bs = [], []
    for _ in range(K):
        A = np.random.randn(d, d)
        Hs.append(A @ A.T + d * np.eye(d))     # SPD, heterogeneous per client
        bs.append(np.random.randn(d))
    Hs, bs = np.array(Hs), np.array(bs)

    w_t = np.random.randn(d)                    # shared per-round start

    def grad_k(k, w):
        return Hs[k] @ w - bs[k]

    # --- Local two-step endpoints w^k_(2), then weighted-average them (FedAvg, tau=2). ---
    w2 = np.zeros(d)
    for k in range(K):
        w = w_t - eta * grad_k(k, w_t)          # step 1
        w = w - eta * grad_k(k, w)              # step 2 (gradient at the moved point)
        w2 += alpha[k] * w
    # --- Two genuine *centralized* GD steps G^(2) on f = sum_k alpha_k F_k. ---
    H = np.tensordot(alpha, Hs, axes=1)         # H = sum_k alpha_k H_k
    b = alpha @ bs                              # so grad f(w) = H w - b
    g = w_t.copy()
    g = g - eta * (H @ g - b)
    g = g - eta * (H @ g - b)

    Delta = w2 - g                              # measured heterogeneity gap

    # --- Prediction: eta^2 * Cov_k(H_k, g_k) = eta^2 ( sum_k a_k H_k g_k - H grad_f ). ---
    g_k = np.array([grad_k(k, w_t) for k in range(K)])   # local grads at w_t
    mean_Hg = np.tensordot(alpha, np.einsum("kij,kj->ki", Hs, g_k), axes=1)
    grad_f = H @ w_t - b
    cov = mean_Hg - H @ grad_f                  # weighted covariance of (H_k, g_k)
    pred = eta ** 2 * cov

    print("Witness of Eq. (3.5): FedAvg(tau=2) - centralized(2 GD steps) = eta^2 Cov_k(H_k,g_k)")
    print("measured gap  Delta = avg_k w^k_(2) - G^(2)(w_t) =", np.round(Delta, 10))
    print("prediction eta^2 * Cov_k(H_k, g_k)              =", np.round(pred, 10))
    print("residual ||Delta - prediction|| =", np.linalg.norm(Delta - pred),
          "(== 0 to machine eps: quadratic => O(eta^3) term vanishes)")
    print("Cov_k norm =", round(float(np.linalg.norm(cov)), 6),
          "-> nonzero because clients are HETEROGENEOUS (this is client drift).")


if __name__ == "__main__":
    main()

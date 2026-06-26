"""Gradient- vs Model-Averaging Equivalence -- concept 05-gradient-model-averaging-equivalence of the paper learning map.

One local gradient step then size-weighted model averaging equals one averaged-gradient
step, because the weights n_k/n sum to 1 (Eq. 3.2); with TWO local steps the two schemes
diverge by a nonzero residual.
Runnable code analog of concepts/05-gradient-model-averaging-equivalence.md.
"""
import numpy as np


def main():
    np.random.seed(0)
    d = 5          # parameter dimension
    K = 4          # number of clients
    eta = 0.1      # learning rate

    # Shared start, client weights n_k/n (a probability vector summing to 1),
    # and a distinct quadratic local objective F_k per client: grad = A_k w - b_k.
    w_t = np.random.randn(d)
    n_k = np.array([100, 200, 50, 150], dtype=float)
    alpha = n_k / n_k.sum()                 # weights n_k / n; sum = 1
    A = [np.diag(0.5 + np.random.rand(d)) for _ in range(K)]   # SPD-ish local Hessians
    b = [np.random.randn(d) for _ in range(K)]

    def grad(k, w):
        return A[k] @ w - b[k]

    print("Concept 05: model-average of one local step == gradient-average step (Eq. 3.2)")
    print("Setup: d=%d params, K=%d clients, eta=%.2f, sum(alpha)=%.1f" % (d, K, eta, alpha.sum()))

    # --- ONE local step --------------------------------------------------------
    # Scheme A (average gradients, then step): w_t - eta * sum_k alpha_k g_k
    g_avg = sum(alpha[k] * grad(k, w_t) for k in range(K))
    wA_1 = w_t - eta * g_avg
    # Scheme B (step locally, then average models): sum_k alpha_k (w_t - eta g_k)
    wB_1 = sum(alpha[k] * (w_t - eta * grad(k, w_t)) for k in range(K))
    res1 = np.linalg.norm(wA_1 - wB_1)
    print("\nONE local step:")
    print("  ||model-average - gradient-average|| = %.3e  (should be ~1e-15)" % res1)
    print("  vectors equal at 1e-12 tol? %s" % np.allclose(wA_1, wB_1, atol=1e-12))

    # --- TWO local steps -------------------------------------------------------
    # Scheme A': two genuine centralized averaged-gradient steps on f = sum_k alpha_k F_k.
    wA = w_t.copy()
    for _ in range(2):
        gA = sum(alpha[k] * grad(k, wA) for k in range(K))
        wA = wA - eta * gA
    # Scheme B': each client runs TWO local steps from w_t, THEN average the models (Eq. 3.4).
    wB = np.zeros(d)
    for k in range(K):
        wk = w_t.copy()
        for _ in range(2):
            wk = wk - eta * grad(k, wk)
        wB += alpha[k] * wk
    res2 = np.linalg.norm(wA - wB)
    print("\nTWO local steps:")
    print("  ||model-average - gradient-average|| = %.3e  (nonzero: schemes diverge)" % res2)
    print("  vectors equal at 1e-12 tol? %s" % np.allclose(wA, wB, atol=1e-12))
    print("\nWitness: equivalence holds at one step (linearity + sum alpha=1);")
    print("the second local step is nonlinear in w_t, so averaging no longer commutes.")


if __name__ == "__main__":
    main()

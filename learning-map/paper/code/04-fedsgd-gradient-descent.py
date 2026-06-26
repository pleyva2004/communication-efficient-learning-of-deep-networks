"""FedSGD as Exact Gradient Descent -- concept 04-fedsgd-gradient-descent of the paper learning map.

At C=1, FedSGD aggregates per-client gradients g_k=grad F_k(w_t) with weights n_k/n;
this reconstructs the full-data gradient grad f(w_t) exactly, so the update is plain
deterministic full-batch gradient descent on f.
Runnable code analog of concepts/04-fedsgd-gradient-descent.py.
"""

import numpy as np


def main():
    np.random.seed(0)

    # Toy convex problem: f_i(w) = 0.5 * (a_i . w - b_i)^2, a least-squares finite sum.
    # Global objective f(w) = (1/n) sum_i f_i(w). We pick a partition of the n examples
    # into K clients and verify the gradient-aggregation identity (Eq. 2.2).
    n, d, K = 12, 3, 4
    A = np.random.randn(n, d)
    b = np.random.randn(n)
    w_t = np.random.randn(d)  # shared model broadcast to every client

    # Per-example gradient: grad f_i(w) = (a_i . w - b_i) * a_i
    def grad_f_i(idx, w):
        return (A[idx] @ w - b[idx]) * A[idx]

    # Partition [n] into K disjoint, exhaustive index sets of unequal sizes (Eq. P).
    perm = np.random.permutation(n)
    P = [perm[0:2], perm[2:5], perm[5:8], perm[8:12]]  # sizes 2,3,3,4 -> sum = 12 = n
    n_k = np.array([len(p) for p in P])
    assert n_k.sum() == n and len(np.unique(np.concatenate(P))) == n

    # Full-data gradient grad f(w_t) = (1/n) sum_i grad f_i(w_t)  (the ground truth).
    grad_f = np.mean([grad_f_i(i, w_t) for i in range(n)], axis=0)

    # Each client computes its LOCAL average gradient g_k = grad F_k(w_t)  (Eq. 2.1).
    g = [np.mean([grad_f_i(i, w_t) for i in P[k]], axis=0) for k in range(K)]

    # FedSGD (C=1) aggregation: weighted sum with weights n_k/n  (Eq. 2.2).
    agg = sum((n_k[k] / n) * g[k] for k in range(K))

    residual = np.linalg.norm(agg - grad_f)

    # The C=1 FedSGD step equals the centralized full-batch GD step (Eq. 2.3).
    eta = 0.1
    w_fedsgd = w_t - eta * agg
    w_gd = w_t - eta * grad_f
    step_residual = np.linalg.norm(w_fedsgd - w_gd)

    print("FedSGD (C=1) gradient-aggregation identity:  sum_k (n_k/n) g_k  ==  grad f(w_t)")
    print("client sizes n_k = {}, total n = {}, weights n_k/n = {}".format(
        n_k.tolist(), n, np.round(n_k / n, 4).tolist()))
    print("residual ||sum_k (n_k/n) g_k - grad f(w_t)|| = {:.3e}  (expect ~1e-12)".format(residual))
    print("step  ||(w_t - eta*agg) - (w_t - eta*grad f)|| = {:.3e}  -> FedSGD step == full-batch GD step".format(
        step_residual))

    assert residual < 1e-12, "aggregation identity must hold to ~1e-12"
    assert step_residual < 1e-12, "FedSGD update must equal centralized GD update"
    print("PASS: C=1 FedSGD is exact distributed full-batch gradient descent.")


if __name__ == "__main__":
    main()

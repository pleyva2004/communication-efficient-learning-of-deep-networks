"""Client-Partition Decomposition -- concept 02-client-partition-decomposition of the paper learning map.

Witnesses FedAvg's exact identity f(w) = sum_k (n_k/n) F_k(w) (Eq. 1.1): the global
finite-sum objective is the size-weighted mixture of per-client local objectives, for
ANY partition of the n examples into K clients (no IID assumption needed).
Runnable code analog of concepts/02-client-partition-decomposition.py.
"""
import numpy as np


def make_partition(n, K, rng):
    """Cut [n] into K disjoint, nonempty, UNEQUAL index blocks P_1..P_K covering all of [n]."""
    cuts = np.sort(rng.choice(np.arange(1, n), size=K - 1, replace=False))
    bounds = np.concatenate(([0], cuts, [n]))
    perm = rng.permutation(n)  # shuffle so blocks are not just contiguous originals
    return [perm[bounds[k]:bounds[k + 1]] for k in range(K)]


def main():
    rng = np.random.default_rng(0)
    np.random.seed(0)

    n, K, d = 60, 5, 4
    # n toy "examples": each f_i(w) is a smooth per-example loss 0.5 * ||A_i w - b_i||^2.
    A = rng.standard_normal((n, d))
    b = rng.standard_normal(n)

    def f_i(w, i):
        r = A[i] @ w - b[i]
        return 0.5 * r * r

    def f(w):  # global objective: mean over all n examples (Eq. 0.1)
        return np.mean([f_i(w, i) for i in range(n)])

    P = make_partition(n, K, rng)
    n_k = np.array([len(Pk) for Pk in P])
    assert n_k.sum() == n and (n_k >= 1).all()              # coverage + nonempty (P)
    assert len(np.unique(np.concatenate(P))) == n           # disjointness (P)

    def F_k(w, k):  # local objective: mean over client k's examples (Eq. 0.2)
        return np.mean([f_i(w, i) for i in P[k]])

    print("Client-Partition Decomposition: f(w) = sum_k (n_k/n) F_k(w)  [FedAvg Eq. 1.1]")
    print("n=%d examples, K=%d clients, sizes n_k=%s (unequal), n_k/n weights sum=%.1f"
          % (n, K, [int(x) for x in n_k], (n_k / n).sum()))
    print("Verifying the exact algebraic identity at several random w:")

    max_residual = 0.0
    for trial in range(6):
        w = rng.standard_normal(d)
        lhs = f(w)
        rhs = sum((n_k[k] / n) * F_k(w, k) for k in range(K))
        res = abs(lhs - rhs)
        max_residual = max(max_residual, res)
        print("  w#%d: f(w)=%+.10f   sum_k (n_k/n)F_k(w)=%+.10f   |residual|=%.2e"
              % (trial, lhs, rhs, res))

    print("max |residual| over 6 random w = %.2e  (~machine epsilon => identity holds)"
          % max_residual)
    assert max_residual < 1e-12, "decomposition identity failed"
    print("WITNESS PASS: the size-weighted mixture reconstructs f exactly, for this partition.")


if __name__ == "__main__":
    main()

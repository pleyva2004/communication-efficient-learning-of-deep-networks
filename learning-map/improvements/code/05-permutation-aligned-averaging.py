"""Permutation-Aligned Averaging -- concept 05-permutation-aligned-averaging of the improvements learning map.

Two one-hidden-layer MLPs from independent inits sit in different elements of the
hidden-unit permutation orbit; weight-matching recovers the permutation P that aligns
their hidden units so averaging blends matched (not unrelated) units.
Runnable code analog of concepts/05-permutation-aligned-averaging.py.md.
"""
import numpy as np


def frob(A, B):
    """Frobenius distance ||A - B||_F."""
    return float(np.linalg.norm(A - B))


def greedy_match(W1, W1p):
    """Greedy column assignment maximizing sum_i <W1[:,i], W1p[:,P(i)]>.

    Returns perm so that W1p[:, perm] best aligns column-by-column with W1.
    Equivalent target to argmin_P ||W1 - W1p P^T||_F^2 (cross-term is the only
    P-dependent part since the column norms are permutation-invariant).
    """
    H = W1.shape[1]
    score = W1.T @ W1p                       # score[i, j] = <W1[:,i], W1p[:,j]>
    perm = -np.ones(H, dtype=int)
    used = np.zeros(H, dtype=bool)
    order = np.dstack(np.unravel_index(np.argsort(-score, axis=None), score.shape))[0]
    for i, j in order:                       # descending inner products
        if perm[i] == -1 and not used[j]:
            perm[i] = j
            used[j] = True
    return perm


def main():
    np.random.seed(0)
    D, H = 4, 6                              # input dim, hidden units

    # Server model w = (W1, W2); independent-init client model w' = (W1p, W2p).
    W1 = np.random.randn(D, H)
    W2 = np.random.randn(H, 1)
    # Build the client as a TRUE permutation+noise of the server: a relabeling Q
    # of hidden units plus small training noise -> same orbit, different element.
    Q = np.random.permutation(H)
    W1p = W1[:, Q] + 0.05 * np.random.randn(D, H)
    W2p = W2[Q, :] + 0.05 * np.random.randn(H, 1)

    print("Permutation-aligned averaging witness (Git Re-Basin weight-matching)")
    print("H =", H, "hidden units; client is a relabeling Q =", Q.tolist())

    # (1) Recover the alignment permutation P from W1 vs W1'.
    perm = greedy_match(W1, W1p)             # perm[i] = which client col matches server col i
    inv = np.argsort(perm)                   # apply P consistently to client weights

    d_unmatched = frob(W1, W1p)
    d_matched = frob(W1, W1p[:, perm])
    print("Hidden-layer Frobenius distance ||W1 - W1'||_F  unmatched = %.4f" % d_unmatched)
    print("Hidden-layer Frobenius distance ||W1 - P(W1')||_F matched  = %.4f" % d_matched)
    print("matched < unmatched :", d_matched < d_unmatched,
          "(recovered perm == Q^-1 :", bool(np.array_equal(perm, np.argsort(Q))), ")")

    # (2) Function-preserving: applying P consistently (W1'->cols, W2'->rows) leaves
    # the client's function unchanged.  Compare outputs on random inputs.
    X = np.random.randn(20, D)
    relu = lambda Z: np.maximum(Z, 0.0)
    f_client = relu(X @ W1p) @ W2p
    W1p_al = W1p[:, perm]                    # permute hidden columns of layer 1
    W2p_al = W2p[perm, :]                    # permute matching rows of layer 2
    f_aligned = relu(X @ W1p_al) @ W2p_al
    max_fn_err = float(np.max(np.abs(f_client - f_aligned)))
    print("max |f_client - f_aligned| = %.2e  (permutation preserves the function)" % max_fn_err)

    # (3) Averaging: naive vs aligned, distance of the average to the server model.
    avg_naive = 0.5 * W1 + 0.5 * W1p
    avg_align = 0.5 * W1 + 0.5 * W1p_al
    print("dist(server, naive avg)   = %.4f" % frob(W1, avg_naive))
    print("dist(server, aligned avg) = %.4f" % frob(W1, avg_align))
    print("VERDICT:", "PASS" if frob(W1, avg_align) < frob(W1, avg_naive) and max_fn_err < 1e-9
          else "FAIL", "-- alignment yields a closer (compatible) average.")


if __name__ == "__main__":
    main()

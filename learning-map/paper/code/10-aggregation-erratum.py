"""Aggregation Erratum: normalize by m_t -- concept 10-aggregation-erratum of the paper learning map.

Shows that FedAvg's corrected server update normalizes by m_t = sum_{k in S} n_k (weights
sum to 1), whereas the buggy n_k/n rule shrinks each round's aggregate by m_t/n ~= C.
Runnable code analog of concepts/10-aggregation-erratum.md.
"""

import numpy as np


def main():
    np.random.seed(0)

    # ---- Setup: K clients, balanced for clean C = m/K, dim-d toy "models" ----
    K = 100          # total clients
    C = 0.1          # fraction selected per round
    m = max(int(C * K), 1)   # |S_t|
    d = 4            # model dimension
    n_per = 50       # examples per client (balanced)
    n = K * n_per    # total examples

    n_k_all = np.full(K, n_per)        # client sizes
    # one toy local model w^k per client (random but fixed by the seed)
    w_clients = np.random.randn(K, d)

    # ---- Round t: server samples a uniform m-subset S_t (without replacement) ----
    S = np.sort(np.random.choice(K, size=m, replace=False))
    n_k = n_k_all[S]                   # sizes of selected clients
    m_t = n_k.sum()                    # = sum_{k in S} n_k          (Eq. 6.1)
    w_sel = w_clients[S]               # selected local models

    # ---- Correct rule (Eq. 6.1): normalize by m_t -> weights sum to 1 ----
    weights_correct = n_k / m_t
    w_correct = (weights_correct[:, None] * w_sel).sum(axis=0)

    # ---- Buggy rule (Eq. 6.4): normalize by n -> weights sum to m_t/n < 1 ----
    weights_buggy = n_k / n
    w_buggy = (weights_buggy[:, None] * w_sel).sum(axis=0)

    shrink = m_t / float(n)            # the m_t/n shrink factor (Eq. 6.4)

    print("FedAvg aggregation erratum: normalize by m_t, not n  (deep dive Eqs. 6.1-6.5)")
    print("K=%d  C=%.2f  m=|S_t|=%d  n_per=%d  n=%d  m_t=sum n_k over S=%d"
          % (K, C, m, n_per, n, m_t))
    print()
    print("CORRECT  weights n_k/m_t  sum = %.6f   (= 1, a true convex combination)"
          % weights_correct.sum())
    print("BUGGY    weights n_k/n    sum = %.6f   (= m_t/n < 1, shrinks the model)"
          % weights_buggy.sum())
    print()
    print("shrink factor m_t/n = %.6f    (C = m/K = %.6f; equal in expectation, Eq. 6.5)"
          % (shrink, m / float(K)))
    print()
    # Witness: buggy aggregate == shrink * correct aggregate, component-wise (Eq. 6.4)
    recon = shrink * w_correct
    print("correct aggregate w_{t+1}      =", np.array2string(w_correct, precision=4))
    print("buggy aggregate (n_k/n)        =", np.array2string(w_buggy, precision=4))
    print("shrink * correct  (predicted)  =", np.array2string(recon, precision=4))
    print("max|buggy - shrink*correct|    = %.2e  -> buggy = (m_t/n) * correct, exactly"
          % np.max(np.abs(w_buggy - recon)))
    print()
    # Geometric decay: feeding w_buggy back as next start shrinks by ~C each round.
    norm0 = np.linalg.norm(w_correct)
    norms = [norm0 * shrink ** t for t in range(5)]
    print("If the buggy rule recurs, ||w_t|| decays like (m_t/n)^t ~ C^t (geometric collapse):")
    print("  ||w_t|| over rounds 0..4 = " +
          ", ".join("%.4f" % v for v in norms))


if __name__ == "__main__":
    main()

"""Client-Sampling Unbiasedness (Horvitz-Thompson) -- concept 09-client-sampling-unbiasedness of the paper learning map.

Monte-Carlo witness that the Horvitz-Thompson estimator g(S)=(K/m) sum_{k in S}(n_k/n)g_k
is unbiased for the true weighted sum sum_k (n_k/n) g_k = grad f(w), while a naive
uncorrected subset-sum is biased (it under-counts by the participation fraction m/K).
Runnable code analog of concepts/09-client-sampling-unbiasedness.py.
"""
import numpy as np


def main():
    rng = np.random.default_rng(0)
    np.random.seed(0)

    K = 12          # total clients
    m = 4           # clients sampled per round (C = m/K = 1/3)
    n_per = 50      # examples per client (balanced n_k = n/K)
    n_k = np.full(K, n_per, dtype=float)
    n = n_k.sum()
    p_k = m / K     # uniform inclusion probability (Eq. A1)

    # Each client's local average gradient g_k (deterministic given w); use d=3 vectors.
    g = rng.standard_normal((K, 3))

    # TRUE target: grad f(w) = sum_k (n_k/n) g_k  (the partition identity, Eq. 2.2).
    weights = n_k / n
    g_true = (weights[:, None] * g).sum(axis=0)

    trials = 200_000
    ht_acc = np.zeros(3)      # Horvitz-Thompson estimator accumulator (Eq. 5.2)
    naive_acc = np.zeros(3)   # naive uncorrected subset-sum: sum_{k in S}(n_k/n)g_k
    for _ in range(trials):
        S = np.random.choice(K, size=m, replace=False)  # uniform m-subset
        contrib = (n_k[S] / n)[:, None] * g[S]           # the (n_k/n) g_k terms on S
        ht_acc += (K / m) * contrib.sum(axis=0)          # 1/p_k = K/m correction
        naive_acc += contrib.sum(axis=0)                 # no 1/p_k correction -> biased

    ht_mean = ht_acc / trials
    naive_mean = naive_acc / trials

    ht_bias = np.linalg.norm(ht_mean - g_true)
    naive_bias = np.linalg.norm(naive_mean - g_true)
    # The naive estimator is the HT mean shrunk by p_k = m/K, so its target is (m/K)*g_true.
    predicted_naive = (m / K) * g_true

    print("Client-Sampling Unbiasedness (Horvitz-Thompson), Eqs. 5.1-5.2")
    print("K={} clients, m={} sampled/round, inclusion prob p_k=m/K={:.3f}, MC trials={}"
          .format(K, m, p_k, trials))
    print("true target  grad f(w) = sum_k (n_k/n) g_k        =", np.round(g_true, 4))
    print("E[HT estimator] (K/m)*sum_S (n_k/n)g_k            =", np.round(ht_mean, 4))
    print("   -> HT bias ||E[HT]-grad f|| = {:.4e}  (~0, UNBIASED)".format(ht_bias))
    print("E[naive subset-sum] sum_S (n_k/n)g_k              =", np.round(naive_mean, 4))
    print("   -> naive bias ||E[naive]-grad f|| = {:.4e}  (BIASED)".format(naive_bias))
    print("naive matches shrunk target (m/K)*grad f          =", np.round(predicted_naive, 4))
    print("   -> ||E[naive]-(m/K)grad f|| = {:.4e}  (confirms bias factor = C = m/K = {:.3f})"
          .format(np.linalg.norm(naive_mean - predicted_naive), m / K))

    assert ht_bias < 1e-2, "HT estimator should be unbiased"
    assert naive_bias > 0.1, "naive estimator should be visibly biased"
    print("VERDICT: PASS -- the 1/p_k=K/m correction makes the sampled gradient unbiased.")


if __name__ == "__main__":
    main()

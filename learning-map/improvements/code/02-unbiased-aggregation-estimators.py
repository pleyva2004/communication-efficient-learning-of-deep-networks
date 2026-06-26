"""Unbiased Aggregation Estimators -- concept 02-unbiased-aggregation-estimators
of the improvements learning map.

FedAvg's erratum-corrected average sum_{k in S} (n_k/m_t) v_k is a Hajek ratio
estimator (biased under client imbalance); the Horvitz-Thompson form
(K/m) sum_{k in S} (n_k/n) v_k is exactly unbiased. This Monte-Carlo witness
prints each estimator's bias norm: self-normalized >> 0 while HT ~ 0.
Runnable code analog of concepts/02-unbiased-aggregation-estimators.md.
"""
import numpy as np

np.random.seed(0)
K = 100              # number of clients
C = 0.1              # client fraction -> m = C*K clients sampled per round
M = max(1, int(round(C * K)))   # = 10
D = 8                # dimension of each client's returned vector v_k
N_DRAWS = 200_000    # Monte-Carlo uniform size-M subsets


def make_population(imbalanced):
    """Return (sizes n_k of shape (K,), vectors v_k of shape (K, D)).
    v_k is correlated with n_k so size-induced bias is visible (worst case)."""
    if imbalanced:
        sizes = np.round(np.geomspace(10.0, 1000.0, K)).astype(np.int64)
    else:
        sizes = np.full(K, 100, dtype=np.int64)
    s = (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-12)   # in [0,1]
    base = np.random.normal(0.0, 1.0, size=(K, D))
    axis = np.random.normal(0.0, 1.0, size=D)
    v = base + 3.0 * np.outer(s, axis)        # large clients shifted along axis
    return sizes, v


def true_target(sizes, v):
    """vbar = sum_k (n_k/n) v_k -- the full size-weighted mean (the estimand)."""
    return (sizes / sizes.sum()) @ v


def mc_uniform(sizes, v):
    """E[self-normalized] and E[Horvitz-Thompson] over uniform size-M subsets."""
    n, p = sizes.sum(), M / K
    acc_sn = np.zeros(D)
    acc_ht = np.zeros(D)
    done = 0
    while done < N_DRAWS:
        b = min(20_000, N_DRAWS - done)
        keys = np.random.random((b, K))
        S = np.argpartition(keys, M, axis=1)[:, :M]        # (b, M) uniform subsets
        nk, vk = sizes[S], v[S]                            # (b, M), (b, M, D)
        mt = nk.sum(axis=1, keepdims=True)                 # RANDOM normalizer
        acc_sn += np.einsum("bm,bmd->bd", nk / mt, vk).sum(axis=0)         # self-norm
        acc_ht += np.einsum("bm,bmd->bd", (nk / n) / p, vk).sum(axis=0)    # HT, fixed p
        done += b
    return acc_sn / N_DRAWS, acc_ht / N_DRAWS


def main():
    print("Unbiased aggregation under client imbalance (FedAvg, 05-improvements.tex M.2)")
    print(f"K={K} clients, m=C*K={M}, dim={D}, {N_DRAWS:,} Monte-Carlo draws")
    print("Target vbar = sum_k (n_k/n) v_k; we report bias ||E[agg]-vbar||\n")

    sz_i, v_i = make_population(imbalanced=True)
    vbar_i = true_target(sz_i, v_i)
    sn_i, ht_i = mc_uniform(sz_i, v_i)
    bias_sn = float(np.linalg.norm(sn_i - vbar_i))
    bias_ht = float(np.linalg.norm(ht_i - vbar_i))

    sz_b, v_b = make_population(imbalanced=False)
    floor = float(np.linalg.norm(mc_uniform(sz_b, v_b)[0] - true_target(sz_b, v_b)))

    print(f"IMBALANCED  self-norm n_k/m_t (FedAvg)   bias = {bias_sn:.3e}   BIASED")
    print(f"IMBALANCED  Horvitz-Thompson (K/m)(n_k/n) bias = {bias_ht:.3e}   ~0 (unbiased)")
    print(f"BALANCED    self-norm n_k/m_t (sanity)    bias = {floor:.3e}   ~0 (MC floor)")
    print(f"\nSelf-normalized bias is {bias_sn / max(bias_ht, 1e-300):.0f}x the HT bias.")

    assert bias_sn > 10 * floor, "self-norm should be clearly biased under imbalance"
    assert bias_ht <= 5 * floor, "HT should sit at the MC floor (unbiased)"
    print("PASS -- Horvitz-Thompson removes the O(1/m) self-normalized bias.")


if __name__ == "__main__":
    main()
"""
unbiased-aggregation.py -- Implements "Unbiased aggregation under client
imbalance" (Mathematical / Code Improvements) of 05-improvements.tex.

CONTEXT (FedAvg paper, McMahan et al. 2017).
The erratum-corrected server aggregation in Algorithm 1 is the SELF-NORMALIZED
weighted average over the SELECTED clients,

    w_{t+1} = sum_{k in S} (n_k / m_t) v_k ,   m_t = sum_{k in S} n_k ,

where v_k = w^k_{t+1} is the client's returned model. The math deep dive (S5-S6)
shows this is a Hajek RATIO estimator of the full weighted mean

    vbar = sum_{k=1}^K (n_k / n) v_k ,   n = sum_k n_k .

Because m_t in the denominator is RANDOM (it depends on which clients were
drawn), E_S[ w_{t+1} ] = vbar EXACTLY only when the clients are BALANCED
(n_k == n/K). Under client IMBALANCE + uniform client sampling the self-
normalized average carries an O(1/m) bias: larger clients are over-represented
because a draw that happens to include a big client inflates BOTH the numerator
and the normalizer m_t in a correlated way.

PROPOSAL -- two fixes that restore EXACT unbiasedness:

  (a) HORVITZ-THOMPSON inverse-probability weighting. With uniform sampling
      p_k = m/K, weight each sampled term by 1/p_k (a FIXED 1/m, not random):

          g_HT(S) = (1/K) sum_{k in S} ((n_k/n) / p_k) v_k
                  = (K/m) sum_{k in S} (n_k/n) v_k .

      E_S[ g_HT ] = sum_k p_k * (1/p_k)(n_k/n) v_k = vbar  -- exactly unbiased,
      for ANY imbalance, no IID assumption (purely combinatorial).

  (b) SIZE-PROPORTIONAL client sampling. If instead p_k proportional to n_k,
      the SELF-NORMALIZED average itself becomes (asymptotically) unbiased
      because large clients are no longer over-represented relative to their
      objective weight n_k/n. (Demonstrated as a secondary panel.)

THIS SCRIPT (pure-numpy Monte-Carlo; NO training needed).
Build a heavily imbalanced population of K=100 clients with n_k spanning
[10, 1000] and a fixed per-client vector v_k. The TRUE target is the full
weighted mean vbar. Over MANY random size-m=10 subsets S (uniform sampling)
estimate E[aggregate] by Monte-Carlo for (i) self-normalized FedAvg and
(ii) Horvitz-Thompson, and report the bias norm ||E[aggregate] - vbar||.

EXPECT: HT bias ~ 0 (within MC error); self-normalized bias clearly NONZERO
under imbalance. Under BALANCED n_k both biases ~ 0 (sanity check).
"""

from __future__ import annotations

import time
import numpy as np

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 0
np.random.seed(SEED)

# --------------------------------------------------------------------------- #
# Configuration (faithful to the paper's defaults where applicable)
# --------------------------------------------------------------------------- #
K = 100            # number of clients
C = 0.1            # client fraction per round  ->  m = C*K
M = max(1, int(round(C * K)))   # clients sampled per round (= 10)
D = 8              # dimension of the per-client vector v_k
N_DRAWS = 400_000  # Monte-Carlo random subsets (large -> tiny MC error)


# --------------------------------------------------------------------------- #
# Client population.  v_k is a FIXED deterministic function of the client
# (its "local mean") -- this stands in for the model w^k a client would return.
# To make the bias VISIBLE we correlate v_k with the client size n_k: large
# clients point in a systematically different direction than small clients.
# (If v_k were independent of n_k the size-induced bias would partly wash out.)
# --------------------------------------------------------------------------- #
def make_population(rng, imbalanced):
    """Return (sizes n_k of shape (K,), vectors v_k of shape (K, D))."""
    if imbalanced:
        # Heavy size spread in [10, 1000] (geometric -> many small, few huge).
        sizes = np.geomspace(10.0, 1000.0, K)
        sizes = np.round(sizes).astype(np.int64)
    else:
        # Balanced sanity case: every client identical size.
        sizes = np.full(K, 100, dtype=np.int64)

    # v_k = a fixed function of the client. We make it depend on a normalized
    # size coordinate s_k in [0,1] so large vs small clients differ in a known,
    # systematic way -- the worst case for the ratio estimator.
    s = (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-12)  # in [0,1]
    base = rng.normal(0.0, 1.0, size=(K, D))            # idiosyncratic part
    direction = rng.normal(0.0, 1.0, size=D)            # "size axis" in R^D
    v = base + 3.0 * np.outer(s, direction)             # large clients shifted
    return sizes, v


def true_target(sizes, v):
    """vbar = sum_k (n_k / n) v_k -- the full weighted mean (what we estimate)."""
    n = sizes.sum()
    w = sizes / n                                       # (K,)
    return w @ v                                         # (D,)


# --------------------------------------------------------------------------- #
# Monte-Carlo expectation of the two aggregates under UNIFORM m-subset sampling.
# Vectorized over draws for speed.
# --------------------------------------------------------------------------- #
def mc_expectations_uniform(sizes, v, n_draws, m, rng):
    """Return (E_selfnorm, E_HT), each a (D,) Monte-Carlo mean over n_draws
    uniform size-m subsets S (sampling without replacement)."""
    n = sizes.sum()
    K_ = sizes.shape[0]
    p = m / K_                                           # uniform inclusion prob

    acc_sn = np.zeros(v.shape[1])
    acc_ht = np.zeros(v.shape[1])

    # Process in chunks to keep memory modest while staying vectorized.
    CHUNK = 20_000
    done = 0
    while done < n_draws:
        b = min(CHUNK, n_draws - done)
        # b independent uniform m-subsets via argsort of random keys (no replace)
        keys = rng.random((b, K_))
        S = np.argpartition(keys, m, axis=1)[:, :m]      # (b, m) selected indices
        nk = sizes[S]                                    # (b, m)
        vk = v[S]                                         # (b, m, D)

        # (i) self-normalized FedAvg: sum (n_k / m_t) v_k, m_t = sum nk over S
        mt = nk.sum(axis=1, keepdims=True)               # (b, 1) RANDOM normalizer
        beta = nk / mt                                    # (b, m) convex weights
        agg_sn = np.einsum("bm,bmd->bd", beta, vk)       # (b, D)

        # (ii) Horvitz-Thompson: (1/K) sum ((n_k/n)/p) v_k = (K/m) sum (n_k/n) v_k
        #      weight = (n_k/n)/p / 1 ... composed cleanly below.
        alpha = (nk / n) / p                              # (b, m) IPW weights (FIXED p)
        agg_ht = np.einsum("bm,bmd->bd", alpha, vk)       # (b, D)

        acc_sn += agg_sn.sum(axis=0)
        acc_ht += agg_ht.sum(axis=0)
        done += b

    return acc_sn / n_draws, acc_ht / n_draws


# --------------------------------------------------------------------------- #
# Fix (b): SIZE-PROPORTIONAL sampling makes the SELF-NORMALIZED average
# (asymptotically) unbiased.  We sample m clients with replacement with prob
# proportional to n_k, then take the PLAIN mean of v_k over the draw (the
# self-normalized average under prob-proportional-to-size sampling reduces to
# the plain mean of v_k, which targets vbar since E[v over one draw] = vbar).
# --------------------------------------------------------------------------- #
def mc_expectation_size_proportional(sizes, v, n_draws, m, rng):
    n = sizes.sum()
    prob = sizes / n                                      # P(pick client k) prop n_k
    K_ = sizes.shape[0]
    acc = np.zeros(v.shape[1])
    CHUNK = 20_000
    done = 0
    while done < n_draws:
        b = min(CHUNK, n_draws - done)
        # b draws of m clients each, with replacement, prob proportional to n_k.
        idx = rng.choice(K_, size=(b, m), replace=True, p=prob)   # (b, m)
        vk = v[idx]                                                # (b, m, D)
        agg = vk.mean(axis=1)                                      # (b, D) plain mean
        acc += agg.sum(axis=0)
        done += b
    return acc / n_draws


# --------------------------------------------------------------------------- #
# Report helpers
# --------------------------------------------------------------------------- #
def bias_norm(estimate, target):
    return float(np.linalg.norm(estimate - target))


def mc_stderr_norm(sizes, m):
    """Rough scale of MC noise in the bias norm: ~ ||spread|| / sqrt(n_draws).
    Reported only to contextualize 'within MC error'."""
    return None  # computed empirically below via the balanced case


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    print("=" * 78)
    print("Unbiased aggregation under client imbalance  (FedAvg, McMahan et al. 2017)")
    print(f"Monte-Carlo over {N_DRAWS:,} uniform size-m subsets | "
          f"K={K} clients  m=C*K={M} (C={C})  dim D={D}")
    print("Target: vbar = sum_k (n_k/n) v_k   (full weighted mean)")
    print("=" * 78)

    # ----------------------- IMBALANCED population ------------------------- #
    sizes_imb, v_imb = make_population(rng, imbalanced=True)
    vbar_imb = true_target(sizes_imb, v_imb)
    e_sn_imb, e_ht_imb = mc_expectations_uniform(sizes_imb, v_imb, N_DRAWS, M, rng)
    bias_sn_imb = bias_norm(e_sn_imb, vbar_imb)
    bias_ht_imb = bias_norm(e_ht_imb, vbar_imb)
    # fix (b): size-proportional sampling + self-normalized (= plain mean)
    e_sp_imb = mc_expectation_size_proportional(sizes_imb, v_imb, N_DRAWS, M, rng)
    bias_sp_imb = bias_norm(e_sp_imb, vbar_imb)

    # ------------------------ BALANCED population -------------------------- #
    sizes_bal, v_bal = make_population(rng, imbalanced=False)
    vbar_bal = true_target(sizes_bal, v_bal)
    e_sn_bal, e_ht_bal = mc_expectations_uniform(sizes_bal, v_bal, N_DRAWS, M, rng)
    bias_sn_bal = bias_norm(e_sn_bal, vbar_bal)
    bias_ht_bal = bias_norm(e_ht_bal, vbar_bal)

    # The balanced self-normalized bias is pure MC noise (both estimators are
    # exactly unbiased there) -> use it as the empirical "MC error floor".
    mc_floor = max(bias_sn_bal, bias_ht_bal)

    # ------------------------------- report -------------------------------- #
    print(f"\nClient sizes n_k : min={sizes_imb.min()}  max={sizes_imb.max()}  "
          f"mean={sizes_imb.mean():.1f}  max/min ratio={sizes_imb.max()/sizes_imb.min():.0f}x "
          f"(heavily imbalanced)")
    print(f"||vbar|| (target norm, imbalanced) = {np.linalg.norm(vbar_imb):.4f}")
    print(f"empirical MC error floor (balanced, both exact) ~ {mc_floor:.2e}")

    print("\n" + "-" * 78)
    print(f"{'population':<14}{'estimator':<26}{'bias ||E[agg]-vbar||':>22}{'verdict':>14}")
    print("-" * 78)

    def row(pop, name, bias):
        tag = "~0 (unbiased)" if bias <= 5 * mc_floor else "BIASED"
        print(f"{pop:<14}{name:<26}{bias:>22.3e}{tag:>14}")

    row("IMBALANCED", "self-norm  n_k/m_t (FedAvg)", bias_sn_imb)
    row("IMBALANCED", "Horvitz-Thompson  (a)", bias_ht_imb)
    row("IMBALANCED", "size-prop sampling (b)", bias_sp_imb)
    print("-" * 78)
    row("BALANCED", "self-norm  n_k/m_t (FedAvg)", bias_sn_bal)
    row("BALANCED", "Horvitz-Thompson  (a)", bias_ht_bal)
    print("-" * 78)

    # ratio: how much smaller is HT bias than self-normalized bias (imbalanced)
    ratio = bias_sn_imb / max(bias_ht_imb, 1e-300)
    print(f"\nUnder imbalance, self-normalized bias is {ratio:.0f}x the HT bias.")
    print(f"  self-norm bias  = {bias_sn_imb:.3e}   (clearly nonzero)")
    print(f"  HT       bias  = {bias_ht_imb:.3e}   (~ MC floor {mc_floor:.1e})")
    print(f"  size-prop bias = {bias_sp_imb:.3e}   (~ MC floor, fix (b) works)")

    # ------------------------------- verdict -------------------------------- #
    # PASS conditions:
    #  1. Imbalanced self-normalized bias is clearly nonzero (>> MC floor).
    #  2. Imbalanced HT bias is at the MC floor (exactly unbiased).
    #  3. HT bias << self-normalized bias under imbalance.
    #  4. Balanced: both biases at the MC floor (sanity).
    cond1 = bias_sn_imb > 10 * mc_floor
    cond2 = bias_ht_imb <= 5 * mc_floor
    cond3 = bias_ht_imb < 0.1 * bias_sn_imb
    cond4 = (bias_sn_bal <= 5 * mc_floor) and (bias_ht_bal <= 5 * mc_floor)
    cond5 = bias_sp_imb <= 5 * mc_floor   # fix (b) restores unbiasedness too
    passed = cond1 and cond2 and cond3 and cond4 and cond5

    print("\n" + "=" * 78)
    print("VERDICT")
    print(f"  (1) imbalanced self-norm bias >> MC floor      : {cond1}")
    print(f"  (2) imbalanced HT bias ~ MC floor (unbiased)   : {cond2}")
    print(f"  (3) HT bias << self-norm bias (>10x smaller)   : {cond3}")
    print(f"  (4) balanced: both biases ~ MC floor (sanity)  : {cond4}")
    print(f"  (5) size-prop sampling also unbiased  (fix b)  : {cond5}")
    print(f"\n  PASS = {passed}  -- HT (and size-proportional sampling) restore "
          f"exact unbiasedness; self-normalized FedAvg is biased under imbalance.")
    print("=" * 78)
    print(f"total runtime: {time.time() - t0:.2f}s")

    assert passed, "proposed fixes did not reproduce the expected unbiasedness"


if __name__ == "__main__":
    main()

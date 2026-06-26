"""
permutation-aligned-averaging.py

Implements the "Permutation-aligned averaging (Git Re-Basin for FL)" proposal
(Theoretical / Conceptual Connections) of 05-improvements.tex.

WHY THIS PROPOSAL
-----------------
The math deep dive explained Figure 1's LOSS BARRIER for INDEPENDENTLY-
initialized models via PERMUTATION SYMMETRY of hidden units: a one-hidden-layer
MLP is invariant under any permutation P of its H hidden neurons
(W1 -> P W1, b1 -> P b1, W2 -> W2 P^T). Two independently trained nets land in
two DIFFERENT elements of the same permutation orbit, so naively averaging
their parameters blends UNRELATED hidden units -> a barrier. FedAvg sidesteps
this by re-broadcasting one shared w_t every round (all clients stay in the
same basin). Git Re-Basin (Ainsworth et al. 2023) showed that if you first
ALIGN one model into the other's basin via weight-matching (find the permutation
that best matches their hidden units), the barrier collapses -- you can then
average models that were NOT born from a shared init.

PROPOSAL FOR FL: align client models modulo permutation BEFORE averaging. This
would RELAX FedAvg's shared-initialization requirement and permit larger E /
fewer synchronization rounds (the regime where client models drift far apart).

THIS PROTOTYPE
--------------
Reuses the tiny one-hidden-layer MLP from sandbox/tiny_fedavg_averaging.py.
Trains two models w, w' from INDEPENDENT inits on DISJOINT data halves -- the
regime that PRODUCES a barrier (Figure 1, LEFT). Then compares:

  (a) NAIVE averaging          0.5 w + 0.5 w'
  (b) PERMUTATION-ALIGNED      find permutation P of w''s hidden units that best
                               matches w's hidden units (weight-matching on the
                               INCOMING rows of W1, a la Git Re-Basin), apply P
                               to w' (permute rows of W1,b1 and columns of W2),
                               THEN average 0.5 w + 0.5 P(w').

Decisive scalar per condition:
    barrier_height = loss(avg) - max(loss(w), loss(w'))
EXPECT: naive barrier > 0 (averaging hurts), aligned barrier < naive barrier
(ideally <= 0).

NOTE ON THE ASSIGNMENT SOLVER: scipy is unavailable, so we implement BOTH a
GREEDY matcher and our own exact Hungarian (Kuhn-Munkres) solver in pure numpy,
and use the exact one. The greedy result is also reported for comparison.
"""
from __future__ import annotations

import warnings
import numpy as np

# numpy 2.x on Apple Silicon (Accelerate BLAS) emits spurious matmul warnings.
warnings.filterwarnings("ignore", message=".*encountered in matmul.*",
                        category=RuntimeWarning)

GLOBAL_SEED = 0
np.random.seed(GLOBAL_SEED)

# --------------------------------------------------------------------------- #
# Problem dimensions (mirror tiny_fedavg_averaging.py)
# --------------------------------------------------------------------------- #
D = 16          # input dimension
H = 32          # hidden units (permutation-symmetric)
C = 4           # output classes
N = 1500        # total synthetic samples


# --------------------------------------------------------------------------- #
# Synthetic 4-class data (copied from tiny_fedavg_averaging.py)
# --------------------------------------------------------------------------- #
def make_data(n, d, c, seed):
    rng = np.random.RandomState(seed)
    means = rng.randn(c, d) * 0.75
    y = rng.randint(0, c, size=n)
    X = means[y] + rng.randn(n, d) * 1.0
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return X.astype(np.float64), y.astype(np.int64)


def one_hot(y, c):
    Y = np.zeros((y.shape[0], c))
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


# --------------------------------------------------------------------------- #
# One-hidden-layer MLP (copied from tiny_fedavg_averaging.py)
# --------------------------------------------------------------------------- #
def init_params(seed, d=D, h=H, c=C):
    rng = np.random.RandomState(seed)
    return {
        "W1": rng.randn(d, h) * np.sqrt(2.0 / d),
        "b1": np.zeros(h),
        "W2": rng.randn(h, c) * np.sqrt(2.0 / h),
        "b2": np.zeros(c),
    }


def forward(params, X):
    z1 = X @ params["W1"] + params["b1"]
    z1 = np.clip(z1, -30.0, 30.0)
    a1 = np.tanh(z1)
    z2 = a1 @ params["W2"] + params["b2"]
    z2 = z2 - z2.max(axis=1, keepdims=True)
    expz = np.exp(z2)
    probs = expz / expz.sum(axis=1, keepdims=True)
    assert np.isfinite(probs).all(), "non-finite probs (real divergence!)"
    cache = (X, z1, a1, probs)
    return probs, cache


def ce_loss(probs, y):
    n = y.shape[0]
    return -np.mean(np.log(probs[np.arange(n), y] + 1e-12))


def backward(params, cache, y):
    X, z1, a1, probs = cache
    n = y.shape[0]
    Y = one_hot(y, probs.shape[1])
    dz2 = (probs - Y) / n
    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0)
    da1 = dz2 @ params["W2"].T
    dz1 = da1 * (1.0 - a1 ** 2)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def train(params, X, y, lr=0.2, epochs=80, batch=50, seed=0):
    rng = np.random.RandomState(seed)
    p = {k: v.copy() for k, v in params.items()}
    n = X.shape[0]
    for _ in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            probs, cache = forward(p, X[idx])
            grads = backward(p, cache, y[idx])
            for k in p:
                p[k] -= lr * grads[k]
    return p


def interpolate(wa, wb, theta):
    """theta*wa + (1-theta)*wb, per-parameter."""
    return {k: theta * wa[k] + (1.0 - theta) * wb[k] for k in wa}


def eval_path(wa, wb, X, y, thetas):
    return np.array([ce_loss(forward(interpolate(wa, wb, t), X)[0], y)
                     for t in thetas])


# --------------------------------------------------------------------------- #
# PERMUTATION ALIGNMENT (Git Re-Basin weight-matching)
# --------------------------------------------------------------------------- #
# A hidden unit j of the MLP is fully described, on the INCOMING side, by the
# column W1[:, j] (its incoming weights) plus the bias b1[j]. Permuting the
# hidden units relabels these columns. We seek the permutation P (a relabeling
# of model w''s hidden units) that makes w' most resemble w on the hidden layer.
#
# Git Re-Basin "weight matching" maximizes the inner product between matched
# weight vectors, equivalently minimizes the squared distance:
#       P* = argmin_P || W1 - W1'[:, P] ||_F^2   (+ bias term)
# Because ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b and the norms are permutation-
# invariant sums, this is the LINEAR ASSIGNMENT problem
#       P* = argmax_P  sum_j  ( W1[:,j] . W1'[:,P(j)] + b1[j] b1'[P(j)] ).
# We build the H x H similarity matrix S[i, j] = <unit i of w, unit j of w'>
# and solve the assignment.
# --------------------------------------------------------------------------- #
def hidden_similarity(w, wp):
    """S[i, j] = incoming-weight (+bias) inner product between hidden unit i of
    w and hidden unit j of w'. Higher = better match. Shape (H, H)."""
    # incoming description of each unit: stack W1 column with the bias scalar.
    Aw = np.vstack([w["W1"], w["b1"][None, :]])     # (D+1, H)
    Ap = np.vstack([wp["W1"], wp["b1"][None, :]])   # (D+1, H)
    return Aw.T @ Ap                                # (H, H), S[i,j]=unit_i . unit_j


def greedy_match(S):
    """Greedy assignment: repeatedly take the globally-largest remaining
    similarity (i, j), fix that pair, strike out row i / column j. Returns perm
    such that hidden unit j of w' is mapped onto slot perm[j] (we express it as
    `col_for_row`: for each row i of w, which column j of w' it matches)."""
    h = S.shape[0]
    Swork = S.astype(np.float64).copy()
    row_used = np.zeros(h, dtype=bool)
    col_used = np.zeros(h, dtype=bool)
    match_col_for_row = np.full(h, -1, dtype=int)
    neg = -np.inf
    for _ in range(h):
        # mask used rows/cols
        masked = Swork.copy()
        masked[row_used, :] = neg
        masked[:, col_used] = neg
        flat = np.argmax(masked)
        i, j = divmod(int(flat), h)
        match_col_for_row[i] = j
        row_used[i] = True
        col_used[j] = True
    return match_col_for_row


def hungarian_min(cost):
    """Exact rectangular-square Hungarian (Kuhn-Munkres) for SQUARE cost matrix,
    MINIMIZING total cost. Pure-numpy O(n^3) implementation (no scipy).
    Returns col index assigned to each row: assignment[row] = col."""
    cost = np.array(cost, dtype=np.float64)
    n = cost.shape[0]
    assert cost.shape[0] == cost.shape[1]
    INF = np.inf
    # potentials and matching arrays (1-indexed columns; 0 = dummy), standard
    # Jonker-Volgenant-style augmenting implementation.
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)      # p[j] = row matched to column j
    way = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, INF)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0 != 0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    assignment = np.zeros(n, dtype=int)
    for j in range(1, n + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def hungarian_match(S):
    """Solve the MAXIMIZATION assignment on similarity S by minimizing -S.
    Returns col_for_row[i] = best-matching unit j of w' for unit i of w."""
    return hungarian_min(-S)


def apply_permutation(wp, col_for_row):
    """Relabel w''s hidden units so that unit `col_for_row[i]` of w' is moved
    into slot i (to align with unit i of w).

    Layer-1 (incoming) params are indexed by hidden unit on their SECOND axis
    (W1: D x H, b1: H); layer-2 (outgoing) by hidden unit on its FIRST axis
    (W2: H x C). We gather columns/rows so the permuted w' matches w slot-for-
    slot. This is the MLP permutation symmetry: it leaves w''s FUNCTION exactly
    unchanged (verified in main)."""
    cfr = np.asarray(col_for_row)
    return {
        "W1": wp["W1"][:, cfr].copy(),   # reorder incoming columns
        "b1": wp["b1"][cfr].copy(),      # reorder hidden biases
        "W2": wp["W2"][cfr, :].copy(),   # reorder outgoing rows
        "b2": wp["b2"].copy(),           # output bias untouched
    }


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #
def barrier_of(w, wp, Xfull, yfull, thetas):
    """Return (loss@theta0=w', loss@0.5=avg, loss@theta1=w, min over path,
    barrier_height)."""
    losses = eval_path(w, wp, Xfull, yfull, thetas)

    def at(target):
        j = int(np.argmin(np.abs(thetas - target)))
        return float(losses[j])

    l0, lhalf, l1 = at(0.0), at(0.5), at(1.0)
    lmin = float(losses.min())
    barrier = float(lhalf - max(l0, l1))
    return l0, lhalf, l1, lmin, barrier, losses


def print_path(thetas, losses, title):
    print("  " + title)
    print("  theta :   loss")
    span = (losses.max() - losses.min()) or 1.0
    for t, l in zip(thetas[::4], losses[::4]):
        bar = "#" * int(round((l - losses.min()) / span * 34))
        print("  {:+5.2f} : {:7.4f}  {}".format(t, l, bar))
    print()


def main():
    # --------------------------------------------------------------------- #
    # Full dataset, disjoint halves (one per model) -- as in the sandbox.
    # --------------------------------------------------------------------- #
    X, y = make_data(N, D, C, seed=7)
    perm = np.random.RandomState(123).permutation(N)
    X, y = X[perm], y[perm]
    half = N // 2
    Xa, ya = X[:half], y[:half]
    Xb, yb = X[half:], y[half:]

    thetas = np.linspace(-0.2, 1.2, 29)   # 0.0, 0.5, 1.0 land on the grid

    # --------------------------------------------------------------------- #
    # Train two models from INDEPENDENT inits on DISJOINT halves.
    # This is the barrier-producing regime (Figure 1, LEFT).
    # --------------------------------------------------------------------- #
    w0_a = init_params(seed=1)
    w0_b = init_params(seed=2)
    w = train(w0_a, Xa, ya, seed=101)     # model w   (theta = 1)
    wp = train(w0_b, Xb, yb, seed=202)    # model w'  (theta = 0)

    print("=" * 74)
    print("Permutation-aligned averaging (Git Re-Basin for FL) -- prototype")
    print("MLP: d={} -> h={} (tanh) -> {} classes  |  INDEPENDENT inits, "
          "DISJOINT data halves".format(D, H, C))
    print("Eval: full-dataset cross-entropy; barrier = loss(avg) - max(parents)")
    print("=" * 74)
    print()

    # parent losses on the full set (for reference)
    loss_w = ce_loss(forward(w, X)[0], y)
    loss_wp = ce_loss(forward(wp, X)[0], y)
    print("Parent losses on full set:  loss(w) = {:.4f}   loss(w') = {:.4f}"
          .format(loss_w, loss_wp))
    print()

    # --------------------------------------------------------------------- #
    # (a) NAIVE averaging
    # --------------------------------------------------------------------- #
    l0_n, lh_n, l1_n, lmin_n, bar_n, losses_n = barrier_of(w, wp, X, y, thetas)

    # --------------------------------------------------------------------- #
    # (b) PERMUTATION-ALIGNED averaging
    #     Find P aligning w' to w on the hidden layer, apply it, then average.
    # --------------------------------------------------------------------- #
    S = hidden_similarity(w, wp)
    cfr_greedy = greedy_match(S)
    cfr_hung = hungarian_match(S)

    # objective values: sum of matched similarities (higher = better match)
    obj_identity = float(np.trace(S))
    obj_greedy = float(S[np.arange(H), cfr_greedy].sum())
    obj_hung = float(S[np.arange(H), cfr_hung].sum())

    # sanity: a permutation must be a bijection
    assert sorted(cfr_hung.tolist()) == list(range(H)), "Hungarian not a permutation"
    assert sorted(cfr_greedy.tolist()) == list(range(H)), "greedy not a permutation"

    # use the EXACT (Hungarian) permutation for the headline result
    wp_aligned = apply_permutation(wp, cfr_hung)

    # the permutation must leave w''s FUNCTION (and thus its loss) unchanged
    loss_wp_aligned = ce_loss(forward(wp_aligned, X)[0], y)
    assert abs(loss_wp_aligned - loss_wp) < 1e-9, \
        "permutation changed w' loss -> not a true symmetry!"

    l0_a, lh_a, l1_a, lmin_a, bar_a, losses_a = barrier_of(
        w, wp_aligned, X, y, thetas)

    # --------------------------------------------------------------------- #
    # Report
    # --------------------------------------------------------------------- #
    print("WEIGHT-MATCHING OBJECTIVE  (sum of matched hidden-unit similarities; "
          "higher=better):")
    print("  identity (no align)     = {:8.4f}".format(obj_identity))
    print("  greedy match            = {:8.4f}".format(obj_greedy))
    print("  exact Hungarian (used)  = {:8.4f}   "
          "(Hungarian >= greedy: {})".format(obj_hung, obj_hung >= obj_greedy - 1e-9))
    print()

    print("-" * 74)
    print("(a) NAIVE averaging   0.5 w + 0.5 w'")
    print("    loss(w')={:.4f}  loss(avg)={:.4f}  loss(w)={:.4f}  "
          "min-on-path={:.4f}".format(l0_n, lh_n, l1_n, lmin_n))
    print("    >>> NAIVE barrier_height = loss(avg) - max(parents) = {:+.4f}"
          .format(bar_n))
    print()
    print("(b) PERMUTATION-ALIGNED averaging   0.5 w + 0.5 P(w')   (P = Hungarian)")
    print("    loss(P w')={:.4f}  loss(avg)={:.4f}  loss(w)={:.4f}  "
          "min-on-path={:.4f}".format(l0_a, lh_a, l1_a, lmin_a))
    print("    >>> ALIGNED barrier_height = loss(avg) - max(parents) = {:+.4f}"
          .format(bar_a))
    print("-" * 74)
    print()

    print_path(thetas, losses_n, "NAIVE interpolation curve (loss vs theta)")
    print_path(thetas, losses_a, "ALIGNED interpolation curve (loss vs theta)")

    # --------------------------------------------------------------------- #
    # Verdict
    # --------------------------------------------------------------------- #
    print("=" * 74)
    print("SUMMARY  (independent inits, disjoint halves -- the barrier regime)")
    print("  NAIVE   barrier_height = {:+.4f}   min-on-path = {:.4f}"
          .format(bar_n, lmin_n))
    print("  ALIGNED barrier_height = {:+.4f}   min-on-path = {:.4f}"
          .format(bar_a, lmin_a))
    print("  barrier reduction (naive - aligned) = {:+.4f}"
          .format(bar_n - bar_a))
    naive_has_barrier = bar_n > 0
    aligned_beats_naive = bar_a < bar_n
    aligned_no_barrier = bar_a <= 0
    print()
    print("  naive has positive barrier (averaging hurts): {}".format(naive_has_barrier))
    print("  aligned barrier < naive barrier:              {}".format(aligned_beats_naive))
    print("  aligned removes barrier (<= 0):               {}".format(aligned_no_barrier))
    verdict = naive_has_barrier and aligned_beats_naive
    print()
    print("  VERDICT: {}  -- permutation alignment {} the barrier"
          .format("PASS" if verdict else "FAIL",
                  "ELIMINATES" if aligned_no_barrier else "REDUCES"))
    print("=" * 74)


if __name__ == "__main__":
    main()

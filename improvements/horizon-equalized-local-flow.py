"""
prototype.py -- Horizon-equalized FedAvg.

SELF-CONTAINED mini-study. Embeds an adapted FedAvg toy (after McMahan et al.
2017, arXiv:1602.05629) and compares, on IDENTICAL data / seeds / partitions:

  BASELINE : vanilla FedAvg with GLOBAL (E, B, eta).
  PROPOSED : horizon-equalized FedAvg with PER-CLIENT eta_k chosen so the
             continuous gradient-flow integration time
                  T_k = eta_k * u_k ,   u_k = E * n_k / B
             is identical (= T*) for every client every round.

WHY.  One forward-Euler step of local SGD,  w <- w - eta * grad F_k(w), advances
the gradient flow  dw/dt = -grad F_k(w)  by physical time eta. A round of E
epochs over n_k examples in batches of B is u_k = E*n_k/B such steps, so the
client integrates its OWN flow for total time T_k = eta * u_k. Vanilla FedAvg
fixes (E,B,eta) globally; under client imbalance n_k varies, so T_k = eta*E*n_k/B
varies. The server then averages models that each flowed for a DIFFERENT amount
of time toward their own optimum w_k* -- a "horizon-mismatch" component of the
heterogeneity drift. Equalizing T_k removes exactly that component.

The equalizer is eta_k = T* / u_k = T* * B / (E * n_k). We set T* equal to the
horizon a SIZE-n_bar client would have under the baseline LR (T* = eta * E *
n_bar / B), so the two arms apply the SAME TOTAL gradient-flow time on average
(no free lunch / handicap): only its DISTRIBUTION across clients differs.

Prints a comparison table and a final VERDICT line. <2 min CPU, deterministic.
"""
import time
import numpy as np

# ----------------------------- reproducibility -----------------------------
SEED = 0
np.random.seed(SEED)
np.seterr(over="ignore", divide="ignore", invalid="ignore")

# ------------------------------ configuration ------------------------------
D = 30              # feature dimension
NUM_CLASSES = 10    # 10 Gaussian blobs (digit-like)
N_TRAIN = 6000      # total training examples (n)
N_TEST = 2000       # total test examples
K = 100             # number of clients
C = 0.1             # fraction of clients sampled per round
LR = 1.0            # baseline local SGD learning rate (global)
COND = 30.0         # feature-scale spread (ill-conditioning) -> slow GD
TARGET_ACC = 0.95   # target test accuracy
MAX_ROUNDS = 800    # cap on communication rounds

E_FIXED = 5         # local epochs (shared by both arms)
B_FIXED = 10        # local minibatch size (shared by both arms)

N_BAR = N_TRAIN / K  # expected examples per client (= 60)


# --------------------------- synthetic data (blobs) ------------------------
def make_dataset(seed):
    rng = np.random.default_rng(seed)
    scales = np.geomspace(1.0, 1.0 / COND, D)
    centers = rng.normal(0.0, 1.0, size=(NUM_CLASSES, D)) * scales

    def draw(n):
        y = rng.integers(0, NUM_CLASSES, size=n)
        X = centers[y] + rng.normal(0.0, 1.0, size=(n, D)) * scales
        return X.astype(np.float64), y.astype(np.int64)

    Xtr, ytr = draw(N_TRAIN)
    Xte, yte = draw(N_TEST)
    return Xtr, ytr, Xte, yte


# ------------------------------ model (softmax) ----------------------------
def softmax(z):
    z = np.clip(z, -60.0, 60.0)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def grad(W, Xb, yb):
    m = Xb.shape[0]
    P = softmax(Xb @ W)
    Y = np.zeros((m, NUM_CLASSES))
    Y[np.arange(m), yb] = 1.0
    return Xb.T @ (P - Y) / m


def accuracy(W, X, y):
    return float((np.argmax(X @ W, axis=1) == y).mean())


# ------------------------------ partitioning -------------------------------
def _noniid_label_assignment(y, k, rng):
    """Pathological NON-IID label structure (paper's MNIST recipe): each client
    gets shards drawn from sorted-by-label data => few classes/client. Returns
    a list of label-sorted index arrays, one shard-pair per client. We rebuild
    the actual per-client index sets AFTER choosing sizes, by sampling WITHOUT
    replacement from the 2 label-shards assigned to the client, so the label
    skew is preserved at any requested client size."""
    order = np.argsort(y, kind="stable")
    shards = np.array_split(order, 2 * k)
    shard_ids = rng.permutation(2 * k)
    pairs = [(shards[shard_ids[2 * c]], shards[shard_ids[2 * c + 1]])
             for c in range(k)]
    return pairs


def partition_noniid_imbalanced(y, k, rng, spread):
    """NON-IID partition with IMBALANCED client sizes. Sizes are drawn
    log-uniform over [n_lo, n_hi] with n_hi/n_lo = `spread`, then rescaled so
    the mean size is ~N_BAR. Each client samples its examples (without
    replacement, with top-up replacement only if a shard is exhausted) from its
    two assigned label-shards, preserving the pathological 2-classes-per-client
    skew at every size. spread=1.0 => all sizes == N_BAR (balanced)."""
    pairs = _noniid_label_assignment(y, k, rng)
    # log-uniform sizes with the requested max/min spread, centered on N_BAR.
    if spread <= 1.0:
        sizes = np.full(k, int(round(N_BAR)), dtype=int)
    else:
        lo = np.log(1.0)
        hi = np.log(spread)
        raw = np.exp(rng.uniform(lo, hi, size=k))      # in [1, spread]
        raw = raw / raw.mean() * N_BAR                 # mean ~ N_BAR
        sizes = np.clip(np.round(raw), 4, None).astype(int)

    client_idx = []
    for c in range(k):
        pool = np.concatenate(pairs[c])
        nk = int(sizes[c])
        if nk <= len(pool):
            sel = rng.choice(pool, size=nk, replace=False)
        else:  # tiny shard, large requested size: allow replacement to top up
            sel = rng.choice(pool, size=nk, replace=True)
        client_idx.append(sel)
    return client_idx


# ------------------------------ local client SGD ---------------------------
def client_update(W_global, Xk, yk, E, B, lr, rng):
    """E local epochs of minibatch SGD from W_global with learning rate `lr`."""
    W = W_global.copy()
    nk = Xk.shape[0]
    bsz = nk if (B is None or B >= nk) else B
    for _ in range(E):
        perm = rng.permutation(nk)
        for start in range(0, nk, bsz):
            sel = perm[start:start + bsz]
            W -= lr * grad(W, Xk[sel], yk[sel])
    return W


def u_of(nk, E, B):
    """Local updates per round for a client of size nk (forward-Euler steps)."""
    bsz = nk if (B is None or B >= nk) else B
    steps_per_epoch = int(np.ceil(nk / bsz))
    return E * steps_per_epoch


# ------------------------------ federated loop -----------------------------
def run_federated(Xtr, ytr, Xte, yte, client_idx, E, B, target, seed,
                  equalize=False, lr=LR):
    """FederatedAveraging (Algorithm 1).

    equalize=False -> BASELINE: every client uses global lr.
    equalize=True  -> PROPOSED: client k uses lr_k = T* / u_k so that
                      T_k = lr_k * u_k = T* for all k, where
                      T* = lr * E * n_bar / B  (horizon of an avg-size client).

    Returns the first round reaching `target` test accuracy, else MAX_ROUNDS+1.
    """
    rng = np.random.default_rng(seed)
    W = np.zeros((D, NUM_CLASSES))                 # SHARED initialization
    m = max(1, int(round(C * K)))
    sizes = np.array([len(ix) for ix in client_idx])
    # target horizon: what an average-size client gets under the baseline LR.
    u_bar = u_of(int(round(N_BAR)), E, B)
    T_star = lr * u_bar

    for t in range(1, MAX_ROUNDS + 1):
        S = rng.choice(K, size=m, replace=False)
        m_t = sizes[S].sum()
        W_new = np.zeros_like(W)
        for k in S:
            nk = sizes[k]
            if equalize:
                u_k = u_of(nk, E, B)
                lr_k = T_star / u_k                # eta_k = T* / u_k
            else:
                lr_k = lr
            Wk = client_update(W, Xtr[client_idx[k]], ytr[client_idx[k]],
                               E, B, lr_k, rng)
            W_new += (sizes[k] / m_t) * Wk         # corrected weighted average
        W = W_new
        acc = accuracy(W, Xte, yte)
        if not np.isfinite(acc):
            return MAX_ROUNDS + 1
        if acc >= target:
            return t
    return MAX_ROUNDS + 1


# --------------------------------- measure ---------------------------------
N_REALIZATIONS = 12   # partition realizations averaged per spread (kills the
                      # integer-round-count noise of any single partition draw)


def measure():
    """Run baseline vs proposed across a sweep of size-imbalance spreads.

    Rounds-to-target is an integer-valued, partition-realization-dependent
    statistic; a single draw is noisy (+/-1 round is meaningless). So at each
    spread we average over N_REALIZATIONS partitions. CRUCIALLY, within each
    realization both arms see the IDENTICAL partition and the IDENTICAL client-
    sampling seed -- they differ ONLY in the per-client learning rate. Returns a
    dict of measured numbers."""
    Xtr, ytr, Xte, yte = make_dataset(SEED)
    spreads = [1.0, 5.0, 20.0, 80.0]    # target max/min client-size ratio
    rows = []
    for spread in spreads:
        b_rounds, p_rounds, real_spreads = [], [], []
        nmins, nmaxs = [], []
        wins = losses = ties = 0
        for s in range(N_REALIZATIONS):
            # IDENTICAL partition + sampling seed for both arms in this draw.
            part_rng = np.random.default_rng(1000 + s)
            idx = partition_noniid_imbalanced(ytr, K, part_rng, spread)
            sizes = np.array([len(ix) for ix in idx])
            real_spreads.append(sizes.max() / max(1, sizes.min()))
            nmins.append(int(sizes.min()))
            nmaxs.append(int(sizes.max()))

            rb = run_federated(Xtr, ytr, Xte, yte, idx, E_FIXED, B_FIXED,
                               TARGET_ACC, seed=2000 + s, equalize=False)
            re = run_federated(Xtr, ytr, Xte, yte, idx, E_FIXED, B_FIXED,
                               TARGET_ACC, seed=2000 + s, equalize=True)
            b_rounds.append(rb)
            p_rounds.append(re)
            if rb > re:
                wins += 1
            elif rb < re:
                losses += 1
            else:
                ties += 1
        b_rounds = np.array(b_rounds, dtype=float)
        p_rounds = np.array(p_rounds, dtype=float)
        rows.append({
            "spread_target": spread,
            "spread_real": float(np.mean(real_spreads)),
            "n_min": int(np.min(nmins)),
            "n_max": int(np.max(nmaxs)),
            "baseline_rounds": float(b_rounds.mean()),
            "proposed_rounds": float(p_rounds.mean()),
            "delta": float((b_rounds - p_rounds).mean()),  # >0 => proposed faster
            "wins": wins, "losses": losses, "ties": ties,
        })
    return {"rows": rows, "max_rounds": MAX_ROUNDS,
            "n_realizations": N_REALIZATIONS}


# --------------------------------- main ------------------------------------
def main():
    t0 = time.time()
    res = measure()
    rows = res["rows"]

    print("=" * 92)
    print("Horizon-equalized FedAvg -- communication rounds to reach "
          f"{TARGET_ACC*100:.0f}% test accuracy")
    print(f"softmax regression | d={D} classes={NUM_CLASSES} K={K} C={C} "
          f"E={E_FIXED} B={B_FIXED} baseline_lr={LR} cond~{COND:.0f}")
    print("BASELINE = global eta.  PROPOSED = per-client eta_k=T*/u_k so "
          "T_k=eta_k*u_k is constant across clients.")
    print("Pathological NON-IID, IMBALANCED (log-uniform) sizes. Both arms share "
          "partition+sampling seed; differ ONLY in eta_k.")
    print(f"Mean over {res['n_realizations']} partition realizations / spread.  "
          f"round cap = {res['max_rounds']}")
    print("=" * 92)
    print(f"{'spread~':>8}{'n_min':>7}{'n_max':>7} | "
          f"{'BASE rnds':>11}{'PROP rnds':>11}{'mean dlt':>10}"
          f"{'  W/L/T':>10}{'  faster?':>10}")
    print("-" * 92)

    for row in rows:
        verdict = ("PROPOSED" if row["delta"] > 1e-9 else
                   ("baseline" if row["delta"] < -1e-9 else "tie"))
        wlt = f"{row['wins']}/{row['losses']}/{row['ties']}"
        print(f"{row['spread_real']:>8.1f}{row['n_min']:>7}{row['n_max']:>7} | "
              f"{row['baseline_rounds']:>11.2f}{row['proposed_rounds']:>11.2f}"
              f"{row['delta']:>10.2f}{wlt:>10}{verdict:>10}")
    print("-" * 92)

    # ---- prediction checks ----
    balanced = rows[0]                       # spread ~ 1 (no-op expected)
    imbalanced = rows[1:]
    deltas = [r["delta"] for r in imbalanced]
    spreads = [r["spread_real"] for r in imbalanced]

    # (P1) no-op on balanced: |mean delta| <= 0.25 round AND zero net losses.
    p1 = abs(balanced["delta"]) <= 0.25 and balanced["losses"] == 0
    # (P2) proposed faster ON AVERAGE on every imbalanced spread, never net-worse.
    p2 = all(d > 1e-9 for d in deltas) and all(r["losses"] <= r["wins"]
                                               for r in imbalanced)
    # (P3) the mean advantage widens (non-decreasing) with imbalance.
    p3 = all(deltas[i + 1] >= deltas[i] - 1e-9 for i in range(len(deltas) - 1))

    print("PREDICTION CHECKS:")
    print(f"  (P1) balanced is a near no-op (|mean delta|<=0.25, 0 losses): "
          f"delta={balanced['delta']:.2f}, losses={balanced['losses']}  "
          f"-> {'OK' if p1 else 'FAIL'}")
    print(f"  (P2) proposed faster on average at every imbalanced spread, never "
          f"net-worse: deltas={[round(d,2) for d in deltas]}  "
          f"-> {'OK' if p2 else 'FAIL'}")
    print(f"  (P3) advantage widens with imbalance (non-decreasing mean delta): "
          f"deltas={[round(d,2) for d in deltas]} at spreads="
          f"{[round(s,1) for s in spreads]}  -> {'OK' if p3 else 'FAIL'}")

    if p1 and p2 and p3:
        verdict = ("PASS -- horizon equalization is a clean no-op when balanced "
                   "and reduces mean rounds-to-target under client-size "
                   "imbalance, with a margin that widens monotonically.")
    elif p1 and p2:
        verdict = ("MIXED(+) -- proposed is a no-op when balanced and faster on "
                   "average under imbalance, but the monotone-widening (P3) "
                   "prediction did not hold cleanly.")
    elif p2:
        verdict = ("MIXED -- proposed beats baseline on average under imbalance, "
                   "but the balanced no-op (P1) prediction did not hold.")
    elif any(d > 1e-9 for d in deltas):
        verdict = ("MIXED -- proposed helps at some imbalance levels but not all; "
                   "horizon mismatch is only part of the heterogeneity drift.")
    else:
        verdict = ("FAIL -- horizon equalization did not reduce mean rounds under "
                   "imbalance; the horizon component is not the bottleneck here.")
    print("=" * 92)
    print(f"VERDICT: {verdict}")
    print(f"total runtime: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()

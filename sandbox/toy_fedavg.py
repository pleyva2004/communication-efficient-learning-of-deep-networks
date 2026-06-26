"""
toy_fedavg.py -- Level-1 (pure numpy) demo of FedAvg's headline
communication-efficiency claim (McMahan et al. 2017, "Communication-Efficient
Learning of Deep Networks from Decentralized Data"; Table 2).

CLAIM REPRODUCED (qualitatively):
  Adding local computation per round -- more local epochs E and/or smaller
  minibatch B, i.e. a larger expected number of local updates per round
  u = nE/(KB) = (n_bar/B)*E -- REDUCES the number of COMMUNICATION ROUNDS
  needed to reach a target test accuracy. FedAvg therefore needs FAR fewer
  rounds than FedSGD (the E=1, B=inf endpoint, first table row). The
  pathological NON-IID partition is HARDER than IID: it needs more rounds and
  yields smaller speedups.

MODEL: softmax (multinomial logistic) regression, w in R^{d x C}. This is a
CONVEX proxy; the paper uses NON-CONVEX nets (MNIST 2NN/CNN, an LSTM). The
convex toy intentionally isolates the communication-vs-computation tradeoff:
the speedups here come purely from doing more local SGD work between
synchronizations, not from any non-convex loss-landscape magic. To make a
convex problem actually require many gradient steps (so FedSGD's one step per
round is visibly slow), we use an ILL-CONDITIONED, anisotropic Gaussian-blob
distribution -- the convex analogue of the slow flat directions that make the
paper's nets need hundreds of rounds.

FAITHFUL to the paper:
  * K=100 clients, client fraction C=0.1.
  * IID partition  = shuffle then split into K shards.
  * NON-IID partition = sort by label, cut into 2K shards, give each client 2
    shards (=> mostly 2 classes/client). This is exactly the paper's MNIST
    "pathological" recipe.
  * Shared model initialization across clients (Sec. 3 / Fig. 1: averaging only
    helps from a common init).
  * Corrected weighted aggregation  w_{t+1} = sum_{k in S_t} (n_k / m_t) w^k.
  * FedSGD = (E=1, B=inf); FedAvg sweeps (E,B) ordered by u, as in Table 2.
"""
import time
import numpy as np

# ----------------------------- reproducibility -----------------------------
SEED = 0
np.random.seed(SEED)
# The ill-conditioned features can make W (hence logits Xb@W) large enough to
# overflow float64 in the matmul on a few transient steps; softmax() clips
# before exp(), so the classifier outputs stay valid. Silence the harmless
# overflow/divide warnings rather than letting them clutter the table.
np.seterr(over="ignore", divide="ignore", invalid="ignore")

# ------------------------------ configuration ------------------------------
D = 30              # feature dimension
NUM_CLASSES = 10    # 10 Gaussian blobs (digit-like)
N_TRAIN = 6000      # total training examples (n)
N_TEST = 2000       # total test examples
K = 100             # number of clients
C = 0.1             # fraction of clients sampled per round (paper's default)
LR = 1.0            # local SGD learning rate (tuned, fixed across configs)
COND = 30.0         # feature-scale spread (ill-conditioning) -> slow GD
TARGET_ACC = 0.95   # target test accuracy (reachable by all configs here)
MAX_ROUNDS = 600    # cap on communication rounds

N_BAR = N_TRAIN / K  # expected examples per client (= 60)


# --------------------------- synthetic data (blobs) ------------------------
def make_dataset(seed):
    """Anisotropic, ILL-CONDITIONED Gaussian blobs. The same class centers and
    the same per-feature scales are shared by train and test (a model must
    generalize across draws). Feature stds span [1, 1/COND], so plain gradient
    descent crawls along the small-scale directions -- the convex stand-in for
    the slow optimization that makes the paper's nets need many rounds."""
    rng = np.random.default_rng(seed)
    scales = np.geomspace(1.0, 1.0 / COND, D)            # per-feature std spread
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
    z = np.clip(z, -60.0, 60.0)              # guard against overflow
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def grad(W, Xb, yb):
    """Gradient of mean cross-entropy loss wrt W (d x C) on a minibatch."""
    m = Xb.shape[0]
    P = softmax(Xb @ W)                      # (m, C)
    Y = np.zeros((m, NUM_CLASSES))
    Y[np.arange(m), yb] = 1.0
    return Xb.T @ (P - Y) / m                # (d, C)


def accuracy(W, X, y):
    return float((np.argmax(X @ W, axis=1) == y).mean())


# ------------------------------ partitioning -------------------------------
def partition_iid(y, k, rng):
    """Shuffle then split into k roughly-equal shards (IID)."""
    return [s for s in np.array_split(rng.permutation(len(y)), k)]


def partition_noniid(y, k, rng):
    """Pathological NON-IID (paper's MNIST recipe): sort by label, cut into 2k
    shards, give each client 2 random shards => mostly 2 classes per client."""
    order = np.argsort(y, kind="stable")          # sort by label
    shards = np.array_split(order, 2 * k)         # 2k contiguous label-shards
    shard_ids = rng.permutation(2 * k)
    return [np.concatenate([shards[shard_ids[2 * c]], shards[shard_ids[2 * c + 1]]])
            for c in range(k)]


# ------------------------------ local client SGD ---------------------------
def client_update(W_global, Xk, yk, E, B, lr, rng):
    """Run E local epochs of minibatch SGD from W_global (B=None => full batch,
    i.e. B=inf). Returns the client's locally-updated weights."""
    W = W_global.copy()
    nk = Xk.shape[0]
    bsz = nk if (B is None or B >= nk) else B
    for _ in range(E):
        perm = rng.permutation(nk)
        for start in range(0, nk, bsz):
            sel = perm[start:start + bsz]
            W -= lr * grad(W, Xk[sel], yk[sel])
    return W


# ------------------------------ federated loop -----------------------------
def run_federated(Xtr, ytr, Xte, yte, client_idx, E, B, lr, target, seed):
    """FederatedAveraging (Algorithm 1). Returns the first round at which test
    accuracy reaches `target`, or MAX_ROUNDS+1 ("DNR") if never reached."""
    rng = np.random.default_rng(seed)
    W = np.zeros((D, NUM_CLASSES))                 # SHARED initialization
    m = max(1, int(round(C * K)))                  # clients sampled per round
    sizes = np.array([len(ix) for ix in client_idx])

    for t in range(1, MAX_ROUNDS + 1):
        S = rng.choice(K, size=m, replace=False)   # sample client set S_t
        m_t = sizes[S].sum()                       # total examples this round
        W_new = np.zeros_like(W)
        for k in S:
            Wk = client_update(W, Xtr[client_idx[k]], ytr[client_idx[k]],
                               E, B, lr, rng)
            # corrected weighted aggregation: w_{t+1} = sum (n_k / m_t) w^k
            W_new += (sizes[k] / m_t) * Wk
        W = W_new
        if accuracy(W, Xte, yte) >= target:
            return t
    return MAX_ROUNDS + 1


def u_of(E, B):
    """Expected local updates per round: u = (n_bar / B) * E  (= nE/(KB))."""
    b = N_BAR if B is None else B
    return (N_BAR / b) * E


# --------------------------------- main ------------------------------------
def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = make_dataset(SEED)

    part_rng = np.random.default_rng(SEED + 7)
    idx_iid = partition_iid(ytr, K, part_rng)
    idx_non = partition_noniid(ytr, K, part_rng)

    # (label, E, B); B=None == B=inf. First row FedSGD, rest FedAvg, by ascending u.
    configs = [
        ("FedSGD", 1, None),
        ("FedAvg", 5, None),
        ("FedAvg", 1, 10),
        ("FedAvg", 20, None),
        ("FedAvg", 5, 50),
        ("FedAvg", 5, 10),
        ("FedAvg", 20, 10),
    ]

    base_iid = base_non = None
    results = []
    for label, E, B in configs:
        r_iid = run_federated(Xtr, ytr, Xte, yte, idx_iid, E, B, LR,
                              TARGET_ACC, seed=SEED + 1)
        r_non = run_federated(Xtr, ytr, Xte, yte, idx_non, E, B, LR,
                              TARGET_ACC, seed=SEED + 1)
        if label == "FedSGD":
            base_iid, base_non = r_iid, r_non
        results.append([label, E, B, u_of(E, B), r_iid, r_non])

    # ------------------------------- print table ---------------------------
    print("=" * 86)
    print("FedAvg toy reproduction of Table 2 -- communication rounds to reach "
          f"{TARGET_ACC*100:.0f}% test accuracy")
    print(f"softmax regression  |  d={D}  classes={NUM_CLASSES}  K={K} clients  "
          f"C={C}  lr={LR}  n_bar={N_BAR:.0f}  cond~{COND:.0f}")
    print(f"FedSGD = (E=1, B=inf).  u = nE/(KB) = (n_bar/B)*E.  "
          f"round cap = {MAX_ROUNDS}  (DNR = did not reach)")
    print("=" * 86)
    print(f"{'config':<8}{'E':>4}{'B':>6}{'u':>7} | "
          f"{'IID rounds':>12}{'speedup':>9} | {'NON-IID rounds':>16}{'speedup':>9}")
    print("-" * 86)

    def cell(r, base):
        txt = "DNR" if r > MAX_ROUNDS else str(r)
        if base is None or base > MAX_ROUNDS or r > MAX_ROUNDS:
            sp = "-"
        else:
            sp = f"{base / r:.1f}x"
        return txt, sp

    for label, E, B, u, r_iid, r_non in results:
        Bs = "inf" if B is None else str(B)
        ti, si = cell(r_iid, None if label == "FedSGD" else base_iid)
        tn, sn = cell(r_non, None if label == "FedSGD" else base_non)
        print(f"{label:<8}{E:>4}{Bs:>6}{u:>7.0f} | "
              f"{ti:>12}{si:>9} | {tn:>16}{sn:>9}")
    print("-" * 86)

    # ----------------------------- claim checks ----------------------------
    fa = results[1:]                                   # FedAvg rows
    fedsgd_iid, fedsgd_non = results[0][4], results[0][5]
    best_fedavg_iid = min(r[4] for r in fa)
    by_u = sorted(fa, key=lambda r: r[3])
    lowest_u, highest_u = by_u[0], by_u[-1]

    # (a) FedAvg << FedSGD on IID
    assert best_fedavg_iid < fedsgd_iid, "(a) FedAvg should beat FedSGD on IID"
    # (b) higher-u needs fewer IID rounds (extremes of the u-ordered sweep)
    assert highest_u[4] < lowest_u[4], "(b) highest-u should need fewer IID rounds"
    # (c) Pathological non-IID is HARDER. As in the paper's own Table 2 (where
    #     the FedSGD baseline can even be slightly FASTER on non-IID -- e.g. CNN
    #     626 IID vs 483 non-IID -- because the averaged global gradient barely
    #     depends on the partition), the effect lives in the FedAvg rows:
    #     (c1) every FedAvg config needs >= as many rounds on non-IID as on IID, and
    #     (c2) every FedAvg speedup vs FedSGD is SMALLER on non-IID than on IID
    #          ("the speedups for the pathologically partitioned non-IID data are
    #           smaller", Sec. 4).
    for label, E, B, u, r_iid, r_non in fa:
        assert r_non >= r_iid, f"(c1) {label} E={E} B={B}: non-IID needs >= IID rounds"
        sp_iid = fedsgd_iid / r_iid
        sp_non = fedsgd_non / r_non
        assert sp_non < sp_iid, f"(c2) {label} E={E} B={B}: non-IID speedup should be smaller"

    best_fedavg_non = min(r[5] for r in fa)
    print("CLAIM CHECKS -- ALL PASSED:")
    print(f"  (a) FedAvg << FedSGD on IID:   best FedAvg IID = {best_fedavg_iid} "
          f"rounds  vs  FedSGD IID = {fedsgd_iid}  -> {fedsgd_iid/best_fedavg_iid:.1f}x fewer rounds")
    print(f"  (b) higher u -> fewer rounds:  u={lowest_u[3]:.0f} needs {lowest_u[4]} IID "
          f"rounds, u={highest_u[3]:.0f} needs {highest_u[4]} IID rounds")
    print(f"  (c) non-IID is harder:         every FedAvg config needs >= IID rounds "
          f"(best FedAvg {best_fedavg_iid} IID -> {best_fedavg_non} non-IID),")
    print(f"                                 and every non-IID speedup is SMALLER than "
          f"its IID speedup (e.g. u={highest_u[3]:.0f}: "
          f"{fedsgd_iid/highest_u[4]:.1f}x IID vs {fedsgd_non/highest_u[5]:.1f}x non-IID).")
    print(f"      (note: the FedSGD baseline itself is partition-insensitive -- IID "
          f"{fedsgd_iid} vs non-IID {fedsgd_non} rounds -- exactly as in the paper's Table 2.)")
    print("=" * 86)
    print(f"total runtime: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()

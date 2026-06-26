"""heterogeneity-control-variate.py

Implements the "Heterogeneity-corrected local steps (control variates)" proposal
of the Mathematical Improvements section of 05-improvements.tex.

MOTIVATION (from the math deep dive, eq. 3.5)
---------------------------------------------
The per-round deviation of FedAvg from CENTRALIZED gradient descent is, to
leading order in the local learning rate,

    Delta = eta^2 * Cov_k(H_k, g_k) + O(eta^3),

a CLIENT-HETEROGENEITY term (the client-weighted covariance between local
Hessians H_k and local gradients g_k). It vanishes in the homogeneous/IID limit
(H_k = H, g_k = grad f) and is the formal counterpart of the paper's empirical
finding that pathological non-IID data degrades per-round progress (the
"client-drift" term).

PROPOSAL (SCAFFOLD-lite, control variates; \citep{karimireddy2020scaffold})
---------------------------------------------------------------------------
Give each client k a CONTROL VARIATE c_k (an estimate of its own local gradient
direction at the global model) and the server a control variate
c = (1/|S_t|) sum_{k in S_t} c_k. The corrected local SGD step replaces the raw
local gradient g_local(w) with the HETEROGENEITY-CORRECTED direction

    g_local(w) - c_k + c,

so that, in expectation, each client tracks the GLOBAL direction instead of its
own biased local one -- cancelling the leading Cov_k drift. This is the variance-
reduction (control-variate) alternative to FedProx's proximal penalty
(\citep{li2020fedprox}), which instead anchors the local iterate to the global
model with a quadratic term.

We reuse the toy federated SOFTMAX-REGRESSION setup of sandbox/toy_fedavg.py
(K=100, C=0.1, ill-conditioned Gaussian blobs, paper's pathological non-IID
partition) and measure ROUNDS-TO-TARGET on the NON-IID partition for
(A) vanilla FedAvg and (B) FedAvg + control variates, across a couple of (E,B)
settings.

EXPECT: the control-variate version needs FEWER (or equal) non-IID rounds.
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
C = 0.1             # fraction of clients sampled per round (paper's default)
LR = 1.0            # local SGD learning rate (tuned, fixed across configs)
COND = 30.0         # feature-scale spread (ill-conditioning) -> slow GD
TARGET_ACC = 0.95   # target test accuracy
MAX_ROUNDS = 600    # cap on communication rounds

N_BAR = N_TRAIN / K  # expected examples per client (= 60)


# --------------------------- synthetic data (blobs) ------------------------
def make_dataset(seed):
    """Anisotropic, ILL-CONDITIONED Gaussian blobs (same recipe as toy_fedavg)."""
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
    """Gradient of mean cross-entropy loss wrt W (d x C) on a minibatch."""
    m = Xb.shape[0]
    P = softmax(Xb @ W)
    Y = np.zeros((m, NUM_CLASSES))
    Y[np.arange(m), yb] = 1.0
    return Xb.T @ (P - Y) / m


def accuracy(W, X, y):
    return float((np.argmax(X @ W, axis=1) == y).mean())


# ------------------------------ partitioning -------------------------------
def partition_iid(y, k, rng):
    return [s for s in np.array_split(rng.permutation(len(y)), k)]


def partition_noniid(y, k, rng):
    """Pathological NON-IID (paper's MNIST recipe): sort by label, cut into 2k
    shards, give each client 2 random shards => mostly 2 classes per client."""
    order = np.argsort(y, kind="stable")
    shards = np.array_split(order, 2 * k)
    shard_ids = rng.permutation(2 * k)
    return [np.concatenate([shards[shard_ids[2 * c]], shards[shard_ids[2 * c + 1]]])
            for c in range(k)]


# ====================== (A) vanilla FedAvg local update ====================
def client_update_vanilla(W_global, Xk, yk, E, B, lr, rng):
    """E local epochs of minibatch SGD from W_global (B=None => full batch)."""
    W = W_global.copy()
    nk = Xk.shape[0]
    bsz = nk if (B is None or B >= nk) else B
    n_steps = 0
    for _ in range(E):
        perm = rng.permutation(nk)
        for start in range(0, nk, bsz):
            sel = perm[start:start + bsz]
            W -= lr * grad(W, Xk[sel], yk[sel])
            n_steps += 1
    return W, n_steps


# ============== (B) SCAFFOLD-lite control-variate local update =============
def client_update_scaffold(W_global, Xk, yk, E, B, lr, rng, c_k, c):
    """SCAFFOLD-lite local update. Each step uses the heterogeneity-corrected
    direction  g_local(w) - c_k + c  (Algorithm in \citep{karimireddy2020scaffold}).

    Returns the locally-updated weights, the NUMBER of local steps taken, and the
    refreshed client control variate c_k_new computed via SCAFFOLD's "Option II"
    (cheap, no extra forward pass):
        c_k_new = c_k - c + (W_global - W) / (n_steps * lr).
    The (-c_k + c) correction is the control-variate that cancels the leading
    Cov_k(H_k, g_k) heterogeneity drift of eq. (3.5)."""
    W = W_global.copy()
    nk = Xk.shape[0]
    bsz = nk if (B is None or B >= nk) else B
    correction = c - c_k                       # constant within the round
    n_steps = 0
    for _ in range(E):
        perm = rng.permutation(nk)
        for start in range(0, nk, bsz):
            sel = perm[start:start + bsz]
            g_local = grad(W, Xk[sel], yk[sel])
            W -= lr * (g_local - c_k + c)       # corrected direction
            n_steps += 1
    # SCAFFOLD Option II control-variate refresh (reuses the steps just taken).
    c_k_new = c_k - c + (W_global - W) / (max(1, n_steps) * lr)
    return W, n_steps, c_k_new


# ------------------------------ federated loop -----------------------------
def run_fedavg(Xtr, ytr, Xte, yte, client_idx, E, B, lr, target, seed):
    """Vanilla FederatedAveraging (Algorithm 1). Returns first round reaching
    `target`, else MAX_ROUNDS+1."""
    rng = np.random.default_rng(seed)
    W = np.zeros((D, NUM_CLASSES))
    m = max(1, int(round(C * K)))
    sizes = np.array([len(ix) for ix in client_idx])

    for t in range(1, MAX_ROUNDS + 1):
        S = rng.choice(K, size=m, replace=False)
        m_t = sizes[S].sum()
        W_new = np.zeros_like(W)
        for k in S:
            Wk, _ = client_update_vanilla(W, Xtr[client_idx[k]], ytr[client_idx[k]],
                                          E, B, lr, rng)
            W_new += (sizes[k] / m_t) * Wk     # corrected weighted aggregation
        W = W_new
        if accuracy(W, Xte, yte) >= target:
            return t
    return MAX_ROUNDS + 1


def run_scaffold(Xtr, ytr, Xte, yte, client_idx, E, B, lr, target, seed):
    """FedAvg + SCAFFOLD-lite control variates. Same aggregation, but local
    steps use g_local - c_k + c and control variates are maintained per round.
    Returns first round reaching `target`, else MAX_ROUNDS+1."""
    rng = np.random.default_rng(seed)
    W = np.zeros((D, NUM_CLASSES))
    m = max(1, int(round(C * K)))
    sizes = np.array([len(ix) for ix in client_idx])

    # Persistent control variates: one per client (c_k) + server (c).
    c_k_all = [np.zeros((D, NUM_CLASSES)) for _ in range(K)]
    c = np.zeros((D, NUM_CLASSES))             # server control variate

    for t in range(1, MAX_ROUNDS + 1):
        S = rng.choice(K, size=m, replace=False)
        m_t = sizes[S].sum()
        W_new = np.zeros_like(W)
        new_ck = {}
        for k in S:
            Wk, _, ck_new = client_update_scaffold(
                W, Xtr[client_idx[k]], ytr[client_idx[k]],
                E, B, lr, rng, c_k_all[k], c)
            W_new += (sizes[k] / m_t) * Wk     # corrected weighted aggregation
            new_ck[k] = ck_new
        W = W_new
        # Commit updated client control variates...
        for k in S:
            c_k_all[k] = new_ck[k]
        # ...and refresh the SERVER control variate as the mean over selected
        # clients' control variates: c = (1/|S_t|) sum_{k in S_t} c_k.
        c = sum(c_k_all[k] for k in S) / len(S)
        if accuracy(W, Xte, yte) >= target:
            return t
    return MAX_ROUNDS + 1


def u_of(E, B):
    b = N_BAR if B is None else B
    return (N_BAR / b) * E


# --------------------------------- main ------------------------------------
def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = make_dataset(SEED)

    part_rng = np.random.default_rng(SEED + 7)
    idx_non = partition_noniid(ytr, K, part_rng)   # focus: NON-IID partition

    # A couple of (E, B) settings (multiple local steps per round -> drift bites).
    configs = [
        ("E=5,  B=inf", 5, None),
        ("E=20, B=inf", 20, None),
        ("E=5,  B=10 ", 5, 10),
        ("E=20, B=10 ", 20, 10),
    ]

    results = []
    for label, E, B in configs:
        r_van = run_fedavg(Xtr, ytr, Xte, yte, idx_non, E, B, LR,
                           TARGET_ACC, seed=SEED + 1)
        r_scf = run_scaffold(Xtr, ytr, Xte, yte, idx_non, E, B, LR,
                             TARGET_ACC, seed=SEED + 1)
        results.append([label, E, B, u_of(E, B), r_van, r_scf])

    # ------------------------------- print table ---------------------------
    print("=" * 84)
    print("Heterogeneity-corrected local steps (SCAFFOLD-lite control variates)")
    print("Rounds to reach %.0f%% test accuracy on the PATHOLOGICAL NON-IID partition"
          % (TARGET_ACC * 100))
    print("softmax regression | d=%d classes=%d K=%d C=%.2f lr=%.1f cond~%.0f | cap=%d"
          % (D, NUM_CLASSES, K, C, LR, COND, MAX_ROUNDS))
    print("local step: vanilla uses g_local(w);  control-variate uses g_local(w) - c_k + c")
    print("=" * 84)
    header = ("%-12s%6s | %16s%18s%14s"
              % ("config", "u", "VANILLA rounds", "CTRL-VARIATE rounds",
                 "improvement"))
    print(header)
    print("-" * 84)

    def fmt(r):
        return "DNR" if r > MAX_ROUNDS else str(r)

    n_better_or_equal = 0
    n_strictly_better = 0
    for label, E, B, u, r_van, r_scf in results:
        if r_scf <= r_van:
            n_better_or_equal += 1
        if r_scf < r_van:
            n_strictly_better += 1
        if r_van > MAX_ROUNDS or r_scf > MAX_ROUNDS:
            imp = "-"
        else:
            imp = "%.2fx fewer (%+d)" % (r_van / r_scf, r_scf - r_van)
        print("%-12s%6.0f | %16s%18s%14s"
              % (label, u, fmt(r_van), fmt(r_scf), imp))
    print("-" * 84)

    # ----------------------------- verdict ---------------------------------
    finite = [(lab, rv, rs) for lab, _, _, _, rv, rs in results
              if rv <= MAX_ROUNDS and rs <= MAX_ROUNDS]
    tot_van = sum(rv for _, rv, _ in finite)
    tot_scf = sum(rs for _, _, rs in finite)
    print("SUMMARY (non-IID):")
    print("  control-variate <= vanilla on %d / %d configs (strictly fewer on %d)"
          % (n_better_or_equal, len(results), n_strictly_better))
    if finite:
        print("  total rounds over %d finite configs:  vanilla = %d   control-variate = %d"
              "   (%.2fx fewer)"
              % (len(finite), tot_van, tot_scf, tot_van / max(1, tot_scf)))
    # PASS criterion: control variates never worse, and strictly help overall.
    passed = (n_better_or_equal == len(results)) and (tot_scf <= tot_van) \
        and (n_strictly_better >= 1)
    print("=" * 84)
    print("VERDICT: %s -- control variates need FEWER-OR-EQUAL non-IID rounds than "
          "vanilla FedAvg" % ("PASS" if passed else "FAIL"))
    print("(cancels the leading eta^2 * Cov_k(H_k, g_k) heterogeneity drift of eq. 3.5)")
    print("total runtime: %.2fs" % (time.time() - t0))
    assert passed, "control variates should match-or-beat vanilla FedAvg on non-IID"


if __name__ == "__main__":
    main()

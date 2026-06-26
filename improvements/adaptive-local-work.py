"""
adaptive-local-work.py -- Implements the "Adaptive local computation
(decay E / grow B)" proposal of 05-improvements.tex (Experimental Extensions),
for "Communication-Efficient Learning of Deep Networks from Decentralized Data"
(McMahan et al., 2017 -- the FedAvg paper).

MOTIVATION (the gap in the paper)
---------------------------------
The paper's Figure 3 shows that for some tasks a LARGE number of local epochs E
can make FedAvg PLATEAU (and they note some configs even diverge): each client
runs many local SGD steps, over-optimizes its OWN local objective, and drifts
out of the shared basin. Averaging heavily-drifted local optima then stalls
short of the global optimum -- the failure mode is the same one that sinks
naive ONE-SHOT averaging (sum local solutions once). McMahan et al. explicitly
SUGGEST "it may be useful to decay the amount of local computation per round
(by decaying E or increasing B) ... analogous to decaying learning rates" but
run NO such experiment. This prototype runs it.

PROPOSAL + TEST
---------------
Treat local work u = (n_bar/B)*E as a hyperparameter you SCHEDULE over rounds,
exactly like a learning rate: start LARGE (big E) for fast early progress out of
the trivial init, then DECAY E toward FedSGD-like steps (E=1) late, so the
clients no longer over-shoot their local optima once the global model nears the
shared solution. We compare, on a FIXED budget of R communication rounds and a
strongly NON-IID partition, the FINAL test accuracy of:
  (a) fixed SMALL E      -- steady but slow; may not finish in R rounds,
  (b) fixed LARGE E      -- fast at first then PLATEAUS below the optimum
                            (client drift; the Fig-3 pathology),
  (c) DECAYED E schedule -- large -> small over rounds.
EXPECT: (c) reaches the HIGHEST final accuracy, and (b) visibly plateaus.

ENGINEERING THE PLATEAU (noted in notes[])
------------------------------------------
A convex softmax problem with the paper's mild 2-class-per-client shards and
WELL-SEPARATED blobs does NOT drift -- there large E only helps, because each
client's local optimum is still roughly compatible with the global one and the
average converges. To surface the large-E pathology in a fast convex toy we
make TWO things severe at once:
  * 1 class per client (extreme label skew), so each client's local objective
    is a degenerate "push ALL probability mass onto my single class" problem,
    whose optimum is at infinity (logits -> +inf for that class); and
  * HEAVILY OVERLAPPING blobs (small class separation `SEP`), so the GLOBAL
    optimum is a delicate COMPROMISE among classes -- exactly the regime where
    a client that over-optimizes its 1-class objective lands far from that
    compromise (large, conflicting weight vectors).
With a fixed, un-decayed local lr, large E then drives each sampled client
deep toward its degenerate local optimum; averaging a handful of such large,
mutually-inconsistent vectors yields a high-variance step that STALLS and even
regresses -- the convex analogue of the paper's Fig-3 plateau. Small/decayed E
keep clients near the shared compromise. The drift is real (it is the SAME
client-drift mechanism \\citep{karimireddy2020scaffold} target); we merely tune
SEP/lr/E so it appears within a <60s convex toy. Engineered, by design.

This is a clean drop-in alternative to the heavier heterogeneity fixes
\\citep{li2020fedprox} (proximal term that penalizes local drift) and
\\citep{karimireddy2020scaffold} (control variates that correct client drift):
adaptive-E needs NO extra state or communication -- only a server-side schedule.
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
LR = 4.0            # local SGD learning rate (FIXED, un-decayed, across arms)
COND = 30.0         # feature-scale spread (ill-conditioning)
SEP = 0.25          # class-center separation: SMALL => heavily overlapping blobs
                    # => global optimum is a compromise => 1-class over-opt drifts
ROUNDS = 120        # FIXED communication-round budget (same for every arm)
B = None            # full-batch local SGD (B=inf) -> E is the sole local-work knob

N_BAR = N_TRAIN / K  # expected examples per client (= 60)


# --------------------------- synthetic data (blobs) ------------------------
def make_dataset(seed):
    """Anisotropic, ILL-CONDITIONED, HEAVILY-OVERLAPPING Gaussian blobs shared by
    train and test. SEP<1 shrinks the class centers relative to the unit blob
    spread, so classes overlap and the Bayes-optimal softmax is a compromise --
    the regime where a 1-class client that over-optimizes drifts far from it."""
    rng = np.random.default_rng(seed)
    scales = np.geomspace(1.0, 1.0 / COND, D)
    centers = rng.normal(0.0, SEP, size=(NUM_CLASSES, D)) * scales

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
def partition_noniid_1class(y, k, rng):
    """EXTREME non-IID: sort by label, cut into k contiguous shards, give each
    client ONE shard => (almost) 1 class per client. This is the paper's
    'pathological' recipe pushed to its 1-shard-per-client limit, which is what
    makes large-E local optima drift far enough to plateau the average."""
    order = np.argsort(y, kind="stable")
    shards = np.array_split(order, k)
    shard_ids = rng.permutation(k)
    return [shards[shard_ids[c]] for c in range(k)]


# ------------------------------ local client SGD ---------------------------
def client_update(W_global, Xk, yk, E, B, lr, rng):
    """E local epochs of (mini)batch SGD from W_global (B=None => full batch)."""
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
def run_federated(Xtr, ytr, Xte, yte, client_idx, E_schedule, B, lr, seed,
                  trace_every=0):
    """FederatedAveraging (Algorithm 1) for len(E_schedule) rounds; E_schedule[t]
    is the number of local epochs used in round t (a CONSTANT list reproduces a
    fixed-E arm; a decaying list is the proposal). Returns (final_acc, trace)
    where trace = [(round, acc), ...] sampled every `trace_every` rounds."""
    rng = np.random.default_rng(seed)
    W = np.zeros((D, NUM_CLASSES))                 # SHARED initialization
    m = max(1, int(round(C * K)))
    sizes = np.array([len(ix) for ix in client_idx])
    trace = []
    for t, E in enumerate(E_schedule, start=1):
        S = rng.choice(K, size=m, replace=False)
        m_t = sizes[S].sum()
        W_new = np.zeros_like(W)
        for k in S:
            Wk = client_update(W, Xtr[client_idx[k]], ytr[client_idx[k]],
                               E, B, lr, rng)
            W_new += (sizes[k] / m_t) * Wk         # corrected weighted average
        W = W_new
        if trace_every and (t % trace_every == 0 or t == 1):
            trace.append((t, accuracy(W, Xte, yte)))
    return accuracy(W, Xte, yte), trace


# ------------------------------ E schedules --------------------------------
def fixed_schedule(E, rounds):
    return [E] * rounds


def decayed_schedule(E_hi, E_lo, rounds):
    """Step-decay E from E_hi down to E_lo over `rounds`, geometric-ish in equal
    blocks (analogous to a step learning-rate decay). Spends the early rounds on
    cheap fast progress (big E) and the late rounds near FedSGD (E=E_lo)."""
    levels = []
    e = E_hi
    while e > E_lo:
        levels.append(e)
        e = max(E_lo, e // 2)
    levels.append(E_lo)
    block = rounds // len(levels)
    sched = []
    for i, lev in enumerate(levels):
        n = block if i < len(levels) - 1 else rounds - block * (len(levels) - 1)
        sched.extend([lev] * n)
    return sched


def avg_u(E_schedule):
    """Mean local updates/round u = (n_bar/B)*E over the schedule (B=inf => B=n_bar)."""
    b = N_BAR if B is None else B
    return float(np.mean([(N_BAR / b) * E for E in E_schedule]))


# ----------------------- MEASUREMENT (Stage 7 F.2) -------------------------
# Headline hyperparameters for the measurement, shared by measure() and main()
# so the printed table and the returned dict are GUARANTEED consistent.
E_SMALL = 1            # fixed small E (FedSGD-like steps; slow, noisy)
E_LARGE = 100          # fixed LARGE E (over-optimizes local shard -> plateau)
E_HI, E_LO = 40, 1     # decayed: start at large E, end at FedSGD-like E
FIXED_ES = [1, 2, 5, 10, 40, 100]   # sweep to locate the best CONSTANT E


def measure():
    """Deterministic quantitative comparison (Stage 7 F.2 MEASUREMENT mode).

    Re-runs the experiment under the module-level fixed SEED and returns the
    FINAL test accuracies (as fractions in [0,1]) for the four headline arms
    plus the centralized ceiling, together with the bookkeeping needed to judge
    PASS (decayed E beats every fixed E AND large-E plateaus). The result is a
    pure dict with no printing, so it can be asserted on / table-formatted by
    callers. Deterministic: same SEED => same numbers on every run.

    Returns dict with keys:
      fixed_small_E_acc, best_fixed_E_acc, best_fixed_E, fixed_large_E_acc,
      decayed_E_acc, centralized_ceiling, E_small, E_large, E_hi, E_lo,
      fixed_sweep (dict E->acc), large_early_gain, large_late_gain,
      decayed_wins (bool), large_plateaus (bool), passed (bool).
    """
    Xtr, ytr, Xte, yte = make_dataset(SEED)
    part_rng = np.random.default_rng(SEED + 7)
    idx_non = partition_noniid_1class(ytr, K, part_rng)

    # Centralized oracle ceiling (no federation): contextualizes every arm.
    W_oracle = np.zeros((D, NUM_CLASSES))
    for _ in range(3000):
        W_oracle -= 2.0 * grad(W_oracle, Xtr, ytr)
    ceiling = accuracy(W_oracle, Xte, yte)

    sched_decay = decayed_schedule(E_HI, E_LO, ROUNDS)

    # Full fixed-E sweep (so "best fixed E" is genuine, not a strawman).
    fixed_results = []  # (E, final_acc, trace)
    for E in FIXED_ES:
        acc, trace = run_federated(Xtr, ytr, Xte, yte, idx_non,
                                   fixed_schedule(E, ROUNDS), B, LR,
                                   seed=SEED + 1, trace_every=ROUNDS // 6)
        fixed_results.append((E, acc, trace))

    small_acc, _ = next((a, t) for E, a, t in fixed_results if E == E_SMALL)
    large_acc, large_tr = next((a, t) for E, a, t in fixed_results if E == E_LARGE)
    decay_acc, _ = run_federated(Xtr, ytr, Xte, yte, idx_non, sched_decay,
                                 B, LR, seed=SEED + 1,
                                 trace_every=ROUNDS // 6)

    best_E, best_fixed, _ = max(fixed_results, key=lambda r: r[1])

    # large-E plateau bookkeeping: most progress is EARLY; late gain is tiny/neg.
    ltr = dict(large_tr)
    lr_rounds = sorted(ltr)
    mid = lr_rounds[len(lr_rounds) // 2]
    early_gain = ltr[mid] - ltr[lr_rounds[0]]
    late_gain = ltr[lr_rounds[-1]] - ltr[mid]

    decayed_wins = bool(decay_acc >= best_fixed - 1e-9)
    large_plateaus = bool((late_gain < early_gain) and (large_acc < decay_acc))

    return {
        "fixed_small_E_acc": float(small_acc),
        "best_fixed_E_acc": float(best_fixed),
        "best_fixed_E": int(best_E),
        "fixed_large_E_acc": float(large_acc),
        "decayed_E_acc": float(decay_acc),
        "centralized_ceiling": float(ceiling),
        "E_small": int(E_SMALL),
        "E_large": int(E_LARGE),
        "E_hi": int(E_HI),
        "E_lo": int(E_LO),
        "fixed_sweep": {int(E): float(a) for E, a, _ in fixed_results},
        "large_early_gain": float(early_gain),
        "large_late_gain": float(late_gain),
        "decayed_wins": decayed_wins,
        "large_plateaus": large_plateaus,
        "passed": bool(decayed_wins and large_plateaus),
    }


# --------------------------------- main ------------------------------------
def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = make_dataset(SEED)
    part_rng = np.random.default_rng(SEED + 7)
    idx_non = partition_noniid_1class(ytr, K, part_rng)

    # Oracle ceiling: best achievable test acc by CENTRALIZED training on all
    # data (no federation). Contextualizes how far each arm gets.
    W_oracle = np.zeros((D, NUM_CLASSES))
    for _ in range(3000):
        W_oracle -= 2.0 * grad(W_oracle, Xtr, ytr)
    ceiling = accuracy(W_oracle, Xte, yte)

    E_SMALL = 1            # fixed small E (FedSGD-like steps; slow, noisy)
    E_LARGE = 100          # fixed LARGE E (over-optimizes local shard -> plateau)
    E_HI, E_LO = 40, 1     # decayed: start at large E, end at FedSGD-like E

    # A small SWEEP of fixed E so "best fixed E" is genuinely the best constant
    # choice, not a strawman. The middle entries expose the interior sweet spot.
    fixed_Es = [1, 2, 5, 10, 40, 100]
    sched_decay = decayed_schedule(E_HI, E_LO, ROUNDS)

    # Run the full fixed-E sweep.
    fixed_results = []  # (E, final_acc, trace)
    for E in fixed_Es:
        acc, trace = run_federated(Xtr, ytr, Xte, yte, idx_non,
                                   fixed_schedule(E, ROUNDS), B, LR,
                                   seed=SEED + 1, trace_every=ROUNDS // 6)
        fixed_results.append((E, acc, trace))

    # The three headline arms (small / large / decayed).
    small_acc, small_tr = next((a, t) for E, a, t in fixed_results if E == E_SMALL)
    large_acc, large_tr = next((a, t) for E, a, t in fixed_results if E == E_LARGE)
    decay_acc, decay_tr = run_federated(Xtr, ytr, Xte, yte, idx_non, sched_decay,
                                        B, LR, seed=SEED + 1,
                                        trace_every=ROUNDS // 6)

    arms = [
        ("fixed small E", str(E_SMALL), fixed_schedule(E_SMALL, ROUNDS),
         small_acc, small_tr),
        ("fixed LARGE E", str(E_LARGE), fixed_schedule(E_LARGE, ROUNDS),
         large_acc, large_tr),
        ("DECAYED E", f"{E_HI}->{E_LO}", sched_decay, decay_acc, decay_tr),
    ]

    # ------------------------------- print -------------------------------
    line = "=" * 78
    print(line)
    print("Adaptive local computation (decay E) on EXTREME non-IID FedAvg")
    print(f"softmax regression | d={D} classes={NUM_CLASSES} K={K} C={C} lr={LR} "
          f"(fixed) | 1 class/client | overlapping blobs (sep={SEP})")
    print(f"rounds={ROUNDS} (fixed budget, same for every arm)  |  "
          f"centralized ceiling = {ceiling*100:.2f}%")
    print(line)
    print(f"{'arm':<16}{'E':>8}{'avg u':>9}{'final acc':>12}")
    print("-" * 78)
    for name, Etag, sched, acc, _ in arms:
        print(f"{name:<16}{Etag:>8}{avg_u(sched):>9.1f}{acc*100:>11.2f}%")
    print("-" * 78)
    print("fixed-E sweep (final test acc %, to locate the best constant E):")
    print("  " + "   ".join(f"E={E}:{a*100:5.2f}" for E, a, _ in fixed_results))
    best_E, best_fixed, _ = max(fixed_results, key=lambda r: r[1])
    print(f"  best fixed E = {best_E} -> {best_fixed*100:.2f}%")

    print("\naccuracy-vs-round trace (test acc %):")
    rounds_axis = sorted({r for _, _, _, _, tr in arms for r, _ in tr})
    print("  round:" + "".join(f"{r:>8}" for r in rounds_axis))
    for name, Etag, sched, acc, trace in arms:
        d = dict(trace)
        row = "".join(f"{d.get(r, float('nan'))*100:>8.2f}" for r in rounds_axis)
        print(f"  {name:<14}" + row)

    # ------------------------------- verdict -----------------------------
    # large-E plateau: most of its progress is made EARLY; the 2nd-half gain is
    # tiny (or negative) AND it finishes below the decayed schedule.
    ltr = dict(large_tr)
    lr_rounds = sorted(ltr)
    mid = lr_rounds[len(lr_rounds) // 2]
    early_gain = ltr[mid] - ltr[lr_rounds[0]]
    late_gain = ltr[lr_rounds[-1]] - ltr[mid]

    print("\n" + line)
    print("VERDICT")
    print(f"  fixed small E ={E_SMALL:>4}  final acc = {small_acc*100:6.2f}%")
    print(f"  fixed LARGE E ={E_LARGE:>4}  final acc = {large_acc*100:6.2f}%   "
          f"(early gain {early_gain*100:+.2f} pts, then late gain only "
          f"{late_gain*100:+.2f} pts -> PLATEAU)")
    print(f"  DECAYED  E    ={E_HI:>3}->{E_LO}  final acc = {decay_acc*100:6.2f}%")
    print(f"  best FIXED-E  (E={best_E:>3})  final acc = {best_fixed*100:6.2f}%")
    decayed_wins = decay_acc >= best_fixed - 1e-9
    large_plateaus = (late_gain < early_gain) and (large_acc < decay_acc)
    print(f"  -> DECAYED E >= best fixed-E in sweep : {decayed_wins} "
          f"({decay_acc*100:.2f}% vs {best_fixed*100:.2f}%, "
          f"+{(decay_acc-best_fixed)*100:.2f} pts)")
    print(f"  -> large-E fast-then-plateau          : {large_plateaus} "
          f"(late gain {late_gain*100:+.2f} < early gain {early_gain*100:+.2f}, "
          f"ends {(large_acc-decay_acc)*100:+.2f} pts vs decay)")
    verdict = decayed_wins and large_plateaus
    print(f"\n  PASS: {verdict}  "
          f"(decayed E beats EVERY fixed E AND large-E plateau reproduced)")
    print(line)

    # ----------------- MEASUREMENT mode (Stage 7 F.2) --------------------
    # Deterministic quantitative comparison via measure(); printed as a table.
    M = measure()
    print("\n" + line)
    print("MEASUREMENT (Stage 7 F.2): deterministic final-acc comparison")
    print(line)
    print(f"{'arm':<22}{'config':>10}{'final test acc':>18}")
    print("-" * 78)
    print(f"{'fixed small E':<22}{('E='+str(M['E_small'])):>10}"
          f"{M['fixed_small_E_acc']*100:>17.2f}%")
    print(f"{'best fixed E':<22}{('E='+str(M['best_fixed_E'])):>10}"
          f"{M['best_fixed_E_acc']*100:>17.2f}%")
    print(f"{'fixed LARGE E':<22}{('E='+str(M['E_large'])):>10}"
          f"{M['fixed_large_E_acc']*100:>17.2f}%")
    print(f"{'DECAYED E schedule':<22}{(str(M['E_hi'])+'->'+str(M['E_lo'])):>10}"
          f"{M['decayed_E_acc']*100:>17.2f}%")
    print("-" * 78)
    print(f"{'centralized ceiling':<22}{'-':>10}"
          f"{M['centralized_ceiling']*100:>17.2f}%")
    print(line)
    print(f"  decayed_wins   = {M['decayed_wins']}  "
          f"(decayed {M['decayed_E_acc']*100:.2f}% vs best fixed "
          f"{M['best_fixed_E_acc']*100:.2f}%)")
    print(f"  large_plateaus = {M['large_plateaus']}  "
          f"(early gain {M['large_early_gain']*100:+.2f} pts, "
          f"late gain {M['large_late_gain']*100:+.2f} pts)")
    print(f"  measure() passed = {M['passed']}")
    print(line)

    print(f"total runtime: {time.time() - t0:.2f}s")
    return verdict


if __name__ == "__main__":
    main()

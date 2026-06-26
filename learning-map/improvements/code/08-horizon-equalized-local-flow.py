"""Horizon-Equalized Local Flow -- concept 08-horizon-equalized-local-flow of the
improvements learning map.

A round of local SGD is forward-Euler integration of dw/dt = -grad F_k(w) for
physical time T_k = eta*u_k = eta*E*n_k/B. Vanilla FedAvg fixes (E,B,eta) globally,
so under client-size imbalance T_k VARIES, injecting a first-order size-imbalance
drift -Cov_k(T_k,g_k) (the unequal-horizon remark of node 07's effective-ODE
bound). Fix: per-client eta_k = T*/u_k with T* = eta*E*n_bar/B, so T_k = T* for all
clients (total flow-time unchanged on average; only its distribution equalized).

This tiny FedAvg witness runs BASELINE (global eta) vs PROPOSED (eta_k = T*/u_k) on
a pathological non-IID partition with log-uniform client sizes; both arms share the
partition + sampling seed and differ ONLY in eta_k. It is a clean no-op when
balanced and needs fewer mean rounds as imbalance grows.
Runnable code analog of concepts/08-horizon-equalized-local-flow.md.
"""
import numpy as np

np.random.seed(0)
np.seterr(over="ignore", divide="ignore", invalid="ignore")

D, C_CLS, K, C, LR = 12, 6, 40, 0.2, 4.0      # small toy softmax-regression FedAvg
E, B, R, TARGET = 5, 10, 150, 0.90            # local epochs, batch, round cap, target acc
N_TRAIN, N_TEST = 2400, 1200
N_BAR = N_TRAIN / K                            # mean examples per client
REALIZATIONS = 6                               # average out integer-round noise
SEP = 2.5                                      # blob separation (headroom above target)


def make_data(n, rng, centers):
    y = rng.integers(0, C_CLS, size=n)
    X = centers[y] + rng.normal(0, 1.0, size=(n, D)) * np.geomspace(1, 0.05, D)
    return X, y


def softmax(z):
    z = np.clip(z - z.max(1, keepdims=True), -60, 60)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def grad(W, X, y):
    P = softmax(X @ W)
    Y = np.zeros_like(P)
    Y[np.arange(len(y)), y] = 1.0
    return X.T @ (P - Y) / len(y)


def acc(W, X, y):
    return float((np.argmax(X @ W, 1) == y).mean())


def u_of(nk):
    """u_k = E * ceil(n_k / B): forward-Euler steps client k runs per round."""
    return E * int(np.ceil(nk / min(B, nk)))


def client_step(W, X, y, lr):                  # E epochs of minibatch SGD, lr per step
    W = W.copy()
    nk = len(y)
    bsz = min(B, nk)
    rng = np.random.default_rng(0)
    for _ in range(E):
        perm = rng.permutation(nk)
        for s in range(0, nk, bsz):
            sel = perm[s:s + bsz]
            W -= lr * grad(W, X[sel], y[sel])
    return W


def partition(y, rng, spread):
    """Pathological non-IID (1-2 classes/client) with log-uniform sizes; mean ~N_BAR.
    spread = max/min size ratio; spread<=1 => balanced (all sizes == N_BAR)."""
    order = np.argsort(y, kind="stable")
    shards = np.array_split(order, K)
    pools = [shards[i] for i in rng.permutation(K)]
    if spread <= 1.0:
        sizes = np.full(K, int(round(N_BAR)), dtype=int)
    else:
        raw = np.exp(rng.uniform(0.0, np.log(spread), size=K))
        sizes = np.clip(np.round(raw / raw.mean() * N_BAR), 4, None).astype(int)
    parts = []
    for c in range(K):
        pool = pools[c]
        nk = int(sizes[c])
        sel = rng.choice(pool, size=nk, replace=(nk > len(pool)))
        parts.append(sel)
    return parts


def fed_run(parts, Xtr, ytr, Xte, yte, seed, equalize):
    rng = np.random.default_rng(seed)
    W = np.zeros((D, C_CLS))                    # SHARED init
    m = max(1, int(round(C * K)))
    sizes = np.array([len(p) for p in parts])
    T_star = LR * u_of(int(round(N_BAR)))       # horizon of an avg-size client
    for t in range(1, R + 1):
        S = rng.choice(K, m, replace=False)
        Wn = np.zeros_like(W)
        for k in S:
            lr_k = (T_star / u_of(sizes[k])) if equalize else LR
            Wk = client_step(W, Xtr[parts[k]], ytr[parts[k]], lr_k)
            Wn += sizes[k] / sizes[S].sum() * Wk    # corrected weighted average
        W = Wn
        if acc(W, Xte, yte) >= TARGET:
            return t
    return R + 1


def main():
    rng = np.random.default_rng(0)
    centers = rng.normal(0, SEP, (C_CLS, D)) * np.geomspace(1, 0.05, D)
    Xtr, ytr = make_data(N_TRAIN, rng, centers)
    Xte, yte = make_data(N_TEST, rng, centers)

    print("Horizon-equalized local flow (05-improvements.tex E.3)")
    print("eta_k = T*/u_k so T_k = eta_k*u_k = T* for every client (T* = eta*E*n_bar/B).")
    print("Rounds to target, pathological NON-IID, imbalanced (log-uniform) sizes:")
    print("  both arms share partition+sampling seed; differ ONLY in eta_k.\n")
    print("  spread~  n_min  n_max | BASE rnds  PROP rnds   mean dlt   W/L/T   faster?")
    print("  " + "-" * 70)

    deltas, spreads = [], []
    balanced_delta = None
    for spread in (1.0, 5.0, 20.0, 80.0):
        b_rounds, p_rounds = [], []
        nmins, nmaxs, real = [], [], []
        w = l = ti = 0
        for s in range(REALIZATIONS):
            prng = np.random.default_rng(1000 + s)
            parts = partition(ytr, prng, spread)
            sizes = np.array([len(p) for p in parts])
            nmins.append(int(sizes.min())); nmaxs.append(int(sizes.max()))
            real.append(sizes.max() / max(1, sizes.min()))
            rb = fed_run(parts, Xtr, ytr, Xte, yte, 2000 + s, equalize=False)
            rp = fed_run(parts, Xtr, ytr, Xte, yte, 2000 + s, equalize=True)
            b_rounds.append(rb); p_rounds.append(rp)
            w += rb > rp; l += rb < rp; ti += rb == rp
        bm, pm = float(np.mean(b_rounds)), float(np.mean(p_rounds))
        dlt = bm - pm
        rs = float(np.mean(real))
        if spread <= 1.0:
            balanced_delta = (dlt, l)
        else:
            deltas.append(dlt); spreads.append(rs)
        verdict = "PROPOSED" if dlt > 1e-9 else ("baseline" if dlt < -1e-9 else "tie")
        print("  %7.1f%6d%7d | %9.2f%10.2f%11.2f  %d/%d/%d %9s"
              % (rs, min(nmins), max(nmaxs), bm, pm, dlt, w, l, ti, verdict))
    print("  " + "-" * 70)

    # prediction checks (mirror the prototype's P1/P2/P3)
    p1 = abs(balanced_delta[0]) <= 0.5 and balanced_delta[1] == 0
    p2 = all(d > 1e-9 for d in deltas)
    p3 = all(deltas[i + 1] >= deltas[i] - 1e-9 for i in range(len(deltas) - 1))
    print("  (P1) balanced no-op (|dlt|<=0.5, 0 losses): dlt=%.2f -> %s"
          % (balanced_delta[0], "OK" if p1 else "FAIL"))
    print("  (P2) faster on every imbalanced spread: deltas=%s -> %s"
          % ([round(d, 2) for d in deltas], "OK" if p2 else "FAIL"))
    print("  (P3) margin widens with imbalance: %s -> %s"
          % ([round(d, 2) for d in deltas], "OK" if p3 else "FAIL"))
    ok = p1 and p2 and p3
    print("VERDICT: %s -- no-op when balanced; fewer mean rounds under imbalance, "
          "margin widens monotonically." % ("PASS" if ok else "MIXED/FAIL"))


if __name__ == "__main__":
    main()

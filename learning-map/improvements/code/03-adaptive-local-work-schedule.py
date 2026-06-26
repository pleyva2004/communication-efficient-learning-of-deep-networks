"""Adaptive Local-Work Schedule -- concept 03-adaptive-local-work-schedule of the
improvements learning map. Schedule the per-round local epochs E_t like a learning
rate (large early, decay toward FedSGD late) to dodge the large-E client-drift
plateau the FedAvg paper flags but never fixes (05-improvements.tex E.1).
Runnable code analog of concepts/03-adaptive-local-work-schedule.md.
"""
import numpy as np

np.random.seed(0)
np.seterr(over="ignore", divide="ignore", invalid="ignore")

D, C_CLS, K, C, LR, SEP = 12, 6, 40, 0.2, 4.0, 0.25  # small toy; SEP<1 => overlap
R, E_MIN, E_MAX, TAU = 60, 1, 64, 12                  # round budget + schedule knobs


def schedule(t):
    """E_t = max(E_min, floor(E_max * 2^(-floor(t/tau))))  (05-improvements.tex E.1)."""
    return max(E_MIN, int(E_MAX * 2.0 ** (-(t // TAU))))


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


def client_step(W, X, y, E):  # full-batch local SGD, E epochs from shared W
    W = W.copy()
    for _ in range(E):
        W -= LR * grad(W, X, y)
    return W


def fed_run(E_of_t, parts, Xtr, ytr, Xte, yte):
    rng = np.random.default_rng(1)
    W = np.zeros((D, C_CLS))                       # shared init each run
    m = max(1, int(round(C * K)))
    sizes = np.array([len(p) for p in parts])
    for t in range(R):
        S = rng.choice(K, m, replace=False)
        Wn = np.zeros_like(W)
        for k in S:                                # corrected n_k/m_t average
            Wk = client_step(W, Xtr[parts[k]], ytr[parts[k]], E_of_t(t))
            Wn += sizes[k] / sizes[S].sum() * Wk
        W = Wn
    return acc(W, Xte, yte)


def main():
    rng = np.random.default_rng(0)
    centers = rng.normal(0, SEP, (C_CLS, D)) * np.geomspace(1, 0.05, D)
    Xtr, ytr = make_data(2400, rng, centers)
    Xte, yte = make_data(1200, rng, centers)
    order = np.argsort(ytr, kind="stable")         # extreme non-IID: 1 class/client
    shards = np.array_split(order, K)
    parts = [shards[i] for i in rng.permutation(K)]

    print("E_t schedule  E_t = max(%d, floor(%d * 2^-floor(t/%d))):" % (E_MIN, E_MAX, TAU))
    print("  " + " ".join("t=%d:E=%d" % (t, schedule(t)) for t in range(0, R, TAU)))
    decayed = fed_run(schedule, parts, Xtr, ytr, Xte, yte)
    print("Final test accuracy over R=%d rounds, extreme non-IID FedAvg:" % R)
    best_fixed, best_E = -1.0, None
    for E in (1, 4, 16, 64):
        a = fed_run(lambda t, E=E: E, parts, Xtr, ytr, Xte, yte)
        tag = "  (fixed large-E plateau)" if E == E_MAX else ""
        print("  fixed E=%-3d  acc=%.2f%%%s" % (E, 100 * a, tag))
        if a > best_fixed:
            best_fixed, best_E = a, E
    print("  DECAYED %d->%d acc=%.2f%%  (best fixed=E=%d:%.2f%%)"
          % (E_MAX, E_MIN, 100 * decayed, best_E, 100 * best_fixed))
    print("VERDICT: decayed schedule beats best fixed E: %s (+%.2f pts)"
          % (decayed >= best_fixed - 1e-9, 100 * (decayed - best_fixed)))


if __name__ == "__main__":
    main()

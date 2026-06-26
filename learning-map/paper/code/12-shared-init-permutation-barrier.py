"""Shared-Init & the Permutation Barrier -- concept 12-shared-init-permutation-barrier of the paper learning map.

A 1-hidden-layer MLP's function is invariant under permuting its hidden units, so
independent inits land in incompatible orbit elements and naive averaging crosses a
loss barrier (avg worse than both parents); a SHARED init keeps both in one basin so
the average beats both. Runnable code analog of concepts/12-shared-init-permutation-barrier.md.
"""
import numpy as np

np.random.seed(0)
np.seterr(all="ignore")  # silence spurious BLAS matmul FPE warnings (numpy 2.0.x; values stay finite)


def make_data(n=400, d=6):
    X = np.random.randn(n, d)
    y = (X @ np.random.randn(d) + 0.3 * np.random.randn(n) > 0).astype(np.float64)
    return X, y


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def forward(p, X):                                   # invariant under hidden-unit perm
    return sigmoid(np.tanh(X @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]).ravel()


def loss(p, X, y):
    yh = np.clip(forward(p, X), 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(yh) + (1 - y) * np.log(1 - yh)))


def init(d, H):
    return {"W1": np.random.randn(d, H) * 0.5, "b1": np.zeros(H),
            "W2": np.random.randn(H) * 0.5, "b2": 0.0}


def train(p, X, y, epochs=400, eta=0.5):
    p = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in p.items()}
    n = len(y)
    for _ in range(epochs):
        h = np.tanh(X @ p["W1"] + p["b1"])
        e = (sigmoid(h @ p["W2"] + p["b2"]).ravel() - y) / n
        dh = np.outer(e, p["W2"]) * (1 - h ** 2)
        p["W2"] -= eta * (h.T @ e)
        p["b2"] -= eta * float(e.sum())
        p["W1"] -= eta * (X.T @ dh)
        p["b1"] -= eta * dh.sum(axis=0)
    return p


def avg(pa, pb):
    return {k: 0.5 * (pa[k] + pb[k]) for k in pa}


def barrier(shared, X, y, d, H):
    p0a = init(d, H)
    p0b = p0a if shared else init(d, H)              # SAME or independent seed
    half = len(y) // 2
    pa = train(p0a, X[:half], y[:half])              # disjoint halves
    pb = train(p0b, X[half:], y[half:])
    la, lb = loss(pa, X, y), loss(pb, X, y)          # full-set loss of each parent
    lavg = loss(avg(pa, pb), X, y)                    # full-set loss of average
    return lavg - max(la, lb), la, lb, lavg


def main():
    X, y = make_data()
    d, H = X.shape[1], 8
    print("FedAvg Fig.1: averaging two 1-hidden-layer MLPs trained on disjoint halves.")
    print("Barrier height = loss(average) - max(loss(parents)); >0 means avg is WORSE.\n")
    for shared in (False, True):
        bh, la, lb, lavg = barrier(shared, X, y, d, H)
        tag = "SHARED init " if shared else "INDEPENDENT inits"
        verdict = "BARRIER (avg worse)" if bh > 0 else "VALLEY (avg better)"
        print("%s : parents=(%.4f, %.4f)  avg=%.4f  barrier=%+.4f  -> %s"
              % (tag, la, lb, lavg, bh, verdict))
    print("\nWitness: independent inits occupy incompatible permutation-orbit elements")
    print("(barrier > 0); a shared init keeps both clients in one basin (barrier < 0).")


if __name__ == "__main__":
    main()

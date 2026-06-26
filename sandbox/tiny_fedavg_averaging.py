"""
tiny_fedavg_averaging.py  --  Level-1 (pure numpy) reproduction of Figure 1
of "Communication-Efficient Learning of Deep Networks from Decentralized Data"
(McMahan et al., 2017 -- the FedAvg paper).

LOAD-BEARING PHENOMENON
-----------------------
For a *non-convex* model, naively averaging the PARAMETERS of two models that
were trained on different data only helps if the two models started from a
SHARED initialization.

  LEFT  (Figure 1, left)  : two MLPs trained from INDEPENDENT random inits.
                            The interpolated model theta*w + (1-theta)*w' has a
                            LOSS BARRIER -- the midpoint (theta=0.5) loss is
                            HIGHER than either parent.
  RIGHT (Figure 1, right) : two MLPs trained from a SHARED init.
                            The midpoint loss is LOWER than either parent
                            (averaging actually helps).

We mirror the paper's recipe in miniature: a one-hidden-layer MLP (smallest
model that is non-convex and has permutation symmetry -- LOGISTIC REGRESSION
IS CONVEX AND WOULD NOT SHOW A BARRIER), synthetic 4-class data split into two
DISJOINT halves (one per model, analogous to the non-overlapping 600-example
MNIST shards), trained "until they begin to overfit" their half, then we sweep
the interpolation weight theta in [-0.2, 1.2] and measure the full-dataset loss.

Decisive scalar per condition:
    barrier_height = loss(theta=0.5) - max(loss(theta=0), loss(theta=1))
    > 0  : averaging HURTS  (a barrier exists)  -> expected for INDEPENDENT init
    <= 0 : averaging HELPS  (midpoint beats both parents) -> expected for SHARED init
"""

import numpy as np
import warnings

# numpy 2.x on Apple Silicon (Accelerate BLAS) emits SPURIOUS
# "divide by zero / overflow / invalid encountered in matmul" RuntimeWarnings
# from np.matmul even when both operands and the result are finite -- it is a
# known false FP-status-flag issue in the vendor BLAS, NOT numerical trouble in
# this code. We silence only that specific message; every forward pass below is
# additionally asserted finite so a *real* blow-up would still be caught.
warnings.filterwarnings("ignore", message=".*encountered in matmul.*",
                        category=RuntimeWarning)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
GLOBAL_SEED = 0
np.random.seed(GLOBAL_SEED)

# --------------------------------------------------------------------------- #
# Problem dimensions (tiny, on purpose)
# --------------------------------------------------------------------------- #
D = 16          # input dimension
H = 32          # hidden units (one hidden layer -> non-convex)
C = 4           # output classes
N = 1500        # total synthetic samples

# Number of trainable params (W1: D*H, b1: H, W2: H*C, b2: C)
PARAM_COUNT = D * H + H + H * C + C


# --------------------------------------------------------------------------- #
# Synthetic 4-class data: C Gaussian blobs with random class means in R^D.
# Modest separation so the task is learnable but each disjoint half lands the
# two models in genuinely different (permutation-incompatible) minima when
# started independently.
# --------------------------------------------------------------------------- #
def make_data(n, d, c, seed):
    rng = np.random.RandomState(seed)
    # Modest class separation + substantial noise so the task is genuinely
    # non-trivial: the parents settle at a clearly NONZERO loss (~0.3-0.6),
    # not a degenerate ~0. This makes the barrier (independent) and the
    # midpoint dip (shared) both clearly visible, as in Figure 1.
    means = rng.randn(c, d) * 0.75         # class centroids (overlapping)
    y = rng.randint(0, c, size=n)
    X = means[y] + rng.randn(n, d) * 1.0   # blob spread (heavy overlap)
    # standardize features
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return X.astype(np.float64), y.astype(np.int64)


def one_hot(y, c):
    Y = np.zeros((y.shape[0], c))
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


# --------------------------------------------------------------------------- #
# One-hidden-layer MLP: tanh activation, softmax + cross-entropy.
# Params packed in a dict. Forward + manual backprop in numpy.
# --------------------------------------------------------------------------- #
def init_params(seed, d=D, h=H, c=C):
    rng = np.random.RandomState(seed)
    # He-ish / Xavier-ish scaling
    return {
        "W1": rng.randn(d, h) * np.sqrt(2.0 / d),
        "b1": np.zeros(h),
        "W2": rng.randn(h, c) * np.sqrt(2.0 / h),
        "b2": np.zeros(c),
    }


def forward(params, X):
    z1 = X @ params["W1"] + params["b1"]
    # Clip pre-activations: at extrapolated theta (e.g. -0.2, 1.2) the
    # interpolated weights of two INDEPENDENT models can be large; tanh
    # already saturates, but clipping keeps everything finite (no overflow).
    z1 = np.clip(z1, -30.0, 30.0)
    a1 = np.tanh(z1)
    z2 = a1 @ params["W2"] + params["b2"]
    z2 = z2 - z2.max(axis=1, keepdims=True)        # numerical stability
    expz = np.exp(z2)
    probs = expz / expz.sum(axis=1, keepdims=True)
    # Guard: prove the silenced matmul warnings are spurious -- a genuine
    # numerical blow-up would trip this assert.
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

    dz2 = (probs - Y) / n                  # (n, C)
    dW2 = a1.T @ dz2                        # (H, C)
    db2 = dz2.sum(axis=0)                   # (C,)

    da1 = dz2 @ params["W2"].T             # (n, H)
    dz1 = da1 * (1.0 - a1 ** 2)            # tanh'
    dW1 = X.T @ dz1                        # (D, H)
    db1 = dz1.sum(axis=0)                  # (H,)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def train(params, X, y, lr=0.2, epochs=80, batch=50, seed=0):
    """Minibatch GD, fixed lr. ~epochs passes -> 'begin to overfit' regime,
    echoing the paper's E=20 passes over the small local datasets (here the
    models drive their own half low while the full-set loss stays nonzero)."""
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


# --------------------------------------------------------------------------- #
# ASCII line-plot of a loss curve over theta.
# --------------------------------------------------------------------------- #
def ascii_plot(thetas, losses, title, height=12, width=58):
    lo, hi = min(losses), max(losses)
    span = (hi - lo) or 1.0
    # build grid
    grid = [[" "] * width for _ in range(height)]
    for t, l in zip(thetas, losses):
        col = int(round((t - thetas[0]) / (thetas[-1] - thetas[0]) * (width - 1)))
        row = int(round((hi - l) / span * (height - 1)))
        grid[row][col] = "*"
    lines = [title]
    for r in range(height):
        yval = hi - (r / (height - 1)) * span
        lines.append("{:7.4f} |".format(yval) + "".join(grid[r]))
    axis = " " * 9 + "+" + "-" * width
    lines.append(axis)
    lines.append(" " * 10 + "theta=-0.2" + " " * (width - 20) + "theta=1.2")
    return "\n".join(lines)


def eval_path(wa, wb, X, y, thetas):
    return np.array([ce_loss(forward(interpolate(wa, wb, t), X)[0], y) for t in thetas])


def run_condition(name, seed_a, seed_b, Xfull, yfull, Xa, ya, Xb, yb, thetas):
    """seed_a / seed_b are the INIT seeds for the two models.
    Same value -> shared init; different -> independent init."""
    w0_a = init_params(seed_a)
    w0_b = init_params(seed_b)
    w = train(w0_a, Xa, ya, seed=101)       # model w  (corresponds to theta=1)
    wp = train(w0_b, Xb, yb, seed=202)      # model w' (corresponds to theta=0)

    losses = eval_path(w, wp, Xfull, yfull, thetas)

    # locate theta = 0, 0.5, 1 (thetas grid includes them by construction)
    def at(target):
        j = int(np.argmin(np.abs(thetas - target)))
        return losses[j]

    l0, lhalf, l1 = at(0.0), at(0.5), at(1.0)
    lmin = float(losses.min())
    barrier = float(lhalf - max(l0, l1))

    print("=" * 70)
    print("CONDITION: {}".format(name))
    print("  init seeds (w', w) = ({}, {})  {}".format(
        seed_b, seed_a, "SHARED" if seed_a == seed_b else "INDEPENDENT"))
    print("  loss(theta=0.0) [parent w'] = {:.4f}".format(l0))
    print("  loss(theta=0.5) [average  ] = {:.4f}".format(lhalf))
    print("  loss(theta=1.0) [parent w ] = {:.4f}".format(l1))
    print("  min loss over path          = {:.4f}".format(lmin))
    print("  >>> barrier_height = loss(0.5) - max(loss(0),loss(1)) = {:+.4f}".format(barrier))
    if barrier > 0:
        print("      => POSITIVE barrier: naive averaging HURTS (midpoint worse than parents).")
    else:
        print("      => NON-POSITIVE: averaging HELPS (midpoint at/under both parents).")
    print()

    # compact table (every other point) + ascii plot
    print("  theta :   loss")
    for t, l in zip(thetas[::2], losses[::2]):
        bar = "#" * int(round((l - losses.min()) / ((losses.max() - losses.min()) or 1) * 30))
        print("  {:+5.2f} : {:7.4f}  {}".format(t, l, bar))
    print()
    print(ascii_plot(thetas.tolist(), losses.tolist(),
                     "  loss vs theta  ({})".format(name)))
    print()

    return {"l0": l0, "lhalf": lhalf, "l1": l1, "lmin": lmin, "barrier": barrier}


def main():
    # Full dataset, then split into two DISJOINT halves (one per model).
    X, y = make_data(N, D, C, seed=7)
    perm = np.random.RandomState(123).permutation(N)
    X, y = X[perm], y[perm]
    half = N // 2
    Xa, ya = X[:half], y[:half]            # shard for model w
    Xb, yb = X[half:], y[half:]            # shard for model w'

    thetas = np.linspace(-0.2, 1.2, 29)    # ensures 0.0, 0.5, 1.0 land on grid

    print("Tiny FedAvg Figure-1 reproduction (pure numpy)")
    print("MLP: d={} -> h={} (tanh) -> {} classes | params/model = {}".format(
        D, H, C, PARAM_COUNT))
    print("Data: {} samples, 4 Gaussian blobs, split into two disjoint halves "
          "of {} each".format(N, half))
    print("Eval: full-dataset cross-entropy over theta in [-0.2, 1.2] (29 pts)")
    print()

    # CONDITION A: INDEPENDENT inits (different seeds) -> expect barrier > 0
    res_indep = run_condition(
        "INDEPENDENT INIT (different random seeds)  [Figure 1, LEFT]",
        seed_a=1, seed_b=2,
        Xfull=X, yfull=y, Xa=Xa, ya=ya, Xb=Xb, yb=yb, thetas=thetas)

    # CONDITION B: SHARED init (identical seed) -> expect barrier <= 0
    res_shared = run_condition(
        "SHARED INIT (identical random seed)        [Figure 1, RIGHT]",
        seed_a=5, seed_b=5,
        Xfull=X, yfull=y, Xa=Xa, ya=ya, Xb=Xb, yb=yb, thetas=thetas)

    print("=" * 70)
    print("SUMMARY")
    print("  INDEPENDENT  barrier_height = {:+.4f}  (expect > 0  : averaging hurts)".format(
        res_indep["barrier"]))
    print("  SHARED       barrier_height = {:+.4f}  (expect <=0  : averaging helps)".format(
        res_shared["barrier"]))
    ok = (res_indep["barrier"] > 0) and (res_shared["barrier"] <= 0)
    print("  Figure-1 phenomenon reproduced: {}".format(ok))
    print("=" * 70)


if __name__ == "__main__":
    main()

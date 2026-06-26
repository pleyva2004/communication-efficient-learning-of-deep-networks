"""IID vs Non-IID -- concept 03-iid-noniid-dichotomy of the paper learning map.

Monte-Carlo witness: a uniform-random (IID) partition makes each local objective
F_k(w) an unbiased proxy for the global f(w), so E[F_k]~f; a sort-by-label
(non-IID) partition makes F_k an arbitrarily bad approximation to f at a fixed w.
Runnable code analog of concepts/03-iid-noniid-dichotomy.py.
"""
import numpy as np


def build_data(n=6000, num_classes=10, d=5):
    """Labelled (x, y); per-example loss f_i(w) is squared-error to label y_i."""
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(num_classes), n // num_classes)  # balanced labels
    # one deterministic feature vector per class so f_i depends on the label
    centers = rng.standard_normal((num_classes, d))
    x = centers[y] + 0.01 * rng.standard_normal((n, d))
    return np.ascontiguousarray(x), y, np.ascontiguousarray(centers)


def f_i(w, x, y, centers):
    """Per-example loss: how badly w 'scores' example i relative to its class."""
    # score = x . w ; target = class-center projection (dot via elementwise sum
    # to avoid spurious numpy-2.0 matmul FPE warnings on (n,d)@(d,) shapes)
    score = (x * w).sum(axis=1)
    target = (centers[y] * w).sum(axis=1)
    return (score - target) ** 2 + (y - 4.5) ** 2  # label-dependent term


def F(w, idx, x, y, centers):
    """Average loss over an index set (F_k if idx=P_k, f if idx=all)."""
    return float(np.mean(f_i(w, x[idx], y[idx], centers)))


def main():
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, K = 6000, 10
    x, y, centers = build_data(n=n)
    all_idx = np.arange(n)
    w = rng.standard_normal(x.shape[1])         # one fixed evaluation point w
    f_w = F(w, all_idx, x, y, centers)          # global objective f(w)

    # IID: uniform-random partition into K equal shards (each shard mixes labels)
    perm = rng.permutation(n)
    iid_parts = np.array_split(perm, K)
    iid_gap = np.mean([abs(F(w, p, x, y, centers) - f_w) for p in iid_parts])

    # Non-IID: sort by label, then split -> each shard holds ~1 label
    sorted_idx = all_idx[np.argsort(y, kind="stable")]
    noniid_parts = np.array_split(sorted_idx, K)
    noniid_gap = np.mean([abs(F(w, p, x, y, centers) - f_w) for p in noniid_parts])

    print("Fixed point w; global objective f(w) = {:.4f}".format(f_w))
    print("IID partition (uniform-random shards): mean |F_k(w) - f(w)| = "
          "{:.4f}   <- unbiased, E[F_k]~f  (Eq. 1.2)".format(iid_gap))
    print("Non-IID partition (sort-by-label shards): mean |F_k(w) - f(w)| = "
          "{:.4f}   <- arbitrarily bad proxy".format(noniid_gap))
    print("Heterogeneity blow-up factor (non-IID / IID) = {:.1f}x".format(
        noniid_gap / iid_gap))
    # Identity (1.1) still holds exactly under BOTH partitions: weighted mix = f
    for name, parts in (("IID", iid_parts), ("non-IID", noniid_parts)):
        mix = sum((len(p) / n) * F(w, p, x, y, centers) for p in parts)
        print("  size-weighted mix of F_k under {:>7s} = {:.4f}  (== f(w), "
              "identity 1.1 unbroken)".format(name, mix))


if __name__ == "__main__":
    main()

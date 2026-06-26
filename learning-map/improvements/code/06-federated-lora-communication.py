"""Federated LoRA Communication -- concept 06-federated-lora-communication
of the improvements learning map.

Freezing a base weight W (d x d) and FedAvg-ing only a low-rank adapter
Delta W = B A (B: d x r, A: r x d) cuts per-round communication from Theta(d^2)
to Theta(2 r d); the ratio 2r/d shrinks as the base model grows. This witness
tabulates base vs adapter param counts and the communication ratio for several
(d, r), including a ~4.3% case matching the sandbox real_fedlora.py.

Runnable code analog of concepts/06-federated-lora-communication.md.
"""

import numpy as np

np.random.seed(0)


def counts(d, r):
    """Base params (d^2), adapter params (2*r*d), and comm ratio (2r/d)."""
    base = d * d
    adapter = 2 * r * d
    ratio = adapter / base          # == 2r/d, the per-round comm fraction
    return base, adapter, ratio


def main():
    print("Federated LoRA communication (05-improvements.tex T.2)")
    print("Freeze base W in R^{d x d}; FedAvg only adapter Delta W = B A,")
    print("  A in R^{r x d}, B in R^{d x r}  ->  per-round comms Theta(2rd) vs Theta(d^2).\n")

    # (d, r) pairs spanning toy -> large; the last targets ~4.3% (sandbox figure).
    configs = [(64, 8), (256, 8), (1024, 16), (4096, 16), (1024, 22)]

    header = "{:>7} {:>5} | {:>14} {:>14} | {:>10} {:>9}".format(
        "d", "r", "base d^2", "adapter 2rd", "ratio 2r/d", "savings"
    )
    print(header)
    print("-" * len(header))
    for d, r in configs:
        base, adapter, ratio = counts(d, r)
        # cross-check the closed form 2r/d against the explicit param counts
        assert abs(ratio - (2 * r / d)) < 1e-12
        assert adapter < base, "LoRA only saves when 2rd < d^2, i.e. 2r < d"
        print("{:>7} {:>5} | {:>14,} {:>14,} | {:>9.2f}% {:>8.1f}x".format(
            d, r, base, adapter, 100.0 * ratio, base / adapter
        ))

    # Numerical witness that B A truly reconstructs a rank-r matrix of shape d x d
    # (so the adapter, not the base, carries the trained update).
    d, r = 1024, 22
    A = np.random.normal(0.0, 0.02, size=(r, d))
    B = np.random.normal(0.0, 0.02, size=(d, r))
    # errstate guards a benign numpy-2.0 + Accelerate BLAS SIMD quirk on Apple
    # Silicon that raises spurious FP flags here; the product is finite & valid.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        dW = B @ A
    assert np.isfinite(dW).all()
    rank = int(np.linalg.matrix_rank(dW))
    base, adapter, ratio = counts(d, r)
    print("\nWitness (d={}, r={}): Delta W = B@A has shape {} and rank {} (<= r).".format(
        d, r, dW.shape, rank
    ))
    print("  base d^2 = {:,} params; adapter 2rd = {:,} params communicated/round.".format(
        base, adapter
    ))
    print("  per-round communication ratio 2r/d = {:.1f}%  ({:.0f}x less than the base).".format(
        100.0 * ratio, base / adapter
    ))
    assert rank <= r, "B@A cannot exceed rank r"
    print("PASS -- adapter is rank <= r and costs only 2r/d of the base each round.")


if __name__ == "__main__":
    main()

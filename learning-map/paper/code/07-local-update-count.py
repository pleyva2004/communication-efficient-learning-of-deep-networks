"""Local-Update Count u = nE/(KB) -- concept 07-local-update-count of the paper learning map.

Computes the per-round local SGD-step count u = nE/(KB) for several (E, B) and
confirms it against an actual ClientUpdate loop that counts the SGD steps taken.
Runnable code analog of concepts/07-local-update-count.md.
"""
import math

import numpy as np

np.random.seed(0)


def client_update_step_count(n_k, E, B):
    """Count SGD steps in ClientUpdate: E epochs, each over ceil(n_k/B) batches.

    B = math.inf means 'one batch = whole local set', i.e. full-batch (FedSGD).
    """
    if math.isinf(B):
        batches_per_epoch = 1  # full batch -> exactly one step per epoch
    else:
        batches_per_epoch = math.ceil(n_k / B)
    steps = 0
    for _epoch in range(E):
        # split the n_k local indices into batches; each batch = one SGD step
        idx = np.arange(n_k)
        np.random.shuffle(idx)
        for start in range(0, n_k, n_k if math.isinf(B) else B):
            steps += 1  # one minibatch -> one local SGD update
    assert steps == E * batches_per_epoch
    return steps


def main():
    K = 100          # number of clients
    n = 60000        # total examples (MNIST-scale)
    n_k = n // K     # balanced partition: 600 examples per client
    print("Local-update count  u = nE/(KB)   (Eqs. 4.1-4.2 of the math deep dive)")
    print("K=%d clients, n=%d total, balanced n_k=%d examples/client" % (K, n, n_k))
    print("FedSGD corner: B=inf, E=1  =>  u_k = 1 (one full-batch step)\n")

    configs = [(1, math.inf), (1, 50), (5, 10), (20, 10), (1, 600)]
    header = "  E |     B  | formula u=nE/(KB) | simulated steps/client | match"
    print(header)
    print("  " + "-" * (len(header) - 2))

    all_match = True
    for E, B in configs:
        # u = nE/(KB); for B=inf the paper's intended reading is u_k = E (one step/epoch)
        if math.isinf(B):
            u_formula = float(E)  # B=inf => one batch => E steps; E=1 => u=1 (FedSGD)
        else:
            u_formula = n * E / (K * B)
        simulated = client_update_step_count(n_k, E, B)
        ok = abs(simulated - u_formula) < 1e-9
        all_match = all_match and ok
        B_str = " inf" if math.isinf(B) else "%4d" % B
        print("  %2d | %s   |       %8.1f    |        %8d        |  %s"
              % (E, B_str, u_formula, simulated, "OK" if ok else "X"))

    print("\nFedSGD endpoint check: (E=1, B=inf) gives u_k = %d"
          % client_update_step_count(n_k, 1, math.inf))
    print("All formula values match the simulated ClientUpdate step counts: %s" % all_match)


if __name__ == "__main__":
    main()

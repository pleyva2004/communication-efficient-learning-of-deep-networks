"""Compute-Accounting Pareto -- concept 04-compute-accounting-pareto of the improvements learning map.

FedAvg reports rounds-to-target alone, assuming on-device compute is "free". Here we
pair each (E,B) config's rounds R with its total local-update FLOP-proxy R*(C*K)*u_k,
revealing the rounds-vs-compute knee where a high-u config is Pareto-dominated.
Runnable code analog of concepts/04-compute-accounting-pareto.py.
"""
import numpy as np

# Toy fixed setup (matches the paper's MNIST family): n_k=600 examples/client,
# C*K = 10 clients selected per round. u_k = E*n_k/B, with B=inf => u_k=1.
N_K = 600          # examples per client
CK = 10            # C*K selected clients per round
INF = float("inf")


def updates_per_round(E, B):
    """u_k = E*n_k/B local SGD updates; B=inf means one full-batch step => u_k=1."""
    if B == INF:
        return float(E)            # E epochs, 1 batch each => u_k = E (and E=1 => FedSGD)
    return E * N_K / B


def main():
    np.random.seed(0)
    # Hardcoded VERIFIED vanilla-FedAvg rounds-to-target for a few (E,B) configs
    # (toy numbers in the spirit of Table 2: more local work -> fewer rounds).
    configs = [
        # (label,           E,   B,    rounds R)
        ("E=1,  B=inf",      1, INF,   40),   # FedSGD endpoint: cheap/round, many rounds
        ("E=5,  B=inf",      5, INF,   23),
        ("E=20, B=inf",     20, INF,   13),
        ("E=5,  B=10",       5,  10,   11),   # u_k=300: cheaper compute AND fewer rounds
        ("E=20, B=10",      20,  10,   12),   # u_k=1200: diminishing returns -> DOMINATED
    ]

    rows = []
    for label, E, B, R in configs:
        u = updates_per_round(E, B)
        compute = R * CK * u           # total on-device local-update FLOP-proxy
        rows.append((label, u, R, compute))

    # Pareto test on (rounds, compute): minimize BOTH. A config is dominated if some
    # other config has rounds <= and compute <= with at least one strictly less.
    def dominated(i):
        Ri, Ci = rows[i][2], rows[i][3]
        for j, (_, _, Rj, Cj) in enumerate(rows):
            if j == i:
                continue
            if Rj <= Ri and Cj <= Ci and (Rj < Ri or Cj < Ci):
                return True
        return False

    print("Compute-Accounting Pareto for FedAvg (toy rounds-to-target, n_k=600, C*K=10)")
    print("config            u_k |  rounds R | total compute (R*CK*u_k) | Pareto?")
    for i, (label, u, R, compute) in enumerate(rows):
        tag = "DOMINATED  " if dominated(i) else "on-frontier"
        print("{:<16}{:>5.0f} | {:>8d}  | {:>22.2e}   | {}".format(label, u, R, compute, tag))

    # Identify the knee: the on-frontier config minimizing rounds, and the one
    # minimizing compute -- the trade lives between them.
    frontier = [r for i, r in enumerate(rows) if not dominated(i)]
    fewest_rounds = min(frontier, key=lambda r: r[2])
    least_compute = min(frontier, key=lambda r: r[3])
    print("---")
    print("Fewest-rounds frontier config : {:<12} ({} rounds, compute {:.2e})".format(
        fewest_rounds[0], fewest_rounds[2], fewest_rounds[3]))
    print("Least-compute frontier config : {:<12} ({} rounds, compute {:.2e})".format(
        least_compute[0], least_compute[2], least_compute[3]))
    dom = [r[0] for i, r in enumerate(rows) if dominated(i)]
    print("Pareto-DOMINATED configs (lose on BOTH axes vs another): {}".format(
        dom if dom else "none"))
    print("LESSON: reporting rounds alone hides that the rounds-winner is not the "
          "compute-winner -- the knee is the trade FedAvg never measured (E.2).")


if __name__ == "__main__":
    main()

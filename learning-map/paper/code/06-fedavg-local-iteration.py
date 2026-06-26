"""FedAvg: Iterated Local Steps -- concept 06-fedavg-local-iteration of the paper learning map.

FedAvg runs tau local gradient steps per client from a shared w_t, then averages
the endpoints (Eq. 3.3). For tau>=2 that round map is NOT tau centralized GD steps;
this witness measures the growing discrepancy (exactly 0 at tau=1) on a 2-client quadratic.
Runnable code analog of concepts/06-fedavg-local-iteration.py.
"""
import numpy as np


def make_quadratic(A, b):
    """Local objective F_k(w) = 0.5 w'Aw - b'w, so grad F_k(w) = A w - b."""
    return lambda w: A @ w - b


def local_run(grad, w0, eta, tau):
    """tau local GD steps from shared start w0 (one client's trajectory)."""
    w = w0.copy()
    for _ in range(tau):
        w = w - eta * grad(w)
    return w


def main():
    np.random.seed(0)
    d = 3
    eta = 0.05
    # Two heterogeneous clients: distinct local curvatures (A_k) and optima.
    A1 = np.array([[2.0, 0.3, 0.0], [0.3, 1.5, 0.1], [0.0, 0.1, 1.0]])
    A2 = np.array([[0.7, -0.2, 0.0], [-0.2, 2.5, 0.4], [0.0, 0.4, 1.8]])
    b1 = np.array([1.0, -0.5, 0.3])
    b2 = np.array([-0.4, 0.8, -1.1])
    n1, n2 = 3, 2                       # client sizes -> weights n_k/n
    n = n1 + n2
    a1, a2 = n1 / n, n2 / n

    g1, g2 = make_quadratic(A1, b1), make_quadratic(A2, b2)
    # Global objective f = sum_k (n_k/n) F_k  =>  grad f(w) = a1*g1(w) + a2*g2(w).
    grad_f = lambda w: a1 * g1(w) + a2 * g2(w)

    w_t = np.array([0.4, -0.2, 0.6])   # shared per-round start w^k_(0) = w_t

    print("FedAvg iterated local steps vs. centralized GD (2-client quadratic, eta=%.2f)" % eta)
    print("  FedAvg round: each client does tau local steps from shared w_t, then average endpoints.")
    print("  Centralized:  tau full GD steps on f = (n1/n)F_1 + (n2/n)F_2.")
    print("  Discrepancy = ||w_FedAvg - w_centralized||  (must be 0 at tau=1, grows with tau).")
    print("  tau | discrepancy ||FedAvg - centralized||")
    print("  ----+--------------------------------------")
    for tau in (1, 2, 3):
        # FedAvg: average the per-client endpoints (Eq. 3.3).
        w_fedavg = a1 * local_run(g1, w_t, eta, tau) + a2 * local_run(g2, w_t, eta, tau)
        # Centralized: tau genuine GD steps on the global f.
        w_central = local_run(grad_f, w_t, eta, tau)
        disc = np.linalg.norm(w_fedavg - w_central)
        print("   %d  | %.6e" % (tau, disc))

    print("Reading: tau=1 averaging gradients == averaging models (discrepancy 0);")
    print("for tau>=2 the nonlinear local trajectory diverges -- the gap IS the local-step free lunch.")


if __name__ == "__main__":
    main()

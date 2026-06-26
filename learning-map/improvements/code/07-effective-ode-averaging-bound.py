"""Effective-ODE Averaging Bound -- concept 07-effective-ode-averaging-bound of the
improvements learning map.

Idealize each client's u_k local full-batch steps as the gradient flow Phi_k^T of
its loss F_k run for time T = eta*u_k. The FedAvg server map S^T = sum_k beta_k
Phi_k^T is compared to the centralized flow Phi_f^T of the global f = sum_k beta_k
F_k. A Lie-Taylor expansion gives the EXACT leading drift

    || S^T(w) - Phi_f^T(w) || = (T^2/2) * ||Cov_k(H_k, g_k)|| + O(T^3),

with leading direction Cov_k(H_k,g_k)/||.||. This witness (i) measures the drift's
log-log slope vs T (predicted 2), (ii) checks D_t / pred -> ~1 as T -> 0, (iii)
checks the drift direction matches Cov_k(H_k,g_k) to cosine ~1, and (iv) shows the
heuristic frozen-eta^2 prefactor underpredicts by u_k^2/2 (~12.5x at E=5).
Runnable code analog of concepts/07-effective-ode-averaging-bound.md.
"""
import numpy as np

np.random.seed(0)
np.seterr(over="ignore", divide="ignore", invalid="ignore")


def make_clients(K=4, d=6):
    """K heterogeneous local quadratics F_k(w) = 0.5 (w-b_k)^T A_k (w-b_k).
    grad F_k(w) = A_k (w-b_k);  Hessian H_k = A_k (constant). Heterogeneous
    A_k => Cov_k(H_k,g_k) != 0, the curvature-gradient covariance that drifts.
    Hessians are kept SMALL and well-conditioned so the O(T^3) remainder
    (~ T^3 * (L^2 G)/3) is negligible against the leading (T^2/2)*sigma_HG term
    over the measured horizons -- i.e. we stay inside the bound's small-T regime."""
    beta = np.full(K, 1.0 / K)               # equal client weights n_k/n
    A, b = [], []
    for _ in range(K):
        M = 0.15 * np.random.randn(d, d)
        A.append(M @ M.T + 0.5 * np.eye(d))  # SPD, spectral norm O(1)
        b.append(0.5 * np.random.randn(d))
    return beta, A, b


def flow(A_k, b_k, w, T, steps):
    """Phi_k^T(w): gradient flow dw/dt = -A_k (w-b_k), integrated to time T by
    fine-grained explicit Euler (steps large => ~exact continuous flow)."""
    dt = T / steps
    for _ in range(steps):
        w = w - dt * (A_k @ (w - b_k))
    return w


def main():
    beta, A, b = make_clients()
    d = A[0].shape[0]
    w = np.random.randn(d)
    K = len(A)

    # Global objective f = sum_k beta_k F_k: Hessian H = sum beta_k A_k, and its
    # gradient flow uses grad f(w) = sum_k beta_k A_k (w - b_k).
    H = sum(beta[k] * A[k] for k in range(K))
    g = sum(beta[k] * (A[k] @ (w - b[k])) for k in range(K))

    def server_map(T, steps):
        return sum(beta[k] * flow(A[k], b[k], w.copy(), T, steps) for k in range(K))

    def central_flow(T, steps):
        wf = w.copy()
        dt = T / steps
        for _ in range(steps):
            wf = wf - dt * sum(beta[k] * (A[k] @ (wf - b[k])) for k in range(K))
        return wf

    # The theorem's leading object: Gamma = Cov_k(H_k, g_k) = sum beta_k H_k g_k - H g.
    gk = [A[k] @ (w - b[k]) for k in range(K)]
    Gamma = sum(beta[k] * (A[k] @ gk[k]) for k in range(K)) - H @ g
    sigma_HG = np.linalg.norm(Gamma)

    print("Effective-ODE averaging-drift bound (05-improvements.tex T.3)")
    print("Server map S^T = sum_k beta_k Phi_k^T  vs centralized flow Phi_f^T.")
    print("K=%d clients, dim=%d, ||Cov_k(H_k,g_k)|| = sigma_HG = %.4f\n" % (K, d, sigma_HG))

    # (1) measure drift at a geometric sweep of small horizons T; fit log-log slope.
    Ts = np.geomspace(2e-3, 4e-2, 7)
    steps = 6000                              # fine Euler => essentially the flow
    drifts, ratios = [], []
    for T in Ts:
        D = np.linalg.norm(server_map(T, steps) - central_flow(T, steps))
        pred = 0.5 * T ** 2 * sigma_HG
        drifts.append(D)
        ratios.append(D / pred)
    drifts = np.array(drifts)
    slope = np.polyfit(np.log(Ts), np.log(drifts), 1)[0]

    print("Measured drift scales as T^p with p ~ 2 (predicted exactly 2):")
    print("   T        drift D_t     pred=(T^2/2)sigma   D_t/pred")
    for T, D, r in zip(Ts, drifts, ratios):
        print("  %.4f   %.4e      %.4e        %.4f" % (T, D, 0.5 * T ** 2 * sigma_HG, r))
    print("  log-log slope of D_t vs T = %.3f  (predicted 2.0)" % slope)
    print("  D_t/pred -> %.3f as T -> 0  (predicted 1.0)" % ratios[0])

    # (2) direction: leading drift should point along Gamma/||Gamma||.
    Tsmall = Ts[0]
    drift_vec = server_map(Tsmall, steps) - central_flow(Tsmall, steps)
    cos = float(drift_vec @ Gamma / (np.linalg.norm(drift_vec) * sigma_HG))
    print("  drift direction . Cov_k(H_k,g_k) cosine = %.4f  (predicted ~1.0)\n" % cos)

    # (3) the heuristic Delta = eta^2 Cov_k(H_k,g_k) is the two-step Taylor sketch.
    # The exact prefactor is T^2/2 = (eta*u_k)^2/2, so relative to the RAW eta^2
    # heuristic the drift grows by u_k^2/2: a factor 2 already at E=2 (where that
    # factor 2 is exactly the explicit-Euler error of the 2-step sketch) and 12.5
    # at E=5. The heuristic UNDERPREDICTS the drift.
    print("Heuristic prefactor (eta^2) vs theorem (T^2/2) at fixed eta, varying E (B=inf => u_k=E):")
    eta = 0.05
    heuristic = eta ** 2                        # raw two-step heuristic prefactor
    for E in (2, 5, 10):
        u_k = E                               # B=inf, full batch => u_k = E
        T = eta * u_k
        correct = 0.5 * T ** 2                 # theorem prefactor (T^2/2)
        underpred = correct / heuristic        # = u_k^2/2
        tag = "  <- factor 2 = Euler error of the 2-step sketch" if E == 2 else ""
        print("  E=%-3d u_k=%-3d  T=%.3f | theorem T^2/2=%.4e  heuristic eta^2=%.4e"
              "  underpred x%.2f (=u_k^2/2)%s"
              % (E, u_k, T, correct, heuristic, underpred, tag))

    ok_slope = abs(slope - 2.0) < 0.15
    ok_ratio = abs(ratios[0] - 1.0) < 0.15
    ok_dir = cos > 0.99
    ok_pref = abs((0.5 * (eta * 5) ** 2) / (eta ** 2) - 5 ** 2 / 2.0) < 1e-9
    ok = ok_slope and ok_ratio and ok_dir and ok_pref
    print("\nVERDICT: %s -- drift = (T^2/2)||Cov_k(H_k,g_k)|| + O(T^3); slope=%.2f, "
          "ratio->%.2f, cos=%.4f, heuristic underpredicts by u_k^2/2."
          % ("PASS" if ok else "FAIL", slope, ratios[0], cos))
    assert ok, "effective-ODE drift law must hold to O(T^2)"


if __name__ == "__main__":
    main()

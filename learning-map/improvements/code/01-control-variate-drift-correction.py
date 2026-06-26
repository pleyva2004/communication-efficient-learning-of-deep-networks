"""Control-Variate Drift Correction -- concept 01-control-variate-drift-correction
of the improvements learning map.

On a 2-client heterogeneous quadratic, SCAFFOLD-style control variates re-center
each client's multi-step local trajectory onto the GLOBAL gradient direction,
cancelling the leading client-heterogeneity drift eta^2*Cov_k(H_k, g_k) (eq. 3.5
of 02-math-deep-dive.md; M.1 of 05-improvements.tex).

Runnable code analog of concepts/01-control-variate-drift-correction.py
(see concepts/01-control-variate-drift-correction.md).
"""
import numpy as np


def main():
    np.random.seed(0)

    # Two heterogeneous local quadratics  F_k(w) = 0.5 (w-b_k)^T H_k (w-b_k).
    # grad F_k(w) = H_k (w - b_k);  Hessians H_k differ => heterogeneity.
    H = [np.array([[3.0, 0.0], [0.0, 0.5]]),
         np.array([[0.5, 0.0], [0.0, 3.0]])]
    b = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    nk = np.array([1.0, 1.0])          # equal client sizes -> weights n_k/n = 1/2
    alpha = nk / nk.sum()

    def gk(k, w):                      # local gradient g_local(w) for client k
        return H[k] @ (w - b[k])

    w0 = np.array([0.7, 0.7])          # shared broadcast point w_t
    eta, tau = 0.10, 5                 # local lr and number of local steps

    # GLOBAL gradient at the shared point: grad f(w0) = sum_k alpha_k g_k(w0)  (eq. 2.2)
    g_global = sum(alpha[k] * gk(k, w0) for k in range(2))

    # Control variates: c_k = g_k(w0), server c = mean_k c_k  (eq. M.1).
    c = [gk(k, w0) for k in range(2)]
    c_server = sum(alpha[k] * c[k] for k in range(2))

    def local_traj(k, corrected):
        w = w0.copy()
        for _ in range(tau):
            d = gk(k, w)
            if corrected:
                d = d - c[k] + c_server     # heterogeneity-corrected direction
            w = w - eta * d
        return (w0 - w) / (tau * eta)        # avg per-step direction client moved

    # Compare each client's effective local direction to the global gradient.
    print("Global gradient grad f(w_t) at shared point w_t={}:".format(w0))
    print("  g_global = {}".format(np.round(g_global, 4)))
    print("Per-client effective local direction vs global gradient:")
    dev_plain, dev_corr = 0.0, 0.0
    for k in range(2):
        dp = local_traj(k, corrected=False)
        dc = local_traj(k, corrected=True)
        ep = np.linalg.norm(dp - g_global)
        ec = np.linalg.norm(dc - g_global)
        dev_plain += alpha[k] * ep
        dev_corr += alpha[k] * ec
        print("  client {}: plain dir ={}  dev={:.4f} | corrected dir ={}  dev={:.4f}"
              .format(k, np.round(dp, 3), ep, np.round(dc, 3), ec))

    print("Client-weighted deviation from global gradient:")
    print("  plain local steps      : {:.4f}".format(dev_plain))
    print("  control-variate steps  : {:.4f}".format(dev_corr))
    assert dev_corr < dev_plain, "corrected direction must track global gradient better"
    print("VERDICT: PASS -- control variates make local steps track grad f "
          "({:.2f}x smaller drift).".format(dev_plain / dev_corr))


if __name__ == "__main__":
    main()

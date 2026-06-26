# Horizon-Equalized Local Flow
**Level:** `extension`  *(novice | intermediate | advanced | graduate | frontier | paper | extension)*
**Concept ID:** `08-horizon-equalized-local-flow`
**Graph:** `improvements`
**Prerequisites:** [paper:07-local-update-count](../../paper/concepts/07-local-update-count.md), [paper:08-heterogeneity-gap-covariance](../../paper/concepts/08-heterogeneity-gap-covariance.md), [gradient descent](https://github.com/pleyva2004/math-foundations/blob/main/concepts/gradient-descent.md), [covariance](https://github.com/pleyva2004/math-foundations/blob/main/concepts/covariance.md)
**Used by:** downstream nodes

## Plain-English intro
One step of local SGD $w\leftarrow w-\eta\,\nabla F_k(w)$ is one forward-Euler step of the gradient flow $\dot w=-\nabla F_k(w)$, so a round of $u_k=En_k/B$ local steps integrates client $k$'s flow for physical time $T_k=\eta u_k=\eta E n_k/B$. Vanilla FedAvg fixes $(E,B,\eta)$ globally, so under client-size imbalance $T_k$ *varies*: the server averages models that each flowed a different amount of time toward their own local optimum. The effective-ODE bound (node 07, its unequal-horizon remark) shows this injects a **first-order** size-imbalance drift $-\mathrm{Cov}_k(T_k,g_k)$ — a pure *size-heterogeneity* term, present even if every $F_k$ has identical shape. The fix: give each client a per-round learning rate $\eta_k=T^\star/u_k$ so $T_k\equiv T^\star$ for all $k$, with $T^\star=\eta E\bar n/B$ (the horizon of an average-size client). Total flow-time is unchanged on average; only its distribution across clients is equalized. No extra communication, no client state.

## Symbols you'll see
| Symbol | Read aloud as | Meaning |
|--------|---------------|---------|
| $u_k=En_k/B$ | "u-sub-k" | Local SGD updates (forward-Euler steps) client $k$ runs per round <!-- TODO add to foundations --> |
| $T_k=\eta u_k$ | "T-sub-k" | Client $k$'s gradient-flow horizon, $=\eta E n_k/B$ <!-- TODO add to foundations --> |
| $T^\star=\eta E\bar n/B$ | "T-star" | Target horizon: that of an average-size client under the baseline LR <!-- TODO add to foundations --> |
| $\eta_k=T^\star/u_k$ | "eta-sub-k" | Per-client equalizing learning rate <!-- TODO add to foundations --> |
| $\bar n=n/K$ | "n-bar" | Mean examples per client <!-- TODO add to foundations --> |
| $g_k=\nabla F_k(w)$ | "g-sub-k" | Client $k$'s local gradient at the broadcast point <!-- TODO add to foundations --> |
| $\mathrm{Cov}_k(T_k,g_k)$ | "cov of T-k and g-k" | The size-imbalance drift cancelled when $T_k\equiv T^\star$ <!-- TODO add to foundations --> |

## Formal definition
$$
\eta_k=\frac{T^\star}{u_k}=\frac{T^\star B}{E\,n_k},\qquad
T^\star=\frac{\eta E\,\bar n}{B}\qquad\Longrightarrow\qquad
T_k=\eta_k u_k\equiv T^\star\ \ \forall k.
$$
Equalizing $T_k$ cancels the first-order size-imbalance drift $-\mathrm{Cov}_k(T_k,g_k)$ of node 07's unequal-horizon remark; the $T^\star$ choice keeps the average flow-time fixed (no free lunch), changing only its distribution across clients.

## Why this matters
Cancels the previously-unnamed size-imbalance drift $-\mathrm{Cov}_k(T_k,g_k)$ that vanilla FedAvg injects under client-size imbalance (the unequal-horizon remark of node 07's effective-ODE bound; 02-math-deep-dive.md §3). Implements 05-improvements.tex E.3 with no extra client state or communication — the actionable corollary of the T.3 theorem.

## Code
The aligned runnable demo lives at [`../code/08-horizon-equalized-local-flow.py`](../code/08-horizon-equalized-local-flow.py). It demonstrates the concept on a finite/low-dimensional example, runnable in <30s on CPU.

**Expected output preview:**
```
Horizon-equalized local flow (05-improvements.tex E.3)
eta_k = T*/u_k so T_k = eta_k*u_k = T* for every client (T* = eta*E*n_bar/B).
Rounds to target, pathological NON-IID, imbalanced (log-uniform) sizes:
  both arms share partition+sampling seed; differ ONLY in eta_k.
```

## Try it yourself
- Exercise 1: Set the size spread to 1 (balanced). Confirm $\eta_k\equiv\eta$, the two arms coincide exactly, and the result is a clean no-op (tie).
- Exercise 2: Grow the size spread and watch the baseline-minus-proposed round margin widen monotonically — the size-imbalance drift $-\mathrm{Cov}_k(T_k,g_k)$ grows with the spread.

## Further reading
- 05-improvements.tex, section E.3; full theorem in `proofs/effective-ode-averaging-bound.tex` (unequal-horizon remark).
- McMahan et al. 2017, *Communication-Efficient Learning of Deep Networks from Decentralized Data* (arXiv:1602.05629), §3.

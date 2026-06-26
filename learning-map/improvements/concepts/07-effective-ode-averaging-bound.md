# Effective-ODE Averaging Bound
**Level:** `extension`  *(novice | intermediate | advanced | graduate | frontier | paper | extension)*
**Concept ID:** `07-effective-ode-averaging-bound`
**Graph:** `improvements`
**Prerequisites:** [paper:08-heterogeneity-gap-covariance](../../paper/concepts/08-heterogeneity-gap-covariance.md), [paper:07-local-update-count](../../paper/concepts/07-local-update-count.md), [taylor theorem](https://github.com/pleyva2004/math-foundations/blob/main/concepts/taylor-theorem.md), [hessian](https://github.com/pleyva2004/math-foundations/blob/main/concepts/hessian.md), [covariance](https://github.com/pleyva2004/math-foundations/blob/main/concepts/covariance.md)
**Used by:** downstream nodes

## Plain-English intro
The math deep dive guessed FedAvg's $E>1$ drift from a two-step Taylor sketch: $\Delta=\eta^2\,\mathrm{Cov}_k(H_k,g_k)$. That was a *heuristic*. Here we make it a *theorem*. Idealize each client's $u_k$ local full-batch steps as the **gradient flow** of its loss $F_k$ run for continuous time $T=\eta u_k$. The FedAvg server map is the weighted average $\mathcal S^T=\sum_k\beta_k\Phi_k^T$ of those per-client flows; the centralized reference is the single flow $\Phi_f^T$ of the global $f$. A Lie–Taylor expansion shows they agree to second order, with the *exact* leading discrepancy $\tfrac{T^2}{2}\lVert\mathrm{Cov}_k(H_k,g_k)\rVert$. This corrects the heuristic prefactor — the frozen $\eta^2$ should be $T^2/2=(\eta u_k)^2/2$, so relative to the raw $\eta^2$ heuristic the drift grows by $u_k^2/2$ (a factor 2 already at $E=2$ — exactly the Euler error of the 2-step sketch — and about $12.5\times$ at $E=5$) — and shows $(E,B,\eta)$ enter only through the single invariant $T=\eta E n_k/B$.

## Symbols you'll see
| Symbol | Read aloud as | Meaning |
|--------|---------------|---------|
| $\Phi_k^{T}(w)$ | "cap-Phi-sub-k of T" | Gradient flow of $F_k$, $\dot\phi=-\nabla F_k(\phi)$, run for time $T$ from $w$ <!-- TODO add to foundations --> |
| $\Phi_f^{T}(w)$ | "cap-Phi-sub-f of T" | Gradient flow of the global $f=\sum_k\beta_kF_k$ for time $T$ <!-- TODO add to foundations --> |
| $\mathcal S^{T}(w)$ | "cap-S of T" | FedAvg server map $\sum_k\beta_k\Phi_k^{T}(w)$ (weighted average of flows) <!-- TODO add to foundations --> |
| $T=\eta u_k$ | "T" | Continuous horizon = (lr) $\times$ (local updates); $T=\eta E n_k/B$ <!-- TODO add to foundations --> |
| $H_k=\nabla^2 F_k(w)$ | "H-sub-k" | Client $k$'s local Hessian at the broadcast point $w$ <!-- TODO add to foundations --> |
| $g_k=\nabla F_k(w)$ | "g-sub-k" | Client $k$'s local gradient at $w$ <!-- TODO add to foundations --> |
| $\beta_k$ | "beta-sub-k" | Client weight $n_k/n$, $\sum_k\beta_k=1$ <!-- TODO add to foundations --> |
| $\Gamma=\mathrm{Cov}_k(H_k,g_k)$ | "cap-Gamma" | Curvature–gradient covariance across clients; $\sigma_{HG}=\lVert\Gamma\rVert$ <!-- TODO add to foundations --> |

## Formal definition
$$
\mathcal S^{T}(w)=\sum_{k=1}^K\beta_k\,\Phi_k^{T}(w),\qquad
\Phi_k^{T}(w)=w-T\,g_k+\tfrac{T^2}{2}\,H_kg_k+R_k(T),\quad \lVert R_k(T)\rVert\le\tfrac{T^3}{6}C_3 .
$$
For clients sharing a horizon $T$, averaging the flow expansions and subtracting $\Phi_f^{T}$ (the $O(1)$ term $w$ and the $O(T)$ term $-Tg$ cancel exactly) gives
$$
\boxed{\;\big\lVert \mathcal S^{T}(w)-\Phi_f^{T}(w)\big\rVert=\frac{T^2}{2}\,\sigma_{HG}+O(T^3),\qquad \sigma_{HG}=\big\lVert\mathrm{Cov}_k(H_k,g_k)\big\rVert\;}
$$
with explicit remainder $\lVert R(T)\rVert\le\tfrac{T^3}{3}C_3$ under bounded third derivatives, and leading drift direction exactly $\Gamma/\lVert\Gamma\rVert$. The single invariant is $T=\eta E n_k/B$.

## Why this matters
Turns the math deep dive's **heuristic** $\Delta=\eta^2\mathrm{Cov}_k(H_k,g_k)$ (02-math-deep-dive.md §3, Eq. 3.5) into a proved **theorem** and *corrects its prefactor*: the right magnitude is $T^2/2=(\eta u_k)^2/2$, not the frozen $\eta^2$, so relative to the raw $\eta^2$ heuristic the drift grows by $u_k^2/2$ ($\approx 12.5\times$ at $E=5$). At $E{=}2,B{=}\infty$ ($T{=}2\eta$) the flow gives $2\eta^2$ — twice the heuristic, the factor 2 being exactly the Euler error of the 2-step sketch. Implements 05-improvements.tex T.3; its unequal-horizon remark $-\mathrm{Cov}_k(T_k,g_k)$ is the size-imbalance drift that node 08 (horizon-equalized local flow) cancels.

## Code
The aligned runnable demo lives at [`../code/07-effective-ode-averaging-bound.py`](../code/07-effective-ode-averaging-bound.py). It demonstrates the concept on a finite/low-dimensional example, runnable in <30s on CPU.

**Expected output preview:**
```
Effective-ODE averaging-drift bound (05-improvements.tex T.3)
Server map S^T = sum_k beta_k Phi_k^T  vs centralized flow Phi_f^T.
K=4 clients, dim=6, ||Cov_k(H_k,g_k)|| = sigma_HG = 0.0694

Measured drift scales as T^p with p ~ 2 (predicted exactly 2):
```

## Try it yourself
- Exercise 1: Set the two client Hessians equal (the homogeneous limit). Confirm $\Gamma=\mathrm{Cov}_k(H_k,g_k)=0$ and the drift collapses to the $O(T^3)$ remainder.
- Exercise 2: Hold $T$ fixed and grow $E$ (shrink the Euler step $\eta=T/u_k$). Confirm the discrete-FedAvg drift converges to the flow drift — the discretization-invariance of Lemma (discrete-to-flow consistency).

## Further reading
- 05-improvements.tex, section T.3; full proof in `proofs/effective-ode-averaging-bound.tex`.
- McMahan et al. 2017, *Communication-Efficient Learning of Deep Networks from Decentralized Data* (arXiv:1602.05629), §3 and Fig. 3.

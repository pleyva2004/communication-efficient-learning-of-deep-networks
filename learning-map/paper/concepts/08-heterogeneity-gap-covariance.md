# Heterogeneity Gap = $\eta^2\,\mathrm{Cov}_k(H_k, g_k)$
**Level:** `paper`  *(novice | intermediate | advanced | graduate | frontier | paper | extension)*
**Concept ID:** `08-heterogeneity-gap-covariance`
**Graph:** `paper`
**Prerequisites:** [06-fedavg-local-iteration](06-fedavg-local-iteration.md), [Taylor theorem](https://github.com/pleyva2004/math-foundations/blob/main/concepts/taylor-theorem.md), [Hessian](https://github.com/pleyva2004/math-foundations/blob/main/concepts/hessian.md), [covariance](https://github.com/pleyva2004/math-foundations/blob/main/concepts/covariance.md)
**Used by:** downstream nodes

## Plain-English intro
Run two local gradient steps on each client, then average the models: this is *not* the same as two centralized gradient steps on the global objective. The difference is the **heterogeneity gap** $\Delta$. A Taylor expansion shows its leading term is $\eta^2$ times the client-weighted **covariance between local Hessians $H_k$ and local gradients $g_k$** — so the gap is zero only when clients are homogeneous, and grows with client drift otherwise.

## Symbols you'll see
| Symbol | Read aloud as | Meaning |
|--------|---------------|---------|
| $w_t$ | "w-sub-t" | Shared per-round starting model <!-- TODO add to foundations --> |
| $w^k_{(2)}$ | "w-k-sub-2" | Client $k$'s model after 2 local steps <!-- TODO add to foundations --> |
| $G^{(2)}(w_t)$ | "cap-G-sup-2 of w-t" | Two centralized GD steps on $f$ <!-- TODO add to foundations --> |
| $g_k=\nabla F_k(w_t)$ | "g-sub-k" | Client $k$'s local gradient at $w_t$ <!-- TODO add to foundations --> |
| $H_k=\nabla^2 F_k(w_t)$ | "H-sub-k" | Client $k$'s local Hessian at $w_t$ <!-- TODO add to foundations --> |
| $H=\sum_k\frac{n_k}{n}H_k$ | "cap-H" | Weighted mean Hessian <!-- TODO add to foundations --> |
| $n_k/n$ | "n-k over n" | Client mixture weight <!-- TODO add to foundations --> |
| $\eta$ | "eta" | Learning rate <!-- TODO add to foundations --> |
| $\mathrm{Cov}_k(H_k,g_k)$ | "cov-sub-k" | $\sum_k\frac{n_k}{n}H_kg_k - H\,\nabla f$ <!-- TODO add to foundations --> |

## Formal definition
$$
\Delta \;=\; \sum_{k}\frac{n_k}{n}\,w^k_{(2)} \;-\; G^{(2)}(w_t)
\;=\; \eta^2\,\mathrm{Cov}_k\!\bigl(H_k,\,g_k\bigr) + O(\eta^3),
\qquad
\mathrm{Cov}_k(H_k,g_k)=\sum_{k}\tfrac{n_k}{n}H_k g_k - H\,\nabla f,\ \ H_k=\nabla^2 F_k.
$$

## Why this matters
This is Eq. (3.5) of `02-math-deep-dive.md`: the exact object behind FedAvg's client drift, and precisely what SCAFFOLD/FedProx (improvement M.1 in `05-improvements.tex`) are built to cancel.

## Code
The aligned runnable demo lives at [`../code/08-heterogeneity-gap-covariance.py`](../code/08-heterogeneity-gap-covariance.py). It demonstrates the concept on a finite/low-dimensional example, runnable in <30s on CPU.

**Expected output preview:**
```
Witness of Eq. (3.5): FedAvg(tau=2) - centralized(2 GD steps) = eta^2 Cov_k(H_k,g_k)
measured gap  Delta = avg_k w^k_(2) - G^(2)(w_t) = [-0.00150894 -0.000443   -0.00110393]
prediction eta^2 * Cov_k(H_k, g_k)              = [-0.00150894 -0.000443   -0.00110393]
```

## Try it yourself
- Exercise 1: Set all $H_k$ equal and all $b_k$ equal (homogeneous clients). Confirm $\Delta\to 0$ and $\mathrm{Cov}_k$ norm collapses.
- Exercise 2: Replace the quadratics with a cubic term so $H_k$ varies with $w$. Watch the residual $\|\Delta-\text{prediction}\|$ grow like $O(\eta^3)$ as you raise $\eta$.

## Further reading
- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, arXiv:1602.05629.
- Karimireddy et al., *SCAFFOLD* (control variates that cancel this $\mathrm{Cov}_k$ term).

# Control-Variate Drift Correction
**Level:** `extension`
**Concept ID:** `01-control-variate-drift-correction`
**Graph:** `improvements`
**Prerequisites:** [paper:08-heterogeneity-gap-covariance](../../paper/concepts/08-heterogeneity-gap-covariance.md), [control variates](https://github.com/pleyva2004/math-foundations/blob/main/concepts/control-variates.md)
**Used by:** downstream nodes

## Plain-English intro
FedAvg lets each client take several local gradient steps before averaging, but on non-IID data each client drifts toward *its own* optimum instead of the global one. A control variate is a per-client correction vector that, added to every local step, re-centers that client's trajectory onto the global gradient direction. This is the SCAFFOLD fix, and it directly cancels the heterogeneity drift identified in paper node 08.

## Symbols you'll see
| Symbol | Read aloud as | Meaning |
|--------|---------------|---------|
| $w$ | "w" | Model parameter vector being updated locally <!-- TODO add to foundations --> |
| $\eta$ | "eta" | Local learning rate <!-- TODO add to foundations --> |
| $g_{\mathrm{local}}(w)$ | "g-local of w" | Client's local gradient at the current local iterate $w$ <!-- TODO add to foundations --> |
| $c_k$ | "c-sub-k" | Client $k$'s control variate (its drift estimate) <!-- TODO add to foundations --> |
| $c$ | "c" | Server control variate, average of selected $c_k$ <!-- TODO add to foundations --> |
| $S_t$ | "S-sub-t" | Set of clients selected in round $t$ <!-- TODO add to foundations --> |
| $H_k=\nabla^2 F_k(w_t)$ | "H-sub-k" | Client $k$'s local Hessian at $w_t$ <!-- TODO add to foundations --> |
| $g_k=\nabla F_k(w_t)$ | "g-sub-k" | Client $k$'s local gradient at the shared point <!-- TODO add to foundations --> |

## Formal definition
$$
\text{Local step (client } k):\quad w \leftarrow w-\eta\bigl(g_{\mathrm{local}}(w)-c_k+c\bigr),
\qquad c=\frac{1}{|S_t|}\sum_{k\in S_t}c_k .
$$
In expectation the $-c_k+c$ term replaces the locally-biased direction with the global one, cancelling the leading client-heterogeneity drift
$$
\Delta=\eta^2\,\mathrm{Cov}_k\!\bigl(H_k,g_k\bigr)+O(\eta^3),
$$
so the multi-step round recovers $w\leftarrow w-\eta\,\nabla f(w)$ to leading order.

## Why this matters
Operationalizes the heterogeneity-gap finding (paper node 08): it cancels the leading $\eta^2\mathrm{Cov}_k(H_k,g_k)$ drift of Eq. (3.5) in `02-math-deep-dive.md`, and is the highest-leverage proposal M.1 of `05-improvements.tex`.

## Code
The aligned runnable demo lives at [`../code/01-control-variate-drift-correction.py`](../code/01-control-variate-drift-correction.py). It demonstrates the concept on a finite/low-dimensional example, runnable in <30s on CPU.

**Expected output preview:**
```
Global gradient grad f(w_t) at shared point w_t=[0.7 0.7]:
  g_global = [-0.275 -0.275]
Per-client effective local direction vs global gradient:
```

## Try it yourself
- Exercise 1: Set the two Hessians equal (`H[1] = H[0]`) — the homogeneous limit. Confirm the plain and corrected deviations both collapse, since $\mathrm{Cov}_k(H_k,g_k)=0$.
- Exercise 2: Increase `tau` (local steps) and `eta`. Watch the *plain* drift grow with $\eta^2\tau$ while the corrected direction stays anchored to the global gradient.

## Further reading
- Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning," ICML 2020.

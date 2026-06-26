# FedSGD as Exact Gradient Descent
**Level:** `paper`  *(novice | intermediate | advanced | graduate | frontier | paper | extension)*
**Concept ID:** `04-fedsgd-gradient-descent`
**Graph:** `paper`
**Prerequisites:** [02-client-partition-decomposition](02-client-partition-decomposition.md), [gradient](https://github.com/pleyva2004/math-foundations/blob/main/concepts/gradient.md), [gradient descent](https://github.com/pleyva2004/math-foundations/blob/main/concepts/gradient-descent.md)
**Used by:** downstream nodes

## Plain-English intro
FedSGD has every client compute the average gradient of its own data at the shared model $w_t$, then the server combines them. When *all* clients participate ($C{=}1$), the size-weighted sum of those local gradients equals the gradient of the global objective exactly. So one FedSGD round is nothing exotic: it is textbook deterministic full-batch gradient descent on $f$, with the clients merely evaluating disjoint partial sums of one global gradient. This makes FedSGD the baseline endpoint against which all of FedAvg is measured.

## Symbols you'll see
| Symbol | Read aloud as | Meaning |
|--------|---------------|---------|
| $w_t$ | "w-sub-t" | Shared model parameters at round $t$ <!-- TODO add to foundations --> |
| $f(w)$ | "f of w" | Global objective: mean loss over all $n$ examples <!-- TODO add to foundations --> |
| $F_k(w)$ | "cap-F-sub-k of w" | Client $k$'s local objective (mean loss on $P_k$) <!-- TODO add to foundations --> |
| $g_k=\nabla F_k(w_t)$ | "g-sub-k" | Client $k$'s local average gradient at $w_t$ <!-- TODO add to foundations --> |
| $n_k$ | "n-sub-k" | Number of examples on client $k$ <!-- TODO add to foundations --> |
| $n$ | "n" | Total number of training examples ($n=\sum_k n_k$) <!-- TODO add to foundations --> |
| $\eta$ | "eta" | Learning rate <!-- TODO add to foundations --> |
| $C$ | "C" | Fraction of clients selected per round ($C{=}1$ here) <!-- TODO add to foundations --> |

## Formal definition
$$
g_k \stackrel{\text{def}}{=} \nabla F_k(w_t),\qquad
\sum_{k=1}^{K}\frac{n_k}{n}\,g_k \;=\; \nabla f(w_t)\quad\text{(Eq. 2.2)};
$$
$$
\text{the }C{=}1\text{ FedSGD update}\quad
w_{t+1}=w_t-\eta\sum_{k=1}^{K}\frac{n_k}{n}g_k \;=\; w_t-\eta\,\nabla f(w_t)\quad\text{(Eq. 2.3)}.
$$
The identity is exact for *any* partition (no IID assumption); it follows by differentiating $f=\sum_k\frac{n_k}{n}F_k$ term-by-term.

## Why this matters
Identifies FedSGD as exact distributed full-batch gradient descent -- the baseline endpoint of the $(C,E,B)$ family. Appears as Eq. (2.2)-(2.3) of `02-math-deep-dive.md` (§2).

## Code
The aligned runnable demo lives at [`../code/04-fedsgd-gradient-descent.py`](../code/04-fedsgd-gradient-descent.py). It demonstrates the concept on a finite/low-dimensional example, runnable in <30s on CPU.

**Expected output preview:**
```
FedSGD (C=1) gradient-aggregation identity:  sum_k (n_k/n) g_k  ==  grad f(w_t)
client sizes n_k = [2, 3, 3, 4], total n = 12, weights n_k/n = [0.1667, 0.25, 0.25, 0.3333]
residual ||sum_k (n_k/n) g_k - grad f(w_t)|| = 5.088e-16  (expect ~1e-12)
```

## Try it yourself
- Exercise 1: Make the partition unbalanced (e.g. one client holds 9 of 12 examples). Confirm the weighted residual stays ~1e-16 but the *unweighted* mean $\frac1K\sum_k g_k$ no longer matches $\nabla f$.
- Exercise 2: Set $C<1$ by aggregating over a random client subset $S_t$ with weights $n_k/m_t$. Observe that the single-round residual is now nonzero -- FedSGD becomes a stochastic estimator of $\nabla f$, not an equality (see §5).

## Further reading
- McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data," AISTATS 2017, §2 (arXiv:1602.05629).

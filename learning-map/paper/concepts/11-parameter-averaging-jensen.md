# Parameter Averaging & Jensen
**Level:** `paper`  *(novice | intermediate | advanced | graduate | frontier | paper | extension)*
**Concept ID:** `11-parameter-averaging-jensen`
**Graph:** `paper`
**Prerequisites:** [02-client-partition-decomposition](02-client-partition-decomposition.md), [convex function](https://github.com/pleyva2004/math-foundations/blob/main/concepts/convex-function.md), [Jensen's inequality](https://github.com/pleyva2004/math-foundations/blob/main/concepts/jensens-inequality.md)
**Used by:** downstream nodes

## Plain-English intro
FedAvg's server step replaces every client's model with one weighted average $\bar w=\sum_k\beta_k w^k$. If the loss $f$ is convex, Jensen's inequality guarantees this averaged model is never worse than the weighted-mean of the parents, and in turn no worse than the *worst* parent. That is the clean reason averaging in parameter space is "safe" in the convex case -- and the reason the non-convex case (real nets) instead needs the shared-initialization trick.

## Symbols you'll see
| Symbol | Read aloud as | Meaning |
|--------|---------------|---------|
| $f(w)$ | "f of w" | Global (convex) objective being averaged over <!-- TODO add to foundations --> |
| $w^k$ | "w-super-k" | Client $k$'s locally-trained model <!-- TODO add to foundations --> |
| $\beta_k=n_k/m_t$ | "beta-sub-k" | Mixture weight; probability vector ($\beta_k\ge0,\sum_k\beta_k=1$) <!-- TODO add to foundations --> |
| $F_k$ | "cap-F-sub-k" | Client $k$'s local objective; $f=\sum_k\alpha_k F_k$ <!-- TODO add to foundations --> |
| $\bar w=\sum_k\beta_k w^k$ | "w-bar" | The averaged (aggregated) model <!-- TODO add to foundations --> |

## Formal definition
For convex $F_k$, the mixture $f=\sum_k\alpha_k F_k$ is convex, so for any convex weights $\beta_k\ge0,\ \sum_k\beta_k=1$:
$$ f\!\Big(\sum_{k}\beta_k w^k\Big)\ \le\ \sum_{k}\beta_k\,f(w^k)\ \le\ \max_k f(w^k). $$
The first inequality is Jensen; the second is that a convex combination is bounded by its largest term.

## Why this matters
Appears as Eq. (7.1) of `02-math-deep-dive.md` (§7a). It formalizes why averaging is harmless under convexity and motivates §7b-c -- non-convex nets have loss barriers, so FedAvg must engineer a shared per-round init to keep the parents in a common basin.

## Code
The aligned runnable demo lives at [`../code/11-parameter-averaging-jensen.py`](../code/11-parameter-averaging-jensen.py). It demonstrates the concept on a finite/low-dimensional example, runnable in <30s on CPU.
**Expected output preview:**
```
Jensen chain on a convex quadratic f (Eq. 7.1), K=5 clients, d=4:
  beta (mixture weights, sum=1.000): [0.34  0.113 0.078 0.339 0.13 ]
  per-client losses f(w^k):        [87.6346 72.7985 83.4229 67.1886 44.7001]
```

## Try it yourself
- Exercise 1: Make $A$ negative-definite (non-convex $f$) and watch the Jensen assertion fail -- the averaged model can exceed every parent.
- Exercise 2: Add the strong-convexity midpoint refinement $f(\tfrac{w+w'}2)\le\tfrac12 f(w)+\tfrac12 f(w')-\tfrac\mu8\lVert w-w'\rVert^2$ and verify the $\mu$-gain numerically.

## Further reading
- `02-math-deep-dive.md` §7 (convex vs non-convex averaging); Jensen's inequality (any convex-analysis text).

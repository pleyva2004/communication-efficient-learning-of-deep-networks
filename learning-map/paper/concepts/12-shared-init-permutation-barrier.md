# Shared-Init & the Permutation Barrier
**Level:** `paper`  *(novice | intermediate | advanced | graduate | frontier | paper | extension)*
**Concept ID:** `12-shared-init-permutation-barrier`
**Graph:** `paper`
**Prerequisites:** [11-parameter-averaging-jensen](11-parameter-averaging-jensen.md), [permutation group](https://github.com/pleyva2004/math-foundations/blob/main/concepts/permutation-group.md)
**Used by:** downstream nodes

## Plain-English intro
A one-hidden-layer network computes the *same function* if you relabel its hidden units, so each minimizer is copied across a huge orbit of permuted twins. Two nets trained from *independent* random seeds land in *incompatible* twins, and naively averaging their weights blends unrelated units, producing a model worse than both parents (a loss "barrier"). FedAvg dodges this by broadcasting the *same* model $w_t$ every round, anchoring all clients in one basin so the average instead beats both.

## Symbols you'll see
| Symbol | Read aloud as | Meaning |
|--------|---------------|---------|
| $W_1, b_1$ | "W-one, b-one" | First-layer weights / bias (hidden layer) <!-- TODO add to foundations --> |
| $W_2$ | "W-two" | Second-layer (output) weights <!-- TODO add to foundations --> |
| $P$ | "P" | A hidden-unit permutation matrix <!-- TODO add to foundations --> |
| $\sigma$ | "sigma" | Elementwise nonlinearity (e.g. $\tanh$) <!-- TODO add to foundations --> |
| $w_t$ | "w-sub-t" | Shared global model broadcast at round $t$ <!-- TODO add to foundations --> |
| $w, w'$ | "w, w-prime" | Two client models being averaged <!-- TODO add to foundations --> |

## Formal definition
A 1-hidden-layer net is invariant under any hidden-unit permutation $P$ (an element of the symmetric group $S_h$ acting as a permutation matrix):
$$ W_2\,\sigma(W_1 x + b_1)=(W_2 P^\top)\,\sigma\bigl(P W_1 x + P b_1\bigr),\qquad (W_1,b_1,W_2)\mapsto(W_1 P^\top,\,P b_1,\,P W_2). $$
Hence every minimizer is replicated across an orbit of size $\prod_\ell |S_{h_\ell}|$. Independent inits occupy incompatible orbit elements, so $f\bigl(\tfrac12 w+\tfrac12 w'\bigr)>\max\{f(w),f(w')\}$ (Fig. 1 barrier); a shared $w_t$ fixes one orbit element, keeping clients in a common basin where averaging instead lowers loss.

## Why this matters
This is the load-bearing empirical justification for FedAvg, formalized in §7(b)-(c) of `02-math-deep-dive.md`; it motivates improvement T.1 (permutation-aligned averaging) in `05-improvements.tex`.

## Code
The aligned runnable demo lives at [`../code/12-shared-init-permutation-barrier.py`](../code/12-shared-init-permutation-barrier.py). It demonstrates the concept on a finite/low-dimensional example, runnable in <30s on CPU.
**Expected output preview:**
```
FedAvg Fig.1: averaging two 1-hidden-layer MLPs trained on disjoint halves.
Barrier height = loss(average) - max(loss(parents)); >0 means avg is WORSE.

INDEPENDENT inits : parents=(0.1982, 0.1849)  avg=0.2231  barrier=+0.0249  -> BARRIER (avg worse)
SHARED init  : parents=(0.1958, 0.1798)  avg=0.1778  barrier=-0.0180  -> VALLEY (avg better)
```

## Try it yourself
- Exercise 1: Permute one trained model's hidden units by a random $P$ (apply $W_1\mapsto W_1 P^\top, b_1\mapsto P b_1, W_2\mapsto P W_2$); confirm `loss` is unchanged to ~1e-9, then average it with the original and watch the barrier reappear.
- Exercise 2: Sweep the number of hidden units $H$. Does a larger (more over-parameterized) net make the shared-init valley deeper and the independent-init barrier higher?

## Further reading
- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017 (Fig. 1).
- Ainsworth, Hayase, Srinivasa, *Git Re-Basin: Merging Models modulo Permutation Symmetries*, ICLR 2023.

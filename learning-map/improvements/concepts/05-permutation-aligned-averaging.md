# Permutation-Aligned Averaging
**Level:** `extension`
**Concept ID:** `05-permutation-aligned-averaging`
**Graph:** `improvements`
**Prerequisites:** [paper:12-shared-init-permutation-barrier](../../paper/concepts/12-shared-init-permutation-barrier.md), [linear assignment problem](https://github.com/pleyva2004/math-foundations/blob/main/concepts/linear-assignment-problem.md)
**Used by:** downstream nodes

## Plain-English intro
A one-hidden-layer net is unchanged if you relabel its hidden units, so two nets trained from independent inits land in *different* relabelings of the same solution and naive averaging blends unrelated units (the loss barrier of paper node 12). Before averaging, we instead solve a linear assignment problem for the permutation $P$ that best matches one net's hidden units to the other's, then average the aligned weights. This is Git Re-Basin weight-matching, and it relaxes FedAvg's shared-initialization requirement.

## Symbols you'll see
| Symbol | Read aloud as | Meaning |
|--------|---------------|---------|
| $W_1$ | "W-one" | Server model's first-layer (hidden) weight matrix <!-- TODO add to foundations --> |
| $W_1'$ | "W-one-prime" | Client model's first-layer weight matrix <!-- TODO add to foundations --> |
| $P$ | "P" | A permutation (matrix) relabeling the $H$ hidden units <!-- TODO add to foundations --> |
| $\Pi_H$ | "cap-Pi-sub-H" | Set of all $H\times H$ permutation matrices <!-- TODO add to foundations --> |
| $H$ | "H" | Number of hidden units <!-- TODO add to foundations --> |
| $\lVert\cdot\rVert_F$ | "Frobenius norm" | Entrywise Euclidean norm of a matrix <!-- TODO add to foundations --> |
| $w,\,w'$ | "w, w-prime" | Full parameter vectors of the server / client models <!-- TODO add to foundations --> |

## Formal definition
$$
P^\star=\arg\min_{P\in\Pi_H}\big\lVert W_1-W_1'P^\top\big\rVert_F^2
       =\arg\max_{P\in\Pi_H}\sum_{i=1}^{H}\big\langle (W_1)_{:,i},\,(W_1')_{:,P(i)}\big\rangle,
$$
since the column norms are permutation-invariant, only the cross-term depends on $P$ — a **linear assignment problem**. The aligned client $w'\mapsto P^\star(w')$ is a function-preserving symmetry ($W_1'\mapsto W_1'P^\top,\ b_1'\mapsto Pb_1',\ W_2'\mapsto PW_2'$), and the server averages $\tfrac12 w+\tfrac12 P^\star(w')$.

## Why this matters
Relaxes FedAvg's shared-init requirement (paper node 12) by aligning independently-trained client models into a common basin via Git Re-Basin weight-matching; this is proposal T.1 of `05-improvements.tex`.

## Code
The aligned runnable demo lives at [`../code/05-permutation-aligned-averaging.py`](../code/05-permutation-aligned-averaging.py). It demonstrates the concept on a finite/low-dimensional example, runnable in <30s on CPU.

**Expected output preview:**
```
Permutation-aligned averaging witness (Git Re-Basin weight-matching)
H = 6 hidden units; client is a relabeling Q = [0, 5, 3, 2, 1, 4]
Hidden-layer Frobenius distance ||W1 - W1'||_F  unmatched = 7.2038
```

## Try it yourself
- Exercise 1: Drop the noise (`0.05 -> 0.0`). The matched distance and `max |f_client - f_aligned|` both go to ~0, and the recovered permutation is exactly $Q^{-1}$.
- Exercise 2: Increase the noise (`0.05 -> 0.5`) until greedy matching mis-pairs a column; check that the matched distance still beats unmatched but the function is no longer preserved.

## Further reading
- Ainsworth, Hayase, Srinivasa, "Git Re-Basin: Merging Models modulo Permutation Symmetries," ICLR 2023.

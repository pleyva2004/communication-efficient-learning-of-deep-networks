# IID vs Non-IID
**Level:** `paper`  *(novice | intermediate | advanced | graduate | frontier | paper | extension)*
**Concept ID:** `03-iid-noniid-dichotomy`
**Graph:** `paper`
**Prerequisites:** [02-client-partition-decomposition](02-client-partition-decomposition.md), [unbiased estimator](https://github.com/pleyva2004/math-foundations/blob/main/concepts/unbiased-estimator.md)
**Used by:** downstream nodes

## Plain-English intro
The decomposition $f=\sum_k\frac{n_k}{n}F_k$ holds for *any* split of the data. The IID/non-IID dichotomy asks a *statistical* question on top of it: does a single client's local objective $F_k$ resemble the global $f$? If the data is sprinkled across clients uniformly at random (IID), each $F_k$ is an unbiased proxy for $f$. If instead each client hoards one kind of example (non-IID), $F_k$ can be an arbitrarily bad approximation to $f$ — yet the size-weighted mixture of all $F_k$ still equals $f$ exactly.

## Symbols you'll see
| Symbol | Read aloud as | Meaning |
|--------|---------------|---------|
| $w$ | "w" | Model parameter vector being evaluated <!-- TODO add to foundations --> |
| $f(w)$ | "f of w" | Global objective: mean loss over all $n$ examples <!-- TODO add to foundations --> |
| $F_k(w)$ | "cap-F-sub-k of w" | Client $k$'s local objective (mean loss on $P_k$) <!-- TODO add to foundations --> |
| $P_k$ | "P-sub-k" | Index set of examples held by client $k$ <!-- TODO add to foundations --> |
| $n_k$ | "n-sub-k" | Number of examples on client $k$ <!-- TODO add to foundations --> |
| $\mathbb{E}_{P_k}[\cdot]$ | "expectation over P-k" | Average over the random draw of $P_k$ <!-- TODO add to foundations --> |

## Formal definition
$$
\textbf{IID:}\quad P_k\ \text{a uniform-random partition}\ \Longrightarrow\ \mathbb{E}_{P_k}\!\big[F_k(w)\big]=f(w)\quad\forall w\ \text{(Eq. 1.2, unbiased proxy)}.
$$
$$
\textbf{Non-IID:}\ \text{the negation —}\ \exists\,w:\ \mathbb{E}_{P_k}\!\big[F_k(w)\big]\neq f(w),\ \text{equivalently}\ \sup_w\big|F_k(w)-f(w)\big|\ \text{unbounded.}
$$
The exact identity $f(w)=\sum_{k}\tfrac{n_k}{n}F_k(w)$ is **unaffected** in either regime; only per-client statistical fidelity changes.

## Why this matters
This is the regime that distinguishes federated optimization from ordinary distributed SGD; it appears as Eq. (1.2) and the dichotomy table of §1 in `02-math-deep-dive.md`, and drives the FedAvg non-IID robustness questions that motivate `05-improvements.tex` M.1 (heterogeneity drift $\Delta=\eta^2\mathrm{Cov}_k(H_k,g_k)$).

## Code
The aligned runnable demo lives at [`../code/03-iid-noniid-dichotomy.py`](../code/03-iid-noniid-dichotomy.py). It demonstrates the concept on a finite/low-dimensional example, runnable in <30s on CPU.

**Expected output preview:**
```
Fixed point w; global objective f(w) = 8.2501
IID partition (uniform-random shards): mean |F_k(w) - f(w)| = 0.2153   <- unbiased, E[F_k]~f  (Eq. 1.2)
Non-IID partition (sort-by-label shards): mean |F_k(w) - f(w)| = 6.4000   <- arbitrarily bad proxy
```

## Try it yourself
- Exercise 1: Increase shard count $K$ (fewer examples per shard). Does the IID gap grow? Relate it to the $1/\sqrt{n_k}$ shrinkage of an unbiased estimator's standard error.
- Exercise 2: Make the non-IID split *two* labels per shard instead of one. Confirm the gap shrinks but the mixture still reconstructs $f(w)$ exactly.

## Further reading
- McMahan et al. 2017, *Communication-Efficient Learning of Deep Networks from Decentralized Data* (arXiv:1602.05629), §3 pathological non-IID MNIST.
- `02-math-deep-dive.md` §1 (Eq. 1.1, 1.2 and the IID/non-IID status table).

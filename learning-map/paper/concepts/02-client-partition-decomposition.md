# Client-Partition Decomposition
**Level:** `paper`  *(novice | intermediate | advanced | graduate | frontier | paper | extension)*
**Concept ID:** `02-client-partition-decomposition`
**Graph:** `paper`
**Prerequisites:** [01-finite-sum-objective](01-finite-sum-objective.md), [partition of a set](https://github.com/pleyva2004/math-foundations/blob/main/concepts/partition-of-a-set.md)
**Used by:** downstream nodes

## Plain-English intro
FedAvg distributes the $n$ training examples across $K$ clients, where client $k$ holds an index set $P_k$. Each client has its own *local* objective $F_k$, the average loss over only its examples. This concept shows the global objective $f$ is exactly the size-weighted average of the local objectives: nothing is approximated, and no assumption is made about *how* the data was split.

## Symbols you'll see
| Symbol | Read aloud as | Meaning |
|--------|---------------|---------|
| $w\in\mathbb{R}^d$ | "w in R-d" | Model parameter vector <!-- TODO add to foundations --> |
| $f(w)$ | "f of w" | Global objective: mean loss over all $n$ examples <!-- TODO add to foundations --> |
| $f_i(w)$ | "f-sub-i of w" | Per-example loss on example $i$ <!-- TODO add to foundations --> |
| $n$ | "n" | Total number of training examples <!-- TODO add to foundations --> |
| $K$ | "K" | Total number of clients <!-- TODO add to foundations --> |
| $P_k$ | "P-sub-k" | Index set of examples held by client $k$ <!-- TODO add to foundations --> |
| $n_k=\lvert P_k\rvert$ | "n-sub-k" | Number of examples on client $k$ <!-- TODO add to foundations --> |
| $F_k(w)$ | "cap-F-sub-k of w" | Client $k$'s local objective <!-- TODO add to foundations --> |

## Formal definition
$$
\bigcup_{k=1}^{K}P_k=[n],\quad P_k\cap P_{k'}=\varnothing\ (k\ne k'),\quad n_k:=|P_k|,\ \textstyle\sum_k n_k=n;
$$
$$
F_k(w):=\frac{1}{n_k}\sum_{i\in P_k}f_i(w)\quad\Longrightarrow\quad f(w)=\sum_{k=1}^{K}\frac{n_k}{n}\,F_k(w)\quad\forall w\in\mathbb{R}^d.
$$
The weight $\frac{n_k}{n}\cdot\frac{1}{n_k}=\frac1n$ collapses, re-indexing the partitioned sum back to $\frac1n\sum_{i=1}^n f_i$.

## Why this matters
This is the structural backbone of FedAvg, appearing as Eq. (1.1) of `02-math-deep-dive.md`. It is an exact algebraic identity for *any* partition, even adversarial/non-IID, which is why the IID assumption is needed only for *quality*, never for *correctness*.

## Code
The aligned runnable demo lives at [`../code/02-client-partition-decomposition.py`](../code/02-client-partition-decomposition.py). It demonstrates the concept on a finite/low-dimensional example, runnable in <30s on CPU.
**Expected output preview:**
```
Client-Partition Decomposition: f(w) = sum_k (n_k/n) F_k(w)  [FedAvg Eq. 1.1]
n=60 examples, K=5 clients, sizes n_k=[7, 25, 15, 1, 12] (unequal), n_k/n weights sum=1.0
max |residual| over 6 random w = 8.88e-16  (~machine epsilon => identity holds)
```

## Try it yourself
- Exercise 1: Make the partition NOT cover all of $[n]$ (drop one index). Confirm the residual is no longer ~1e-15, showing coverage is load-bearing.
- Exercise 2: Replace `n_k/n` with uniform client weights `1/K` and observe the residual blow up unless all $n_k$ are equal.

## Further reading
- McMahan et al. (2017), "Communication-Efficient Learning of Deep Networks from Decentralized Data," arXiv:1602.05629, Eq. (1).

# Client-Sampling Unbiasedness (Horvitz-Thompson)
**Level:** `paper`  *(novice | intermediate | advanced | graduate | frontier | paper | extension)*
**Concept ID:** `09-client-sampling-unbiasedness`
**Graph:** `paper`
**Prerequisites:** [03-iid-noniid-dichotomy](03-iid-noniid-dichotomy.md), [04-fedsgd-gradient-descent](04-fedsgd-gradient-descent.md), [unbiased estimator](https://github.com/pleyva2004/math-foundations/blob/main/concepts/unbiased-estimator.md)
**Used by:** downstream nodes

## Plain-English intro
When only a fraction $C<1$ of clients report each round, the server sees a random subset $S$, not everyone. If you weight each sampled client's gradient by the inverse of its chance of being picked, the random average lands *on the true full-data gradient in expectation* — no bias. This Horvitz-Thompson trick is what makes partial participation legitimate, and it needs no IID assumption: the unbiasedness is purely combinatorial.

## Symbols you'll see
| Symbol | Read aloud as | Meaning |
|--------|---------------|---------|
| $S,\ m=\lvert S\rvert$ | "S, m" | Random selected client set / its size, $m=\max(CK,1)$ <!-- TODO add to foundations --> |
| $K$ | "K" | Total number of clients <!-- TODO add to foundations --> |
| $p_k=m/K$ | "p-sub-k" | Inclusion probability of client $k$ (uniform sampling) <!-- TODO add to foundations --> |
| $n_k,\ n$ | "n-sub-k, n" | Examples on client $k$ / total examples <!-- TODO add to foundations --> |
| $g_k=\nabla F_k(w)$ | "g-sub-k" | Client $k$'s local average gradient at $w$ <!-- TODO add to foundations --> |
| $g(S)$ | "g of S" | Horvitz-Thompson sampled gradient estimator <!-- TODO add to foundations --> |
| $\nabla f(w)$ | "grad f" | True global gradient (the estimand) <!-- TODO add to foundations --> |

## Formal definition
Assume (A1): $S$ is a uniform $m$-subset of $[K]$ (sampling without replacement), so $p_k:=\Pr[k\in S]=m/K$ for every $k$. The Horvitz-Thompson / inverse-probability estimator weights each sampled term by $1/p_k$:
$$
g(S):=\sum_{k\in S}\frac{1}{p_k}\frac{n_k}{n}g_k=\frac{K}{m}\sum_{k\in S}\frac{n_k}{n}g_k,
\qquad
\mathbb{E}_S[g]=\sum_{k=1}^{K}p_k\cdot\frac{1}{p_k}\frac{n_k}{n}g_k=\sum_{k=1}^{K}\frac{n_k}{n}g_k=\nabla f(w).
$$
The $p_k$ cancels exactly (Eqs. 5.1-5.2). The uncorrected subset-sum $\sum_{k\in S}\frac{n_k}{n}g_k$ has expectation $\frac{m}{K}\nabla f(w)=C\,\nabla f(w)$ — biased toward $0$.

## Why this matters
Justifies partial participation ($C<1$): appears as Eqs. (5.1)-(5.2) of `02-math-deep-dive.md` §5, and underpins the unbiased-aggregation fix `05-improvements.tex` M.2(a). The unbiasedness is purely combinatorial — no IID needed; non-IID affects only $\mathrm{Var}(g)$, never $\mathbb{E}[g]$.

## Code
The aligned runnable demo lives at [`../code/09-client-sampling-unbiasedness.py`](../code/09-client-sampling-unbiasedness.py). It demonstrates the concept on a finite/low-dimensional example, runnable in <30s on CPU.

**Expected output preview:**
```
Client-Sampling Unbiasedness (Horvitz-Thompson), Eqs. 5.1-5.2
K=12 clients, m=4 sampled/round, inclusion prob p_k=m/K=0.333, MC trials=200000
   -> HT bias ||E[HT]-grad f|| = 1.0375e-03  (~0, UNBIASED)
```

## Try it yourself
- Exercise 1: Change $m$ (e.g. $m=2$ then $m=8$) and confirm the naive estimator's bias norm tracks the shrink factor $C=m/K$, while HT stays ~0.
- Exercise 2: Break (A1) by sampling clients with probability $\propto n_k$ instead of uniformly; show plain $\frac1m\sum_{k\in S}g_k$ then becomes unbiased while $K/m$-weighting does not (M.2(b)).

## Further reading
- McMahan et al. 2017, *Communication-Efficient Learning of Deep Networks from Decentralized Data*, footnote 2 (arXiv:1602.05629).
- Horvitz & Thompson 1952, *A generalization of sampling without replacement from a finite universe*, JASA.

# Gradient- vs Model-Averaging Equivalence
**Level:** `paper`  *(novice | intermediate | advanced | graduate | frontier | paper | extension)*
**Concept ID:** `05-gradient-model-averaging-equivalence`
**Graph:** `paper`
**Prerequisites:** [04-fedsgd-gradient-descent](04-fedsgd-gradient-descent.md)
**Used by:** downstream nodes

## Plain-English intro
FedSGD can be run two ways that give the *same* answer for one local step: the server can average everyone's gradients and then take a step, OR each client can take its own step and the server averages the resulting models. They coincide because all clients start from the *same* point $w_t$ and the weights $n_k/n$ sum to 1. Take a *second* local step and the two schemes no longer agree — that gap is exactly what FedAvg exploits.

## Symbols you'll see
| Symbol | Read aloud as | Meaning |
|--------|---------------|---------|
| $w_t$ | "w-sub-t" | Shared model at round $t$ (common start for every client) <!-- TODO add to foundations --> |
| $w^k$ | "w-super-k" | Client $k$'s model after its local step(s) <!-- TODO add to foundations --> |
| $g_k=\nabla F_k(w_t)$ | "g-sub-k" | Client $k$'s local gradient at $w_t$ <!-- TODO add to foundations --> |
| $n_k$ | "n-sub-k" | Number of examples on client $k$ <!-- TODO add to foundations --> |
| $n=\sum_k n_k$ | "n" | Total examples; makes $n_k/n$ a weight summing to 1 <!-- TODO add to foundations --> |
| $\eta$ | "eta" | Learning rate <!-- TODO add to foundations --> |

## Formal definition
$$
\sum_{k}\frac{n_k}{n}\,w^k=\sum_{k}\frac{n_k}{n}\bigl(w_t-\eta g_k\bigr)=\underbrace{\Bigl(\sum_k\tfrac{n_k}{n}\Bigr)}_{=\,1}w_t-\eta\sum_k\frac{n_k}{n}g_k=w_t-\eta\sum_k\frac{n_k}{n}g_k.
$$
For $\tau\ge2$ local steps the composite local map is nonlinear in $w_t$, so $\sum_k\frac{n_k}{n}w^k_{(\tau)}\neq$ the centralized averaged-gradient iterate (Eq. 3.4).

## Why this matters
This one-step identity is Eq. (3.2) of `02-math-deep-dive.md` (§3): the hinge that lets FedSGD be reorganized as model averaging, which FedAvg then generalizes by iterating the local step.

## Code
The aligned runnable demo lives at [`../code/05-gradient-model-averaging-equivalence.py`](../code/05-gradient-model-averaging-equivalence.py). It demonstrates the concept on a finite/low-dimensional example, runnable in <30s on CPU.

**Expected output preview:**
```
Concept 05: model-average of one local step == gradient-average step (Eq. 3.2)
ONE local step:
  ||model-average - gradient-average|| = 1.110e-16  (should be ~1e-15)
```

## Try it yourself
- Exercise 1: Change the client weights so they do *not* sum to 1 (e.g. drop the normalization). Confirm the one-step residual is no longer ~1e-16, and identify which term in the formal definition broke.
- Exercise 2: Give clients *different* starting points instead of the shared $w_t$. Verify the one-step equivalence now fails too, isolating "shared initialization" as the load-bearing assumption.

## Further reading
- McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data," AISTATS 2017 (arXiv:1602.05629), §2-3.
- `02-math-deep-dive.md`, §3 (one-step equivalence and the $\Delta=\eta^2\mathrm{Cov}_k(H_k,g_k)$ heterogeneity gap).

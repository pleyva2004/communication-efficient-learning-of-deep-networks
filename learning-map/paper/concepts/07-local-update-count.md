# Local-Update Count u = nE/(KB)

**Level:** `paper`
**Concept ID:** `07-local-update-count`
**Graph:** `paper`
**Prerequisites:** [06-fedavg-local-iteration](06-fedavg-local-iteration.md)
**Used by:** downstream nodes

## Plain-English intro
Each round, a client runs `ClientUpdate`: `E` passes (epochs) over its local data, each pass split into minibatches of size `B`, and every minibatch is one SGD step. So one client does `u_k = E*ceil(n_k/B)` local updates. Averaged over a uniformly random client (with `E[n_k]=n/K`), the expected per-round local work is `u = nE/(KB)`. This single number, not raw `(E,B)`, orders the configurations in the paper's Table 2 and quantifies how much extra on-device compute each round buys.

## Symbols you'll see
| Symbol | Read aloud as | Meaning |
|--------|---------------|---------|
| $u_k$ | "u-sub-k" | Local SGD updates client $k$ runs per round <!-- TODO add to foundations --> |
| $u$ | "u" | Expected local updates per round, over a random client <!-- TODO add to foundations --> |
| $E$ | "E" | Local epochs per client per round <!-- TODO add to foundations --> |
| $B$ | "B" | Local minibatch size ($B=\infty$ ⇒ full batch) <!-- TODO add to foundations --> |
| $n_k$ | "n-sub-k" | Number of examples on client $k$ <!-- TODO add to foundations --> |
| $n$ | "n" | Total number of training examples <!-- TODO add to foundations --> |
| $K$ | "K" | Total number of clients <!-- TODO add to foundations --> |

## Formal definition
$$
u_k \;=\; E\Bigl\lceil\tfrac{n_k}{B}\Bigr\rceil
   \;\overset{B\,\mid\,n_k}{=}\; \frac{E\,n_k}{B}
\qquad\text{(Eq. 4.1)},
$$
$$
u \;:=\; \mathbb{E}[u_k] \;=\; \frac{E}{B}\,\mathbb{E}[n_k]
   \;=\; \frac{nE}{KB}\quad\bigl(\mathbb{E}[n_k]=\tfrac{n}{K}\bigr)
\qquad\text{(Eq. 4.2)}.
$$
The corner $B=\infty,\,E=1$ means one full-batch step, so $u_k=1$ (FedSGD) — not the naive $En_k/\infty=0$.

## Why this matters
$u=nE/(KB)$ is the statistic that orders the rows of Table 2 (e.g. MNIST CNN $n/K=600$: $E{=}1,B{=}50\Rightarrow u{=}12$; $E{=}20,B{=}10\Rightarrow u{=}1200$); it quantifies the compute-for-communication trade, appearing as Eqs. (4.1)-(4.2) of `02-math-deep-dive.md`.

## Code
The aligned runnable demo lives at [`../code/07-local-update-count.py`](../code/07-local-update-count.py). It demonstrates the concept on a finite/low-dimensional example, runnable in <30s on CPU.

**Expected output preview:**
```
Local-update count  u = nE/(KB)   (Eqs. 4.1-4.2 of the math deep dive)
K=100 clients, n=60000 total, balanced n_k=600 examples/client
FedSGD corner: B=inf, E=1  =>  u_k = 1 (one full-batch step)
```

## Try it yourself
- Exercise 1: Add the unbalanced case $B\nmid n_k$ (e.g. $n_k=600, B=64$) and watch the simulated count match $E\lceil n_k/B\rceil$, not the clean $En_k/B$.
- Exercise 2: Reproduce the full Table 2 ordering by sweeping $(E,B)$ and sorting configs by $u$; confirm $(20,10)$ is the heaviest at $u=1200$.

## Further reading
- McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (arXiv:1602.05629), §3 / Table 2.
- `02-math-deep-dive.md` §4 (Eqs. 4.1-4.2).

# Compute-Accounting Pareto
**Level:** `extension`  *(novice | intermediate | advanced | graduate | frontier | paper | extension)*
**Concept ID:** `04-compute-accounting-pareto`
**Graph:** `improvements`
**Prerequisites:** [paper:07-local-update-count](../../paper/concepts/07-local-update-count.md)
**Used by:** downstream nodes

## Plain-English intro
FedAvg reports only *rounds of communication*, justified by the unmeasured premise that on-device computation is "essentially free." But each round each client runs $u_k=En_k/B$ local SGD updates, and total on-device work scales as $R\cdot C\cdot K\cdot u_k$. If you plot rounds **and** total compute together, high-$u$ configs that win on the rounds axis can be *Pareto-dominated* on the compute axis. This node turns rounds-alone reporting into a rounds-vs-compute trade.

## Symbols you'll see
| Symbol | Read aloud as | Meaning |
|--------|---------------|---------|
| $u_k=En_k/B$ | "u-sub-k" | Local SGD updates client $k$ runs per round <!-- TODO add to foundations --> |
| $E$ | "E" | Local epochs per client per round <!-- TODO add to foundations --> |
| $n_k$ | "n-sub-k" | Number of examples on client $k$ <!-- TODO add to foundations --> |
| $B$ | "B" | Local minibatch size ($B=\infty\Rightarrow$ full batch, $u_k=1$) <!-- TODO add to foundations --> |
| $R$ | "R" | Rounds of communication to hit the target <!-- TODO add to foundations --> |
| $C$ | "C" | Fraction of clients selected per round <!-- TODO add to foundations --> |
| $K$ | "K" | Total number of clients <!-- TODO add to foundations --> |

## Formal definition
$$
u_k=\frac{E\,n_k}{B},\qquad
\text{Compute}_{\text{total}}\ \propto\ R\cdot C\cdot K\cdot u_k
\;=\; R\cdot(CK)\cdot\frac{E\,n_k}{B}.
$$
Report the **Pareto frontier** $\{(R,\ \text{Compute}_{\text{total}})\}$ over $(E,B)$ configs, not $R$ alone: a config is dominated if some other config has both fewer rounds and less total compute.

## Why this matters
Tests FedAvg's unquantified "computation is free" premise (Gaps Flagged, 02-math-deep-dive.md; 05-improvements.tex E.2, design): high-$u$ configurations such as $E{=}20,B{=}10$ ($u_k$ up to 1200/round) look best on rounds but may be Pareto-dominated on total compute, revealing a genuine knee.

## Code
The aligned runnable demo lives at [`../code/04-compute-accounting-pareto.py`](../code/04-compute-accounting-pareto.py). It demonstrates the concept on a finite/low-dimensional example, runnable in <30s on CPU.
**Expected output preview:**
```
Compute-Accounting Pareto for FedAvg (toy rounds-to-target, n_k=600, C*K=10)
config            u_k |  rounds R | total compute (R*CK*u_k) | Pareto?
E=1,  B=inf         1 |       40  |               4.00e+02   | on-frontier
```

## Try it yourself
- Exercise 1: Change the hardcoded rounds so a high-$u$ config also wins on compute; watch the frontier shrink to one point.
- Exercise 2: Add a per-round communication cost (one model up+down) and plot a 3-way rounds/compute/comm trade.

## Further reading
- McMahan et al. 2017, "Communication-Efficient Learning of Deep Networks from Decentralized Data" (arXiv:1602.05629), Table 2 and the "computation is free" remark.

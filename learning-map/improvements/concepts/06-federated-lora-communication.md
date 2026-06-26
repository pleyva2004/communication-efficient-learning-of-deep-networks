# Federated LoRA Communication
**Level:** `extension`  *(novice | intermediate | advanced | graduate | frontier | paper | extension)*
**Concept ID:** `06-federated-lora-communication`
**Graph:** `improvements`
**Prerequisites:** [paper:04-fedsgd-gradient-descent](../../paper/concepts/04-fedsgd-gradient-descent.md), [low-rank factorization](https://github.com/pleyva2004/math-foundations/blob/main/concepts/low-rank-factorization.md)
**Used by:** downstream nodes

## Plain-English intro
FedAvg's whole premise is that sending the model each round costs more than the on-device compute. For a big model that hurts: a full layer $W$ is $d\times d$ parameters. LoRA freezes $W$ (shared once) and trains only a thin low-rank adapter $\Delta W = BA$; running FedAvg over *just the adapter* drops per-round communication from $\Theta(d^2)$ to $\Theta(2rd)$, a fraction of $2r/d$.

## Symbols you'll see
| Symbol | Read aloud as | Meaning |
|--------|---------------|---------|
| $W\in\mathbb{R}^{d\times d}$ | "cap-W in R d-by-d" | Frozen base weight matrix (never communicated) <!-- TODO add to foundations --> |
| $\Delta W$ | "delta-W" | Trained low-rank update added to $W$ <!-- TODO add to foundations --> |
| $A\in\mathbb{R}^{r\times d}$ | "cap-A" | Down-projection factor of the adapter <!-- TODO add to foundations --> |
| $B\in\mathbb{R}^{d\times r}$ | "cap-B" | Up-projection factor of the adapter <!-- TODO add to foundations --> |
| $r$ | "r" | Adapter rank, $r\ll d$ <!-- TODO add to foundations --> |
| $d$ | "d" | Layer width / parameter dimension <!-- TODO add to foundations --> |
| $w$ | "w" | The FedAvg model — here only the adapter $(A,B)$ <!-- TODO add to foundations --> |

## Formal definition
$$
W\in\mathbb{R}^{d\times d}\ \text{frozen},\qquad
\Delta W = BA,\quad B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times d},\ \operatorname{rank}(\Delta W)\le r,
$$
$$
\text{FedAvg over } w=(A,B):\qquad
\underbrace{\lvert A\rvert+\lvert B\rvert = 2rd}_{\text{per-round comms}}\ \;\text{vs.}\;\ \underbrace{\lvert W\rvert = d^2}_{\text{full model}},
\qquad \text{ratio}=\frac{2rd}{d^2}=\frac{2r}{d}.
$$

## Why this matters
Makes FedAvg's compute-for-communication trade dramatic for large models: per-round comms become $\Theta(2rd)\ll\Theta(d^2)$, the modern realization of the communication-reduction program of konecny2016 (05-improvements.tex T.2, design).

## Code
The aligned runnable demo lives at [`../code/06-federated-lora-communication.py`](../code/06-federated-lora-communication.py). It demonstrates the concept on a finite/low-dimensional example, runnable in <30s on CPU.

**Expected output preview:**
```
Federated LoRA communication (05-improvements.tex T.2)
Freeze base W in R^{d x d}; FedAvg only adapter Delta W = B A,
  A in R^{r x d}, B in R^{d x r}  ->  per-round comms Theta(2rd) vs Theta(d^2).
```

## Try it yourself
- Exercise 1: Add a non-square base $W\in\mathbb{R}^{d_\text{out}\times d_\text{in}}$ and rederive the ratio (now $r(d_\text{in}+d_\text{out})/(d_\text{in}d_\text{out})$).
- Exercise 2: Pick a target budget (say 1% comms) and solve $2r/d$ for the largest admissible rank $r$ at each $d$ in the table.

## Further reading
- 05-improvements.tex, section T.2; sandbox `real_fedlora.py` (FedAvg over LoRA adapters, ~4.3% of params).
- Konecny et al., *Federated Learning: Strategies for Improving Communication Efficiency*, arXiv:1610.05492.

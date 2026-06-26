# Adaptive Local-Work Schedule
**Level:** `extension`  *(novice | intermediate | advanced | graduate | frontier | paper | extension)*
**Concept ID:** `03-adaptive-local-work-schedule`
**Graph:** `improvements`
**Prerequisites:** [paper:07-local-update-count](../../paper/concepts/07-local-update-count.md), [paper:08-heterogeneity-gap-covariance](../../paper/concepts/08-heterogeneity-gap-covariance.md)
**Used by:** downstream nodes

## Plain-English intro
FedAvg fixes the local epochs $E$ for the whole run. But a large $E$ lets each client over-optimize its own (non-IID) data and drift out of the shared basin, so progress plateaus. The fix: treat $E$ like a learning rate — start large for fast early progress, then step-decay it toward FedSGD-like single steps ($E\to1$) so late-round clients stop overshooting.

## Symbols you'll see
| Symbol | Read aloud as | Meaning |
|--------|---------------|---------|
| $E_t$ | "E-sub-t" | Local epochs used in round $t$ (now scheduled, not fixed) <!-- TODO add to foundations --> |
| $E_{\min},E_{\max}$ | "E-min, E-max" | Floor / starting value of the schedule <!-- TODO add to foundations --> |
| $\tau$ | "tau" | Halving period: $E_t$ halves every $\tau$ rounds <!-- TODO add to foundations --> |
| $t$ | "t" | Communication-round index <!-- TODO add to foundations --> |
| $u_k=En_k/B$ | "u-sub-k" | Local SGD updates client $k$ runs per round <!-- TODO add to foundations --> |
| $E$ | "E" | Local epochs per client per round <!-- TODO add to foundations --> |
| $B$ | "B" | Local minibatch size ($B=\infty$ ⇒ full batch) <!-- TODO add to foundations --> |

## Formal definition
$$ E_t \;=\; \max\!\Big(E_{\min},\ \big\lfloor E_{\max}\,2^{-\lfloor t/\tau\rfloor}\big\rfloor\Big),\qquad t=0,1,\dots,R-1. $$
Large $E$ early; geometric step-decay toward FedSGD-like steps ($E_t\to E_{\min}=1$) late. The per-round local work $u_k=E_t n_k/B$ inherits the same decay.

## Why this matters
Dodges the large-$E$ plateau that paper node 08's heterogeneity drift $\Delta=\eta^2\mathrm{Cov}_k(H_k,g_k)$ predicts (02-math-deep-dive.md §3, Eq. 3.5) — the failure McMahan et al. flag (Fig. 3) but never fix. Implements 05-improvements.tex E.1 with no extra client state or communication.

## Code
The aligned runnable demo lives at [`../code/03-adaptive-local-work-schedule.py`](../code/03-adaptive-local-work-schedule.py). It demonstrates the concept on a finite/low-dimensional example, runnable in <30s on CPU.

**Expected output preview:**
```
E_t schedule  E_t = max(1, floor(64 * 2^-floor(t/12))):
  t=0:E=64 t=12:E=32 t=24:E=16 t=36:E=8 t=48:E=4
Final test accuracy over R=60 rounds, extreme non-IID FedAvg:
```

## Try it yourself
- Exercise 1: Set `SEP` larger (well-separated blobs). The drift vanishes and fixed-large $E$ stops plateauing — confirming the plateau is a heterogeneity effect, not a generic one.
- Exercise 2: Replace the $2^{-\lfloor t/\tau\rfloor}$ decay with a linear or cosine schedule on $E_t$; compare final accuracy against the geometric one.

## Further reading
- McMahan et al. 2017, *Communication-Efficient Learning of Deep Networks from Decentralized Data* (arXiv:1602.05629), §3 and Fig. 3.

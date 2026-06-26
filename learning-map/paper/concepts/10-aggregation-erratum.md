# Aggregation Erratum: normalize by m_t
**Level:** `paper`
**Concept ID:** `10-aggregation-erratum`
**Graph:** `paper`
**Prerequisites:** [09-client-sampling-unbiasedness](09-client-sampling-unbiasedness.md)
**Used by:** downstream nodes

## Plain-English intro
FedAvg's server combines the selected clients' models into a weighted average. The published erratum (footnote 4) fixes the normalizer: divide each client's weight $n_k$ by $m_t$ (the example count *over the selected set*), not by the global $n$. Using $n$ makes the weights sum to only $m_t/n<1$, so the aggregate is shrunk toward the origin every round.

## Symbols you'll see
| Symbol | Read aloud as | Meaning |
|--------|---------------|---------|
| $w^{k}$ | "w-super-k" | Client $k$'s locally updated model <!-- TODO add to foundations --> |
| $w_{t+1}$ | "w-sub-t-plus-one" | Server's aggregated model after round $t$ <!-- TODO add to foundations --> |
| $S_t$ | "S-sub-t" | Set of clients selected in round $t$ <!-- TODO add to foundations --> |
| $n_k$ | "n-sub-k" | Number of examples on client $k$ <!-- TODO add to foundations --> |
| $n$ | "n" | Total number of training examples <!-- TODO add to foundations --> |
| $m_t=\sum_{k\in S_t}n_k$ | "m-sub-t" | Total examples on the selected clients <!-- TODO add to foundations --> |
| $C$ | "C" | Fraction of clients selected per round ($m/K$) <!-- TODO add to foundations --> |

## Formal definition
$$
\textbf{Correct (Eq. 6.1):}\quad w_{t+1}=\sum_{k\in S_t}\frac{n_k}{m_t}\,w^{k},\quad m_t=\sum_{k\in S_t}n_k,\qquad \sum_{k\in S_t}\frac{n_k}{m_t}=\frac{m_t}{m_t}=1.
$$
$$
\textbf{Buggy (Eq. 6.4):}\quad w_{t+1}^{\text{wrong}}=\sum_{k\in S_t}\frac{n_k}{n}\,w^{k}=\frac{m_t}{n}\,w_{t+1},\quad \sum_{k\in S_t}\frac{n_k}{n}=\frac{m_t}{n}<1,\qquad \mathbb{E}_{S_t}\!\big[w_{t+1}^{\text{wrong}}\big]=\frac{m}{K}\,\bar v=C\,\bar v\ \ (\text{Eq. 6.5}).
$$

## Why this matters
This is the published erratum (footnote 4), formalized as Eqs. (6.1)-(6.5) of `02-math-deep-dive.md` §6: the buggy $n_k/n$ rule multiplies the aggregate by $m_t/n\approx C$ every round, so the model decays geometrically like $C^t$.

## Code
The aligned runnable demo lives at [`../code/10-aggregation-erratum.py`](../code/10-aggregation-erratum.py). It demonstrates the concept on a finite/low-dimensional example, runnable in <30s on CPU.

**Expected output preview:**
```
FedAvg aggregation erratum: normalize by m_t, not n  (deep dive Eqs. 6.1-6.5)
K=100  C=0.10  m=|S_t|=10  n_per=50  n=5000  m_t=sum n_k over S=500

CORRECT  weights n_k/m_t  sum = 1.000000   (= 1, a true convex combination)
```

## Try it yourself
- Exercise 1: Make the clients unbalanced (vary `n_per` per client). Confirm the correct weights still sum to 1, while the shrink factor $m_t/n$ now fluctuates around $C$ across re-samples of $S_t$.
- Exercise 2: Iterate the buggy rule for 20 rounds (feed `w_buggy` back as the next start) and plot $\|w_t\|$; verify the $\sim C^t$ geometric collapse.

## Further reading
- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017 (arXiv:1602.05629), Algorithm 1 and footnote 4.
- `02-math-deep-dive.md` §6 (Eqs. 6.1-6.5).

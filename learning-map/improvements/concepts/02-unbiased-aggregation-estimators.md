# Unbiased Aggregation Estimators
**Level:** `extension`  *(novice | intermediate | advanced | graduate | frontier | paper | extension)*
**Concept ID:** `02-unbiased-aggregation-estimators`
**Graph:** `improvements`
**Prerequisites:** [paper:10-aggregation-erratum](../../paper/concepts/10-aggregation-erratum.md), [Horvitz-Thompson estimator](https://github.com/pleyva2004/math-foundations/blob/main/concepts/horvitz-thompson-estimator.md)
**Used by:** downstream nodes

## Plain-English intro
FedAvg's corrected server step averages the selected clients' models weighted by their data counts and normalized by the total examples *in the drawn set*. Because that normalizer is itself random and tends to be large exactly when a big client is sampled, the average systematically over-weights large clients — it is a Hajek *ratio* estimator, biased under client imbalance. Swapping the random normalizer for the fixed inclusion probability (Horvitz-Thompson) restores exact unbiasedness for any imbalance.

## Symbols you'll see
| Symbol | Read aloud as | Meaning |
|--------|---------------|---------|
| $v_k$ | "v-sub-k" | Client $k$'s returned model $w^k_{t+1}$ <!-- TODO add to foundations --> |
| $n_k$ | "n-sub-k" | Number of examples on client $k$ <!-- TODO add to foundations --> |
| $n=\sum_k n_k$ | "n" | Total examples over all $K$ clients <!-- TODO add to foundations --> |
| $S$ | "S" | Random set of selected clients, $\lvert S\rvert=m$ <!-- TODO add to foundations --> |
| $m_t=\sum_{k\in S} n_k$ | "m-sub-t" | Random normalizer: examples on the selected clients <!-- TODO add to foundations --> |
| $m,\,K$ | "m, K" | Clients sampled per round; total clients <!-- TODO add to foundations --> |
| $p_k=m/K$ | "p-sub-k" | Uniform inclusion probability of client $k$ <!-- TODO add to foundations --> |
| $\bar v$ | "v-bar" | Target: full size-weighted mean $\sum_k\frac{n_k}{n}v_k$ <!-- TODO add to foundations --> |

## Formal definition
$$
\bar v=\sum_{k=1}^{K}\tfrac{n_k}{n}\,v_k,\qquad
\underbrace{\sum_{k\in S}\tfrac{n_k}{m_t}\,v_k}_{\text{self-normalized (Hajek ratio, }O(1/m)\text{ bias)}}\!,\qquad
\underbrace{g_{\mathrm{HT}}(S)=\tfrac{K}{m}\sum_{k\in S}\tfrac{n_k}{n}\,v_k}_{\mathbb{E}_S[g_{\mathrm{HT}}]=\bar v\ \text{(exactly unbiased)}}.
$$
With $p_k=m/K$, inverse-probability weighting gives $\mathbb{E}_S[g_{\mathrm{HT}}]=\sum_k p_k\frac{1}{p_k}\frac{n_k}{n}v_k=\bar v$. Size-proportional sampling ($p_k\propto n_k$) makes the plain mean over $S$ unbiased too.

## Why this matters
Fixes the $O(1/m)$ bias of the erratum-corrected average (paper node 10) under client imbalance; appears as M.2, Eqs. (a)–(b) of 05-improvements.tex and the §5–§6 ratio-vs-IPW distinction in 02-math-deep-dive.md.

## Code
The aligned runnable demo lives at [`../code/02-unbiased-aggregation-estimators.py`](../code/02-unbiased-aggregation-estimators.py). It demonstrates the concept on a finite/low-dimensional example, runnable in <30s on CPU.

**Expected output preview:**
```
Unbiased aggregation under client imbalance (FedAvg, 05-improvements.tex M.2)
K=100 clients, m=C*K=10, dim=8, 200,000 Monte-Carlo draws
Target vbar = sum_k (n_k/n) v_k; we report bias ||E[agg]-vbar||
```

## Try it yourself
- Exercise 1: Reduce the size spread in `make_population` toward balanced ($n_k\equiv$ const) and watch the self-normalized bias collapse to the Monte-Carlo floor.
- Exercise 2: Add the size-proportional sampling fix ($p_k\propto n_k$, then plain mean) and confirm it is unbiased too.

## Further reading
- McMahan et al. 2017, *Communication-Efficient Learning of Deep Networks*, arXiv:1602.05629 (Algorithm 1, footnotes 2 & 4).
- 05-improvements.tex, section M.2; 02-math-deep-dive.md, §5–§6.

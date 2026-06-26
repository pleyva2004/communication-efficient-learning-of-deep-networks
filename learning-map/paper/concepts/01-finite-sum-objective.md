# Finite-Sum Objective
**Level:** `paper`  *(novice | intermediate | advanced | graduate | frontier | paper | extension)*
**Concept ID:** `01-finite-sum-objective`
**Graph:** `paper`
**Prerequisites:** [expectation](https://github.com/pleyva2004/math-foundations/blob/main/concepts/expectation.md), [empirical risk](https://github.com/pleyva2004/math-foundations/blob/main/concepts/empirical-risk.md)
**Used by:** downstream nodes

## Plain-English intro
Almost every supervised-learning goal is to make the *average* loss over a fixed training set as small as possible. FedAvg writes this goal as a **finite sum**: one term per training example, divided by the number of examples. This single average is the global objective; every later idea (client splits, local updates, averaging) is just a way of taking it apart and minimizing it across machines.

## Symbols you'll see
| Symbol | Read aloud as | Meaning |
|--------|---------------|---------|
| $w\in\mathbb{R}^d$ | "w in R-d" | Model parameter vector ($d$ parameters) <!-- TODO add to foundations --> |
| $f(w)$ | "f of w" | Global objective: mean loss over all $n$ examples <!-- TODO add to foundations --> |
| $f_i(w)=\ell(x_i,y_i;w)$ | "f-sub-i of w" | Per-example loss on $(x_i,y_i)$ <!-- TODO add to foundations --> |
| $n$ | "n" | Total number of training examples <!-- TODO add to foundations --> |
| $\ell(\cdot)$ | "ell" | Loss function comparing prediction to label <!-- TODO add to foundations --> |

## Formal definition
$$
\min_{w\in\mathbb{R}^d} f(w),\qquad
f(w)\ \stackrel{\text{def}}{=}\ \frac{1}{n}\sum_{i=1}^{n} f_i(w),
\qquad f_i(w)=\ell(x_i,y_i;w). \tag{0.1}
$$
No convexity or smoothness of $\ell$ is assumed; $f$ is in general non-convex.

## Why this matters
This is the global objective FedAvg minimizes; everything else decomposes it — e.g. the partition identity $f(w)=\sum_k \frac{n_k}{n}F_k(w)$ in §1 / Eq. (1.1) of `02-math-deep-dive.md` rests directly on Eq. (0.1).

## Code
The aligned runnable demo lives at [`../code/01-finite-sum-objective.py`](../code/01-finite-sum-objective.py). It demonstrates the concept on a finite/low-dimensional example, runnable in <30s on CPU.

**Expected output preview:**
```
Finite-Sum Objective: f(w) = (1/n) * sum_{i=1}^n f_i(w)
n = 12 toy 1-D regression examples; f_i = squared error.
------------------------------------------------------------
```

## Try it yourself
- Exercise 1: Replace the squared-error $f_i$ with absolute error $|w x_i - y_i|$ and confirm $f$ is still exactly the mean of the $f_i$.
- Exercise 2: Add a second parameter (intercept $b$) so $w\in\mathbb{R}^2$, and check the grid-minimizer recovers both the true slope and intercept.

## Further reading
- McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data," AISTATS 2017, Eq. (1). https://arxiv.org/abs/1602.05629

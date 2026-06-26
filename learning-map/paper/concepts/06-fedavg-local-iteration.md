# FedAvg: Iterated Local Steps
**Level:** `paper`  *(novice | intermediate | advanced | graduate | frontier | paper | extension)*
**Concept ID:** `06-fedavg-local-iteration`
**Graph:** `paper`
**Prerequisites:** [05-gradient-model-averaging-equivalence](05-gradient-model-averaging-equivalence.md)
**Used by:** downstream nodes

## Plain-English intro
FedAvg keeps the shared start $w_t$ but lets each client take $\tau\ge1$ local gradient steps before the server averages the endpoints. At $\tau=1$ this is exactly the one-step equivalence you saw in node 05 (averaging gradients = averaging models). For $\tau\ge2$ the per-client trajectories curve, so averaging their endpoints is no longer equal to running $\tau$ centralized GD steps — and that controlled mismatch is where the communication savings come from.

## Symbols you'll see
| Symbol | Read aloud as | Meaning |
|--------|---------------|---------|
| $w_t$ | "w-sub-t" | Shared model broadcast at round $t$; the common start $w^k_{(0)}=w_t$ <!-- TODO add to foundations --> |
| $w^k_{(s)}$ | "w-sub-paren-s superscript k" | Client $k$'s local iterate after $s$ local steps <!-- TODO add to foundations --> |
| $\tau$ | "tau" | Number of local steps per client per round ($\ge1$) <!-- TODO add to foundations --> |
| $\eta$ | "eta" | Learning rate <!-- TODO add to foundations --> |
| $F_k(w)$ | "cap-F-sub-k of w" | Client $k$'s local objective <!-- TODO add to foundations --> |
| $n_k$ | "n-sub-k" | Number of examples on client $k$ (weight $n_k/n$) <!-- TODO add to foundations --> |
| $H_k$ | "H-sub-k" | Local Hessian $\nabla^2 F_k(w_t)$ <!-- TODO add to foundations --> |
| $g_k$ | "g-sub-k" | Local gradient $\nabla F_k(w_t)$ <!-- TODO add to foundations --> |

## Formal definition
$$
w^{k}_{(s+1)}=w^{k}_{(s)}-\eta\,\nabla F_k\!\bigl(w^{k}_{(s)}\bigr),\quad w^k_{(0)}=w_t,\qquad
w_{t+1}=\sum_{k}\frac{n_k}{n}\,w^{k}_{(\tau)}.\tag{3.3}
$$
For $\tau\ge2$ the composite local map is nonlinear in $w_t$, so $\sum_k\frac{n_k}{n}G_k^{(\tau)}(w_t)\ne G^{(\tau)}(w_t)$ (centralized GD). The leading deviation is
$$
\Delta=\eta^2\Bigl(\textstyle\sum_k\frac{n_k}{n}H_k g_k-H\nabla f\Bigr)+O(\eta^3)=\eta^2\,\mathrm{Cov}_k(H_k,g_k)+O(\eta^3),\quad \Delta\equiv 0\text{ at }\tau=1.\tag{3.5}
$$

## Why this matters
This is the core FedAvg algorithm: Eq. (3.3) of `02-math-deep-dive.md` §3. The extra local steps ($\tau\ge2$) are exactly where communication savings come from, and the drift $\Delta$ of Eq. (3.5) is the object that `05-improvements.tex` M.1 cancels with control variates.

## Code
The aligned runnable demo lives at [`../code/06-fedavg-local-iteration.py`](../code/06-fedavg-local-iteration.py). It demonstrates the concept on a finite/low-dimensional example, runnable in <30s on CPU.

**Expected output preview:**
```
   1  | 5.551115e-17
   2  | 1.107564e-03
   3  | 3.153887e-03
```

## Try it yourself
- Exercise 1: Set both clients' Hessians equal ($A_1=A_2$) and confirm the discrepancy collapses toward 0 for all $\tau$ — the homogeneous/IID limit where $\mathrm{Cov}_k(H_k,g_k)=0$.
- Exercise 2: Shrink $\eta$ by 10x and check the $\tau=2$ gap scales like $\eta^2$, matching Eq. (3.5).

## Further reading
- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017 (arXiv:1602.05629), §3.

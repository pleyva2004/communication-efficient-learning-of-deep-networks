# Math Deep Dive — Communication-Efficient Learning of Deep Networks from Decentralized Data

**Arxiv:** https://arxiv.org/abs/1602.05629 — v4 (26 Jan 2023); AISTATS 2017
**Studied:** 2026-06-25

> Mathematician-grade walk-through of the paper's central technical content. Definitions, derivations, and load-bearing assumptions — no paraphrase. Each result below was independently derived and then adversarially checked against the paper text; the verifier's refinements are folded in.
>
> **Equation numbering** is local to this note (e.g. (3.2) is equation 2 of §3). The *paper* numbers only its objective, Eq. (1) — every other equation here is this note's own reconstruction. Where I supply a step the paper omits ("by standard arguments", a one-line "since…"), I say so; genuine hand-waves are collected in **Gaps Flagged**.

---

## Setup & Notation

The paper targets a generic **finite-sum** objective (its Eq. (1), p. 3):

$$
\min_{w\in\mathbb{R}^d} f(w),\qquad f(w)\ \stackrel{\text{def}}{=}\ \frac{1}{n}\sum_{i=1}^{n} f_i(w). \tag{0.1}
$$

- $w\in\mathbb{R}^d$ — model parameter vector; $d$ the parameter count.
- $n$ — total number of training examples, indexed $i\in[n]:=\{1,\dots,n\}$.
- $f_i(w)=\ell(x_i,y_i;w)$ — loss of model $w$ on labelled example $(x_i,y_i)$. The decompositions below need *nothing* about $\ell$ beyond $f_i:\mathbb{R}^d\to\mathbb{R}$ being well-defined (no convexity/smoothness); differentiability is invoked only where gradients appear. $f$ is in general **non-convex** (p. 3).

**Client partition.** There are $K$ clients; client $k$ holds the index set $P_k\subseteq[n]$ with $n_k:=|P_k|$. The word "partitioned" (p. 3) is load-bearing: $\{P_k\}_{k=1}^K$ is a genuine partition,

$$
\bigcup_{k=1}^{K}P_k=[n]\ \ (\text{coverage}),\qquad P_k\cap P_{k'}=\varnothing\ (k\ne k')\ \ (\text{disjointness}). \tag{P}
$$

A counting consequence of (P) (which the paper leaves implicit in the word "partitioned") is the conservation of examples $n=\sum_{k}n_k$. Each participating client has $n_k\ge1$, so the **local objective**

$$
F_k(w)\ :=\ \frac{1}{n_k}\sum_{i\in P_k}f_i(w) \tag{0.2}
$$

is well-defined. Algorithm hyperparameters: $C$ (fraction of clients selected per round), $E$ (local epochs), $B$ (local minibatch size; $B=\infty$ means "treat the whole local set as one batch"), $\eta$ (learning rate), and the round-$t$ selected set $S_t$ with $m=\max(CK,1)=|S_t|$, $m_t:=\sum_{k\in S_t}n_k$.

---

## Notation key

The full notation reference lives at the [math-foundations glossary](https://github.com/pleyva2004/math-foundations/blob/main/NOTATION.md) (canonical `notation.json`). *That glossary was unreachable from this sandbox (network-blocked), so the table below was built directly from the paper and every row is flagged for upstream propagation on the next foundations sync.*

| Symbol | Read aloud as | Plain-English meaning |
|--------|---------------|------------------------|
| $w\in\mathbb{R}^d$ | "w in R-d" | Model parameter vector ($d$ parameters) <!-- TODO add to foundations --> |
| $f(w)$ | "f of w" | Global objective: mean loss over all $n$ examples <!-- TODO add to foundations --> |
| $f_i(w)=\ell(x_i,y_i;w)$ | "f-sub-i of w" | Per-example loss on $(x_i,y_i)$ <!-- TODO add to foundations --> |
| $n$ | "n" | Total number of training examples <!-- TODO add to foundations --> |
| $K$ | "K" | Total number of clients <!-- TODO add to foundations --> |
| $P_k$ | "P-sub-k" | Index set of examples held by client $k$ <!-- TODO add to foundations --> |
| $n_k=\lvert P_k\rvert$ | "n-sub-k" | Number of examples on client $k$ <!-- TODO add to foundations --> |
| $F_k(w)$ | "cap-F-sub-k of w" | Client $k$'s local objective (mean loss on $P_k$) <!-- TODO add to foundations --> |
| $g_k=\nabla F_k(w_t)$ | "g-sub-k" | Client $k$'s local average gradient at $w_t$ <!-- TODO add to foundations --> |
| $C$ | "C" | Fraction of clients selected per round <!-- TODO add to foundations --> |
| $E$ | "E" | Local epochs per client per round <!-- TODO add to foundations --> |
| $B$ | "B" | Local minibatch size ($B=\infty$ ⇒ full batch) <!-- TODO add to foundations --> |
| $S_t,\ m=\lvert S_t\rvert$ | "S-sub-t" | Set / count of clients selected in round $t$ <!-- TODO add to foundations --> |
| $m_t=\sum_{k\in S_t}n_k$ | "m-sub-t" | Total examples on the selected clients <!-- TODO add to foundations --> |
| $u_k=En_k/B$ | "u-sub-k" | Local SGD updates client $k$ runs per round <!-- TODO add to foundations --> |
| $\eta$ | "eta" | Learning rate <!-- TODO add to foundations --> |
| $H_k=\nabla^2 F_k(w_t)$ | "H-sub-k" | Client $k$'s local Hessian at $w_t$ <!-- TODO add to foundations --> |

---

## §1. The finite-sum decomposition and the IID / non-IID dichotomy

**Claim (partition identity).** Under (P),

$$
f(w)\ =\ \sum_{k=1}^{K}\frac{n_k}{n}\,F_k(w)\qquad\text{for all }w\in\mathbb{R}^d. \tag{1.1}
$$

**Derivation.** Substitute (0.2) into the RHS; the factor $\frac{n_k}{n}\cdot\frac{1}{n_k}=\frac1n$ collapses exactly (this is *why* $F_k$ carries weight $n_k/n$ — the global mixture weight undoes the local normalization):

$$
\sum_{k=1}^{K}\frac{n_k}{n}F_k(w)=\frac1n\sum_{k=1}^{K}\sum_{i\in P_k}f_i(w)\overset{(P)}{=}\frac1n\sum_{i=1}^{n}f_i(w)\overset{(0.1)}{=}f(w).
$$

The middle equality is pure regrouping: by disjointness no $f_i$ is double-counted, by coverage none is omitted, so $\sum_k\sum_{i\in P_k}=\sum_{i=1}^n$ is a re-indexing of the same $n$ numbers. **(1.1) is an exact algebraic identity, pointwise in $w$ — no sampling, no expectation, no assumption on *how* the data is split.** The weights $\alpha_k:=n_k/n$ are a probability vector ($\alpha_k\ge0$, $\sum_k\alpha_k=1$), so $f$ is literally a **convex combination** of the $F_k$. This holds for an arbitrary, even adversarial, partition; it is the structural backbone of everything below. $\blacksquare$

**The IID condition.** The paper now treats the partition as *random* and asserts $\mathbb{E}_{P_k}[F_k(w)]=f(w)$ "if the partition was formed by distributing examples uniformly at random." Made precise (the paper does not pin the mechanism; both standard models give the same mean):

- *Model A — i.i.d. draws.* Form $P_k$ by drawing $m=n_k$ indices i.i.d. uniform on $[n]$. Then $F_k(w)=\frac1m\sum_{j}f_{I_j}(w)$ and, by linearity + identical distribution,
$$
\mathbb{E}_{P_k}[F_k(w)]=\mathbb{E}[f_{I_1}(w)]=\sum_{i=1}^n\frac1n f_i(w)=f(w). \tag{1.2}
$$
- *Model B — uniform $m$-subset (without replacement).* By exchangeability $\Pr[i\in P_k]=m/n$ for every $i$ (since $\sum_i \mathbf 1\{i\in P_k\}=m$ deterministically). Writing $F_k(w)=\frac1m\sum_i\mathbf 1\{i\in P_k\}f_i(w)$,
$$
\mathbb{E}_{P_k}[F_k(w)]=\frac1m\sum_{i=1}^n\frac{m}{n}f_i(w)=f(w).
$$
The two models differ only in the **variance** of $F_k$, not the mean: under IID, each $F_k$ is an *unbiased* proxy for the global $f$.

**The non-IID setting** is defined by negation: any partition for which (1.2) fails — "$F_k$ could be an arbitrarily bad approximation to $f$." Precisely, either $\mathbb{E}_{P_k}[F_k(w)]\ne f(w)$ for some $w$ (biased), or, in the paper's stronger deterministic reading, $\sup_w|F_k(w)-f(w)|$ is unbounded.

*Concrete instance (the paper's pathological MNIST, §3).* Sort the $n=60{,}000$ examples by digit and give each client two shards (mostly two digits). A client $k$ holding only, say, digits $\{3,8\}$ has $F_k(w^\star)\approx0$ for a $w^\star$ that is good on $\{3,8\}$ and bad elsewhere, while $f(w^\star)\gg0$ from the other eight digits — the gap is "arbitrarily bad." **Crucially, identity (1.1) is unaffected**: the correctly size-weighted mixture of *all* clients' $F_k$ still reconstructs $f$ exactly. What breaks under non-IID is per-client statistical fidelity, never the algebra — which is why FedAvg's empirical non-IID robustness is noteworthy rather than guaranteed.

| Object | Status | Holds when |
|---|---|---|
| $n=\sum_k n_k$ | exact identity | $\{P_k\}$ disjoint |
| $f=\sum_k\frac{n_k}{n}F_k$ (1.1) | exact **algebraic** identity, $\forall w$ | $\{P_k\}$ a partition; $n_k\ge1$ |
| $\mathbb{E}_{P_k}[F_k(w)]=f(w)$ (1.2) | **statistical** identity (unbiasedness) | uniform random partition (IID) |
| non-IID | *negation* of (1.2) | any non-uniform/adversarial partition |

---

## §2. FedSGD with $C=1$ is exact distributed full-batch gradient descent

Each client computes its local average gradient at the shared $w_t$ (gradient passes through the finite sum (0.2)):

$$
g_k\ \stackrel{\text{def}}{=}\ \nabla F_k(w_t)=\frac{1}{n_k}\sum_{i\in P_k}\nabla f_i(w_t). \tag{2.1}
$$

**Gradient-aggregation identity.** Differentiating (1.1) ($\nabla$ linear, finite sum, constant weights):

$$
\nabla f(w_t)=\sum_{k=1}^{K}\frac{n_k}{n}\,\nabla F_k(w_t)=\sum_{k=1}^{K}\frac{n_k}{n}\,g_k. \tag{2.2}
$$

This is the identity the paper asserts in a one-line "since…"; (2.2) is exact for **any** partition (no IID needed). The server update with $C=1$ (all clients, $S_t=[K]$) is

$$
w_{t+1}\leftarrow w_t-\eta\sum_{k=1}^{K}\frac{n_k}{n}g_k \overset{(2.2)}{=}\ w_t-\eta\,\nabla f(w_t). \tag{2.3}
$$

So **FedSGD at $C=1$ is not stochastic at all** — it is textbook deterministic full-batch GD on $f$, the clients merely evaluating disjoint partial sums of one global gradient. This is exactly "$C$ controls the global batch size, with $C=1$ corresponding to full-batch (non-stochastic) gradient descent." For $C<1$ the sampled aggregate is instead an *unbiased estimator* of $\nabla f$ (see §5), not an equality.

*Honest note:* passing $\nabla$ through the sum needs each $f_i$ differentiable; the paper's ReLU nets are differentiable only a.e., a (minor) hand-wave the paper does not flag.

---

## §3. Gradient-averaging $\equiv$ model-averaging — and how iterating breaks it

The conceptual hinge of the whole method. Two schemes:

**Scheme A (average gradients, then step):** $w_{t+1}\leftarrow w_t-\eta\sum_k\frac{n_k}{n}g_k$ — Eq. (2.3).

**Scheme B (step locally, then average models):**
$$
\text{(local)}\quad w^{k}_{t+1}\leftarrow w_t-\eta\,g_k,\qquad\text{(server)}\quad w_{t+1}\leftarrow\sum_{k=1}^{K}\frac{n_k}{n}w^{k}_{t+1}. \tag{3.1}
$$

**Exact equivalence (one step).** Substitute the local step into the average and distribute:

$$
w_{t+1}=\sum_{k}\frac{n_k}{n}\bigl(w_t-\eta g_k\bigr)=\underbrace{\Bigl(\sum_k\tfrac{n_k}{n}\Bigr)}_{=1}w_t-\eta\sum_k\frac{n_k}{n}g_k=w_t-\eta\sum_k\frac{n_k}{n}g_k. \tag{3.2}
$$

This is character-for-character Scheme A, as identical maps $w_t\mapsto w_{t+1}$, for *every* $w_t$, partition, and $\eta$. The crucial enabling fact is **term (I)**: a weighted average of *identical* starting points $w_t$ returns $w_t$ exactly because $\sum_k n_k/n=1$. Had clients started from *different* points the equivalence would already fail here — this is precisely where "shared initialization" enters the algebra. No convexity, no expectation; pure linearity of $w\mapsto w-\eta(\cdot)$.

**FedAvg generalizes by iterating.** Run $\tau\ge1$ local steps from the common start $w^k_{(0)}=w_t$:

$$
w^{k}_{(s+1)}=w^{k}_{(s)}-\eta\,\nabla F_k\bigl(w^{k}_{(s)}\bigr),\qquad w_{t+1}=\sum_{k}\frac{n_k}{n}w^{k}_{(\tau)}. \tag{3.3}
$$

(Algorithm 1 is the minibatch realization with $\tau=u_k=En_k/B$; $B=\infty,E=1\Rightarrow\tau=1$ recovers FedSGD.) For $\tau\ge2$ the composite local map $G_k^{(\tau)}=L_k\circ\cdots\circ L_k$, $L_k(w)=w-\eta\nabla F_k(w)$, is **nonlinear** in $w_t$ because later gradients are evaluated at points that themselves depend on $w_t$. Writing the $\tau=2$ endpoint explicitly,

$$
w^{k}_{(2)}=w_t-\eta\nabla F_k(w_t)-\eta\nabla F_k\bigl(w_t-\eta\nabla F_k(w_t)\bigr), \tag{3.4}
$$

averaging no longer commutes with the trajectory: $\sum_k\frac{n_k}{n}G_k^{(\tau)}(w_t)\ne G^{(\tau)}(w_t)$ (centralized GD) for $\tau\ge2$.

**Quantifying the gap (the free lunch, made precise).** Taylor-expand the inner gradient in (3.4) with $H_k=\nabla^2F_k(w_t)$: $\nabla F_k(w_t-\eta g_k)=g_k-\eta H_k g_k+O(\eta^2)$, so $w^k_{(2)}=w_t-2\eta g_k+\eta^2H_kg_k+O(\eta^3)$. Averaging and comparing to two genuine centralized steps $G^{(2)}(w_t)=w_t-2\eta\nabla f+\eta^2 H\nabla f+O(\eta^3)$ (with $H=\sum_k\frac{n_k}{n}H_k$), the $O(1)$ and $O(\eta)$ terms cancel and the leading discrepancy is

$$
\Delta=\eta^2\Bigl(\sum_k\tfrac{n_k}{n}H_kg_k-H\nabla f\Bigr)+O(\eta^3)=\eta^2\,\mathrm{Cov}_k\!\bigl(H_k,g_k\bigr)+O(\eta^3), \tag{3.5}
$$

the client-weighted **covariance between local Hessians and local gradients**. Three consequences:
1. $\Delta\equiv0$ at $\tau=1$ — the $O(\eta^2)$ term *is* the price/payoff of the second local step.
2. $\Delta=0$ in the homogeneous limit ($H_k\equiv H$, $g_k\equiv\nabla f$): the gap is driven entirely by **client heterogeneity** — the formal counterpart of the paper's empirical finding that non-IID data degrades per-round progress.
3. Communication is unchanged (one model up/down per client per round) while local work grows with $\tau$. *"Use additional computation to decrease rounds of communication"* is exactly the decision to **operate in the regime $\Delta\ne0$**: the nonlinear local trajectory travels further toward each $F_k$'s minimizer than one averaged gradient step would.

As $\tau\to\infty$ (convex $F_k$) each client reaches its own $w_k^\star=\arg\min F_k$, independent of $w_t$, and the round degenerates to **one-shot averaging** $\sum_{k\in S_t}\frac{n_k}{m_t}w_k^\star$ — worst-case "no better than a single client." FedAvg lives in the productive middle: $\tau=1$ (no gap, slow) ↔ $\tau\to\infty$ (maximal gap, can break).

*(The $\Delta$ formula and the composite-map non-commutativity are this note's reconstruction; the paper argues the point only qualitatively + empirically.)*

> **Remark 3.1 (the heuristic $\Delta$ is a theorem — with a corrected prefactor).** The two-step Taylor sketch above is *heuristic*: it freezes the local learning rate at $\eta$ and stops at $\tau=2$. An effective-ODE / averaging-theory argument makes it rigorous. Idealize client $k$'s $u_k$ local full-batch steps as the gradient flow $\Phi_k^T$ of $F_k$ run for physical time $T=\eta u_k$; then the FedAvg server map $\mathcal S^T=\sum_k\beta_k\Phi_k^T$ drifts from the centralized flow $\Phi_f^T$ by exactly $\|\mathcal S^T(w)-\Phi_f^T(w)\|=\tfrac{T^2}{2}\|\mathrm{Cov}_k(H_k,g_k)\|+O(T^3)$, with leading direction $\mathrm{Cov}_k(H_k,g_k)/\|\cdot\|$. This recovers (3.5)'s object $\mathrm{Cov}_k(H_k,g_k)$ exactly (the $E{=}2,B{=}\infty$, i.e. $T{=}2\eta$, special case) but **corrects its prefactor**: the right magnitude is $T^2/2=(\eta u_k)^2/2$, not the frozen $\eta^2$, so the heuristic underpredicts by a factor $u_k^2/4$ (e.g. $\approx 6\times$ at $E{=}5$). The single invariant is $T=\eta E n_k/B$. See [`05-improvements.tex` §T.3](05-improvements.tex) and the full proof in [`proofs/effective-ode-averaging-bound.tex`](proofs/effective-ode-averaging-bound.tex) (numerically confirmed: drift $\propto T^{2.06}$, direction cosine $0.9999$).

---

## §4. Local-update count $u_k=En_k/B$ and the $(C,E,B)$ family

`ClientUpdate` runs, per round, $E$ epochs over $\lceil n_k/B\rceil$ batches, each batch = one SGD step:

$$
u_k=E\Bigl\lceil\frac{n_k}{B}\Bigr\rceil\ \overset{B\,\mid\,n_k}{=}\ \frac{En_k}{B}. \tag{4.1}
$$

The paper writes the clean form $u_k=En_k/B$ (dropping the ceiling); it is exact when $B\mid n_k$, which holds in every reported MNIST run ($n_k=600$, $B\in\{10,50,600{=}\infty\}$). **Expected updates per round**, over a uniformly random client ($\Pr[k=j]=1/K$, so $\mathbb{E}[n_k]=\frac1K\sum_j n_j=n/K$ — needs no IID assumption, only the counts):

$$
u:=\mathbb{E}[u_k]=\frac{E}{B}\,\mathbb{E}[n_k]=\frac{nE}{KB}. \tag{4.2}
$$

This is the statistic ordering the rows of Table 2; e.g. MNIST CNN ($n/K=600$): $E{=}1,B{=}50\Rightarrow u{=}12$; $E{=}5,B{=}10\Rightarrow u{=}300$; $E{=}20,B{=}10\Rightarrow u{=}1200$ — all matching the paper.

**The $B=\infty,E=1$ corner is FedSGD.** Here "$B=\infty$" means one batch ($|\mathcal B_k|=1$), so $u_k=1$ (not $0$ — the literal substitution $En_k/\infty$ is *not* the intended reading; the paper's $u=1$ FedSGD rows confirm this). The single step is $w^k_{t+1}=w_t-\eta\nabla F_k(w_t)$ since the full-batch loss equals $F_k$; aggregating (with $C=1$, $m_t=n$) gives $w_{t+1}=w_t-\eta\nabla f(w_t)$ — exactly §2. Thus **FedSGD is the $(C,1,\infty)$ endpoint** of the family; moving off it (↓$B$, ↑$E$) raises $u$ above $1$, which is the entire compute-for-communication mechanism.

---

## §5. Unbiasedness of the FedSGD batch gradient under client sampling (footnote 2)

With $C<1$, the *only* randomness is which clients are drawn ($g_k=\nabla F_k(w)$ is deterministic given $w$). Under the natural assumption:

> **(A1)** $S$ is a uniform $m$-subset of $[K]$ (sampling without replacement), so every client has equal marginal inclusion probability $p_k:=\Pr[k\in S]=m/K$.

(Internal corroboration: the paper's own $u=nE/(KB)$ presumes $\mathbb{E}[n_k]=n/K$, i.e. uniform marginals.) The target is the fixed-weight sum $\sum_{k=1}^K\frac{n_k}{n}g_k=\nabla f(w)$. The **Horvitz–Thompson / inverse-probability** estimator weights each sampled term by $1/p_k$:

$$
g(S):=\sum_{k\in S}\frac{1}{p_k}\frac{n_k}{n}g_k=\sum_{k=1}^{K}\mathbf 1\{k\in S\}\frac{1}{p_k}\frac{n_k}{n}g_k,\qquad
\mathbb{E}_S[g]=\sum_{k=1}^{K}p_k\cdot\frac{1}{p_k}\frac{n_k}{n}g_k=\nabla f(w). \tag{5.1}
$$

The $p_k$ cancels exactly. Substituting $p_k=m/K$:

$$
g(S)=\frac{K}{m}\sum_{k\in S}\frac{n_k}{n}g_k=\frac1m\sum_{k\in S}\frac{Kn_k}{n}g_k. \tag{5.2}
$$

Two weights compose, with distinct jobs: **$n_k/n$** is *intrinsic to the objective* (it makes the target $\nabla f$, not the uniform-over-clients $\frac1K\sum_k F_k$); **$K/m=1/p_k$** is the *partial-participation correction*. **No IID assumption is used** — unbiasedness is purely combinatorial via the partition identity; non-IID affects only $\mathrm{Var}(g)$, never $\mathbb{E}(g)$.

**The subtlety the paper hand-waves.** Algorithm 1's *server aggregation* uses the **self-normalized** weights $n_k/m_t$ ($m_t=\sum_{j\in S}n_j$ random) — a Hájek **ratio estimator**, $\mathbb{E}_S[\cdot]\ne\nabla f$ exactly for unbalanced $n_k$ (it is only consistent, $O(1/m)$ bias). It coincides with the unbiased form **only in the balanced case** $n_k\equiv n/K$, where both reduce to the plain mean $\frac1m\sum_{k\in S}g_k$ — and the paper's MNIST/CIFAR experiments are all balanced. So footnote 2's unbiasedness claim is cleanest exactly in the regime the experiments use. The paper never writes the partial-participation estimator, the inclusion probability, or this gradient-vs-self-normalized-average distinction.

---

## §6. The aggregation erratum: normalize by $m_t$, not $n$

Algorithm 1 (corrected, footnote 4) aggregates over the **selected** set with normalizer $m_t$:

$$
w_{t+1}\leftarrow\sum_{k\in S_t}\frac{n_k}{m_t}\,w^{k}_{t+1},\qquad m_t=\sum_{k\in S_t}n_k. \tag{6.1}
$$

The buggy earlier version summed $\frac{n_k}{n}$ over $S_t$. The error is purely the denominator. Why $m_t$ is right: the coefficients $n_k/m_t$ sum to $1$ **for every realization** of $S_t$,

$$
\sum_{k\in S_t}\frac{n_k}{m_t}=\frac{m_t}{m_t}=1, \tag{6.2}
$$

so (6.1) is always a genuine convex combination of the available models (it lies in their convex hull and keeps the *relative* weights $n_j:n_k$ identical to the full target $\bar v=\sum_{k}\frac{n_k}{n}v_k$). In the balanced case $n_k\equiv n/K$ it is the plain mean over $S_t$, and since $\Pr[k\in S_t]=m/K$,

$$
\mathbb{E}_{S_t}\Bigl[\tfrac1m\sum_{k\in S_t}v_k\Bigr]=\tfrac1m\sum_{k}\tfrac{m}{K}v_k=\tfrac1K\sum_k v_k=\bar v, \tag{6.3}
$$

i.e. **exactly unbiased**. (For unbalanced $n_k$, (6.1) is the same Hájek ratio estimator as §5: consistent, $O(1/m)$ bias — the paper provides no expectation analysis at all.)

**Why the wrong rule is catastrophic.** With denominator $n$ but a sum only over $S_t$, the weights total $\sum_{k\in S_t}\frac{n_k}{n}=m_t/n<1$. Hence, on the same realization,

$$
w_{t+1}^{\text{wrong}}=\sum_{k\in S_t}\frac{n_k}{n}v_k=\frac{m_t}{n}\,w_{t+1}, \tag{6.4}
$$

the correct aggregate **shrunk toward the origin by $m_t/n$**. In expectation (balanced) the bias factor is exactly the participation fraction $m/K=C$:

$$
\mathbb{E}_{S_t}\bigl[w_{t+1}^{\text{wrong}}\bigr]=\frac{m}{K}\,\bar v=C\,\bar v. \tag{6.5}
$$

At the paper's default $C=0.1$ the buggy aggregate is only **10%** of the intended magnitude *each round*. Since $w_{t+1}$ feeds back as the next round's shared start, the server model decays like $\sim C^t$ — geometric collapse toward $0$. The correct rule (6.1), being a true convex combination, has no such shrinkage. *(The dynamical decay argument is a mechanism sketch; the paper reports only that the formula was wrong, not the observed failure. $m/K=C$ exactly requires $CK\in\mathbb{Z}_{\ge1}$, true for $C=0.1,K=100\Rightarrow m=10$.)*

---

## §7. When parameter-space averaging helps vs hurts

Write the aggregate as a convex combination $\bar w=\sum_k\beta_k w^k$, $\beta_k=n_k/m_t\ge0$, $\sum_k\beta_k=1$.

**(a) Convex case — Jensen makes averaging safe.** $f=\sum_k\alpha_k F_k$ is a nonnegative combination of convex functions, hence convex; the finite Jensen inequality gives

$$
f\Bigl(\sum_k\beta_k w^k\Bigr)\le\sum_k\beta_k f(w^k)\le\max_k f(w^k). \tag{7.1}
$$

The averaged model is never worse than the (weighted) mean parent loss — at least as good as the worst parent, strictly better when the $w^k$ scatter around a common minimizer. If $f$ is $\mu$-strongly convex, the two-client midpoint gains explicitly: $f(\tfrac{w+w'}2)\le\tfrac12 f(w)+\tfrac12 f(w')-\tfrac\mu8\|w-w'\|^2$. Structurally: convex sublevel sets are connected, so the segment between two low-loss points stays low-loss. *(The paper invokes convexity only as an idealized limit; (7.1) is standard but not written there.)*

**(b) Non-convex, independent inits — the average is worse than either parent (Fig. 1, left).** Two MNIST 2NNs trained on disjoint 600-example IID subsets from *different* seeds: the interpolation $\theta\mapsto f(\theta w+(1-\theta)w')$ shows a loss **barrier** — the midpoint exceeds both endpoints. Jensen fails (no convexity), and the mechanism is **permutation symmetry**: a feed-forward net's function is invariant under permuting hidden units,
$$
W_2\,\sigma(W_1x+b_1)=(W_2P^\top)\,\sigma(PW_1x+Pb_1),
$$
so every minimum is replicated across an orbit of size $\prod_\ell |S_{h_\ell}|$ (for the 2NN, $(200!)^2$ — astronomical). Independent seeds break this symmetry independently, landing in incompatible orbit elements; the naive average then blends *unrelated* hidden units (feature $i$ of $w$ with feature $i$ of $w'$), producing a degraded near-random net. This is the "arbitrarily bad model" warning. *(The permutation-symmetry account is the standard explanation, supplied here; the paper only shows Fig. 1 and cites Goodfellow et al. for the technique.)*

**(c) Shared init — the average beats both parents (Fig. 1, right), and why FedAvg engineers it.** Same protocol but a *shared* seed $w_0$: now the midpoint achieves *lower* full-set loss than either parent, with no barrier. Under the over-parameterized "well-behaved basin" hypothesis (Dauphin et al.: critical points are mostly saddles; Goodfellow et al.: barrier-free init→solution paths; Choromanska et al.: minima concentrate at similar loss), shared init fixes the *same* orbit element once, so both runs stay in a **common basin** where $f$ is locally near-convex and (7.1) applies *locally*. The strict gain is an **ensembling / variance-reduction** effect: each parent over-fit its own subset with idiosyncratic error directions, and averaging within a shared basin cancels the uncorrelated overfit while keeping the shared signal (the paper's "regularization benefit similar to dropout" conjecture). **FedAvg's design is exactly this:** broadcasting the same $w_t$ each round (the per-round analogue of the shared seed) anchors every client's update at a common point, with drift $\|w^k_{t+1}-w_t\|$ bounded by $E,B,\eta$, keeping the averaged parents mutually compatible — the right-panel regime, by construction. *(Asserted as intuition; no bound that the basin stays common across $t$ or large $E$.)*

**(d) The $E\to\infty$ limit and Fig. 3.** Since $w_t$ enters `ClientUpdate` *only* as initialization, in the convex case $\lim_{E\to\infty}\text{ClientUpdate}(k,w_t)=w_k^\star$ independent of $w_t$, so the round degenerates to the memoryless one-shot average $\sum_{k\in S_t}\frac{n_k}{m_t}w_k^\star$ and *further rounds cannot help*. The conjectured non-convex analogue ("same basin ⇒ same local min, large $E$ harmless") is **refuted empirically** by Fig. 3: on the Shakespeare LSTM, large $E$ lets clients over-optimize their non-IID $F_k$, leave the common basin, and re-enter the incompatible-basin regime (b) — FedAvg "can plateau or diverge." Hence the prescription to *decay* local computation (smaller $E$ / larger $B$) like a learning rate. The failure is model-specific: the MNIST CNN shows no degradation at large $E$, the word-LSTM does slightly better at $E{=}1$ than $E{=}5$.

---

## Alternative Formulations

- **FedSGD as one endpoint; one-shot averaging as the other.** The $(C,E,B)$ family interpolates between large-batch synchronous SGD ($B{=}\infty,E{=}1$; §2/§4) and one-shot averaging of fully-converged local minimizers ($E{\to}\infty$; §3/§7d). FedAvg is the productive interior.
- **Gradient form ⇄ model form.** §3 shows these are the *same map* at one local step and *diverge by $\Delta=\eta^2\mathrm{Cov}_k(H_k,g_k)$* (3.5) thereafter — a clean lens on "what the extra local computation buys."
- **Objective as expectation.** $f=\sum_k\alpha_k F_k=\mathbb{E}_{k\sim\alpha}[F_k]$ with $\alpha_k=n_k/n$; the client-sampling estimator (§5) is then a two-level (clients, then examples) Monte-Carlo estimate of $\nabla\mathbb{E}_{k}[F_k]$.
- **Unbiased gradient vs. self-normalized model average.** The Horvitz–Thompson gradient (5.2) and the Hájek model average (6.1) coincide only when balanced — a distinction worth carrying into any convergence analysis.

---

## Load-Bearing Assumptions

| Assumption | Used in | Failure mode if violated |
|------------|---------|--------------------------|
| $\{P_k\}$ a genuine partition (P); $n_k\ge1$ | (1.1), §2–§6 | Decomposition over/under-counts; $\nabla f\ne\sum_k\frac{n_k}{n}g_k$; weights mis-normalized |
| Each $f_i$ differentiable at the iterates | (2.1),(2.2),(3.x) | Gradient cannot pass through the sum; FedSGD≠GD identity loses meaning (ReLU kinks: a.e. only) |
| **Shared per-round initialization** $w^k_{(0)}=w_t$ | (3.2), §7c | One-step equivalence fails at term (I); averaging crosses loss barriers (Fig. 1 left) |
| Common-basin / over-parameterized well-behaved landscape | §7c, §7d | Independent-basin regime ⇒ averaging produces an arbitrarily bad model |
| Moderate $E$ (clients stay in the shared basin) | §3, §7d | Plateau or divergence (Fig. 3); one-shot-averaging worst case |
| **Uniform client sampling (A1)**, $p_k=m/K$ | §5, §6 | $p_k$ no longer cancels $n_k/n$; estimator (5.2) biased; (6.3) unbiasedness lost |
| **Balanced** $n_k\equiv n/K$ (for *exact* unbiasedness) | §5, §6 | Self-normalized average $n_k/m_t$ becomes a ratio estimator with $O(1/m)$ bias |
| IID ($\mathbb{E}_{P_k}[F_k]=f$) | only for *quality*, not correctness | Higher $\mathrm{Var}(g)$ and large heterogeneity gap $\Delta$ (3.5); slower / non-monotone convergence — **never** breaks any identity |
| "Computation $\ll$ communication cost" | the entire compute-for-rounds thesis | If local compute/upload dominates (large models), trading rounds for $u_k$ updates stops paying off |

---

## Gaps Flagged

Steps the paper hand-waves or that cannot be reconstructed from what is given:

- **No convergence theory whatsoever.** For $E\ge2$ the paper offers no fixed-point, rate, or non-divergence result. The §3.5 covariance gap and the §7 basin story are *this note's* characterizations, not the paper's; the paper argues qualitatively + empirically only.
- **Sampling distribution unspecified.** "$S_t\leftarrow$ random set of $m$ clients" never fixes with/without replacement or whether $n_k$-weighted. All of §5/§6 assume uniform-without-replacement to get $p_k=m/K$.
- **Partial-participation FedSGD estimator never written.** Footnote 2 asserts $\mathbb{E}[g]=\nabla f$ without the estimator, $p_k$, or the inverse-probability factor; (5.1)–(5.2) supply them.
- **Unbiased gradient vs. self-normalized model average conflated.** The paper does not distinguish the exactly-unbiased $n_k/(np_k)$ gradient from Algorithm 1's $n_k/m_t$ ratio average (biased for unbalanced $n_k$).
- **Erratum given without analysis.** Footnote 4 only states the fix; the bias factor $C$ (6.5) and the geometric-decay mechanism are reconstructions. No record of the actually-observed failure.
- **$u_k=En_k/B$ drops the ceiling.** The $B\nmid n_k$ case (a trailing partial batch) is never addressed; harmless in the paper's $n_k=600$ runs.
- **Differentiability of ReLU nets** assumed silently for every gradient step.
- **Figure 1 numeric loss values** are visual reads of axis ranges, not text-stated; only the qualitative claims (left: above both; right: below both) are textually confirmed.
- **"Computation is essentially free"** asserted, never measured — no wall-clock/energy for $u_k$ up to 1200 on-device updates.

---

## Connections

- **The decomposition $f=\sum_k\frac{n_k}{n}F_k$** is the same finite-sum structure underlying SVRG/SAGA-style variance reduction and any distributed-SGD analysis; what is special here is that the $F_k$ are *fixed and heterogeneous*, not IID minibatches.
- **The heterogeneity gap $\Delta=\eta^2\mathrm{Cov}_k(H_k,g_k)$ (3.5)** prefigures the *client-drift* term that later FL theory (e.g. SCAFFOLD's control variates, FedProx's proximal term) is explicitly designed to cancel — both attack exactly the $\mathrm{Cov}_k$ object.
- **Permutation symmetry / linear mode connectivity (§7b–c)** is the same phenomenon studied in "Git Re-Basin" and the linear-mode-connectivity literature; FedAvg's shared-init trick is the cheapest possible way to stay on the connected side.
- **Horvitz–Thompson / Hájek estimators (§5–§6)** tie federated aggregation to classical survey sampling — the unbiased-vs-ratio distinction is the same one that governs importance-weighted estimators throughout ML.
- **Privacy framing** (data minimization + data-processing-inequality bound) is the seed that grows into the secure-aggregation (Bonawitz et al.) and DP-FedAvg (Abadi et al.; McMahan et al. follow-ups) lines — none of which deliver a *guarantee* in this paper.

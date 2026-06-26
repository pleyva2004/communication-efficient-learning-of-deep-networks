# Interview Prep — Communication-Efficient Learning of Deep Networks from Decentralized Data

**Authors:** H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agüera y Arcas (Google)
**Arxiv:** https://arxiv.org/abs/1602.05629 — v4 (26 Jan 2023); published AISTATS 2017
**Studied:** 2026-06-25

---

## One-sentence elevator
The paper that named **Federated Learning** and introduced **FederatedAveraging (FedAvg)** — run several local SGD epochs on each device, then average the *models* (not the gradients) on a coordinating server — cutting required communication rounds **10–100×** versus synchronized SGD while the training data never leaves the device.

## What's novel
- **The problem and framing, not the optimizer.** The authors are candid that iterative model averaging predates them (McDonald et al. for perceptrons; Povey et al. for speech DNNs). The real contributions are (i) *naming and formalizing* "federated optimization" via four properties — **non-IID, unbalanced, massively-distributed, limited-communication** — which jointly violate *every* standing assumption of the convex distributed-optimization literature ($K\!\ll\!n_k$, IID partition, equal $n_k$); and (ii) an evaluation methodology built for that regime.
- **One knob dominates: local computation.** FedAvg exposes $(C,E,B)$ = client fraction, local epochs, local minibatch size, and shows the speedups come almost entirely from doing *more local work per round* ($E\!\uparrow$, $B\!\downarrow$), trading "essentially free" on-device compute for the scarce resource — rounds of communication.

## What's mathematically clever
- **Gradient-averaging $\equiv$ model-averaging, for exactly one step.** The identity $\sum_k\frac{n_k}{n}g_k=\nabla f(w_t)$ makes FedSGD ($B\!=\!\infty,E\!=\!1$) *literally* distributed full-batch gradient descent; averaging one local step equals one global step purely because $\sum_k n_k/n=1$. FedAvg is just "iterate the local step *before* averaging" — and the instant you take $\ge 2$ steps the equivalence breaks. *That breakage is the free lunch:* to leading order the per-round gap is $\eta^2\,\mathrm{Cov}_k(H_k,g_k)$ — a pure client-**heterogeneity** term that vanishes in the IID/quadratic limit.
- **Shared initialization is load-bearing.** Figure 1: averaging two nets from *independent* seeds is worse than either parent (permutation symmetry $\Rightarrow$ incompatible basins); from a *shared* seed the average beats both. FedAvg re-broadcasts $w_t$ every round, manufacturing the favorable regime by construction.

## What I'd push back on
- **No theory at all.** Zero convergence results, no rate, not even a non-divergence guarantee — the entire justification is one 1-D loss-interpolation plot plus borrowed loss-landscape folklore. They concede averaging "could produce an arbitrarily bad model" and that FedAvg "can plateau or diverge" for large $E$.
- **The evaluation flatters the method.** Per-$x$-axis-point learning-rate grid search (11–13 rates, ">2000 models"), monotonized learning curves, linear-interpolated fractional round counts, and **no error bars anywhere**. The "FedSGD baseline" is one full-batch step per round — the weakest possible per-round comparator, so a large per-round speedup is half-definitional.
- **"Computation is essentially free" is asserted, never measured** — no wall-clock or energy for $u_k$ up to **1200** local updates on a phone. And the flagship 95× is the *unbalanced* Shakespeare case the authors admit is *easier*, not the pathological-non-IID robustness story; several non-IID cells are actually **slower** than FedSGD (0.5×).

## Open questions
- What fixed point (if any) does FedAvg approach for $E>1$ under genuine non-IID data, and is there a principled rule for choosing $(E,B)$ — given Table 2 shows more local work sometimes *hurts*?
- The aggregation erratum ($n_k/m_t$ vs $n_k/n$): the corrected self-normalized average is only *exactly* unbiased when clients are **balanced**; for unbalanced $n_k$ it's a Hájek ratio estimator with $O(1/m)$ bias. How much does this matter for the unbalanced settings the paper actually motivates?
- Privacy is framed as "data minimization," but a footnote concedes gradients can leak exact tokens — what does it take to turn the qualitative story into a guarantee (DP, secure aggregation)?

## My proposed extensions
- **Cancel the heterogeneity drift directly.** The $E>1$ "free lunch" is exactly $\eta^2\mathrm{Cov}_k(H_k,g_k)$ — a client-Hessian/gradient covariance. Add SCAFFOLD-style control variates ($g_{\text{local}}-c_k+c$) to cancel it; prototype shows **1.62× fewer non-IID rounds**, no extra backward pass. *(This is the one I'd lead with — it operationalizes the paper's own weakest spot and connects straight to SCAFFOLD/FedProx.)*
- **Fix the aggregation bias.** The erratum's $n_k/m_t$ average is only *exactly* unbiased when balanced; under imbalance it's a Hájek ratio estimator. Horvitz–Thompson reweighting or size-proportional sampling restores exactness — prototype cuts bias **1.6e-1 → 2e-3**.
- **Schedule local work like a learning rate.** Decay $E$ (40→1) over rounds to dodge the large-$E$ plateau the paper flags but never fixes — prototype beats *every* fixed $E$, zero extra communication. *(Bonus: permutation-align client models pre-averaging — Git Re-Basin — to drop the shared-init requirement; flips the Fig-1 barrier +0.024 → −0.023.)*
- **Make the drift $\Delta$ a theorem (effective-ODE bound).** Idealizing each client's $u_k$ local steps as gradient flow for time $T=\eta u_k$ turns the heuristic $\Delta$ into the *exact* leading drift $\frac{T^2}{2}\lVert\mathrm{Cov}_k(H_k,g_k)\rVert+O(T^3)$ — same object, but it *corrects the prefactor* ($T^2/2$, not the frozen $\eta^2$; vs the raw $\eta^2$ the drift grows by $u_k^2/2$, ~12.5× at $E{=}5$, the factor 2 at $E{=}2$ being the 2-step Euler error); numerics confirm $T^{2.06}$ scaling, direction cosine 0.9999.
- **Horizon-equalized local work.** Per-client $\eta_k=T^\star/u_k$ holds the flow-time $T_k=\eta E n_k/B$ equal across clients, cancelling the first-order size-imbalance drift $-\mathrm{Cov}_k(T_k,g_k)$ at zero extra comms/state — measured *fewer* non-IID rounds as imbalance grows (margin widens to +2.17 rounds), a clean no-op when balanced.

*(These last two — the continuous-time / dynamics-lens pair — are the proposals that additionally survived independent adversarial re-verification.)*

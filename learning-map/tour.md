# Tour — Communication-Efficient Learning of Deep Networks from Decentralized Data

> A self-contained walking tour of this study. Read top-to-bottom for a complete first pass; jump by section if you already know the foundations. Estimated total time: **90 minutes**.

**Paper:** [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) — McMahan et al., 2017
**Repo:** [`pleyva2004/communication-efficient-learning-of-deep-networks`](https://github.com/pleyva2004/communication-efficient-learning-of-deep-networks)
**Foundations graph:** [`pleyva2004/math-foundations`](https://github.com/pleyva2004/math-foundations)

---

## 1. Reader's contract

By the end of this tour you will be able to:

- Derive why FedAvg cuts communication rounds and where non-IID data breaks it
- Reproduce the headline tradeoff and the Figure-1 barrier from runnable witnesses
- Map each proposed extension to the exact gap it closes (drift, bias, plateau, shared-init)

What this tour assumes:

- Comfort with multivariable calculus / gradients (refresh: https://github.com/pleyva2004/math-foundations/blob/main/concepts/gradient.md)
- Comfort with basic probability / expectation (refresh: https://github.com/pleyva2004/math-foundations/blob/main/concepts/expectation.md)

What this tour does **not** cover:

- differential-privacy / secure-aggregation guarantees (only framed, not delivered)
- convergence theory for E>1 (the paper proves none)

If anything in §3 lands as undefined, jump to §2 first.

---

## 2. Foundations walk

Each row is one foundation node. Click through to the canonical concept page in `math-foundations`. Local copies of the same notation live under [`learning-map/notation.json`](./notation.json).

| Concept | Why it matters here | Link |
|---|---|---|
| expectation | defines the IID identity E[F_k]=f and all unbiasedness arguments | [expectation](https://github.com/pleyva2004/math-foundations/blob/main/concepts/expectation.md) |
| empirical risk | the finite-sum objective f is the empirical risk being minimized | [empirical-risk](https://github.com/pleyva2004/math-foundations/blob/main/concepts/empirical-risk.md) |
| partition of a set | the client decomposition f=sum_k (n_k/n)F_k is exact because {P_k} partitions the data | [partition-of-a-set](https://github.com/pleyva2004/math-foundations/blob/main/concepts/partition-of-a-set.md) |
| unbiased estimator | IID means each F_k is an unbiased proxy for f; Horvitz-Thompson restores it under sampling | [unbiased-estimator](https://github.com/pleyva2004/math-foundations/blob/main/concepts/unbiased-estimator.md) |
| gradient | FedSGD aggregates per-client gradients into grad f | [gradient](https://github.com/pleyva2004/math-foundations/blob/main/concepts/gradient.md) |
| gradient descent | FedSGD with C=1 is exactly full-batch GD | [gradient-descent](https://github.com/pleyva2004/math-foundations/blob/main/concepts/gradient-descent.md) |
| taylor theorem | the E>1 heterogeneity gap is a second-order Taylor remainder | [taylor-theorem](https://github.com/pleyva2004/math-foundations/blob/main/concepts/taylor-theorem.md) |
| hessian | the gap is a covariance of local Hessians H_k with gradients | [hessian](https://github.com/pleyva2004/math-foundations/blob/main/concepts/hessian.md) |
| covariance | the drift term is exactly Cov_k(H_k,g_k) | [covariance](https://github.com/pleyva2004/math-foundations/blob/main/concepts/covariance.md) |
| convex function | Jensen safety of averaging needs convex F_k | [convex-function](https://github.com/pleyva2004/math-foundations/blob/main/concepts/convex-function.md) |
| jensens inequality | gives f(avg) <= avg f for convex objectives | [jensens-inequality](https://github.com/pleyva2004/math-foundations/blob/main/concepts/jensens-inequality.md) |
| permutation group | hidden-unit relabelings are the symmetry behind the Fig-1 barrier | [permutation-group](https://github.com/pleyva2004/math-foundations/blob/main/concepts/permutation-group.md) |
| control variates | the SCAFFOLD fix subtracts a control variate to cancel drift | [control-variates](https://github.com/pleyva2004/math-foundations/blob/main/concepts/control-variates.md) |
| horvitz thompson estimator | the unbiased aggregation fix is inverse-probability weighting | [horvitz-thompson-estimator](https://github.com/pleyva2004/math-foundations/blob/main/concepts/horvitz-thompson-estimator.md) |
| linear assignment problem | permutation alignment solves an assignment problem | [linear-assignment-problem](https://github.com/pleyva2004/math-foundations/blob/main/concepts/linear-assignment-problem.md) |
| low rank factorization | federated LoRA communicates a low-rank adapter only | [low-rank-factorization](https://github.com/pleyva2004/math-foundations/blob/main/concepts/low-rank-factorization.md) |

> Foundations live in their own repo so they can be reused across every paper study. If a row 404s, the foundation hasn't been authored yet — open an issue at [`pleyva2004/math-foundations`](https://github.com/pleyva2004/math-foundations/issues).

---

## 3. Paper concepts walk

The paper graph in this repo. Read in topological order; each row links to a `.md` (prose + formal definition) and a `.py` (runnable witness, CPU, <30s).

| # | Concept | Role in the paper | Files |
|---|---|---|---|
| 01 | Finite-Sum Objective | the objective everything minimizes | [md](paper/concepts/01-finite-sum-objective.md) · [py](paper/code/01-finite-sum-objective.py) |
| 02 | Client-Partition Decomposition | rewrites the objective per client (exact) | [md](paper/concepts/02-client-partition-decomposition.md) · [py](paper/code/02-client-partition-decomposition.py) |
| 03 | IID vs Non-IID | defines the regime FedAvg must survive | [md](paper/concepts/03-iid-noniid-dichotomy.md) · [py](paper/code/03-iid-noniid-dichotomy.py) |
| 04 | FedSGD as Exact Gradient Descent | the FedSGD baseline = exact GD | [md](paper/concepts/04-fedsgd-gradient-descent.md) · [py](paper/code/04-fedsgd-gradient-descent.py) |
| 05 | Gradient- vs Model-Averaging Equivalence | why model-averaging = gradient-averaging (1 step) | [md](paper/concepts/05-gradient-model-averaging-equivalence.md) · [py](paper/code/05-gradient-model-averaging-equivalence.py) |
| 06 | FedAvg: Iterated Local Steps | the FedAvg algorithm itself | [md](paper/concepts/06-fedavg-local-iteration.md) · [py](paper/code/06-fedavg-local-iteration.py) |
| 07 | Local-Update Count u=nE/(KB) | the (C,E,B) compute-for-communication knob | [md](paper/concepts/07-local-update-count.md) · [py](paper/code/07-local-update-count.py) |
| 08 | Heterogeneity Gap = eta^2 Cov_k(H_k,g_k) | the leading E>1 deviation (client drift) | [md](paper/concepts/08-heterogeneity-gap-covariance.md) · [py](paper/code/08-heterogeneity-gap-covariance.py) |
| 09 | Client-Sampling Unbiasedness (Horvitz-Thompson) | justifies partial participation C<1 | [md](paper/concepts/09-client-sampling-unbiasedness.md) · [py](paper/code/09-client-sampling-unbiasedness.py) |
| 10 | Aggregation Erratum: normalize by m_t | the published aggregation erratum | [md](paper/concepts/10-aggregation-erratum.md) · [py](paper/code/10-aggregation-erratum.py) |
| 11 | Parameter Averaging & Jensen | why averaging is safe when convex | [md](paper/concepts/11-parameter-averaging-jensen.md) · [py](paper/code/11-parameter-averaging-jensen.py) |
| 12 | Shared-Init & the Permutation Barrier | why shared init is load-bearing | [md](paper/concepts/12-shared-init-permutation-barrier.md) · [py](paper/code/12-shared-init-permutation-barrier.py) |

**How the pieces fit:** Start from the finite-sum objective (01) and its exact client decomposition (02); the IID/non-IID dichotomy (03) says how faithful each client's slice is. FedSGD (04) is exact GD, and the one-step gradient/model-averaging equivalence (05) is the hinge FedAvg (06) generalizes by iterating local steps — quantified by the update count (07). The two deep facts are the heterogeneity gap (08), which is exactly why non-IID hurts, and the sampling/erratum pair (09,10), which fix the aggregation. Finally (11,12) explain when averaging parameters is safe and why a shared per-round initialization is non-negotiable.

For the full mermaid DAG see [`learning-map/paper/README.md`](./paper/README.md).

---

## 4. Improvements walk

Each proposal in [`05-improvements.tex`](../05-improvements.tex) gets exactly one validation file:

- **PROOF mode** — a standalone LaTeX proof under [`proofs/`](../proofs/) with theorem statement, complete proof, and discussion of where it would break.
- **MEASUREMENT mode** — an extended Python prototype under [`improvements/`](../improvements/) exposing `measure() -> dict` that prints a quantitative comparison table; deterministic under a fixed seed.

- **Control-Variate Drift Correction** — *PROOF* — `proofs/heterogeneity-control-variate.tex`  
  SCAFFOLD-style local step g_local - c_k + c cancels the eta^2 Cov_k(H_k,g_k) drift; prototype: 1.62x fewer non-IID rounds
- **Unbiased Aggregation Estimators** — *PROOF* — `proofs/unbiased-aggregation.tex`  
  Horvitz-Thompson / size-proportional sampling restore exact unbiasedness lost by the self-normalized n_k/m_t average under imbalance
- **Adaptive Local-Work Schedule** — *MEASUREMENT* — `improvements/adaptive-local-work.py → measure()`  
  Decay E (or grow B) over rounds like a learning rate to dodge the large-E plateau; prototype beats every fixed E
- **Compute-Accounting Pareto (design)** — *link-only* — `design (E.2); see 05-improvements.tex`  
  Measure the unquantified 'computation is free' premise: report a rounds-vs-total-on-device-compute Pareto frontier. Design proposal (E.2); link-only.
- **Permutation-Aligned Averaging (Git Re-Basin for FL)** — *PROOF* — `proofs/permutation-aligned-averaging.tex`  
  Align client models modulo permutation before averaging to relax the shared-init requirement; prototype flips barrier +0.024 -> -0.023
- **Federated LoRA (design)** — *link-only* — `design (T.2); see 05-improvements.tex`  
  Run FedAvg over LoRA adapters only: per-round communication drops from Theta(B) to Theta(rank*dim) (4.3% in the sandbox). Design proposal (T.2); link-only.

> Heuristic for mode selection: math/theoretical proposals → PROOF (override to MEASUREMENT for tighter empirical bounds); code/experimental proposals → MEASUREMENT (override to PROOF for complexity-class statements). Pure citations of existing literature → "link only".

---

## 5. What to do next

You've finished the tour. Pick one:

1. **Reproduce the sandbox claim.** `cd sandbox && pip install -r requirements.txt && python experiment.py`. Expected output and falsification criteria live in [`sandbox/README.md`](../sandbox/README.md).
2. **Run a measurement.** Pick any improvement in §4 marked MEASUREMENT, run `python improvements/<slug>.py`, eyeball the table, then change one knob and re-run. Note what moved.
3. **Verify a proof.** Pick any improvement in §4 marked PROOF, open `proofs/<slug>.tex` (or its rendered `.pdf`), and audit the proof. The "Discussion" section names where it would break — try to break it.
4. **Form an opinion.** Fill in [`03-opinions.md`](../03-opinions.md). The skill never fabricates opinions; this row is yours.
5. **Extend the foundations graph.** If §2 sent you to a missing concept, draft it at [`pleyva2004/math-foundations`](https://github.com/pleyva2004/math-foundations) — that's how the foundations grow.

---

*Generated by the [`study-paper`](https://github.com/pleyva2004/claude-skill-study-paper) Claude Code skill, Stage 7.*

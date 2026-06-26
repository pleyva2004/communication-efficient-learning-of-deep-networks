# Quick Tour — paper graph (~30 min)

A fast top-to-bottom read of the 12 paper concepts. Skim each `.md`, run its `.py`.

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

**Fit:** Start from the finite-sum objective (01) and its exact client decomposition (02); the IID/non-IID dichotomy (03) says how faithful each client's slice is. FedSGD (04) is exact GD, and the one-step gradient/model-averaging equivalence (05) is the hinge FedAvg (06) generalizes by iterating local steps — quantified by the update count (07). The two deep facts are the heterogeneity gap (08), which is exactly why non-IID hurts, and the sampling/erratum pair (09,10), which fix the aggregation. Finally (11,12) explain when averaging parameters is safe and why a shared per-round initialization is non-negotiable.

For the full DAG see [`../README.md`](../README.md); for foundation refreshers see the [foundations repo](https://github.com/pleyva2004/math-foundations).

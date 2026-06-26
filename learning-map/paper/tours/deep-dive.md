# Deep Dive — paper graph (~3 hr, with foundation refresh)

Walk every node, run its witness, AND refresh each foundation prerequisite first.

## Foundations to refresh

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

## Paper concepts (run each `.py`, read each `.md`)

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

Then open [`../../02-math-deep-dive.md`](../../02-math-deep-dive.md) and match each §/equation to its concept node. Finish with the [improvements graph](../../improvements/README.md).

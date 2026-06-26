# Paper Graph — FedAvg concepts

Concepts the paper itself defines or uses, in dependency order, derived from [`../../02-math-deep-dive.md`](../../02-math-deep-dive.md). 12 nodes, each with a prose page + runnable witness. arXiv:[1602.05629](https://arxiv.org/abs/1602.05629).

```mermaid
flowchart TD
  P01["01 Finite-Sum Objective"]
  P02["02 Client-Partition Decomposition"]
  P03["03 IID vs Non-IID"]
  P04["04 FedSGD as Exact Gradient Descent"]
  P05["05 Gradient- vs Model-Averaging Equivalence"]
  P06["06 FedAvg: Iterated Local Steps"]
  P07["07 Local-Update Count u=nE/(KB)"]
  P08["08 Heterogeneity Gap = eta^2 Cov_k(H_k,g_k)"]
  P09["09 Client-Sampling Unbiasedness (Horvitz-Thompson)"]
  P10["10 Aggregation Erratum: normalize by m_t"]
  P11["11 Parameter Averaging & Jensen"]
  P12["12 Shared-Init & the Permutation Barrier"]
  F03["partition of a set"]
  F08["hessian"]
  F10["convex function"]
  F11["jensens inequality"]
  F09["covariance"]
  F06["gradient descent"]
  F02["empirical risk"]
  F04["unbiased estimator"]
  F01["expectation"]
  F12["permutation group"]
  F07["taylor theorem"]
  F05["gradient"]
  F01 --> P01
  F02 --> P01
  P01 --> P02
  F03 --> P02
  P02 --> P03
  F04 --> P03
  P02 --> P04
  F05 --> P04
  F06 --> P04
  P04 --> P05
  P05 --> P06
  P06 --> P07
  P06 --> P08
  F07 --> P08
  F08 --> P08
  F09 --> P08
  P03 --> P09
  P04 --> P09
  F04 --> P09
  P09 --> P10
  P02 --> P11
  F10 --> P11
  F11 --> P11
  P11 --> P12
  F12 --> P12
  click P01 "concepts/01-finite-sum-objective.md"
  click P02 "concepts/02-client-partition-decomposition.md"
  click P03 "concepts/03-iid-noniid-dichotomy.md"
  click P04 "concepts/04-fedsgd-gradient-descent.md"
  click P05 "concepts/05-gradient-model-averaging-equivalence.md"
  click P06 "concepts/06-fedavg-local-iteration.md"
  click P07 "concepts/07-local-update-count.md"
  click P08 "concepts/08-heterogeneity-gap-covariance.md"
  click P09 "concepts/09-client-sampling-unbiasedness.md"
  click P10 "concepts/10-aggregation-erratum.md"
  click P11 "concepts/11-parameter-averaging-jensen.md"
  click P12 "concepts/12-shared-init-permutation-barrier.md"
  click F03 "https://github.com/pleyva2004/math-foundations/blob/main/concepts/partition-of-a-set.md"
  click F08 "https://github.com/pleyva2004/math-foundations/blob/main/concepts/hessian.md"
  click F10 "https://github.com/pleyva2004/math-foundations/blob/main/concepts/convex-function.md"
  click F11 "https://github.com/pleyva2004/math-foundations/blob/main/concepts/jensens-inequality.md"
  click F09 "https://github.com/pleyva2004/math-foundations/blob/main/concepts/covariance.md"
  click F06 "https://github.com/pleyva2004/math-foundations/blob/main/concepts/gradient-descent.md"
  click F02 "https://github.com/pleyva2004/math-foundations/blob/main/concepts/empirical-risk.md"
  click F04 "https://github.com/pleyva2004/math-foundations/blob/main/concepts/unbiased-estimator.md"
  click F01 "https://github.com/pleyva2004/math-foundations/blob/main/concepts/expectation.md"
  click F12 "https://github.com/pleyva2004/math-foundations/blob/main/concepts/permutation-group.md"
  click F07 "https://github.com/pleyva2004/math-foundations/blob/main/concepts/taylor-theorem.md"
  click F05 "https://github.com/pleyva2004/math-foundations/blob/main/concepts/gradient.md"
  classDef paper fill:#cce5ff,stroke:#3399ff;
  classDef fnd fill:#e0e0e0,stroke:#888,color:#333;
  class P01,P02,P03,P04,P05,P06,P07,P08,P09,P10,P11,P12 paper;
  class F03,F08,F10,F11,F09,F06,F02,F04,F01,F12,F07,F05 fnd;
```

Grey = foundations (click → shared repo). Each node links to its concept page; the aligned runnable witness is in `code/<NN>-<slug>.py`.

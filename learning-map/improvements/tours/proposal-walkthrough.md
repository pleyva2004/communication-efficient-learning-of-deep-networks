# Proposal Walkthrough — improvements graph

Each proposal, the paper gap it closes, and its validation file.

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

Validation modes: PROOF → standalone LaTeX in [`../../proofs/`](../../proofs/); MEASUREMENT → `measure()` in the matching [`../../improvements/`](../../improvements/) prototype; link-only → design proposal in [`../../05-improvements.tex`](../../05-improvements.tex).

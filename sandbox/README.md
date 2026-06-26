# Sandbox: communication-efficient-learning-of-deep-networks

> Minimal runnable experiments probing claims from **Communication-Efficient Learning of Deep Networks from Decentralized Data** (McMahan et al., AISTATS 2017 — the FedAvg paper). https://arxiv.org/abs/1602.05629

This is a subdirectory of the larger study repo — see [`../README.md`](../README.md) for the full artifact set (interview prep, math deep dive, lit review). Measured numbers live in [`../findings.md`](../findings.md).

This sandbox demonstrates the paper's claims at **two levels**, every script self-contained and deterministic.

---

## The claims being probed

1. **Communication efficiency (Table 2):** doing more *local* computation per round (more local epochs `E`, smaller minibatch `B` → more local updates `u = nE/(KB)`) sharply reduces the number of **communication rounds** to reach a target accuracy; FedAvg needs far fewer rounds than FedSGD (`E=1, B=∞`), and the **pathological non-IID** partition is harder (smaller speedups) than IID.
2. **Why parameter averaging works at all (Figure 1):** averaging two non-convex nets in parameter space helps *only when they share an initialization*. From independent inits the average is **worse** than either parent (a loss barrier); from a shared init the average **beats** both. FedAvg engineers the good regime by re-broadcasting the same `w_t` every round.

## What would falsify the claims

- If higher `u` did **not** reduce rounds-to-accuracy (claim 1), or if FedAvg matched FedSGD round-for-round, the communication-efficiency thesis fails.
- If shared-init averaging showed the **same** loss barrier as independent-init averaging (claim 2), the justification for FedAvg's per-round broadcast collapses.

---

## Level 1: CPU baseline (runs anywhere, pure numpy)

No GPU, fast, deterministic. Verifies the *algorithmic* claims. (Pure `numpy` — no torch needed.)

| Script | Demonstrates | Runtime |
|--------|--------------|---------|
| `toy_fedavg.py` | Claim 1 — rounds-to-accuracy vs `u`, IID vs non-IID, on convex softmax regression (K=100, C=0.1) | ~0.4 s |
| `tiny_fedavg_averaging.py` | Claim 2 — the Figure 1 interpolation barrier (independent) vs valley (shared) on a 1-hidden-layer MLP | ~0.3 s |

```bash
# from the study root:
.venv/bin/python sandbox/toy_fedavg.py
.venv/bin/python sandbox/tiny_fedavg_averaging.py
```

**Measured (this machine, deterministic):**
- `toy_fedavg.py`: FedSGD = 62 rounds (IID); best FedAvg (`E=20,B=10`, `u=120`) = **2 rounds → 31.0× fewer**; non-IID speedups strictly smaller (5.5× at `u=120`); higher `u` ⇒ fewer rounds, monotonically.
- `tiny_fedavg_averaging.py`: **independent-init barrier_height = +0.024** (avg worse than both parents); **shared-init barrier_height = −0.027** (avg beats both). Figure-1 phenomenon reproduced.

## Level 2: Hardware-upsized (`tier_mid_gpu` — Apple M5 Max, 36 GB unified)

Sized to the detected tier. Real torch autograd on **MPS**; the federated-LoRA script uses transformers + peft.

| Script | Demonstrates | Runtime | Needs |
|--------|--------------|---------|-------|
| `torch_fedavg.py` | Claim 1 with the paper's **exact** MNIST CNN (1,663,370 params) on MPS | smoke ~1 s; full `--rounds 30` ~5 s | torch (MPS/CUDA) |
| `real_fedlora.py` | FedAvg applied to **LoRA adapters** of a causal LM — only 4.3% of params communicated | smoke ~8 s | transformers + peft |

```bash
.venv/bin/python sandbox/torch_fedavg.py --smoke      # 2 rounds, IID, FedAvg vs FedSGD
.venv/bin/python sandbox/torch_fedavg.py --rounds 30  # full IID+non-IID rounds-to-target table
.venv/bin/python sandbox/real_fedlora.py --smoke      # offline: tiny from_config LM + LoRA FedAvg
.venv/bin/python sandbox/real_fedlora.py --train      # real Qwen2.5-1.5B base (needs HF download + GPU host)
```

**Measured (this machine):**
- `torch_fedavg.py` (MPS, **1,663,370 params** = paper's CNN exactly): smoke IID FedAvg `0.09 → 1.00` vs FedSGD `0.09 → 0.25` over 2 rounds; full run 3.0× (IID) / 2.0× (non-IID) speedup to the 0.80 target.
- `real_fedlora.py`: LoRA trainable = **16,384 / 377,472 = 4.34%**; avg client loss `5.04 → 4.11` over 15 FedAvg rounds (erratum-corrected `n_k/m_t` weighting; only the adapter is communicated).

---

## Faithful vs. simplified (read before citing these numbers)

- **Faithful:** K=100 clients, C=0.1; IID = shuffle-then-split; **non-IID = sort-by-label / 2K shards / 2-per-client** (the paper's exact pathological MNIST recipe); shared per-round init; **erratum-corrected** aggregation `w_{t+1}=Σ_{k∈S_t}(n_k/m_t)w^k`; FedSGD = `E=1,B=∞`; `torch_fedavg.py` reproduces the CNN's parameter count to the digit.
- **Simplified (so it runs offline in seconds):**
  - **Data is synthetic** (Gaussian-blob classes / structured token streams) — no MNIST/Shakespeare/CIFAR downloads (network-blocked here).
  - `toy_fedavg.py` uses **convex** softmax regression with *ill-conditioned* features (per-feature std spanning `[1, 1/30]`). Convexity isolates the pure compute-for-communication tradeoff; the ill-conditioning is the convex stand-in for the slow flat directions of the paper's deep nets, so FedSGD's single full-batch step/round is visibly slow.
  - `tiny_fedavg_averaging.py` needs the **hidden layer** — logistic regression is convex and shows *no* barrier; one hidden layer adds the permutation symmetry that creates incompatible basins.
  - Absolute round counts / losses are smaller than the paper's (toy tasks converge faster); the **qualitative ordering and the headline ~30× IID gap reproduce**.

## Hardware tier note

`../metadata.json` lists `hardware_tier = "tier_mid_gpu"`. Refresh with:

```bash
python3 ~/.claude/skills/study-paper/templates/detect-hardware.py --force --summary
```

## Notes

Sandbox-grade code: clarity over performance, single-purpose, not production. All scripts soft-import their heavy deps and degrade gracefully (Level-2 scripts print an install message and exit 0 if torch/peft are absent), so they parse and run on any reader's laptop. Level 1 prioritizes inspectability; Level 2 prioritizes fidelity to the paper's regime.

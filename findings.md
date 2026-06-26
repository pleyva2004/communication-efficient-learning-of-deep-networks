# Findings — FedAvg sandbox

Measured numbers from the Stage 4 sandbox, run on the study machine
(**Apple M5 Max, 36 GB unified, `tier_mid_gpu`**, Python 3.9.6 in `.venv`, torch 2.8.0 / MPS).
All scripts are deterministic; reruns reproduce these figures.

> **Reading guide.** The paper's headline is *10–100× fewer communication rounds vs synchronized SGD*. These toys reproduce the **qualitative shape** of that claim at a scale that runs in seconds; absolute round counts are smaller than the paper's because the synthetic tasks converge faster. See `sandbox/README.md` → "Faithful vs. simplified" for exactly which design choices are paper-faithful.

---

## Claim 1 — more local computation ⇒ fewer communication rounds (Table 2)

### Level 1: `toy_fedavg.py` (convex softmax regression, K=100, C=0.1, target 95%)

| Algorithm | E | B | u=nE/(KB) | IID rounds | IID speedup | non-IID rounds | non-IID speedup |
|-----------|---|---|-----------|-----------:|------------:|---------------:|----------------:|
| FedSGD | 1 | ∞ | 1 | 62 | — | 55 | — |
| FedAvg | 5 | ∞ | 5 | 14 | 4.4× | 24 | 2.3× |
| FedAvg | 1 | 10 | 6 | 11 | 5.6× | 17 | 3.2× |
| FedAvg | 20 | ∞ | 20 | 6 | 10.3× | 14 | 3.9× |
| FedAvg | 5 | 50 | 6 | 8 | 7.8× | 14 | 3.9× |
| FedAvg | 5 | 10 | 30 | 4 | 15.5× | 10 | 5.5× |
| **FedAvg** | **20** | **10** | **120** | **2** | **31.0×** | **10** | **5.5×** |

- **(a)** Best FedAvg reaches the target in **2 IID rounds vs FedSGD's 62 → 31.0× fewer communication rounds.**
- **(b)** Higher `u` monotonically reduces IID rounds (u=5→14 rounds … u=120→2 rounds).
- **(c)** Non-IID is harder: every FedAvg config's non-IID speedup is **strictly smaller** than its IID speedup. The FedSGD *baseline* is partition-insensitive (62 IID vs 55 non-IID) — matching the paper's own Table 2, where FedSGD is even slightly faster on non-IID. The "non-IID is harder" effect lives in the FedAvg rows.

### Level 2: `torch_fedavg.py` (paper's exact MNIST CNN, **1,663,370 params**, MPS)

- **Smoke (2 rounds, IID):** FedAvg test acc `0.090 → 0.798 → 1.000` (+0.910) vs FedSGD `0.090 → 0.100 → 0.250` (+0.160) — FedAvg far ahead at equal rounds. (~1 s on MPS.)
- **Full (`--rounds 30`, target 0.80):** FedAvg @ round 3 vs FedSGD @ round 9 → **3.0× (IID)**; FedAvg @ round 9 vs FedSGD @ round 18 → **2.0× (pathological non-IID)**. (~5 s on MPS.)
- Gap is smaller than the paper's 10–100× because the synthetic task is easy (both hit ~1.0); raising `--noise` / lowering `--lr` widens it.

---

## Claim 2 — parameter averaging works only with shared initialization (Figure 1)

### Level 1: `tiny_fedavg_averaging.py` (1-hidden-layer MLP, 676 params/model, two disjoint data halves)

Loss along the interpolation `θ·w + (1−θ)·w'`, evaluated on the full dataset:

| Condition | loss(θ=0) parent w' | loss(θ=0.5) average | loss(θ=1) parent w | **barrier_height** = loss(0.5) − max(loss(0),loss(1)) |
|-----------|--------------------:|--------------------:|-------------------:|------------------------------------------------------:|
| **Independent init** (Fig 1 left) | 0.0964 | 0.1206 | 0.0790 | **+0.0243** → barrier, averaging **hurts** |
| **Shared init** (Fig 1 right) | 0.0754 | 0.0486 | 0.0660 | **−0.0268** → valley, averaging **beats both parents** |

The sign flip is the whole point: independent inits land in incompatible basins (permutation symmetry), so the midpoint sits on a ridge; a shared init keeps both runs in one basin, so the average lands *below* both parents (an ensembling/variance-reduction gain). FedAvg re-broadcasts `w_t` each round to stay permanently in the right-panel regime.

---

## Claim 3 (extension) — FedAvg transfers to LoRA fine-tuning of an LM

### Level 2: `real_fedlora.py` (tiny from-config Llama, offline; K=8, C=0.5, E=4)

- Trainable params = **16,384 / 377,472 = 4.34%** — *only the LoRA adapter is communicated each round*; the frozen 361,088 base params are never sent. (At the documented `--train` scale, the base is Qwen2.5-1.5B and the adapter is a few MB.)
- Avg client loss `5.04 → 4.11` over 15 FedAvg rounds (18.5% reduction), using the erratum-corrected `n_k/m_t` weighting over selected clients.
- **Takeaway:** the same model-averaging algorithm works when the "model" is a low-rank adapter — a natural communication-efficiency amplifier the 2017 paper predates but is structurally compatible with.

---

## Improvements (forward-looking) — validated extensions

Two proposed extensions (the continuous-time / dynamics-lens pair) were lifted into the parent study after surviving independent adversarial re-verification. Full write-ups in `05-improvements.tex` (§T.3, §E.3); validation artifacts under `proofs/` and `improvements/`.

### Node 07 — effective-ODE drift bound (theoretical, §T.3, `proofs/effective-ode-averaging-bound.tex`)

Turns the math deep dive's *heuristic* per-round drift $\Delta=\eta^2\mathrm{Cov}_k(H_k,g_k)$ into a theorem: idealizing each client's $u_k$ local steps as gradient flow for time $T=\eta u_k$, the server-vs-centralized drift is exactly $\|\mathcal S^T(w)-\Phi_f^T(w)\|=\tfrac{T^2}{2}\|\mathrm{Cov}_k(H_k,g_k)\|+O(T^3)$. **Corrects the prefactor** ($T^2/2$, not the frozen $\eta^2$; the heuristic underpredicts by $u_k^2/4$, ~6× at E=5). Single invariant: $T=\eta E n_k/B$.

Drift law verified numerically:
- drift scales as **T^2.06** (log-log slope vs predicted 2),
- ratio D_t/pred → **1.05** as T→0,
- leading-direction cosine vs $\mathrm{Cov}_k(H_k,g_k)$ = **0.9999**,
- drift is E-independent at fixed T (**<5%** gap, E=25 vs E=100).

**VERDICT: PASS** (proof, not a separate prototype).

### Node 08 — horizon-equalized local computation (experimental, §E.3, `improvements/horizon-equalized-local-flow.py::measure`)

Vanilla FedAvg fixes $(E,B,\eta)$ globally, so under client-size imbalance the flow-time $T_k=\eta E n_k/B$ varies, injecting a first-order size-imbalance drift $-\mathrm{Cov}_k(T_k,g_k)$ (the unequal-horizon remark of node 07's bound). Fix: per-client $\eta_k=T^\star/u_k$ with $T^\star=\eta E\bar n/B$, so $T_k\equiv T^\star$ for all clients (total flow-time unchanged on average; only its distribution equalized). No extra comms/state. Both arms share partition + sampling seed and differ *only* in $\eta_k$.

Rounds to 95% test acc, pathological non-IID, log-uniform client sizes (mean over 12 partition realizations):

| spread~ | n_min | n_max | BASE rnds | PROP rnds | mean Δ | W/L/T | faster? |
|--------:|------:|------:|----------:|----------:|-------:|:-----:|:--------|
| 1.0 | 60 | 60 | 9.75 | 9.75 | 0.00 | 0/0/12 | tie (no-op) |
| 4.9 | 23 | 127 | 11.83 | 11.17 | +0.67 | 3/0/9 | PROPOSED |
| 19.1 | 9 | 213 | 15.17 | 14.42 | +0.75 | 5/1/6 | PROPOSED |
| 66.8 | 4 | 320 | 18.33 | 16.17 | +2.17 | 6/0/6 | PROPOSED |

All 3 prediction checks pass: (P1) balanced no-op, (P2) faster on every imbalanced spread, (P3) margin widens monotonically. **VERDICT: PASS.**

---

## Reproduce

```bash
cd ~/ai-research-studies/communication-efficient-learning-of-deep-networks
uv pip install --python .venv/bin/python -r sandbox/requirements.txt   # already done in this study
.venv/bin/python sandbox/toy_fedavg.py
.venv/bin/python sandbox/tiny_fedavg_averaging.py
.venv/bin/python sandbox/torch_fedavg.py --smoke      # add --rounds 30 for the full table
.venv/bin/python sandbox/real_fedlora.py --smoke
```

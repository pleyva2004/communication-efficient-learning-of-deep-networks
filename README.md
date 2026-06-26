# ai-study: FedAvg (McMahan et al., 2017)

[![Render LaTeX](https://github.com/pleyva2004/communication-efficient-learning-of-deep-networks/actions/workflows/render.yml/badge.svg)](https://github.com/pleyva2004/communication-efficient-learning-of-deep-networks/actions/workflows/render.yml)

> Layered study artifacts — interview prep, mathematician-grade deep dive, opinion template, a LaTeX literature-review entry, forward-looking improvement proposals with runnable prototypes, and a three-graph interactive learning map — for [arxiv:1602.05629](https://arxiv.org/abs/1602.05629), the paper that introduced **Federated Learning** and the **FederatedAveraging (FedAvg)** algorithm.

**Compiled PDFs** (auto-built by GitHub Actions on every push to `.tex` / `.bib`): [`pdfs/04-literature-review.pdf`](./pdfs/04-literature-review.pdf) · [`pdfs/05-improvements.pdf`](./pdfs/05-improvements.pdf) · [`pdfs/interview-handout.pdf`](./pdfs/interview-handout.pdf) · [`learning-map/tour.pdf`](./learning-map/) · [`proofs/*.pdf`](./proofs/).

Generated with the `study-paper` Claude Code skill.

---

## The paper

**Title:** Communication-Efficient Learning of Deep Networks from Decentralized Data
**Authors:** H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agüera y Arcas (Google)
**Venue:** AISTATS 2017 (JMLR W&CP vol. 54); arXiv v4, 26 Jan 2023

**Headline claim.** Train a shared model on data that never leaves the device — each client runs several epochs of local SGD, the server averages the resulting *models* — cutting required communication rounds **10–100×** versus synchronized SGD, robustly across non-IID and unbalanced data.

## What's in this repo

| Path | Purpose |
|------|---------|
| [`01-interview-prep.md`](./01-interview-prep.md) | ~630-word opinionated talking-points doc (elevator, novelty, what-to-push-back-on, proposed extensions) |
| [`02-math-deep-dive.md`](./02-math-deep-dive.md) | Mathematician-grade walk-through: 22 numbered equations across 7 derivations, load-bearing assumptions, gaps flagged. Notation key prelude links the [math-foundations glossary](https://github.com/pleyva2004/math-foundations/blob/main/NOTATION.md) |
| [`03-opinions.md`](./03-opinions.md) | Opinion-capture template (filled in by hand, never by AI) |
| [`04-literature-review.tex`](./04-literature-review.tex) + [`references.bib`](./references.bib) | Research-ready LaTeX literature-review entry, standalone-compilable, 18 citations |
| [`05-improvements.tex`](./05-improvements.tex) | **Forward-looking proposals** — control variates, unbiased aggregation, adaptive local work, permutation-aligned averaging, federated LoRA |
| [`improvements/`](./improvements/) | 4 runnable numpy prototypes for the proposals, each printing a baseline-vs-proposed comparison |
| [`proofs/`](./proofs/) | 3 standalone, complete LaTeX proofs for the math/theoretical proposals (CI-rendered) |
| [`sandbox/`](./sandbox/) | Minimal experiments probing the paper's claims: numpy Level-1 + torch/MPS Level-2 (incl. the paper's exact 1,663,370-param CNN) + a federated-LoRA demo |
| [`learning-map/`](./learning-map/) | Three-graph interactive learning map (paper + improvements + foundations links); every concept has a `.md` page, a runnable `.py` witness, a mermaid node, a notebook cell, and a D3 HTML node |
| [`findings.md`](./findings.md) | Measured sandbox numbers per level |
| [`metadata.json`](./metadata.json) | Slug, arxiv ID, authors, completed stages, hardware tier, environment notes |
| [`source.pdf`](./source.pdf) / [`paper.txt`](./paper.txt) | The paper itself + extracted text |

## Navigate the learning map

Start at [`learning-map/README.md`](./learning-map/README.md) — it has a skill-level entry-point table (total beginner → researcher) and the combined mermaid DAG. Three views: mermaid (GitHub browse), [`learning-map/notebook/combined.ipynb`](./learning-map/notebook/combined.ipynb) (runnable cells), and [`learning-map/html/index.html`](./learning-map/html/index.html) (D3 force graph). A guided tour lives in [`learning-map/tour.md`](./learning-map/tour.md) (also `.tex`/`.ipynb`).

## Reproduce the sandbox (uses a local `uv` venv)

```bash
uv venv .venv --python python3
uv pip install --python .venv/bin/python -r sandbox/requirements.txt
.venv/bin/python sandbox/toy_fedavg.py                 # Level 1: 31× IID rounds, numpy, <1s
.venv/bin/python sandbox/tiny_fedavg_averaging.py      # Level 1: the Figure-1 barrier
.venv/bin/python sandbox/torch_fedavg.py --smoke       # Level 2: paper's CNN on MPS/CUDA/CPU
.venv/bin/python sandbox/real_fedlora.py --smoke       # Level 2: FedAvg over LoRA adapters
```

Expected numbers and the faithful-vs-simplified caveats are in [`sandbox/README.md`](./sandbox/README.md) and [`findings.md`](./findings.md).

## Build the LaTeX locally (optional — CI does this automatically)

```bash
# any TeX install; tectonic is the lightest:
tectonic 04-literature-review.tex
tectonic 05-improvements.tex
tectonic interview-handout.tex
```


## Forward-looking extensions — 14 sibling studies

This study spawned **14 standalone mathematical extensions to FedAvg**, each its own repo with the same structure (derivation + runnable prototype + proof-or-measurement + adversarially-verified findings + auto-rendering CI). They were generated across six mathematical lenses, then each claim was **re-run and refuted by an independent skeptic agent**. Result: **2 clean wins, 7 nuanced/partial, 5 honest negative results** — negatives are kept, not hidden.

| | Extension | Lens | Verdict | Result |
|---|---|---|---|---|
| ✅ | [effective-ode-averaging-bound](https://github.com/pleyva2004/fedavg-effective-ode-averaging-bound) | Dynamics | PASS (confirmed) | Closed-form averaging-theory drift bound $D_t=\tfrac{T^2}{2}\lVert\mathrm{Cov}_k(H_k,g_k)\rVert+O(T^3)$, exact to $O(T^2)$ (log-log slope 2.06, direction cosine 0.9999) — a rigorous replacement for the parent study's *heuristic* $\Delta$. |
| ✅ | [horizon-equalized-local-flow](https://github.com/pleyva2004/fedavg-horizon-equalized-local-flow) | Dynamics | PASS (confirmed) | Per-client $\eta_k=T^\star/u_k$ equalizes gradient-flow integration time, cancelling an imbalance drift $-\mathrm{Cov}(T_k,g_k)$; gains grow monotonically with client size-spread (+0→+2.2 rounds), never net-worse. |
| ◐ | [fisher-precision-weighted-aggregation](https://github.com/pleyva2004/fedavg-fisher-precision-weighted-aggregation) | Geometry | MIXED (confirmed) | Per-coordinate Fisher-precision barycenter: **~29% fewer non-IID rounds, strictly faster on all 6/6 configs** (and faster on IID); MIXED only because it *beat* its own 'IID-neutral' sub-claim. |
| ◐ | [quantization-drift-additive-bound](https://github.com/pleyva2004/fedavg-quantization-drift-additive-bound) | Compression | MIXED (confirmed) | Bound proving compression-noise and heterogeneity-drift are **additive/separable** (no cross-term — verified ~1e-4 vs scale 1.23); predicts a harmless bit-rate $b^\star\approx3$, measured threshold exactly $b=3$. |
| ◐ | [debiased-hajek-jackknife](https://github.com/pleyva2004/fedavg-debiased-hajek-jackknife) | Estimation | MIXED (confirmed) | Jackknife-of-ratios cuts the Hájek $O(1/m)$ bias 3.5× and beats Horvitz–Thompson in MSE 5/5 populations — but plain Hájek still wins MSE 5/5 (bias wasn't the dominant term). |
| ◐ | [neyman-optimal-drift-sampling](https://github.com/pleyva2004/fedavg-neyman-optimal-drift-sampling) | Estimation | MIXED (confirmed) | Neyman (size×drift) client sampling cuts estimator MSE 26% below the best baseline on non-IID — but EMA cold-start makes median rounds 5 vs 4. |
| ◐ | [rao-blackwell-stale-aggregation](https://github.com/pleyva2004/fedavg-rao-blackwell-stale-aggregation) | Estimation | MIXED (confirmed) | Memory-bank Rao-Blackwell control variate: frozen-target variance up to **1114× smaller**, bias ~1e-4; round-count ~1.09× IID / ~0.95× non-IID (washes out at high bank staleness). |
| ◐ | [adaptive-diagonal-server-step](https://github.com/pleyva2004/fedavg-adaptive-diagonal-server-step) | Server-opt | MIXED (unconfirmed) | Adam-style server step. Proof $\kappa(D^{-1}A)=\sqrt{\kappa(A)}$ is **valid**; but on the toy the diagonal is inert (ε-dominated) — the 15.5× headline is a step-size confound (~2.25× vs a fairly-tuned scalar baseline). |
| ◐ | [server-lookahead-drift-contractor](https://github.com/pleyva2004/fedavg-server-lookahead-drift-contractor) | Server-opt | MIXED (unconfirmed) | Slow/fast server interpolation damps non-IID oscillation but missed the strict 'fewer rounds **and** ≥50% smaller amplitude' bar. |
| ✗ | [anderson-extrapolation](https://github.com/pleyva2004/fedavg-anderson-extrapolation) | Server-opt | FAIL (confirmed) | Type-II Anderson on the global-model sequence: **3.7× slower** at the FedSGD endpoint — per-round sub-sampling noise destabilizes the sign-unconstrained coefficients (sum-to-one invariant held). |
| ✗ | [kl-bregman-mirror-aggregation](https://github.com/pleyva2004/fedavg-kl-bregman-mirror-aggregation) | Geometry | FAIL (confirmed) | Dual-space (logit) KL/Bregman centroid aggregation: no better than the plain arithmetic average on the convex toy. |
| ✗ | [jl-shared-sketch-upload](https://github.com/pleyva2004/fedavg-jl-shared-sketch-upload) | Compression | FAIL (confirmed) | One shared Johnson–Lindenstrauss projection + pseudo-inverse reconstruction: too noisy at $p=d/4$ to stay within 25% of full-precision rounds. |
| ✗ | [server-sam-aggregation](https://github.com/pleyva2004/fedavg-server-sam-aggregation) | Landscape | FAIL (confirmed) | Server-side sharpness-aware step (ascend-then-average): no faster than vanilla FedAvg on non-IID. |
| ✗ | [weight-noise-averaging](https://github.com/pleyva2004/fedavg-weight-noise-averaging) | Landscape | FAIL (confirmed) | 'Dropout-on-the-mean' Gaussian weight noise scaled by client disagreement: neutral-to-harmful; the hand-waved Sec.7c regularization benefit didn't materialize on the convex toy. |

*Verdict legend:* ✅ PASS · ◐ MIXED (partial / fails a sub-claim) · ✗ FAIL. *confirmed* = the adversarial verifier reproduced the numbers **and** judged the comparison fair; *unconfirmed* = numbers reproduce but the verifier flagged the claim as mis-attributed (e.g. a step-size confound).

## License

- **Paper (`source.pdf`):** © 2017 the authors (AISTATS / JMLR W&CP vol. 54), redistributed here for research/study use under arXiv's non-exclusive distribution license.
- **All other files in this repo:** MIT.

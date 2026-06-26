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

## License

- **Paper (`source.pdf`):** © 2017 the authors (AISTATS / JMLR W&CP vol. 54), redistributed here for research/study use under arXiv's non-exclusive distribution license.
- **All other files in this repo:** MIT.

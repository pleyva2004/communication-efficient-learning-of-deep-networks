# improvements/

> Runnable Python prototypes for the proposals in [`../05-improvements.tex`](../05-improvements.tex).

Each file implements one proposal and prints a **baseline-vs-proposed comparison** with a `PASS` verdict. All are pure-`numpy`, CPU-only, deterministic, and finish in well under a minute.

---

## Prototypes

| File | Implements (§ in 05-improvements.tex) | One-line result |
|------|---------------------------------------|-----------------|
| `heterogeneity-control-variate.py` | §M.1 (Mathematical) | SCAFFOLD-lite control variates → **1.62× fewer** non-IID rounds than vanilla FedAvg |
| `unbiased-aggregation.py` | §M.2 (Mathematical) | Horvitz–Thompson / size-prop sampling cut aggregation bias **1.6e-1 → 2e-3** under imbalance |
| `adaptive-local-work.py` | §E.1 (Experimental) | Decaying `E` (40→1) beats **every** fixed `E`; reproduces the large-`E` plateau |
| `permutation-aligned-averaging.py` | §T.1 (Theoretical) | Git-Re-Basin weight-matching turns the Fig-1 barrier **+0.024 → −0.023** |

## Run

```bash
# from the study root (uses the project venv):
.venv/bin/python improvements/heterogeneity-control-variate.py
.venv/bin/python improvements/unbiased-aggregation.py
.venv/bin/python improvements/adaptive-local-work.py
.venv/bin/python improvements/permutation-aligned-averaging.py
```

(Or `pip install -r requirements.txt` then `python <file>.py` in any numpy environment.)

## Notes

These are prototypes, not production code: each demonstrates the *direction* of the proposed improvement on the same small, transparent harness as `../sandbox/` (convex softmax regression / a tiny MLP), not a SOTA result. Margins are deliberately modest because the toy problems have low ceilings; the verified fact is the **sign** of the effect, which is robust and deterministic across reruns. Production re-implementations on the paper's actual non-convex nets belong in a separate repo. The `permutation-aligned-averaging.py` solver is an exact in-house Hungarian (no scipy); the control-variate refresh is SCAFFOLD's no-extra-pass "Option II".

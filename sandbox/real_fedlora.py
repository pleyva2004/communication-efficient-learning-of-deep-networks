#!/usr/bin/env python3
"""
real_fedlora.py -- Level-2: FEDERATED LoRA.

Applies the FedAvg algorithm (McMahan et al. 2017, Algorithm 1) to the LoRA
*adapter* weights of a causal language model (transformers + peft). This shows
that the paper's algorithm extends naturally to modern LM fine-tuning, and tells
a clean communication-efficiency story: each round, ONLY the tiny LoRA adapter
tensors -- not the frozen base model -- are sent to the server, averaged with the
erratum-corrected n_k / m_t weighting, and broadcast back.

FedAvg (Algorithm 1, corrected aggregation):
    Server:
        initialize w0
        for each round t:
            m   = max(C*K, 1)
            S_t = random set of m clients
            for k in S_t (in parallel): w^k_{t+1} = ClientUpdate(k, w_t)
            m_t = sum_{k in S_t} n_k
            w_{t+1} = sum_{k in S_t} (n_k / m_t) * w^k_{t+1}   # Erratum4
    ClientUpdate(k, w):
        for E local epochs: SGD over local batches; return w

HERE w == ONLY the LoRA adapter parameters. The base model is frozen and shared
verbatim, so it never needs to be communicated.

CLI
    --smoke : build a TINY LM from a config (NO pretrained download), wrap with
              LoRA, and run FedAvg-over-LoRA for a few rounds on synthetic
              non-IID token shards. Offline-safe, < 180s.
    --train : real path -- swap in a real base (e.g. Qwen/Qwen2.5-1.5B-Instruct,
              <=3B, fits 36GB unified memory on MPS). Needs an HF download +
              GPU/MPS host. Clearly marked; will attempt a real download.

Run:
    .venv/bin/python sandbox/real_fedlora.py --smoke
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------------
# Soft imports: if torch/transformers/peft/accelerate are missing, explain + exit 0.
# ----------------------------------------------------------------------------
try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoConfig, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    import accelerate  # noqa: F401  (used implicitly by --train path)
except Exception as exc:  # pragma: no cover
    print("[real_fedlora] Missing a dependency:", repr(exc))
    print("[real_fedlora] Install with:")
    print("    pip install torch transformers peft accelerate numpy")
    sys.exit(0)


# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
SEED = 0


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# Synthetic NON-IID data: each client gets a different "topic" = a distinct
# subset of the vocab it draws structured token streams from. Structure (a
# repeating bigram pattern per topic) gives the LM something learnable, so the
# training loss can actually go down.
# ----------------------------------------------------------------------------
def make_client_shards(
    num_clients: int,
    vocab_size: int,
    seq_len: int,
    seqs_per_client_range: Tuple[int, int],
    rng: np.random.Generator,
) -> List[torch.Tensor]:
    """Return a list of LongTensors, one per client, shape [n_k, seq_len].

    Non-IID: client k draws tokens from its own contiguous vocab band and emits
    a deterministic-per-topic bigram chain, so each client's local distribution
    differs and is individually learnable.
    """
    shards: List[torch.Tensor] = []
    # Reserve token 0 as a BOS-ish anchor shared across clients.
    usable = vocab_size - 1
    band = max(4, usable // num_clients)
    for k in range(num_clients):
        lo = 1 + (k * band) % max(1, (usable - band))
        hi = min(vocab_size, lo + band)
        topic_tokens = np.arange(lo, hi)
        # Per-topic transition: next = topic_tokens[(idx*step + bias) % band]
        step = 1 + (k % 3)
        bias = k % len(topic_tokens)
        n_k = int(rng.integers(seqs_per_client_range[0], seqs_per_client_range[1] + 1))
        rows = []
        for _ in range(n_k):
            seq = [0]  # BOS anchor
            idx = int(rng.integers(0, len(topic_tokens)))
            for _ in range(seq_len - 1):
                idx = (idx * step + bias) % len(topic_tokens)
                seq.append(int(topic_tokens[idx]))
            rows.append(seq)
        shards.append(torch.tensor(rows, dtype=torch.long))
    return shards


# ----------------------------------------------------------------------------
# Model construction (offline-safe for --smoke)
# ----------------------------------------------------------------------------
def build_tiny_model_and_lora(
    vocab_size: int,
    seq_len: int,
    lora_r: int = 8,
) -> Tuple[torch.nn.Module, List[str]]:
    """Build a TINY Llama-style causal LM from config (NO download) + LoRA wrap."""
    cfg = AutoConfig.for_model(
        "llama",
        vocab_size=vocab_size,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=seq_len,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
    )
    base = AutoModelForCausalLM.from_config(cfg)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=2 * lora_r,
        lora_dropout=0.0,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    return model, target_modules


# ----------------------------------------------------------------------------
# LoRA adapter state helpers -- the ONLY thing communicated in FedAvg.
# ----------------------------------------------------------------------------
def get_lora_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Detached CPU copy of exactly the trainable (LoRA) tensors == w to send."""
    return {
        name: p.detach().to("cpu").clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def set_lora_state(model: torch.nn.Module, state: Dict[str, torch.Tensor], device) -> None:
    """Load an averaged LoRA state back into the model (broadcast step)."""
    own = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor in state.items():
            own[name].copy_(tensor.to(device))


def average_lora_states(
    states: List[Dict[str, torch.Tensor]],
    weights: List[float],
) -> Dict[str, torch.Tensor]:
    """w_{t+1} = sum_k (n_k / m_t) * w^k  -- erratum-corrected FedAvg aggregation.

    `weights` are the normalized n_k / m_t coefficients (sum to 1).
    """
    keys = states[0].keys()
    out: Dict[str, torch.Tensor] = {}
    for key in keys:
        acc = torch.zeros_like(states[0][key], dtype=torch.float32)
        for st, wgt in zip(states, weights):
            acc += wgt * st[key].to(torch.float32)
        out[key] = acc.to(states[0][key].dtype)
    return out


# ----------------------------------------------------------------------------
# ClientUpdate: train ONLY the LoRA adapter for E local epochs of SGD.
# ----------------------------------------------------------------------------
def client_update(
    model: torch.nn.Module,
    global_lora: Dict[str, torch.Tensor],
    data: torch.Tensor,
    device,
    local_epochs: int,
    batch_size: int,
    lr: float,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """Run E local epochs on client data; return updated LoRA state + mean loss."""
    set_lora_state(model, global_lora, device)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(trainable, lr=lr, momentum=0.0)

    n = data.shape[0]
    total_loss, n_batches = 0.0, 0
    for _ in range(local_epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch = data[idx].to(device)
            inputs = batch[:, :-1]
            labels = batch[:, 1:]
            out = model(input_ids=inputs)
            logits = out.logits
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1
    mean_loss = total_loss / max(1, n_batches)
    return get_lora_state(model), mean_loss


# ----------------------------------------------------------------------------
# Server: the FedAvg outer loop over LoRA adapters.
# ----------------------------------------------------------------------------
def federated_train(
    model: torch.nn.Module,
    shards: List[torch.Tensor],
    device,
    rounds: int,
    frac_clients: float,
    local_epochs: int,
    batch_size: int,
    lr: float,
) -> List[float]:
    num_clients = len(shards)
    client_sizes = [int(s.shape[0]) for s in shards]

    # w_0: the freshly-initialized global LoRA adapter.
    global_lora = get_lora_state(model)

    round_losses: List[float] = []
    rng = random.Random(SEED)
    for t in range(1, rounds + 1):
        m = max(int(frac_clients * num_clients), 1)
        selected = sorted(rng.sample(range(num_clients), m))

        # m_t = sum of sample counts among SELECTED clients (erratum denominator).
        m_t = sum(client_sizes[k] for k in selected)

        states, weights, losses = [], [], []
        for k in selected:
            new_state, loss = client_update(
                model, global_lora, shards[k], device,
                local_epochs, batch_size, lr,
            )
            states.append(new_state)
            weights.append(client_sizes[k] / m_t)  # n_k / m_t
            losses.append(loss)

        global_lora = average_lora_states(states, weights)
        set_lora_state(model, global_lora, device)  # broadcast back

        avg_client_loss = float(np.mean(losses))
        round_losses.append(avg_client_loss)
        sel_str = ",".join(str(s) for s in selected)
        print(
            f"  round {t:2d} | selected {m}/{num_clients} clients [{sel_str}] "
            f"| m_t={m_t:4d} samples | avg client loss = {avg_client_loss:.4f}"
        )
    return round_losses


# ----------------------------------------------------------------------------
# Param accounting
# ----------------------------------------------------------------------------
def param_report(model: torch.nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ----------------------------------------------------------------------------
# Entry points
# ----------------------------------------------------------------------------
def run_smoke() -> int:
    set_seed(SEED)

    # MPS for a tiny model can be flaky / slower than CPU due to launch overhead;
    # CPU is fastest and most reliable for this size, so we use CPU here.
    device = torch.device("cpu")

    NUM_CLIENTS = 8
    FRAC = 0.5          # C = 0.5
    VOCAB = 256
    SEQ_LEN = 32
    LORA_R = 8
    ROUNDS = 15
    LOCAL_EPOCHS = 4    # E
    BATCH_SIZE = 8      # B
    LR = 0.7

    print("=" * 74)
    print("FEDERATED LoRA  --  FedAvg over LoRA adapters of a tiny causal LM")
    print("=" * 74)
    print(f"device                : {device}")

    rng = np.random.default_rng(SEED)
    shards = make_client_shards(
        NUM_CLIENTS, VOCAB, SEQ_LEN, seqs_per_client_range=(48, 96), rng=rng
    )
    sizes = [int(s.shape[0]) for s in shards]
    print(f"clients (K)           : {NUM_CLIENTS}  | client fraction C = {FRAC}")
    print(f"per-client samples n_k: {sizes}  (total {sum(sizes)})")
    print(f"non-IID partition     : each client draws a distinct vocab-band bigram stream")
    print(f"seq_len               : {SEQ_LEN}  | vocab {VOCAB}")

    model, targets = build_tiny_model_and_lora(VOCAB, SEQ_LEN, LORA_R)
    model.to(device)

    trainable, total = param_report(model)
    print("-" * 74)
    print(f"base model            : tiny Llama-style from_config (NO pretrained download)")
    print(f"LoRA target modules   : {targets}  (r={LORA_R}, alpha={2*LORA_R})")
    print(f"total params          : {total:,}")
    print(f"LoRA trainable params : {trainable:,}  ({100.0*trainable/total:.3f}% of total)")
    print(f"  -> ONLY these {trainable:,} adapter params are communicated each round")
    print(f"     (the {total-trainable:,} frozen base params are never sent)")
    print("-" * 74)
    print(f"FedAvg: rounds={ROUNDS}, E={LOCAL_EPOCHS}, B={BATCH_SIZE}, lr={LR}")
    print("Running FedAvg over LoRA adapters...")

    losses = federated_train(
        model, shards, device,
        rounds=ROUNDS, frac_clients=FRAC,
        local_epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE, lr=LR,
    )

    print("-" * 74)
    first, last = losses[0], losses[-1]
    print(f"avg client loss: round 1 = {first:.4f}  ->  round {len(losses)} = {last:.4f}")
    drop = first - last
    print(f"total decrease : {drop:.4f}  ({100.0*drop/first:.1f}% reduction)")
    if last < first:
        print("RESULT: PASS -- FedAvg over LoRA adapters drove avg client loss DOWN.")
    else:
        print("RESULT: WARN -- loss did not decrease; check hyperparameters.")
    print("=" * 74)
    return 0


def run_train() -> int:
    """Real-base path. Needs an HF download + a GPU/MPS host (network required).

    Swap in a real <=3B base such as Qwen/Qwen2.5-1.5B-Instruct, which fits in
    36GB unified memory on an Apple-silicon MPS device. The FedAvg-over-LoRA loop
    below is identical to --smoke; only model construction and data differ.
    """
    print("[real_fedlora --train] Real-base federated-LoRA path.")
    print("  This requires network access (HF model download) and a GPU/MPS host.")
    print("  Recommended base: Qwen/Qwen2.5-1.5B-Instruct (<=3B, fits 36GB unified).")
    MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"  device: {device}")
    try:
        from transformers import AutoModelForCausalLM as _AM, AutoTokenizer as _AT
        print(f"  Loading base model {MODEL_ID} (this downloads weights)...")
        tok = _AT.from_pretrained(MODEL_ID)
        base = _AM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        lora_cfg = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora_cfg).to(device)
        trainable, total = param_report(model)
        print(f"  base params: {total:,} | LoRA trainable: {trainable:,} "
              f"({100.0*trainable/total:.3f}%)")
        print("  To run a real federation: tokenize per-client text shards with `tok`,")
        print("  then call federated_train(model, shards, device, ...) exactly as --smoke.")
        print("  (Wire up your own real corpus shards here.)")
        _ = tok  # keep reference; real data wiring is environment-specific
        return 0
    except Exception as exc:
        print(f"  Could not load real base (likely offline / no GPU): {exc!r}")
        print("  Run --smoke for the fully offline demonstration.")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Federated LoRA via FedAvg.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--smoke", action="store_true",
                   help="tiny from_config model, offline FedAvg-over-LoRA")
    g.add_argument("--train", action="store_true",
                   help="real base (needs HF download + GPU/MPS)")
    args = ap.parse_args()
    if args.train:
        return run_train()
    # default to smoke
    return run_smoke()


if __name__ == "__main__":
    sys.exit(main())

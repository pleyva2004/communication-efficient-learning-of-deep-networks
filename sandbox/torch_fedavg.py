#!/usr/bin/env python3
"""
torch_fedavg.py -- Level-2 (PyTorch + MPS) REAL FederatedAveraging.

Reproduces the central claim of McMahan et al. 2017 ("Communication-Efficient
Learning of Deep Networks from Decentralized Data", arXiv:1602.05629):

  * FedAvg (E local epochs, minibatch B, weighted n_k/m_t averaging) reaches
    a target accuracy in FAR FEWER communication rounds than FedSGD
    (E=1, B=full-batch), i.e. the 10-100x communication speedup.
  * It does so under both IID and the pathological non-IID (2 classes/client)
    partition that defines the federated setting.

This is the faithful PyTorch counterpart of the numpy `toy_fedavg`, but with:
  * a real ~1.6M-param CNN matching the paper's "MNIST CNN" architecture
    (two 5x5 conv layers 32->64, each + 2x2 max-pool, 512-unit FC, softmax),
  * genuine torch autograd SGD on-device (MPS > CUDA > CPU),
  * SYNTHETIC MNIST-like data (no downloads): each class has a structured
    1x28x28 mean pattern + Gaussian noise, so the 10-way task is learnable
    but non-trivial.

Algorithm 1 (paper):
  Server: for each round t: m = max(C*K, 1); S_t = random m clients;
          w_{t+1} = sum_{k in S_t} (n_k / m_t) * w^k_{t+1}   (Erratum: m_t = sum n_k)
  ClientUpdate(k, w): for E epochs, for each size-B batch: w <- w - eta * grad.

Self-contained; no shared utils. Deterministic via fixed seeds.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---- Soft-import torch so this file PARSES even without torch installed. ----
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except Exception:  # pragma: no cover - portability path
    _TORCH_OK = False


# --------------------------------------------------------------------------- #
# Device selection: MPS > CUDA > CPU
# --------------------------------------------------------------------------- #
def pick_device(requested: Optional[str]) -> "torch.device":
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Synthetic "MNIST-like" data: 10 classes of 1x28x28 images.
# Each class gets a fixed structured spatial template (a couple of Gaussian
# blobs at class-dependent locations) + per-sample Gaussian noise. This is
# learnable by a CNN but requires actual feature extraction (not memorizable
# from a single pixel), giving a realistic round-vs-accuracy curve.
# --------------------------------------------------------------------------- #
def _class_templates(num_classes: int = 10, size: int = 28, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    templates = np.zeros((num_classes, size, size), dtype=np.float32)
    for c in range(num_classes):
        # 2-3 blobs per class at class-specific random centers.
        n_blobs = 2 + (c % 2)
        for _ in range(n_blobs):
            cy = rng.uniform(4, size - 4)
            cx = rng.uniform(4, size - 4)
            sigma = rng.uniform(2.5, 4.5)
            amp = rng.uniform(0.7, 1.3)
            templates[c] += amp * np.exp(
                -((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2)
            )
    # Normalize each template to roughly unit scale.
    templates -= templates.mean(axis=(1, 2), keepdims=True)
    templates /= (templates.std(axis=(1, 2), keepdims=True) + 1e-6)
    return templates


def make_dataset(
    n_per_class: int, templates: np.ndarray, noise: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    num_classes, size, _ = templates.shape
    X = np.empty((num_classes * n_per_class, 1, size, size), dtype=np.float32)
    y = np.empty((num_classes * n_per_class,), dtype=np.int64)
    idx = 0
    for c in range(num_classes):
        base = templates[c][None, :, :]  # (1,28,28)
        block = base + rng.normal(0.0, noise, size=(n_per_class, size, size)).astype(
            np.float32
        )
        X[idx : idx + n_per_class, 0] = block
        y[idx : idx + n_per_class] = c
        idx += n_per_class
    return X, y


# --------------------------------------------------------------------------- #
# Client partitioning (balanced, like the paper).
#   IID:     shuffle all examples, split evenly across K clients.
#   non-IID: sort by label, cut into 2*K shards, give each client 2 shards
#            -> most clients see only ~2 classes (pathological).
# --------------------------------------------------------------------------- #
def partition_iid(n: int, K: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return [np.array(s) for s in np.array_split(perm, K)]


def partition_noniid(y: np.ndarray, K: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(y)
    order = np.argsort(y, kind="stable")  # group by label
    n_shards = 2 * K
    shards = [np.array(s) for s in np.array_split(order, n_shards)]
    shard_ids = rng.permutation(n_shards)
    clients: List[np.ndarray] = []
    for k in range(K):
        a, b = shard_ids[2 * k], shard_ids[2 * k + 1]
        clients.append(np.concatenate([shards[a], shards[b]]))
    return clients


# --------------------------------------------------------------------------- #
# Model: paper's "MNIST CNN" -> 1,663,370 params.
#   conv1: 1->32, 5x5 (same pad) ; 2x2 maxpool  (28->28->14)
#   conv2: 32->64, 5x5 (same pad); 2x2 maxpool  (14->14->7)
#   fc1: 64*7*7=3136 -> 512 (ReLU)
#   fc2: 512 -> 10 (softmax via cross-entropy)
# With 'same' padding the FC input is 64*7*7=3136, giving 3136*512+512 +
# the conv & fc2 params = 1,663,370 -- exactly the paper's MNIST CNN count.
# --------------------------------------------------------------------------- #
if _TORCH_OK:

    class MnistCNN(nn.Module):
        def __init__(self, num_classes: int = 10):
            super().__init__()
            # padding=2 keeps 28x28 ('same' for 5x5), matching the paper so the
            # post-pool feature map is 7x7 and total params == 1,663,370.
            self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)   # 28 -> 28
            self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)  # 14 -> 14
            self.fc1 = nn.Linear(64 * 7 * 7, 512)
            self.fc2 = nn.Linear(512, num_classes)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 28 -> 14
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 14 -> 7
            x = x.flatten(1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- #
# FedAvg core
# --------------------------------------------------------------------------- #
def get_flat_state(model) -> Dict[str, "torch.Tensor"]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def client_update(
    model,
    global_state: Dict[str, "torch.Tensor"],
    Xc: "torch.Tensor",
    yc: "torch.Tensor",
    E: int,
    B: int,
    lr: float,
    device,
    gen: "torch.Generator",
) -> Dict[str, "torch.Tensor"]:
    """ClientUpdate(k, w): E epochs of SGD with batch B (B<=0 means full-batch)."""
    model.load_state_dict(global_state)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    n = Xc.shape[0]
    bs = n if (B is None or B <= 0 or B >= n) else B
    for _ in range(E):
        perm = torch.randperm(n, generator=gen, device=Xc.device)
        for start in range(0, n, bs):
            idx = perm[start : start + bs]
            opt.zero_grad(set_to_none=True)
            logits = model(Xc[idx])
            loss = F.cross_entropy(logits, yc[idx])
            loss.backward()
            opt.step()
    return get_flat_state(model)


@torch.no_grad()
def aggregate(
    states: List[Dict[str, "torch.Tensor"]], weights: List[int]
) -> Dict[str, "torch.Tensor"]:
    """w_{t+1} = sum_k (n_k / m_t) * w^k   (paper Eq./Erratum: m_t = sum n_k)."""
    total = float(sum(weights))
    out: Dict[str, "torch.Tensor"] = {}
    for key in states[0]:
        acc = torch.zeros_like(states[0][key], dtype=torch.float32)
        for st, w in zip(states, weights):
            acc += (w / total) * st[key].float()
        out[key] = acc.to(states[0][key].dtype)
    return out


@torch.no_grad()
def evaluate(model, state, Xte, yte, device, batch: int = 2000) -> float:
    model.load_state_dict(state)
    model.eval()
    correct = 0
    n = Xte.shape[0]
    for start in range(0, n, batch):
        logits = model(Xte[start : start + batch])
        correct += (logits.argmax(1) == yte[start : start + batch]).sum().item()
    return correct / n


def run_fed(
    *,
    mode: str,          # "fedavg" or "fedsgd"
    partition: str,     # "iid" or "noniid"
    rounds: int,
    K: int,
    C: float,
    E: int,
    B: int,
    lr: float,
    Xtr: "torch.Tensor",
    ytr: "torch.Tensor",
    client_idx: List[np.ndarray],
    Xte: "torch.Tensor",
    yte: "torch.Tensor",
    model,
    device,
    eval_every: int,
    seed: int,
) -> List[Tuple[int, float]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    # Fresh init for a fair comparison between algorithms.
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(layer.weight, a=5 ** 0.5)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    global_state = get_flat_state(model)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 999)
    rng = np.random.default_rng(seed + 7)

    m = max(int(C * K), 1)
    history: List[Tuple[int, float]] = []

    # Round 0 baseline.
    acc0 = evaluate(model, global_state, Xte, yte, device)
    history.append((0, acc0))

    for t in range(1, rounds + 1):
        selected = rng.choice(K, size=m, replace=False)
        states: List[Dict[str, "torch.Tensor"]] = []
        weights: List[int] = []
        for k in selected:
            idx = client_idx[k]
            Xc = Xtr[idx]
            yc = ytr[idx]
            if mode == "fedsgd":
                # FedSGD == one full-batch gradient step (E=1, B=inf).
                st = client_update(model, global_state, Xc, yc, 1, 0, lr, device, gen)
            else:
                st = client_update(model, global_state, Xc, yc, E, B, lr, device, gen)
            states.append(st)
            weights.append(len(idx))
        global_state = aggregate(states, weights)

        if t % eval_every == 0 or t == rounds:
            acc = evaluate(model, global_state, Xte, yte, device)
            history.append((t, acc))
    return history


def rounds_to_target(history: List[Tuple[int, float]], target: float) -> Optional[int]:
    for r, a in history:
        if a >= target:
            return r
    return None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Torch + MPS FedAvg (McMahan 2017).")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--smoke", action="store_true",
                    help="Quick ~2-round sanity run (IID FedAvg vs FedSGD).")
    ap.add_argument("--device", type=str, default="auto",
                    help="auto|mps|cuda|cpu (default auto: MPS>CUDA>CPU).")
    ap.add_argument("--clients", type=int, default=50)
    ap.add_argument("--frac", type=float, default=0.1, help="C: client fraction.")
    ap.add_argument("--epochs", type=int, default=5, help="E: local epochs (FedAvg).")
    ap.add_argument("--batch", type=int, default=10, help="B: local minibatch (FedAvg).")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--per-class", type=int, default=120,
                    help="train examples per class (total train = 10x this).")
    ap.add_argument("--noise", type=float, default=1.4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not _TORCH_OK:
        print("PyTorch is not installed in this environment.")
        print("Install it with:  pip install torch")
        print("(This script needs torch + autograd; exiting cleanly.)")
        return 0

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    print("=" * 70)
    print("Torch FedAvg  (McMahan et al. 2017, arXiv:1602.05629)")
    print("=" * 70)
    print(f"device         : {device}")
    print(f"torch version  : {torch.__version__}")

    # ---- Build model + report param count ----
    model = MnistCNN(num_classes=10).to(device)
    p = count_params(model)
    print(f"model          : MNIST CNN (2x conv 5x5 [32->64] + FC512 + softmax)")
    print(f"param_count    : {p:,}  (paper MNIST CNN = 1,663,370)")

    # ---- Smoke vs full configuration ----
    if args.smoke:
        rounds = 2
        K = 20
        per_class = 60
        eval_every = 1
        E = 5
        B = 10
        print("mode           : SMOKE (2 rounds, IID, FedAvg vs FedSGD)")
    else:
        rounds = args.rounds
        K = args.clients
        per_class = args.per_class
        eval_every = max(1, rounds // 10)
        E = args.epochs
        B = args.batch

    C = args.frac
    lr = args.lr

    # ---- Data ----
    templates = _class_templates(10, 28, seed=123)
    Xtr_np, ytr_np = make_dataset(per_class, templates, args.noise, seed=args.seed + 1)
    Xte_np, yte_np = make_dataset(max(40, per_class // 2), templates, args.noise,
                                  seed=args.seed + 2)

    Xtr = torch.from_numpy(Xtr_np).to(device)
    ytr = torch.from_numpy(ytr_np).to(device)
    Xte = torch.from_numpy(Xte_np).to(device)
    yte = torch.from_numpy(yte_np).to(device)

    print(f"train examples : {Xtr.shape[0]} ({per_class}/class)   "
          f"test examples: {Xte.shape[0]}")
    print(f"clients K      : {K}   C={C}   m=max(C*K,1)={max(int(C*K),1)}")
    print(f"FedAvg local   : E={E} epochs, B={B} minibatch, lr={lr}")
    print("-" * 70)

    t0 = time.time()

    if args.smoke:
        # Only IID for smoke (keep < 120s); run both algorithms.
        idx_iid = partition_iid(Xtr.shape[0], K, seed=args.seed)
        print("[SMOKE] IID partition, FedAvg:")
        h_avg = run_fed(mode="fedavg", partition="iid", rounds=rounds, K=K, C=C,
                        E=E, B=B, lr=lr, Xtr=Xtr, ytr=ytr, client_idx=idx_iid,
                        Xte=Xte, yte=yte, model=model, device=device,
                        eval_every=eval_every, seed=args.seed)
        for r, a in h_avg:
            print(f"   round {r:3d}  test_acc = {a:.4f}")
        print("[SMOKE] IID partition, FedSGD (E=1, full-batch):")
        h_sgd = run_fed(mode="fedsgd", partition="iid", rounds=rounds, K=K, C=C,
                        E=1, B=0, lr=lr, Xtr=Xtr, ytr=ytr, client_idx=idx_iid,
                        Xte=Xte, yte=yte, model=model, device=device,
                        eval_every=eval_every, seed=args.seed)
        for r, a in h_sgd:
            print(f"   round {r:3d}  test_acc = {a:.4f}")

        dt = time.time() - t0
        avg_gain = h_avg[-1][1] - h_avg[0][1]
        sgd_gain = h_sgd[-1][1] - h_sgd[0][1]
        print("-" * 70)
        print(f"[SMOKE] FedAvg accuracy gain over {rounds} rounds: "
              f"{h_avg[0][1]:.3f} -> {h_avg[-1][1]:.3f}  (+{avg_gain:.3f})")
        print(f"[SMOKE] FedSGD accuracy gain over {rounds} rounds: "
              f"{h_sgd[0][1]:.3f} -> {h_sgd[-1][1]:.3f}  (+{sgd_gain:.3f})")
        print(f"[SMOKE] elapsed: {dt:.1f}s on {device}")
        ok = (h_avg[-1][1] > h_avg[0][1]) and (h_avg[-1][1] >= h_sgd[-1][1] - 0.02)
        print(f"[SMOKE] PASS (FedAvg improved & >= FedSGD per round): {ok}")
        return 0

    # ---- Full run: IID and non-IID, FedAvg vs FedSGD ----
    results: Dict[str, List[Tuple[int, float]]] = {}
    for partition in ("iid", "noniid"):
        if partition == "iid":
            cidx = partition_iid(Xtr.shape[0], K, seed=args.seed)
        else:
            cidx = partition_noniid(ytr_np, K, seed=args.seed)

        print(f"\n### {partition.upper()} partition ###")
        for mode in ("fedavg", "fedsgd"):
            label = f"{partition}/{mode}"
            E_use, B_use = (E, B) if mode == "fedavg" else (1, 0)
            hist = run_fed(mode=mode, partition=partition, rounds=rounds, K=K, C=C,
                           E=E_use, B=B_use, lr=lr, Xtr=Xtr, ytr=ytr,
                           client_idx=cidx, Xte=Xte, yte=yte, model=model,
                           device=device, eval_every=eval_every, seed=args.seed)
            results[label] = hist
            print(f"  -- {mode.upper():7s} (E={E_use}, B={'inf' if B_use==0 else B_use}) --")
            for r, a in hist:
                print(f"     round {r:3d}  test_acc = {a:.4f}")

    # ---- Communication-efficiency summary (the paper's headline claim) ----
    print("\n" + "=" * 70)
    print("COMMUNICATION-EFFICIENCY SUMMARY  (rounds to reach target acc)")
    print("=" * 70)
    for partition in ("iid", "noniid"):
        best = max(results[f"{partition}/fedavg"][-1][1],
                   results[f"{partition}/fedsgd"][-1][1])
        target = round(0.8 * best, 3)
        r_avg = rounds_to_target(results[f"{partition}/fedavg"], target)
        r_sgd = rounds_to_target(results[f"{partition}/fedsgd"], target)
        speed = (f"{r_sgd / r_avg:.1f}x" if (r_avg and r_sgd and r_avg > 0)
                 else "n/a")
        print(f"  {partition:7s}  target={target:.3f}  "
              f"FedAvg@{r_avg}  FedSGD@{r_sgd}  speedup={speed}")
    print(f"\ntotal elapsed: {time.time() - t0:.1f}s on {device}")
    print("Claim: FedAvg reaches the target in far fewer rounds than FedSGD,")
    print("under both IID and pathological non-IID partitions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

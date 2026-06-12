#!/usr/bin/env python3
"""T9f Phase 2 — Supervised router pretraining (BAR-style decoupled training).

Trains the MoE router as a multi-class classifier using bottleneck features
collected by ``collect_router_data.py``.  Only the router parameters (1540)
are updated; backbone, experts, and heads stay frozen.

After supervised pretraining, the full model checkpoint can be loaded by
``train.py --resume`` for Phase 3 joint fine-tuning at very low LR.

Usage::

    # Step 1: collect features
    uv run python scripts/collect_router_data.py \\
        checkpoints/t9f-phase1/best.pt \\
        --configs configs/even_clash.json configs/example.json \\
                  configs/dragon_battle.json configs/mage_duel.json \\
        --output data/router_dataset.pt --device cuda

    # Step 2: train router
    uv run python scripts/train_router_supervised.py \\
        checkpoints/t9f-phase1/best.pt data/router_dataset.pt \\
        --output checkpoints/t9f-phase2-supervised/best.pt \\
        --epochs 50 --lr 1e-3 --device cuda

    # Step 3: verify
    uv run python scripts/train_router_supervised.py \\
        checkpoints/t9f-phase1/best.pt data/router_dataset.pt \\
        --verify-only --configs configs/even_clash.json configs/example.json \\
                      configs/dragon_battle.json configs/mage_duel.json \\
        --device cuda
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from ai.deep.model import BattleNet


class RouterDataset(Dataset):
    """Simple (features, labels) dataset for router classification."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def _to_router_input(model: BattleNet, features: torch.Tensor) -> torch.Tensor:
    """Convert bottleneck features to router input (expert hidden concat).

    If features already match router input dim, return as-is.
    Otherwise, compute expert hidden features from bottleneck.
    """
    router_dim = model.moe.num_experts * model.moe.hidden_dim
    if features.shape[-1] == router_dim:
        return features  # already expert features
    # Compute expert features from bottleneck
    parts = []
    for i in range(model.moe.num_experts):
        parts.append(model.moe.experts[i](features))
    return torch.cat(parts, dim=-1)


def train_router(
    model: BattleNet,
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    val_feats: torch.Tensor,
    val_labels: torch.Tensor,
    num_epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
) -> dict[str, list]:
    """Train only the router layer via cross-entropy classification.

    Freezes backbone, experts, and heads.  Only the router parameters
    are updated.  Accepts either bottleneck features (384-dim) or
    pre-computed expert features (E*hidden_dim).

    Returns history dict with train/val loss and accuracy per epoch.
    """
    # Freeze everything except router
    model.freeze_backbone()
    model.moe.freeze_all_experts()

    # Convert to router input if needed
    with torch.no_grad():
        train_router_in = _to_router_input(model, train_feats)
        val_router_in = _to_router_input(model, val_feats)

    # Verify only router has gradients
    router_params = list(model.moe.router.parameters())
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) == len(router_params), (
        f"Expected {len(router_params)} trainable params, "
        f"got {len(trainable)}"
    )
    print(f"Training {sum(p.numel() for p in router_params)} router params",
          flush=True)

    optimizer = torch.optim.Adam(router_params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_ds = RouterDataset(train_router_in.to(device), train_labels.to(device))
    val_ds = RouterDataset(val_router_in.to(device), val_labels.to(device))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        # Train
        model.moe.router.train()
        total_loss, total_correct, total_n = 0.0, 0, 0
        for feats, labs in train_loader:
            logits = model.moe.router(feats)  # (B, num_experts)
            loss = criterion(logits, labs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labs)
            total_correct += (logits.argmax(dim=1) == labs).sum().item()
            total_n += len(labs)

        train_loss = total_loss / total_n
        train_acc = total_correct / total_n

        # Validate
        model.moe.router.eval()
        val_loss, val_correct, val_n = 0.0, 0, 0
        with torch.no_grad():
            for feats, labs in val_loader:
                logits = model.moe.router(feats)
                loss = criterion(logits, labs)
                val_loss += loss.item() * len(labs)
                val_correct += (logits.argmax(dim=1) == labs).sum().item()
                val_n += len(labs)

        val_loss = val_loss / val_n
        val_acc = val_correct / val_n

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if epoch % 10 == 0 or epoch == 1 or epoch == num_epochs:
            print(f"  Epoch {epoch:3d}: "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                  f"(best={best_val_acc:.4f})",
                  flush=True)

    print(f"\nBest val accuracy: {best_val_acc:.4f}", flush=True)
    return history


def evaluate_router(
    model: BattleNet,
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 512,
    device: str = "cpu",
) -> dict:
    """Evaluate router classification accuracy and confusion matrix.

    Accepts either bottleneck features (384-dim) or pre-computed
    expert features (E*hidden_dim).
    """
    model.moe.router.eval()
    with torch.no_grad():
        router_in = _to_router_input(model, features)
    ds = TensorDataset(router_in.to(device), labels.to(device))
    loader = DataLoader(ds, batch_size=batch_size)
    num_classes = labels.max().item() + 1

    all_preds = []
    all_labels = []
    correct, total = 0, 0

    with torch.no_grad():
        for feats, labs in loader:
            logits = model.moe.router(feats)
            preds = logits.argmax(dim=1)
            correct += (preds == labs).sum().item()
            total += len(labs)
            all_preds.append(preds.cpu())
            all_labels.append(labs.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t][p] += 1

    per_class_acc = {}
    for i in range(num_classes):
        class_total = cm[i].sum()
        per_class_acc[i] = cm[i][i] / class_total if class_total > 0 else 0.0

    return {
        "accuracy": correct / total,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm,
    }


def verify_router(
    model: BattleNet,
    config_paths: list[str],
    device: str = "cpu",
    samples_per_config: int = 100,
) -> dict:
    """Verify router assigns correct config->expert mapping with live env.

    Runs episodes in each config, collects bottleneck features, and checks
    that the router's argmax matches the expected expert index.
    """
    from ai.deep.pipeline import load_battle_config
    from ai.env import BattleEnv

    model.eval()
    results = {}

    for idx, cfg_path in enumerate(config_paths):
        config = load_battle_config(cfg_path)
        name = os.path.splitext(os.path.basename(cfg_path))[0]

        env = BattleEnv(config)
        obs, _ = env.reset(seed=42 + idx)

        correct, total = 0, 0
        import random
        rng = random.Random(42 + idx)

        with torch.no_grad():
            for _ in range(samples_per_config):
                grid_t = torch.tensor(
                    obs["grid"], dtype=torch.float32
                ).unsqueeze(0).to(device)
                gvec_t = torch.tensor(
                    obs["global"], dtype=torch.float32
                ).unsqueeze(0).to(device)

                feat = model.extract_bottleneck(grid_t, gvec_t)
                # Expert-aware routing: compute expert features first
                expert_feats = []
                for ei in range(model.moe.num_experts):
                    expert_feats.append(model.moe.experts[ei](feat))
                router_input = torch.cat(expert_feats, dim=-1)
                logits = model.moe.router(router_input)
                pred = logits.argmax(dim=1).item()

                if pred == idx:
                    correct += 1
                total += 1

                # Step
                legal = [a for a in range(13566) if obs["mask"][a]]
                if legal:
                    obs, _, terminated, truncated, _ = env.step(
                        rng.choice(legal))
                    if terminated or truncated:
                        obs, _ = env.reset(seed=rng.randint(0, 2**31))
                else:
                    obs, _ = env.reset(seed=rng.randint(0, 2**31))

        acc = correct / total if total > 0 else 0.0
        results[name] = {
            "accuracy": acc,
            "expected_expert": idx,
            "correct": correct,
            "total": total,
        }
        status = "✅" if acc >= 0.7 else "❌"
        print(f"  [{idx}] {name}: {acc:.2%} ({correct}/{total}) {status}",
              flush=True)

    overall = sum(r["correct"] for r in results.values()) / \
              max(sum(r["total"] for r in results.values()), 1)
    print(f"\n  Overall: {overall:.2%}", flush=True)
    return {"per_config": results, "overall_accuracy": overall}


def main():
    parser = argparse.ArgumentParser(
        description="Supervised MoE router pretraining")
    parser.add_argument("checkpoint", help="Phase 1 model checkpoint (.pt)")
    parser.add_argument("dataset", help="Feature dataset from collect_router_data.py")
    parser.add_argument("--output", default="checkpoints/t9f-phase2-supervised/best.pt",
                        help="Output checkpoint path")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Validation split ratio (default 0.2)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing router (no training)")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Config paths for --verify-only mode")
    args = parser.parse_args()

    # Load model
    model = BattleNet(num_experts=4, moe_hidden_dim=384, routing_topk=2)
    ckpt = torch.load(args.checkpoint, map_location=args.device,
                      weights_only=False)
    # Router weight shape changed (expert-aware routing: 384→1536 input dim).
    # Filter out mismatched keys and load rest, then init router randomly.
    pretrained = ckpt["model"]
    model_dict = model.state_dict()
    matched = {
        k: v for k, v in pretrained.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    model_dict.update(matched)
    model.load_state_dict(model_dict)
    skipped = set(pretrained.keys()) - set(matched.keys())
    model.to(args.device)
    model.eval()
    print(f"Loaded model from {args.checkpoint} "
          f"({len(matched)} keys matched, {len(skipped)} skipped: {skipped})",
          flush=True)

    if args.verify_only:
        if not args.configs:
            print("Error: --verify-only requires --configs", file=sys.stderr)
            sys.exit(1)
        print("\n=== Router Verification ===")
        verify_router(model, args.configs, device=args.device)
        return

    # Load dataset
    data = torch.load(args.dataset, weights_only=False)
    features = data["features"]
    labels = data["labels"]
    config_names = data["config_names"]
    print(f"Loaded {len(features)} samples from {args.dataset}", flush=True)
    print(f"  Configs: {config_names}")
    print(f"  Labels:  {torch.bincount(labels).tolist()}", flush=True)

    # Train/val split
    n = len(labels)
    n_val = int(n * args.val_split)
    perm = torch.randperm(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_feats, train_labels = features[train_idx], labels[train_idx]
    val_feats, val_labels = features[val_idx], labels[val_idx]
    print(f"  Train: {len(train_labels)}, Val: {len(val_labels)}", flush=True)

    # Train
    print(f"\n=== Router Supervised Training ===")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}",
          flush=True)

    t0 = time.time()
    history = train_router(
        model, train_feats, train_labels, val_feats, val_labels,
        num_epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, device=args.device,
    )
    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.1f}s", flush=True)

    # Final evaluation
    print("\n=== Final Evaluation ===")
    results = evaluate_router(model, val_feats, val_labels, device=args.device)
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Per-class: {results['per_class_accuracy']}")
    print(f"  Confusion matrix:\n{results['confusion_matrix']}")

    # Pass/fail
    acc = results["accuracy"]
    if acc >= 0.85:
        print(f"\n✅ PASSED: accuracy {acc:.2%} >= 85% threshold")
    elif acc >= 0.70:
        print(f"\n⚠️  MARGINAL: accuracy {acc:.2%} between 70-85%")
    else:
        print(f"\n❌ FAILED: accuracy {acc:.2%} < 70% — experts not differentiated enough")

    # Save checkpoint
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({
        "step": 0,
        "model": model.state_dict(),
        "router_history": history,
        "router_accuracy": acc,
    }, args.output)
    print(f"\nSaved checkpoint to {args.output}", flush=True)

    # Router weight analysis
    w = model.moe.router[2].weight.data
    print(f"\n=== Router Weight Analysis ===")
    print(f"  Weight shape: {w.shape}")
    print(f"  Row norms: {w.norm(dim=1).tolist()}")
    print(f"  Inter-row cosine similarities:")
    for i in range(w.shape[0]):
        for j in range(i + 1, w.shape[0]):
            cos = torch.nn.functional.cosine_similarity(
                w[i].unsqueeze(0), w[j].unsqueeze(0)
            ).item()
            print(f"    Expert {i} vs {j}: {cos:.4f}")


if __name__ == "__main__":
    main()

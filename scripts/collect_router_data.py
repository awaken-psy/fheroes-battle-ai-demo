#!/usr/bin/env python3
"""T9f Phase 2 — Collect bottleneck features for supervised router pretraining.

Runs the Phase 1 model through each config using random-legal actions,
collecting (bottleneck_feature, config_id) pairs.  The saved dataset is
used by ``train_router_supervised.py`` to train the router as a classifier.

Usage::

    uv run python scripts/collect_router_data.py \
        checkpoints/t9f-phase1/best.pt \
        --configs configs/even_clash.json configs/example.json \
                  configs/dragon_battle.json configs/mage_duel.json \
        --samples-per-config 2000 \
        --output data/router_dataset.pt --device cuda
"""

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from ai.deep.model import BattleNet
from ai.deep.pipeline import load_battle_config
from ai.env import BattleEnv


def _random_legal_action(mask: list) -> int:
    """Pick a random legal action from the mask."""
    legal = [a for a in range(len(mask)) if mask[a]]
    return random.choice(legal) if legal else 0


def collect_features(
    model: torch.nn.Module,
    config: dict,
    config_id: int,
    num_samples: int = 2000,
    max_steps_per_episode: int = 200,
    device: str = "cpu",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect bottleneck features for one config.

    Uses random-legal actions for speed (no model inference needed for
    action selection).  The model is only used for ``extract_bottleneck()``.

    Parameters
    ----------
    model : BattleNet
        Model with MoE layer, loaded from Phase 1 checkpoint.
    config : dict
        BattleEnv configuration dict.
    config_id : int
        Integer label (0..num_configs-1) for classification.
    num_samples : int
        Target number of (feature, label) pairs to collect.
    device : str
        Torch device.

    Returns
    -------
    features : Tensor (N, 384)
    labels : Tensor (N,)
    """
    model.eval()
    features_list = []
    rng = random.Random(seed)

    env = BattleEnv(config)
    obs, _ = env.reset(seed=seed)
    collected = 0

    with torch.no_grad():
        while collected < num_samples:
            grid_t = torch.tensor(
                obs["grid"], dtype=torch.float32
            ).unsqueeze(0).to(device)
            gvec_t = torch.tensor(
                obs["global"], dtype=torch.float32
            ).unsqueeze(0).to(device)

            feat = model.extract_bottleneck(grid_t, gvec_t)  # (1, 384)
            features_list.append(feat.squeeze(0).cpu())
            collected += 1

            # Step with random legal action
            action = _random_legal_action(obs["mask"])
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if done or env._step_count > max_steps_per_episode:
                obs, _ = env.reset(seed=rng.randint(0, 2**31))

    features = torch.stack(features_list)  # (N, 384)
    labels = torch.full((collected,), config_id, dtype=torch.long)
    return features, labels


def main():
    parser = argparse.ArgumentParser(
        description="Collect bottleneck features for supervised router training")
    parser.add_argument("checkpoint", help="Phase 1 model checkpoint (.pt)")
    parser.add_argument("--configs", nargs="+", required=True,
                        help="Battle config JSON paths (one per expert)")
    parser.add_argument("--samples-per-config", type=int, default=2000,
                        help="Number of feature samples per config (default 2000)")
    parser.add_argument("--output", default="data/router_dataset.pt",
                        help="Output dataset path (default data/router_dataset.pt)")
    parser.add_argument("--device", default="cpu",
                        help="Torch device (default cpu)")
    args = parser.parse_args()

    # Load model
    model = BattleNet(num_experts=len(args.configs), moe_hidden_dim=384,
                      routing_topk=2)
    ckpt = torch.load(args.checkpoint, map_location=args.device,
                      weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(args.device)
    model.eval()
    print(f"Loaded model from {args.checkpoint}", flush=True)

    # Collect features for each config
    all_features = []
    all_labels = []
    config_names = []

    for idx, cfg_path in enumerate(args.configs):
        config = load_battle_config(cfg_path)
        name = os.path.splitext(os.path.basename(cfg_path))[0]
        config_names.append(name)

        t0 = time.time()
        feats, labs = collect_features(
            model, config, config_id=idx,
            num_samples=args.samples_per_config,
            device=args.device,
            seed=42 + idx,
        )
        elapsed = time.time() - t0
        print(f"  [{idx}] {name}: {len(feats)} samples in {elapsed:.1f}s",
              flush=True)

        all_features.append(feats)
        all_labels.append(labs)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # Shuffle
    perm = torch.randperm(len(labels))
    features = features[perm]
    labels = labels[perm]

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({
        "features": features,
        "labels": labels,
        "config_names": config_names,
    }, args.output)

    print(f"\nSaved {len(features)} samples to {args.output}")
    print(f"Label distribution: {torch.bincount(labels).tolist()}")


if __name__ == "__main__":
    main()

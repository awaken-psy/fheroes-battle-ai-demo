"""R7/T9c — Training pipeline utilities.

Config loading, curriculum scheduling, and checkpoint management shared
between the training script and its tests.
"""

import json
import os
from typing import Optional, Set, Tuple

import torch

from ai.deep.trainer import PPOTrainer


# ── Config loading ──────────────────────────────────────────────


def default_battle_config() -> dict:
    """Simple 1v1 Swordsman default for quick validation."""
    return {
        "units": [
            ("Swordsman", 0, 5, 3),
            ("Swordsman", 1, 5, 6),
        ],
    }


def load_battle_config(path: Optional[str] = None) -> dict:
    """Load battle config from a JSON file, or return the default.

    Handles both dict-style units (from ``configs/*.json``) and
    tuple/list-style units (native BattleEnv format).
    """
    if path is None:
        return default_battle_config()

    with open(path) as f:
        raw = json.load(f)

    # Support both dict-style {"units": [...]} and bare list [...]
    if isinstance(raw, list):
        raw = {"units": raw}

    units = []
    for u in raw.get("units", []):
        if isinstance(u, dict):
            name = u.get("name") or u.get("type")
            count = u.get("count")
            tup: tuple = (name, u["team"], u["col"], u["row"])
            units.append(tup if count is None else tup + (count,))
        else:
            units.append(u)  # already a tuple/list

    config: dict = {"units": units}
    for key in ("heroes", "siege", "morale", "luck"):
        if key in raw:
            config[key] = raw[key]
    return config


# ── Curriculum scheduling ───────────────────────────────────────


def get_curriculum_phase(
    total_steps: int,
    phase1_steps: int = 10_000,
    phase2_steps: int = 30_000,
) -> Tuple[int, float]:
    """Return ``(reward_phase, dense_weight)`` for *total_steps*.

    Phase 1 (``step < phase1_steps``): dense + sparse, weight = 1.0
    Phase 2 (``phase1_steps <= step < phase2_steps``): transition, weight
        linearly decays 1 → 0.
    Phase 3 (``step >= phase2_steps``): sparse only, weight = 0.0
    """
    if total_steps < phase1_steps:
        return 1, 1.0
    elif total_steps < phase2_steps:
        progress = (total_steps - phase1_steps) / max(
            phase2_steps - phase1_steps, 1)
        return 2, max(0.0, 1.0 - progress)
    else:
        return 3, 0.0


# ── Checkpoint management ───────────────────────────────────────


def save_checkpoint(
    trainer: PPOTrainer,
    step: int,
    path: str,
) -> None:
    """Save model, optimizer state, and step counter to *path*."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "step": step,
        "model": trainer.model.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
    }, path)


def load_checkpoint(
    trainer: PPOTrainer,
    path: str,
) -> int:
    """Restore trainer from *path*.  Returns the saved step count."""
    ckpt = torch.load(path, map_location=trainer.device,
                      weights_only=False)
    trainer.model.load_state_dict(ckpt["model"])
    trainer.optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"]


def load_backbone_weights(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: str = "cpu",
) -> Tuple[int, Set[str]]:
    """Load backbone weights from a non-MoE checkpoint into a MoE model.

    Only keys whose name **and** shape match are copied.  New MoE layers
    keep their random initialisation.  The optimizer is **not** loaded
    (MoE layers have no prior momentum).

    Returns (num_matched_keys, set_of_skipped_keys).

    Typical usage::

        model = BattleNet(num_experts=4)
        matched, skipped = load_backbone_weights(
            model, "checkpoints/t9b-replay/best.pt", device="cuda")
    """
    ckpt = torch.load(checkpoint_path, map_location=device,
                       weights_only=False)
    pretrained = ckpt["model"]
    model_dict = model.state_dict()

    matched = {
        k: v for k, v in pretrained.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    model_dict.update(matched)
    model.load_state_dict(model_dict)

    skipped = set(pretrained.keys()) - set(matched.keys())
    return len(matched), skipped

#!/usr/bin/env python3
"""R7/T2 — Training pipeline for fheroes2 battle DeepAI.

Usage::

    # Train from scratch
    python scripts/train.py --total-steps 100000 --eval-interval 5000

    # Resume from checkpoint
    python scripts/train.py --resume checkpoints/checkpoint_5000.pt

    # Custom battle config
    python scripts/train.py --config configs/even_clash.json

    # With all T2 improvements
    python scripts/train.py --device cuda --lr-decay --grad-accum 4 --tensorboard

All training progress is printed as JSON lines (one per rollout).
Evaluation results are JSON lines with an ``"eval"`` key.
"""

import argparse
import json
import os
import sys
import time

# Add project root to import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from ai.deep.model import BattleNet
from ai.deep.pipeline import (
    get_curriculum_phase,
    load_battle_config,
    load_checkpoint,
    save_checkpoint,
)
from ai.deep.player import make_agent_fn
from ai.deep.trainer import PPOTrainer
from ai.self_play import eval_vs_classic


# ── CLI ──────────────────────────────────────────────────────────


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train fheroes2 battle DeepAI via PPO self-play")

    # Training
    p.add_argument("--total-steps", type=int, default=100_000,
                   help="Total env steps to train (default: 100k)")
    p.add_argument("--rollout-steps", type=int, default=2048,
                   help="Steps per rollout collection (default: 2048)")
    p.add_argument("--config", type=str, default=None,
                   help="Path to battle config JSON (default: 1v1 Swordsman)")
    p.add_argument("--device", type=str, default="cpu",
                   help="torch device (default: cpu)")

    # PPO hyperparameters
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--entropy-coeff", type=float, default=0.01)
    p.add_argument("--value-coeff", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)

    # T2 improvements
    p.add_argument("--lr-decay", action="store_true",
                   help="Enable linear LR decay to 0 over training")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Gradient accumulation steps (default: 1, no accumulation)")
    p.add_argument("--tensorboard", action="store_true",
                   help="Enable TensorBoard logging to runs/")

    # Curriculum
    p.add_argument("--phase1-steps", type=int, default=10_000,
                   help="Steps for phase 1 (dense+sparse)")
    p.add_argument("--phase2-steps", type=int, default=30_000,
                   help="Step at which phase 3 (sparse-only) begins")

    # Evaluation
    p.add_argument("--eval-interval", type=int, default=5_000,
                   help="Evaluate every N steps")
    p.add_argument("--eval-games", type=int, default=20,
                   help="Eval games vs ClassicAI")

    # Checkpointing
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")

    return p.parse_args(argv)


# ── Logging helpers ──────────────────────────────────────────────


def log_step(step: int, info: dict) -> None:
    """Print one JSON line per training step."""
    entry = {"step": step, "type": "train"}
    entry.update(info)
    print(json.dumps(entry, ensure_ascii=False), flush=True)


def log_eval(step: int, eval_info: dict) -> None:
    """Print one JSON line for evaluation results."""
    print(json.dumps(
        {"step": step, "type": "eval", **eval_info},
        ensure_ascii=False,
    ), flush=True)


# ── Main loop ────────────────────────────────────────────────────


def main(argv=None):
    args = parse_args(argv)

    # ── Setup ─────────────────────────────────────────────────────
    env_config = load_battle_config(args.config)
    model = BattleNet()
    trainer = PPOTrainer(
        model, env_config,
        lr=args.lr, gamma=args.gamma, gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps, update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size, entropy_coeff=args.entropy_coeff,
        value_coeff=args.value_coeff, max_grad_norm=args.max_grad_norm,
        grad_accum_steps=args.grad_accum, device=args.device,
    )

    # ── LR scheduler (T2) ─────────────────────────────────────────
    scheduler = None
    if args.lr_decay:
        # Total optimizer steps ≈ total_rollouts × update_epochs
        # We approximate: scheduler steps once per rollout
        # LinearLR decays from lr to end_lr over total_iters
        scheduler = torch.optim.lr_scheduler.LinearLR(
            trainer.optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=args.total_steps // args.rollout_steps,
        )

    # ── TensorBoard writer (T2) ──────────────────────────────────
    writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir="runs")

    # ── Resume ────────────────────────────────────────────────────
    total_steps = 0
    if args.resume:
        total_steps = load_checkpoint(trainer, args.resume)
        log_step(total_steps, {"msg": f"resumed from {args.resume}"})

    last_eval = total_steps
    t0 = time.time()

    # ── Train ─────────────────────────────────────────────────────
    while total_steps < args.total_steps:
        phase, dense_weight = get_curriculum_phase(
            total_steps, args.phase1_steps, args.phase2_steps)

        info = trainer.train_step(
            num_steps=args.rollout_steps,
            reward_phase=phase,
            dense_weight=dense_weight,
        )
        total_steps += info.pop("steps")

        info["phase"] = phase
        info["dense_w"] = round(dense_weight, 3)
        info["elapsed"] = round(time.time() - t0, 1)

        # Log current LR
        current_lr = trainer.optimizer.param_groups[0]["lr"]
        info["lr"] = current_lr

        log_step(total_steps, info)

        # TensorBoard logging
        if writer is not None:
            writer.add_scalar("train/policy_loss", info["policy_loss"],
                              total_steps)
            writer.add_scalar("train/value_loss", info["value_loss"],
                              total_steps)
            writer.add_scalar("train/entropy", info["entropy"],
                              total_steps)
            writer.add_scalar("train/total_loss", info["total_loss"],
                              total_steps)
            writer.add_scalar("train/approx_kl", info["approx_kl"],
                              total_steps)
            writer.add_scalar("train/mean_reward", info["mean_reward"],
                              total_steps)
            writer.add_scalar("train/mean_length", info["mean_length"],
                              total_steps)
            writer.add_scalar("train/lr", current_lr, total_steps)

        # Step LR scheduler
        if scheduler is not None:
            scheduler.step()

        # ── Periodic eval + checkpoint ────────────────────────────
        if total_steps - last_eval >= args.eval_interval:
            model.eval()
            agent_fn = make_agent_fn(model, device=args.device)
            eval_info = eval_vs_classic(
                env_config, agent_fn,
                learning_team=0, games=args.eval_games, seed=42)
            model.train()
            log_eval(total_steps, eval_info)

            # TensorBoard eval metrics
            if writer is not None:
                writer.add_scalar("eval/win_rate", eval_info["win_rate"],
                                  total_steps)
                writer.add_scalar("eval/avg_rounds", eval_info["avg_rounds"],
                                  total_steps)

            ckpt_path = os.path.join(
                args.checkpoint_dir, f"checkpoint_{total_steps}.pt")
            save_checkpoint(trainer, total_steps, ckpt_path)
            last_eval = total_steps

    # ── Final checkpoint ──────────────────────────────────────────
    final_path = os.path.join(args.checkpoint_dir, "final.pt")
    save_checkpoint(trainer, total_steps, final_path)

    if writer is not None:
        writer.close()

    elapsed = round(time.time() - t0, 1)
    print(json.dumps({
        "step": total_steps, "type": "done",
        "msg": (f"Training complete in {elapsed}s. "
                f"Saved to {final_path}"),
    }))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""R7/T2/T9c — Training pipeline for fheroes2 battle DeepAI.

Usage::

    # Train from scratch
    python scripts/train.py --total-steps 100000 --eval-interval 5000

    # Resume from checkpoint
    python scripts/train.py --resume checkpoints/checkpoint_5000.pt

    # Custom battle config
    python scripts/train.py --config configs/even_clash.json

    # With all T2/T6 improvements
    python scripts/train.py --device cuda --lr-schedule cosine --grad-accum 4 --tensorboard

    # T9c MoE Stage 2: per-expert round-robin training
    python scripts/train.py --use-moe --num-experts 4 --train-stage 2 \
        --load-backbone checkpoints/t9b-replay/best.pt \
        --config configs/even_clash.json configs/example.json \
                configs/dragon_battle.json configs/mage_duel.json \
        --total-steps 150000 --device cuda

    # T9c MoE Stage 3: router-only training
    python scripts/train.py --use-moe --num-experts 4 --train-stage 3 \
        --resume checkpoints/t9c-stage2/best.pt \
        --config configs/even_clash.json configs/example.json \
                configs/dragon_battle.json configs/mage_duel.json \
        --total-steps 50000 --device cuda

All training progress is printed as JSON lines (one per rollout).
Evaluation results are JSON lines with an ``"eval"`` key.
"""

import argparse
import json
import os
import random
import sys
import time

# Add project root to import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from ai.deep.model import BattleNet
from ai.action_space import ACTION_DIM
from ai.deep.opponent_pool import OpponentPool
from ai.deep.replay_buffer import ReplayBuffer
from ai.deep.pipeline import (
    get_curriculum_phase,
    load_backbone_weights,
    load_battle_config,
    load_checkpoint,
    save_checkpoint,
)
from ai.deep.player import make_agent_fn
from ai.deep.trainer import PPOTrainer
from ai.self_play import eval_vs_classic


# ── T9e helpers ──────────────────────────────────────────────────


def _transfer_shared_head_weights(
    model: BattleNet, checkpoint_path: str, device: str,
) -> None:
    """Copy shared policy/value head weights to every per-expert head.

    This gives each expert the same initial policy as the pre-trained shared
    heads (hot start), solving T9d's cold-start failure where 7M randomly-
    initialized parameters couldn't learn in 300K steps.

    Only works when ``moe.hidden_dim == moe.input_dim`` (both 384) so that
    the per-expert head dimensions match the old shared head dimensions.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = ckpt.get("model", ckpt)

    required_keys = [
        "policy_head.weight", "policy_head.bias",
        "value_head.weight", "value_head.bias",
    ]
    missing = [k for k in required_keys if k not in state]
    if missing:
        print(f"WARNING: Cannot transfer shared heads (missing keys: {missing})")
        return

    moe = model.moe
    for i in range(moe.num_experts):
        moe.policy_heads[i].weight.data.copy_(state["policy_head.weight"])
        moe.policy_heads[i].bias.data.copy_(state["policy_head.bias"])
        moe.value_heads[i].weight.data.copy_(state["value_head.weight"])
        moe.value_heads[i].bias.data.copy_(state["value_head.bias"])

    print(f"[T9e] Transferred shared head weights to "
          f"{moe.num_experts} per-expert heads")


# ── CLI ──────────────────────────────────────────────────────────


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train fheroes2 battle DeepAI via PPO self-play")

    # Training
    p.add_argument("--total-steps", type=int, default=100_000,
                   help="Total env steps to train (default: 100k)")
    p.add_argument("--rollout-steps", type=int, default=2048,
                   help="Steps per rollout collection (default: 2048)")
    p.add_argument("--config", type=str, nargs="+", default=None,
                   help="Path(s) to battle config JSON. Multiple files enable "
                        "multi-config training (T5). Default: 1v1 Swordsman")
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

    # T2/T6 improvements
    p.add_argument("--lr-schedule", type=str, default="none",
                   choices=["none", "linear", "cosine"],
                   help="LR schedule: none (constant), linear (decay to 0), "
                        "cosine (CosineAnnealingLR, T6)")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Gradient accumulation steps (default: 1, no accumulation)")
    p.add_argument("--tensorboard", action="store_true",
                   help="Enable TensorBoard logging to runs/")

    # T3 opponent pool
    p.add_argument("--opponent-pool", type=int, default=0,
                   help="Opponent pool capacity (0 = disabled, default: 0)")

    # T9b replay buffer
    p.add_argument("--replay-buffer", type=int, default=0,
                   help="Replay buffer capacity in rollouts (0 = disabled, default: 0)")

    # T9c/T9d MoE
    p.add_argument("--use-moe", action="store_true",
                   help="Enable Soft MoE layer (T9c)")
    p.add_argument("--num-experts", type=int, default=4,
                   help="Number of MoE experts (default: 4)")
    p.add_argument("--moe-hidden-dim", type=int, default=384,
                   help="Hidden dim for each MoE expert (default: 384, "
                        "must match bottleneck for identity init)")
    p.add_argument("--routing-topk", type=int, default=2,
                   help="Number of experts to activate per input (default: 2)")
    p.add_argument("--balance-loss-weight", type=float, default=0.01,
                   help="Weight for router load-balancing loss (default: 0.01, 0=disabled)")
    p.add_argument("--diversity-loss-weight", type=float, default=0.0,
                   help="Weight for expert output diversity loss (T9g). "
                        "Penalizes pairwise cosine similarity between experts "
                        "(default: 0.0=disabled, try 0.1-1.0)")
    p.add_argument("--train-stage", type=int, default=0,
                   choices=[0, 2, 3],
                   help="0=normal training, 2=per-expert (Stage 2), "
                        "3=router-only (Stage 3)")
    p.add_argument("--load-backbone", type=str, default=None,
                   help="Load backbone weights from a non-MoE checkpoint "
                        "(for Stage 2 bootstrap)")

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
    # Load one or more battle configs (T5 multi-config)
    config_paths = args.config or [None]
    configs = [load_battle_config(p) for p in config_paths]
    config_names = [
        os.path.splitext(os.path.basename(p))[0] if p else "default"
        for p in config_paths
    ]
    env_config = configs[0]  # default for trainer init

    # ── Replay buffer (T9b) ─────────────────────────────────────
    replay_buffer = None
    if args.replay_buffer > 0:
        replay_buffer = ReplayBuffer(capacity=args.replay_buffer)

    model = BattleNet(
        num_experts=args.num_experts if args.use_moe else 0,
        moe_hidden_dim=args.moe_hidden_dim,
        routing_topk=args.routing_topk,
    )

    # ── T9c: Load backbone from non-MoE checkpoint ────────────────
    if args.load_backbone:
        matched, skipped = load_backbone_weights(
            model, args.load_backbone, args.device)
        log_step(0, {"msg": f"loaded backbone: {matched} keys matched, "
                          f"{len(skipped)} skipped"})

    # ── T9e: Transfer shared head weights to per-expert heads ──────
    if (args.use_moe and args.load_backbone
            and model.moe is not None
            and model.moe.hidden_dim == model.moe.input_dim):
        _transfer_shared_head_weights(model, args.load_backbone, args.device)

    trainer = PPOTrainer(
        model, env_config,
        lr=args.lr, gamma=args.gamma, gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps, update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size, entropy_coeff=args.entropy_coeff,
        value_coeff=args.value_coeff, max_grad_norm=args.max_grad_norm,
        grad_accum_steps=args.grad_accum, device=args.device,
        replay_buffer=replay_buffer,
        balance_loss_weight=args.balance_loss_weight,
        diversity_loss_weight=args.diversity_loss_weight,
    )

    # ── T9c: Stage-specific freezing ──────────────────────────────
    if args.train_stage == 2:
        model.freeze_backbone()
        log_step(0, {"msg": f"Stage 2: backbone frozen, "
                          f"{model.num_experts} experts round-robin"})
    elif args.train_stage == 3:
        model.freeze_backbone()
        model.freeze_experts_and_heads()
        log_step(0, {"msg": "Stage 3: backbone + experts + heads frozen, "
                          "router-only training"})

    multi_config = len(configs) > 1
    if multi_config:
        log_step(0, {"msg": f"multi-config training: {config_names}"})

    # ── LR scheduler (T2/T6) ──────────────────────────────────────────
    scheduler = None
    total_iters = args.total_steps // args.rollout_steps
    if args.lr_schedule == "linear":
        # LinearLR decays from lr to 0 over total_iters
        scheduler = torch.optim.lr_scheduler.LinearLR(
            trainer.optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=total_iters,
        )
    elif args.lr_schedule == "cosine":
        # CosineAnnealingLR decays LR following a cosine curve (T6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer,
            T_max=total_iters,
            eta_min=0.0,
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

    # ── Opponent pool (T3) ────────────────────────────────────────
    pool = None
    if args.opponent_pool > 0:
        pool_dir = os.path.join(args.checkpoint_dir, "opponent_pool")
        pool = OpponentPool(capacity=args.opponent_pool, save_dir=pool_dir)
        if args.resume:
            pool.load_from_disk()
        log_step(total_steps, {"msg": f"opponent pool enabled (capacity={args.opponent_pool}, loaded={len(pool)})"})

    if replay_buffer is not None:
        log_step(total_steps, {"msg": f"replay buffer enabled (capacity={args.replay_buffer})"})



    last_eval = total_steps
    best_win_rate = 0.0
    t0 = time.time()

    # ── Train ─────────────────────────────────────────────────────
    while total_steps < args.total_steps:
        # Stage 2/4: skip curriculum, use fixed dense reward to avoid
        # value-head-frozen PPO collapse on reward distribution shift.
        if args.train_stage == 2:
            phase, dense_weight = 1, 1.0
        else:
            phase, dense_weight = get_curriculum_phase(
                total_steps, args.phase1_steps, args.phase2_steps)

        # Decide opponent for this rollout (T3)
        opponent_model = None
        if pool is not None and len(pool) > 0 and random.random() < 0.5:
            state_dict = pool.sample()
            if state_dict is not None:
                opponent_model = BattleNet(
                    num_experts=args.num_experts if args.use_moe else 0,
                    moe_hidden_dim=args.moe_hidden_dim,
                    routing_topk=args.routing_topk,
                )
                opponent_model.load_state_dict(state_dict)
                opponent_model.to(args.device)
                opponent_model.eval()

        # Select config for this rollout (T5)
        if multi_config:
            idx = random.randrange(len(configs))
            rollout_config = configs[idx]
            rollout_config_name = config_names[idx]
        else:
            idx = 0
            rollout_config = None
            rollout_config_name = config_names[0]

        # Stage 2: activate matching expert for this config
        if args.train_stage == 2 and model.moe is not None:
            expert_idx = idx % model.num_experts
            model.set_active_expert(expert_idx)

        info = trainer.train_step(
            num_steps=args.rollout_steps,
            reward_phase=phase,
            dense_weight=dense_weight,
            opponent_model=opponent_model,
            env_config=rollout_config,
        )
        total_steps += info.pop("steps")

        info["phase"] = phase
        info["dense_w"] = round(dense_weight, 3)
        info["elapsed"] = round(time.time() - t0, 1)
        if multi_config:
            info["config"] = rollout_config_name

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
            writer.add_scalar("train/diversity_loss",
                              info.get("diversity_loss", 0.0), total_steps)
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

            # Eval across all training configs (T5)
            per_config_wr = {}
            total_wins = 0
            total_games = 0
            for i, cfg in enumerate(configs):
                eval_info = eval_vs_classic(
                    cfg, agent_fn,
                    learning_team=0, games=args.eval_games, seed=42)
                per_config_wr[config_names[i]] = eval_info["win_rate"]
                total_wins += eval_info["wins"]
                total_games += eval_info["games"]

                if writer is not None:
                    writer.add_scalar(
                        f"eval/{config_names[i]}/win_rate",
                        eval_info["win_rate"], total_steps)

            avg_win_rate = total_wins / total_games if total_games > 0 else 0.0
            model.train()

            eval_log = {
                "win_rate": round(avg_win_rate, 4),
                "configs": {k: round(v, 4) for k, v in per_config_wr.items()},
                "avg_rounds": eval_info["avg_rounds"],
            }

            # T9c: log router weight distribution using real game states
            if model.moe is not None:
                from ai.env import BattleEnv
                with torch.no_grad():
                    all_features = []
                    for cfg in configs:
                        env = BattleEnv(cfg)
                        obs, _ = env.reset(seed=123)
                        for _ in range(16):
                            grid_t = torch.tensor(
                                obs["grid"], dtype=torch.float32
                            ).unsqueeze(0).to(args.device)
                            gvec_t = torch.tensor(
                                obs["global"], dtype=torch.float32
                            ).unsqueeze(0).to(args.device)
                            feat = model.extract_bottleneck(grid_t, gvec_t)
                            all_features.append(feat)
                            legal = [a for a in range(ACTION_DIM)
                                     if obs["mask"][a]]
                            if legal:
                                obs, _, done, _, _ = env.step(
                                    random.choice(legal))
                            if done:
                                obs, _ = env.reset()
                    features = torch.cat(all_features, dim=0)
                    _, _, rw = model.moe(features)
                    mean_w = rw.mean(dim=0).tolist()
                eval_log["router_weights"] = [round(w, 4) for w in mean_w]
                if writer is not None:
                    for ei, w in enumerate(mean_w):
                        writer.add_scalar(
                            f"router/expert_{ei}_weight", w, total_steps)

            log_eval(total_steps, eval_log)

            if writer is not None:
                writer.add_scalar("eval/win_rate", avg_win_rate,
                                  total_steps)
                writer.add_scalar("eval/avg_rounds", eval_info["avg_rounds"],
                                  total_steps)

            ckpt_path = os.path.join(
                args.checkpoint_dir, f"checkpoint_{total_steps}.pt")
            save_checkpoint(trainer, total_steps, ckpt_path)
            last_eval = total_steps

            # Add to opponent pool (T3)
            if pool is not None:
                pool.add(model.state_dict(), total_steps)

            # Track best checkpoint by average win rate (T5)
            if avg_win_rate > best_win_rate:
                best_win_rate = avg_win_rate
                best_path = os.path.join(args.checkpoint_dir, "best.pt")
                save_checkpoint(trainer, total_steps, best_path)

            # TensorBoard best win rate
            if writer is not None:
                writer.add_scalar("eval/best_win_rate", best_win_rate,
                                  total_steps)

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

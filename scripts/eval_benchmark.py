#!/usr/bin/env python3
"""T1+T7 — Benchmark evaluation framework for fheroes2 battle DeepAI.

Loads a trained checkpoint and evaluates against ClassicAI across multiple
battle configurations, reporting win rates with confidence intervals and
pass/fail against target thresholds.

Supports two modes:
  - Default: auto-discover all ``configs/*.json`` training configs
  - Legacy:  ``--legacy-4`` to use the original 4-config T6 benchmark

Usage::

    # Auto-discover all training configs
    python scripts/eval_benchmark.py checkpoints/final.pt

    # Legacy T6 benchmark (4 configs)
    python scripts/eval_benchmark.py checkpoints/final.pt --legacy-4

    # Custom number of games per config
    python scripts/eval_benchmark.py checkpoints/final.pt --games 100

    # GPU evaluation
    python scripts/eval_benchmark.py checkpoints/final.pt --device cuda

    # Save results to JSON
    python scripts/eval_benchmark.py checkpoints/final.pt --json results.json
"""

import argparse
import json
import math
import os
import sys
import time

# Add project root to import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.deep.model import BattleNet
from ai.deep.pipeline import load_battle_config
from ai.deep.player import make_agent_fn
from ai.self_play import eval_vs_classic


# ── Legacy T6 benchmark configs (4 configs) ────────────────────────

LEGACY_BENCHMARK_CONFIGS = [
    ("configs/example.json", "Mirror Melee", 0.50),
    ("configs/even_clash.json", "Asymmetric w/ Heroes", 0.40),
    ("configs/mage_duel.json", "Spell-Heavy", 0.30),
    ("configs/dragon_battle.json", "Tier-7 Units", 0.20),
]

# Default target win rate for auto-discovered configs
DEFAULT_TARGET_WIN_RATE = 0.30


# ── Auto-discovery ─────────────────────────────────────────────────

# Files in configs/ that are NOT training configs (test/dev/validation)
_EXCLUDE_CONFIGS = {
    "validation_results.json",
    "ability_showcase.json",
    "ranged_fest.json",
    "siege_wizards.json",
    "spell_duel.json",
    "spell_mage_duel.json",
    "wide_melee.json",
}


def discover_training_configs(configs_dir: str = "configs") -> list:
    """Auto-discover training config files from configs/ directory.

    Returns list of (config_path, display_name, target_win_rate).
    Excludes known test/dev configs.
    """
    results = []
    if not os.path.isdir(configs_dir):
        return results

    for fname in sorted(os.listdir(configs_dir)):
        if not fname.endswith(".json"):
            continue
        if fname in _EXCLUDE_CONFIGS:
            continue

        path = os.path.join(configs_dir, fname)
        # Generate display name from filename: "melee_brawl" -> "Melee Brawl"
        display = fname.replace(".json", "").replace("_", " ").title()
        results.append((path, display, DEFAULT_TARGET_WIN_RATE))

    return results


# ── Statistics ─────────────────────────────────────────────────────


def wilson_interval(wins: int, n: int, z: float = 1.96):
    """Wilson score 95% confidence interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


# ── Model loading ─────────────────────────────────────────────────


def load_model(checkpoint_path: str, device: str = "cpu") -> BattleNet:
    """Load BattleNet from a checkpoint file."""
    import torch

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model"]
    num_experts = 0
    moe_hidden_dim = 384
    if any(k.startswith("moe.") for k in state):
        num_experts = sum(
            1 for k in state
            if k.startswith("moe.experts.") and k.endswith(".linear1.weight")
        )
    model = BattleNet(
        num_experts=num_experts,
        moe_hidden_dim=moe_hidden_dim if num_experts > 0 else 384,
    )
    model.load_state_dict(state)
    model.to(device).eval()
    return model


# ── Core evaluation ───────────────────────────────────────────────


def run_benchmark(
    model: BattleNet,
    games: int = 50,
    device: str = "cpu",
    seed: int = 42,
    benchmark_configs: list = None,
) -> list:
    """Run all benchmark configs against ClassicAI.

    Returns a list of result dicts, one per config.
    """
    if benchmark_configs is None:
        benchmark_configs = discover_training_configs()

    results = []

    for config_path, name, target in benchmark_configs:
        env_config = load_battle_config(config_path)
        agent_fn = make_agent_fn(model, device=device)

        t0 = time.time()
        eval_info = eval_vs_classic(
            env_config, agent_fn,
            learning_team=0, games=games, seed=seed)
        elapsed = round(time.time() - t0, 1)

        p, lo, hi = wilson_interval(eval_info["wins"], games)

        results.append({
            "config": config_path,
            "name": name,
            "target": target,
            "wins": eval_info["wins"],
            "games": games,
            "win_rate": round(p, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "avg_rounds": round(eval_info["avg_rounds"], 2),
            "elapsed": elapsed,
            "pass": p >= target,
        })

    return results


# ── Output formatting ─────────────────────────────────────────────


def format_table(results: list) -> str:
    """Format benchmark results as a readable table."""
    lines = []
    lines.append(f"{'Config':<30} {'Wins':>5} {'Rate':>7} {'95% CI':>14} "
                 f"{'Target':>7} {'Pass':>5} {'Time':>6}")
    lines.append("-" * 82)

    total_pass = 0
    for r in results:
        ci = f"[{r['ci95'][0]*100:.1f}%, {r['ci95'][1]*100:.1f}%]"
        mark = "Y" if r["pass"] else "N"
        lines.append(
            f"{r['name']:<30} {r['wins']:>3}/{r['games']:<2} "
            f"{r['win_rate']*100:>6.1f}% {ci:>14} "
            f"{r['target']*100:>6.0f}%  {mark:>4} {r['elapsed']:>4.1f}s"
        )
        if r["pass"]:
            total_pass += 1

    lines.append("-" * 82)
    lines.append(f"Passed: {total_pass}/{len(results)}")
    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Evaluate a trained DeepAI checkpoint against ClassicAI benchmarks")
    p.add_argument("checkpoint", help="Path to model checkpoint (.pt)")
    p.add_argument("--games", type=int, default=50,
                   help="Games per benchmark config (default: 50)")
    p.add_argument("--device", type=str, default="cpu",
                   help="torch device (default: cpu)")
    p.add_argument("--seed", type=int, default=42,
                   help="Base RNG seed (default: 42)")
    p.add_argument("--json", type=str, default=None,
                   help="Save results to JSON file")
    p.add_argument("--legacy-4", action="store_true",
                   help="Use legacy T6 benchmark (4 configs only)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if not os.path.isfile(args.checkpoint):
        sys.exit(f"Checkpoint not found: {args.checkpoint}")

    # Select benchmark configs
    if args.legacy_4:
        benchmark_configs = LEGACY_BENCHMARK_CONFIGS
        mode = "legacy T6 (4 configs)"
    else:
        benchmark_configs = discover_training_configs()
        mode = f"auto-discovered ({len(benchmark_configs)} configs)"

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device=args.device)
    print(f"Model loaded ({model.count_parameters():,} params)")
    print(f"Running benchmark: {mode} × {args.games} games")
    print()

    results = run_benchmark(model, games=args.games, device=args.device,
                            seed=args.seed, benchmark_configs=benchmark_configs)

    # Print results table
    print(format_table(results))

    # Save JSON if requested
    if args.json:
        output = {
            "checkpoint": args.checkpoint,
            "games_per_config": args.games,
            "device": args.device,
            "results": results,
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.json}")

    # Exit code: 0 if all pass, 1 otherwise
    all_pass = all(r["pass"] for r in results)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

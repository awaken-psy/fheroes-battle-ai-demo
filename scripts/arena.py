"""Arena — batch AI-vs-AI self-play for measuring AI strength.

Runs many headless battles and reports win rate with a confidence interval,
so "fidelity to the original" can finally be checked by play rather than by
eyeballing code.

Examples:
    # 500 mirror games (identical armies both sides) — should be ~50%
    python scripts/arena.py --preset Balanced --games 500 --mirror

    # asymmetric matchup from a config file
    python scripts/arena.py --config configs/example.json --games 200

    # reproducible run + machine-readable output
    python scripts/arena.py --preset Balanced --games 500 --mirror --seed 0 --json out.json
"""

import argparse
import json
import math
import os
import sys

# allow running as `python scripts/arena.py` from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from engine.unit import Unit
from headless import simulate

# A placement spec is a list of (type_name, team, col, row) tuples.


def spec_from_preset(name: str):
    if name not in config.PRESETS:
        sys.exit(f"Unknown preset '{name}'. Available: {', '.join(config.PRESETS)}")
    spec = []
    for team, placements in config.PRESETS[name].items():
        for type_name, col, row in placements:
            spec.append((type_name, team, col, row))
    return spec


def spec_from_config(path: str):
    with open(path, encoding="utf-8") as f:
        placements = json.load(f)
    return [(p["type"], p["team"], p["col"], p["row"]) for p in placements]


def mirror_spec(spec):
    """Replace team 1 with a board-mirrored copy of team 0.

    Both sides end up with identical armies, so a fair AI/engine should win
    ~50%. Any systematic deviation reveals a side or first-move bias.
    """
    cols = config.GRID_COLS
    team0 = [(t, 0, c, r) for (t, team, c, r) in spec if team == 0]
    if not team0:
        sys.exit("--mirror needs at least one team-0 unit to mirror.")
    mirrored = [(t, 1, cols - 1 - c, r) for (t, _, c, r) in team0]
    return team0 + mirrored


def build_units(spec):
    return [Unit.from_type(t, team, c, r) for (t, team, c, r) in spec]


def wilson_interval(wins: int, n: int, z: float = 1.96):
    """Wilson score 95% confidence interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def run(args):
    spec = (spec_from_config(args.config) if args.config
            else spec_from_preset(args.preset))
    if args.mirror:
        spec = mirror_spec(spec)

    default_hero = {"power": 3, "spell_points": 15}
    hero_configs = {0: default_hero if args.hero0 else None,
                    1: default_hero if args.hero1 else None}
    use_heroes = args.hero0 or args.hero1

    wins = {0: 0, 1: 0}
    early = 0
    total_rounds = 0
    for i in range(args.games):
        seed = (args.seed + i) if args.seed is not None else None
        # alternate initiative tie-break and attacker side to cancel any
        # first-move / attacker-retreat bias across the batch
        side = i % 2
        winner, rounds, ended_early = simulate(
            build_units(spec), seed=seed, first_team=side, attacker_team=side,
            hero_configs=hero_configs if use_heroes else None)
        wins[winner] += 1
        total_rounds += rounds
        if ended_early:
            early += 1

    n = args.games
    p, lo, hi = wilson_interval(wins[0], n)
    result = {
        "games": n,
        "source": args.config or f"preset:{args.preset}",
        "mirror": args.mirror,
        "seed": args.seed,
        "team0_wins": wins[0],
        "team1_wins": wins[1],
        "team0_winrate": round(p, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "ended_early": early,
        "avg_rounds": round(total_rounds / n, 2) if n else 0,
    }

    print(f"Games:        {n}")
    print(f"Source:       {result['source']}{'  (mirror)' if args.mirror else ''}")
    print(f"Team 0 wins:  {wins[0]}  ({p*100:.1f}%)")
    print(f"Team 1 wins:  {wins[1]}  ({wins[1]/n*100:.1f}%)")
    print(f"95% CI:       [{lo*100:.1f}%, {hi*100:.1f}%]")
    print(f"Ended early:  {early}  ({early/n*100:.1f}%)  (stalemate / round cap)")
    print(f"Avg rounds:   {result['avg_rounds']}")
    if args.mirror:
        # The odd-r board is not perfectly mirror-symmetric (a plain column
        # reflection is not an isometry) and moving first is a structural
        # disadvantage, so even a perfect AI won't hit exactly 50% on a
        # mirror. We gate on a wide "no gross side bias" band instead.
        fair = 0.40 <= p <= 0.60
        print(f"Mirror fairness (40-60% band): {'PASS' if fair else 'FAIL'}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote {args.json}")
    return result


def main():
    ap = argparse.ArgumentParser(description="Batch AI-vs-AI self-play arena.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--preset", default="Balanced",
                     help=f"preset name ({', '.join(config.PRESETS)})")
    src.add_argument("--config", help="path to a battle config JSON")
    ap.add_argument("--games", type=int, default=100, help="number of games")
    ap.add_argument("--mirror", action="store_true",
                    help="mirror team 0 onto team 1 (identical armies; expect ~50%%)")
    ap.add_argument("--seed", type=int, default=None,
                    help="base RNG seed for reproducibility (game i uses seed+i)")
    ap.add_argument("--hero0", action="store_true", help="give team 0 a default spellcasting hero")
    ap.add_argument("--hero1", action="store_true", help="give team 1 a default spellcasting hero")
    ap.add_argument("--json", help="also write the summary to this JSON file")
    run(ap.parse_args())


if __name__ == "__main__":
    main()

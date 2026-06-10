#!/usr/bin/env python3
"""AI validation batch — runs all presets & configs headless, reports summary.

Exercises the full ClassicAI decision path (A1–A4) across diverse scenarios:
  - Mirror fairness (identical armies → ~50%)
  - Asymmetric matchups (cross-faction, abilities, spells)
  - Siege battles (castle walls, moat, towers)
  - Wide/flying/archer unit handling
  - Spell casting (AOE, buff, debuff, control)
  - Special abilities (spell_caster, all_adjacent_attack, hp_drain, etc.)

Usage:
    python scripts/ai_validation.py              # default 200 games each
    python scripts/ai_validation.py --games 500  # more statistical power
    python scripts/ai_validation.py --json results.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.arena import run as arena_run


def build_args(preset=None, config=None, games=200, extra_flags=None):
    """Build a namespace that arena.run() expects."""
    args = argparse.Namespace(
        preset=preset, config=config, games=games,
        mirror=False, siege=False, seed=42,
        hero0=False, hero1=False, difficulty="Normal", json=None,
    )
    if extra_flags:
        for k, v in extra_flags.items():
            setattr(args, k, v)
    return args


SCENARIOS = [
    # ── Mirror fairness (core test: should be ~50%) ──
    {"name": "Mirror: Balanced",    "preset": "Balanced",    "mirror": True},
    {"name": "Mirror: Wide Clash",  "preset": "Wide Clash",  "mirror": True},
    {"name": "Mirror: Dragon Horde","preset": "Dragon Horde", "mirror": True},

    # ── Asymmetric preset matchups ──
    {"name": "Archer Defense",      "preset": "Archer Defense"},
    {"name": "Flyer Threat",        "preset": "Flyer Threat"},
    {"name": "Knight vs Barbarian", "preset": "Knight vs Barbarian"},
    {"name": "Clash of Titans",     "preset": "Clash of Titans"},
    {"name": "Sorceress vs Necro",  "preset": "Sorceress vs Necromancer"},
    {"name": "Wizard's Tower",      "preset": "Wizard's Tower"},

    # ── Siege ──
    {"name": "Siege: Assault",      "preset": "Siege: Assault", "siege": True},

    # ── Custom config scenarios (exercise A1–A4 features) ──
    {"name": "Spell Duel",          "config": "configs/spell_duel.json"},
    {"name": "Ability Showcase",    "config": "configs/ability_showcase.json"},
    {"name": "Wide Melee",          "config": "configs/wide_melee.json"},
    {"name": "Dragon Battle",       "config": "configs/dragon_battle.json"},
    {"name": "Ranged Fest",         "config": "configs/ranged_fest.json"},
    {"name": "Siege + Wizards",     "config": "configs/siege_wizards.json"},
]


def main():
    ap = argparse.ArgumentParser(description="Batch AI validation across all scenarios")
    ap.add_argument("--games", type=int, default=200, help="games per scenario")
    ap.add_argument("--json", default=None, help="write results to JSON file")
    cli = ap.parse_args()

    results = []
    total_games = 0
    total_time = 0.0
    pass_count = 0
    warn_count = 0

    print("=" * 72)
    print(f"  AI Validation Batch — {len(SCENARIOS)} scenarios × {cli.games} games")
    print("=" * 72)

    for sc in SCENARIOS:
        extra = {}
        if sc.get("mirror"):
            extra["mirror"] = True
        if sc.get("siege"):
            extra["siege"] = True

        if "config" in sc:
            args = build_args(config=sc["config"], games=cli.games, extra_flags=extra)
        else:
            args = build_args(preset=sc["preset"], games=cli.games, extra_flags=extra)

        t0 = time.time()
        res = arena_run(args)
        elapsed = time.time() - t0

        total_games += res["games"]
        total_time += elapsed

        # ── Classify result ──
        is_mirror = sc.get("mirror", False)
        wr = res["team0_winrate"]

        if is_mirror:
            fair = 0.40 <= wr <= 0.60
            status = "PASS" if fair else "WARN"
            detail = f"mirror fairness 40-60% band (got {wr*100:.1f}%)"
        else:
            # asymmetric: just check the battle ran to completion
            retreat_rate = res["retreats"] / res["games"] if res["games"] else 0
            early_rate = res["ended_early"] / res["games"] if res["games"] else 0
            if retreat_rate > 0.5:
                status = "WARN"
                detail = f"high retreat rate ({retreat_rate*100:.0f}%)"
            elif early_rate > 0.8:
                status = "WARN"
                detail = f"high stalemate/cap rate ({early_rate*100:.0f}%)"
            else:
                status = "PASS"
                detail = f"T0={res['team0_wins']} T1={res['team1_wins']} avg_rnd={res['avg_rounds']}"

        if status == "PASS":
            pass_count += 1
        else:
            warn_count += 1

        res["scenario"] = sc["name"]
        res["status"] = status
        res["detail"] = detail
        res["elapsed_s"] = round(elapsed, 2)
        results.append(res)

        tag = "✅" if status == "PASS" else "⚠️"
        print(f"  {tag} {sc['name']:<28s}  {detail}")
        print(f"     ({res['games']} games, {elapsed:.1f}s, CI [{res['ci95'][0]*100:.1f}%-{res['ci95'][1]*100:.1f}%])")
        print()

    # ── Summary ──
    print("=" * 72)
    print(f"  SUMMARY: {pass_count} PASS / {warn_count} WARN / {len(SCENARIOS)} total")
    print(f"  Games played: {total_games}  |  Time: {total_time:.1f}s  "
          f"({total_games/total_time:.0f} games/s)")
    print("=" * 72)

    if cli.json:
        with open(cli.json, "w", encoding="utf-8") as f:
            json.dump({"scenarios": results, "total_games": total_games,
                        "total_time_s": round(total_time, 2),
                        "pass": pass_count, "warn": warn_count}, f, indent=2)
        print(f"\n  Results written to {cli.json}")

    return 0 if warn_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

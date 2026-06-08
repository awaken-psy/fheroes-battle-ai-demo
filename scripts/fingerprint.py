"""Deterministic battle fingerprint — a regression safety net across refactors.

Runs every preset against a fixed set of seeds, records a detailed per-action
trace (AI intent + execution result, including damage numbers and positions),
and prints one hash over everything. A matching hash before and after a change
proves **zero behavior change** for the covered single-hex, open-field battles.

This is the safety net for the M5b/M6 rule-replication work: structural changes
(wide units, siege) must not alter any of these existing single-hex battles, so
this hash must stay identical until M6a deliberately rebuilds the baseline when
switching to fheroes2's exact monster stats.

Usage:
    python scripts/fingerprint.py          # print the hash
    python scripts/fingerprint.py --trace  # also dump the full trace to stderr
"""

import hashlib
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from engine.hex_grid import HexGrid  # noqa: E402
from engine.unit import Unit  # noqa: E402
from engine.battle_state import BattleState  # noqa: E402
from ai import create_ai  # noqa: E402
from headless import _take_unit_turn  # noqa: E402

SEEDS = list(range(10))

# The baseline scenarios are pinned by name so that adding new presets (e.g. a
# wide-unit demo in M5b) never perturbs the safety-net hash. These three are the
# single-hex, open-field battles whose behavior must not change.
BASELINE_PRESETS = ["Archer Defense", "Balanced", "Flyer Threat"]


def _build_units(preset):
    units = []
    for team, placements in preset.items():
        for name, col, row in placements:
            units.append(Unit.from_type(name, team, col, row))
    return units


def run_traced(preset_name, seed):
    """Run one deterministic battle, returning its full text trace."""
    random.seed(seed)
    units = _build_units(config.PRESETS[preset_name])
    battle = BattleState(HexGrid(), units)
    ai = create_ai("classic")

    trace = [f"=== {preset_name} seed={seed} ==="]
    while not battle.is_over():
        order = battle.turn_order()
        if not order:
            break
        battle.start_round()
        trace.append(f"R{battle.round_num}")
        for unit in order:
            if not unit.is_alive:
                continue
            if battle.is_over():
                break
            _take_unit_turn(
                battle, ai, unit,
                log=lambda desc, res: trace.append(f"  {desc} || {res}"),
            )
        if battle._retreated is not None:
            break
    trace.append(f"WINNER={battle.winner()} ROUNDS={battle.round_num}")
    return "\n".join(trace)


def fingerprint():
    chunks = []
    for preset_name in BASELINE_PRESETS:
        for seed in SEEDS:
            chunks.append(run_traced(preset_name, seed))
    full = "\n".join(chunks)
    return hashlib.blake2b(full.encode(), digest_size=8).hexdigest(), full


def main():
    digest, full = fingerprint()
    if "--trace" in sys.argv:
        print(full, file=sys.stderr)
    n = len(BASELINE_PRESETS) * len(SEEDS)
    print(f"FINGERPRINT {digest}  ({n} battles: "
          f"{len(BASELINE_PRESETS)} presets x {len(SEEDS)} seeds)")


if __name__ == "__main__":
    main()

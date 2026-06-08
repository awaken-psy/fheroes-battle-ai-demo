"""Headless battle runner — no GUI, reads config, writes log.

Called from main.py when config files are provided as arguments.
"""

import json
import os
import random
import sys
from typing import List, Optional, Tuple

from engine.hex_grid import HexGrid
from engine.unit import Unit
from engine.battle_state import BattleState
from engine.battle_logger import BattleLogger
from ai.planner import BattleAI


def simulate(units: List[Unit], seed: Optional[int] = None,
             first_team: int = 0, attacker_team: int = 0) -> Tuple[int, int, bool]:
    """Run one battle to completion with no logging or IO.

    Returns ``(winner_team, rounds, ended_early)`` where ``ended_early`` means
    the battle stopped on a stalemate or the absolute round cap rather than by
    elimination. Reused by ``scripts/arena.py``; the given ``units`` are
    mutated, so pass a fresh list per game.
    """
    if seed is not None:
        random.seed(seed)
    grid = HexGrid()
    battle = BattleState(grid, units, first_team=first_team,
                         attacker_team=attacker_team)
    ai = BattleAI()
    while not battle.is_over():
        order = battle.turn_order()
        if not order:
            break
        battle.start_round()
        for unit in order:
            if not unit.is_alive:
                continue
            if battle.is_over():
                break
            battle.execute(ai.decide(battle, unit)[0])
    ended_early = battle.is_stalemate() or battle.round_num >= BattleState.MAX_ROUNDS
    return battle.winner(), battle.round_num, ended_early


def run_battle(config_path: str, output_path: str | None = None) -> str:
    """Run a battle from config file, return the log file path."""
    with open(config_path, encoding="utf-8") as f:
        placements = json.load(f)

    # build units
    units = []
    for p in placements:
        try:
            units.append(Unit.from_type(p["type"], p["team"], p["col"], p["row"]))
        except KeyError as e:
            print(f"Invalid entry in {config_path}: missing {e}", file=sys.stderr)
            sys.exit(1)

    # run battle
    grid = HexGrid()
    battle = BattleState(grid, units)
    ai = BattleAI()
    logger = BattleLogger()
    logger.start(units)

    while not battle.is_over():
        order = battle.turn_order()
        if not order:
            break
        battle.start_round()
        logger.round_start(battle.round_num)
        for unit in order:
            if not unit.is_alive:
                continue
            if battle.is_over():
                break
            action, desc = ai.decide(battle, unit)
            result = battle.execute(action)
            logger.action(desc, result["desc"])

    timeout = battle.round_num >= BattleState.MAX_ROUNDS
    logger.end(battle.winner(), battle.round_num, timeout=timeout)

    # move log if custom output path given
    if output_path and logger._path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        os.replace(logger._path, output_path)
        return output_path

    return logger._path

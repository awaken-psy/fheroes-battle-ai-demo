"""Headless battle runner — no GUI, reads config, writes log.

Called from main.py when config files are provided as arguments.
"""

import json
import os
import sys

from engine.hex_grid import HexGrid
from engine.unit import Unit
from engine.battle_state import BattleState
from engine.battle_logger import BattleLogger
from ai.planner import BattleAI


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
    grid = HexGrid(scale=1.0)
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

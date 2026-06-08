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
             first_team: int = 0, attacker_team: int = 0,
             hero_configs: Optional[dict] = None, difficulty: str = "Normal",
             morale: Optional[dict] = None, luck: Optional[dict] = None
             ) -> Tuple[int, int, bool, str]:
    """Run one battle to completion with no logging or IO.

    Returns ``(winner_team, rounds, ended_early, end_reason)`` where end_reason
    is one of "elim" / "retreat" / "stalemate" / "cap". ``hero_configs`` is an
    optional {team: dict} of hero configs. Reused by ``scripts/arena.py``; the
    given ``units`` are mutated, so pass a fresh list per game.
    """
    from engine.hero import Hero

    if seed is not None:
        random.seed(seed)
    heroes = {0: None, 1: None}
    if hero_configs:
        heroes = {0: Hero.from_config(hero_configs.get(0)),
                  1: Hero.from_config(hero_configs.get(1))}
    grid = HexGrid()
    battle = BattleState(grid, units, first_team=first_team,
                         attacker_team=attacker_team, heroes=heroes,
                         difficulty=difficulty, morale=morale, luck=luck)
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
            _take_unit_turn(battle, ai, unit)
        if battle._retreated is not None:
            break

    if battle._retreated is not None:
        reason = "retreat"
    elif battle.is_stalemate():
        reason = "stalemate"
    elif battle.round_num >= BattleState.MAX_ROUNDS:
        reason = "cap"
    else:
        reason = "elim"
    return battle.winner(), battle.round_num, reason != "elim", reason


def _take_unit_turn(battle, ai, unit, log=None):
    """One unit's full activation: retreat / morale / cast / act.

    Shared by simulate and run_battle. ``log`` is an optional callable
    (ai_desc, result_desc) -> None for logging.
    """
    def execute(action, desc):
        result = battle.execute(action)
        if log is not None:
            log(desc, result["desc"])

    # Step 2: retreat (with a farewell damage spell)
    retreat = ai.check_retreat(battle, unit)
    if retreat is not None:
        farewell, retreat_action = retreat
        if farewell is not None:
            execute(farewell[0], farewell[1])
        execute(retreat_action, "[RETREAT]")
        return

    # Morale: bad -> skip the turn; good -> an extra action after acting.
    morale = battle.roll_morale(unit.team)
    if morale < 0:
        return
    for _ in range(2 if morale > 0 else 1):
        if not unit.is_alive or battle.is_over():
            break
        # Step 3: hero spellcast (once per round)
        cast = ai.maybe_cast_spell(battle, unit)
        if cast is not None:
            execute(cast[0], cast[1])
            if battle.is_over() or not unit.is_alive:
                break
        # Step 4: the unit's own action
        action, desc = ai.decide(battle, unit)
        execute(action, desc)


def run_battle(config_path: str, output_path: str | None = None) -> str:
    """Run a battle from config file, return the log file path."""
    from engine.hero import Hero

    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)

    # Config is either a bare list of placements (legacy) or a dict with
    # "units" and optional "heroes": {"0": {...}, "1": {...}}.
    if isinstance(data, dict):
        placements = data.get("units", [])
        hero_cfg = data.get("heroes", {})
    else:
        placements = data
        hero_cfg = {}

    units = []
    for p in placements:
        try:
            units.append(Unit.from_type(p["type"], p["team"], p["col"], p["row"]))
        except KeyError as e:
            print(f"Invalid entry in {config_path}: missing {e}", file=sys.stderr)
            sys.exit(1)

    heroes = {0: Hero.from_config(hero_cfg.get("0") or hero_cfg.get(0)),
              1: Hero.from_config(hero_cfg.get("1") or hero_cfg.get(1))}

    cfg = data if isinstance(data, dict) else {}
    morale = {int(k): v for k, v in cfg.get("morale", {}).items()} or None
    luck = {int(k): v for k, v in cfg.get("luck", {}).items()} or None

    # run battle
    grid = HexGrid()
    battle = BattleState(grid, units, heroes=heroes,
                         difficulty=cfg.get("difficulty", "Normal"),
                         morale=morale, luck=luck)
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
            _take_unit_turn(battle, ai, unit, log=logger.action)
        if battle._retreated is not None:
            break

    timeout = battle.round_num >= BattleState.MAX_ROUNDS
    logger.end(battle.winner(), battle.round_num, timeout=timeout)

    # move log if custom output path given
    if output_path and logger._path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        os.replace(logger._path, output_path)
        return output_path

    return logger._path

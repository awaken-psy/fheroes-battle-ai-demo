"""Action scoring heuristics — threat evaluation and position value.

Maps to scoring functions scattered through ai_battle.cpp.
"""

from typing import List

from engine.unit import Unit
from engine.battle_state import BattleState


def threat(attacker: Unit, defender: Unit) -> float:
    """Simplified evaluateThreatForUnit() — scores a target's danger."""
    base = defender.strength
    if defender.is_archer:
        base *= 1.2
    if defender.speed > attacker.speed:
        base *= 1.1
    if defender.is_flying:
        base *= 1.3
    return base


def pos_value(battle: BattleState, unit: Unit,
              pos: tuple, enemies: List[Unit]) -> float:
    """evaluatePotentialAttackPositions() — simplified."""
    val = 0.0
    for e in enemies:
        if battle.grid.distance(pos, e.pos) == 1:
            t = threat(unit, e)
            if e.is_archer:
                val += t
            else:
                val = max(val, t)
    return val

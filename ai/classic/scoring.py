"""Action scoring heuristics — threat evaluation and position value.

Maps to scoring functions in fheroes2's battle_troop.cpp / ai_battle.cpp.
"""

from typing import List

from engine.unit import Unit
from engine.battle_state import BattleState


def threat(battle: BattleState, attacker: Unit, defender: Unit) -> float:
    """How valuable is `defender` as a target for `attacker`.

    Core of fheroes2's Unit::evaluateThreatForUnit (battle_troop.cpp:1007):
    the expected damage the attacker deals, discounted by how far it has to
    travel to land the hit, then scaled up by the attacker's special abilities.
    """
    dmg = float(battle.expected_damage(attacker, defender, ranged=attacker.is_archer))

    # Distance modifier: shooters and flyers reach anywhere; melee units that
    # cannot strike this turn (distance beyond speed+1) are discounted.
    if attacker.is_flying or attacker.is_archer:
        dist_mod = 1.0
    else:
        attack_range = attacker.speed + 1
        dist = battle.grid.distance(attacker.pos, defender.pos)
        dist_mod = 1.0 if dist <= attack_range else 1.5 * dist / attacker.speed

    threat_value = dmg / dist_mod

    # Special-ability multipliers (evaluateThreatForUnit ability terms).
    if attacker.has_ability("death_gaze"):   # enemy-halving (legacy)
        threat_value *= 2
    if attacker.has_ability("enemy_halving"):  # enemy-halving (Genie)
        threat_value *= 2
    if attacker.has_ability("hp_drain"):
        threat_value *= 1.3

    # Reduce the priority of enemies that have already got their turn
    # this round — they can't act again until next round.
    # fheroes2: battle_troop.cpp evaluateThreatForUnit, TR_MOVED check.
    if defender._acted:
        threat_value /= 1.25

    return threat_value


def pos_value(battle: BattleState, unit: Unit,
              pos: tuple, enemies: List[Unit]) -> float:
    """Value of standing at `pos` — the melee damage reachable from there.

    Archers adjacent to `pos` are summed (every shooter silenced is worth it);
    for other adjacent enemies we take the best single target.
    """
    val = 0.0
    for e in enemies:
        if battle.grid.distance(pos, e.pos) == 1:
            d = float(battle.expected_damage(unit, e, ranged=False))
            if e.is_archer:
                val += d
            else:
                val = max(val, d)
    return val

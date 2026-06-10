"""Action scoring heuristics — threat evaluation and position value.

Maps to scoring functions in fheroes2's battle_troop.cpp / ai_battle.cpp.
"""

from typing import Dict, List, Optional, Set, Tuple

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


# ────────────────────────────────────────────────────────────────
#  Attack position evaluation — evaluatePotentialAttackPositions
#  and optimalAttackValue — ai_battle.cpp:98-270
# ────────────────────────────────────────────────────────────────

def splash_value(battle: BattleState, attacker: Unit, target: Unit,
                 attack_pos: Tuple[int, int]) -> float:
    """doubleCellAttackValue — ai_battle.cpp:98.

    When a wide or two_cell attacker strikes `target` from `attack_pos`,
    the cell *behind* the target (relative to the attack direction) may
    contain a secondary target.  Return its threat value if so.
    """
    grid = battle.grid
    # Direction from attack_pos toward target
    dx = target.col - attack_pos[0]
    dy = target.row - attack_pos[1]
    # "Behind" the target: one more step in the same direction.
    # Hex grids have 6 directions so we use a simpler approach:
    # find the neighbor of target that is furthest from attack_pos
    # and not target's own cells.
    target_cells = set(target.occupied_cells())
    best_behind = None
    best_dist = -1
    for nb in grid.neighbors(*target.pos):
        if nb in target_cells or nb == attack_pos:
            continue
        d = grid.distance(attack_pos, nb)
        if d > best_dist:
            best_dist = d
            best_behind = nb
    if best_behind is None:
        return 0.0
    # Check tail cell for wide targets too
    if target.is_wide:
        for nb in grid.neighbors(*target.tail_cell):
            if nb in target_cells or nb == attack_pos:
                continue
            d = grid.distance(attack_pos, nb)
            if d > best_dist:
                best_dist = d
                best_behind = nb
    # Is there a unit at the behind cell?
    for u in battle.alive():
        if u is attacker or u is target:
            continue
        if best_behind in u.occupied_cells():
            return threat(battle, attacker, u)
    return 0.0


def optimal_attack_value(battle: BattleState, attacker: Unit,
                         target: Unit, attack_pos: Tuple[int, int],
                         enemies: List[Unit]) -> float:
    """optimalAttackValue — ai_battle.cpp:157.

    Value of attacking `target` from `attack_pos`, including:
    - base threat of target
    - all_adjacent_attack: sum all adjacent enemy threats
    - two_cell_melee / wide attacker: splash damage from behind target
    """
    # allAdjacentCellsAttack: Hydra — sum threats of all adjacent enemies
    if attacker.has_ability("all_adjacent_attack"):
        total = 0.0
        adj_cells = set()
        for nb in battle.grid.neighbors(*attack_pos):
            adj_cells.add(nb)
        if attacker.is_wide:
            for nb in battle.grid.neighbors(*attacker.tail_cell):
                adj_cells.add(nb)
        seen = set()
        for e in enemies:
            if e is attacker:
                continue
            for ec in e.occupied_cells():
                if ec in adj_cells and id(e) not in seen:
                    seen.add(id(e))
                    total += threat(battle, attacker, e)
        return total

    # Base: threat of the primary target
    val = threat(battle, attacker, target)

    # Double-cell attack splash (Cavalry/Champion/Wolf/Cyclops/etc.)
    if attacker.has_ability("two_cell_melee") or attacker.is_wide:
        val += splash_value(battle, attacker, target, attack_pos)

    return val


def build_attack_position_map(
    battle: BattleState, unit: Unit, enemies: List[Unit],
    reachable: Set[Tuple[int, int]]
) -> Dict[Tuple[int, int], float]:
    """evaluatePotentialAttackPositions — ai_battle.cpp:202.

    Pre-compute a mapping from every reachable attack position to its
    aggregate attack value.  For each enemy, find positions adjacent to
    it that are in `reachable`; compute the attack value; merge across
    enemies (archer enemies → sum, non-archer → max).
    """
    grid = battle.grid
    result: Dict[Tuple[int, int], float] = {}

    # Sort: non-archers first, archers second (archers always add to value).
    sorted_enemies = sorted(enemies,
                            key=lambda e: (not e.is_archer, id(e)))

    for e in sorted_enemies:
        # Positions adjacent to the enemy that the unit can reach.
        seen_pos: set = set()
        # Wide attackers can be 2 tiles from target and still be "adjacent".
        max_reach = 2 if unit.is_wide else 1
        for target_cell in e.occupied_cells():
            for nb in grid.neighbors(*target_cell):
                if nb in seen_pos:
                    continue
                if nb in battle._move_occupied(unit):
                    continue
                # Must be adjacent (dist 1) to some part of the target.
                dist = min(grid.distance(nb, tc) for tc in e.occupied_cells())
                if dist != 1:
                    continue
                if nb not in reachable and nb != unit.pos:
                    continue
                seen_pos.add(nb)
                a_val = optimal_attack_value(battle, unit, e, nb, enemies)
                if nb in result:
                    if e.is_archer:
                        result[nb] += a_val
                    else:
                        result[nb] = max(result[nb], a_val)
                else:
                    result[nb] = a_val
    return result


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

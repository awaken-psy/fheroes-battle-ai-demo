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

    # §8.1-#6: SPELL_CASTER threat — battle_troop.cpp:1093-1125.
    # Units with spell_caster ability (e.g. Genie, Unicorn) add probabilistic
    # spell damage to threat based on the triggered spell type.
    if attacker.has_ability("spell_caster"):
        params = attacker.ability_params.get("spell_caster", {})
        spell_name = params.get("spell", "")
        chance = params.get("chance", 20)
        # Defender's average damage (used to value disabling them)
        def_avg_dmg = (defender.damage_min + defender.damage_max) / 2.0
        if spell_name in ("Blind", "Paralyze", "Petrify"):
            # Blind/Paralyze/Petrify: probability × defender avg damage
            # C++ checks AllowApplySpell but we skip immunity check for simplicity
            threat_value += def_avg_dmg * chance / 100.0
        elif spell_name == "Curse":
            # Curse: lower impact, divided by 10
            threat_value += def_avg_dmg * chance / 100.0 / 10.0
        # Dispel: TODO (C++ also has TODO here)

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
    """doubleCellAttackValue + optimalAttackVector — ai_battle.cpp:98-155.

    Enumerate all valid (attackCell, targetCell) pairs where they are
    adjacent, compute the cell *behind* the target for each attack
    direction using ``grid.cell_behind``, and return the highest splash
    value from secondary targets hit by the two-cell attack.
    """
    grid = battle.grid
    best_splash = 0.0

    # Attack cells: head and (for wide attackers) tail at attack_pos
    attack_cells = [attack_pos]
    if attacker.is_wide:
        tail_offset = -1 if attacker.team == 0 else 1
        tail = (attack_pos[0] + tail_offset, attack_pos[1])
        if grid.is_valid(*tail):
            attack_cells.append(tail)

    # Target cells: all occupied cells (head + tail for wide targets)
    target_cells = list(target.occupied_cells())

    for ac in attack_cells:
        for tc in target_cells:
            if grid.distance(ac, tc) != 1:
                continue
            # Cell behind the target from this attack direction
            behind = grid.cell_behind(ac, tc)
            if behind is None:
                continue
            # Check for a secondary unit at the behind cell
            for u in battle.alive():
                if u is attacker or u is target:
                    continue
                if behind in u.occupied_cells():
                    best_splash = max(best_splash,
                                      threat(battle, attacker, u))
    return best_splash


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
                    # §8.2-#5: allAdjacentAttack yields the same total value
                    # regardless of which enemy triggered the evaluation,
                    # so no merge is needed (C++ asserts equality).
                    if unit.has_ability("all_adjacent_attack"):
                        pass  # value already correct
                    elif e.is_archer:
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

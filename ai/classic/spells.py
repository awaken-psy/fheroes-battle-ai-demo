"""Spell AI — pick the best spell for a hero to cast this turn.

Reimplemented from fheroes2's selectBestSpell / spell*Value
(ai_battle_spell.cpp). The hero casts at most one spell per round, and only
if its scored value clears a threshold scaled by the army balance.
"""

import math
from typing import List, Optional, Tuple

from engine.unit import Unit
from engine.battle_state import BattleState
from engine.hero import Hero
from engine.spells import (Spell, DAMAGE, AOE, BUFF, DEBUFF, CONTROL,
                            DISPEL, CURE, UTILITY, spell_damage)

from .evaluation import AIState

# fheroes2 Speed::INSTANT — maximum battlefield speed (HoMM2 range 1-11).
SPEED_INSTANT = 11


def select_best_spell(battle: BattleState, team: int, s: AIState,
                      retreating: bool = False
                      ) -> Optional[Tuple[Spell, Unit]]:
    """Return (spell, target) for the hero of `team`, or None.

    selectBestSpell — ai_battle_spell.cpp:71. When ``retreating`` is set this is
    the farewell cast: only damage spells, no threshold and no cost discount.
    """
    hero: Hero = battle.heroes.get(team)
    if hero is None:
        return None

    my_str = max(s.my_army, 1.0)
    enemy_str = max(s.enemy_army, 1.0)

    # Threshold: 0.04 of (myStr^2 / enemyStr) — ~20% of a single unit when even.
    threshold = my_str * my_str / enemy_str * 0.04
    if s.enemy_shooters / enemy_str > 0.5:
        threshold *= 0.5
    if hero.spell_points * 2 < hero.max_spell_points:
        threshold *= 2

    friendly = battle.alive(team)
    enemies = battle.alive(1 - team)

    best: Optional[Tuple[Spell, Unit]] = None
    best_value = 0.0
    for spell in hero.spellbook:
        if not hero.can_cast(spell):
            continue
        if retreating and spell.kind not in (DAMAGE, AOE):
            continue  # farewell cast: only damage
        value, target = _spell_value(battle, hero, spell, friendly, enemies, s)
        if target is None or value <= 0:
            continue
        if retreating:
            if value > best_value:
                best_value, best = value, (spell, target)
            continue
        # Diminish by spell-point cost: sqrt so high-level spells aren't linear.
        spv = value / math.sqrt(spell.cost / 3.0)
        if spv > best_value and spv > threshold:
            best_value, best = spv, (spell, target)
    return best


# ── dispatch ─────────────────────────────────────────────────────

def _spell_value(battle: BattleState, hero: Hero, spell: Spell,
                 friendly: List[Unit], enemies: List[Unit], s: AIState
                 ) -> Tuple[float, Optional[Unit]]:
    """Top-level dispatcher — returns (value, primary_target)."""

    if spell.kind == DAMAGE:
        return _damage_value(battle, hero, spell, enemies, s)

    if spell.kind == AOE:
        return _aoe_value(battle, hero, spell, friendly, enemies, s)

    if spell.kind == DISPEL:
        return _dispel_value(battle, hero, spell, friendly, enemies, s)

    if spell.kind == CURE:
        return _cure_value(battle, hero, spell, friendly, s)

    if spell.kind == UTILITY:
        return _utility_value(battle, hero, spell, friendly, enemies, s)

    # BUFF / DEBUFF / CONTROL — single or mass
    return _effect_value(battle, hero, spell, friendly, enemies, s)


# ── shared helpers ────────────────────────────────────────────────

def _spell_duration_multiplier(hero: Hero, target: Unit) -> int:
    """spellDurationMultiplier — ai_battle_spell.cpp:275.

    Returns 0 if the spell would have no meaningful duration
    (hero power < 2 and target already acted this round), else 1.
    """
    if hero.power < 2 and target._acted:
        return 0
    return 1


def _distance_from_starting_edge(target: Unit, grid) -> int:
    """ReduceEffectivenessByDistance — ai_battle_spell.cpp:56-63.

    Distance from the target's own starting board edge along the X axis.
    fheroes2: GetDistanceFromBoardEdgeAlongXAxis(headIndex, isReflect).
    Team 0 (facing right): col + 1  (1-based from left edge).
    Team 1 (facing left):  cols - col (from right edge).
    """
    if target.team == 0:
        return target.col + 1
    return grid.cols - target.col


# ── damage ───────────────────────────────────────────────────────

def _damage_value(battle: BattleState, hero: Hero, spell: Spell,
                  enemies: List[Unit], s: AIState
                  ) -> Tuple[float, Optional[Unit]]:
    """damageHeuristic over each enemy — kill bonus or fraction of strength."""
    damage = spell_damage(spell, hero.power)
    best_v, best_t = 0.0, None
    for e in enemies:
        # Skip units immune to this spell
        if e.is_immune_to_spells:
            continue
        v = _damage_heuristic(e, damage, s.enemy_army, s.enemy_avg_speed)
        if v > best_v:
            best_v, best_t = v, e
    return best_v, best_t


def _damage_heuristic(unit: Unit, damage: int,
                      army_str: float, army_speed: float) -> float:
    """Value of dealing *damage* to *unit* (original damageHeuristic)."""
    if unit.has_ability("magic_resistance"):
        params = unit.ability_params.get("magic_resistance", {})
        chance = params.get("chance", 0)
        if chance >= 100:
            return 0.0  # immune

    hp = unit._total_hp
    if damage >= hp:
        bonus = 0.07 if unit.speed > army_speed else 0.035
        return unit.strength + army_str * bonus
    fraction = min(damage / hp, 1.0)
    # Penalty for waking up a disabled unit (Blind/Paralyze).
    if unit.skip_turn:
        fraction += fraction - 1.0
    return fraction * unit.strength


# ── AOE ──────────────────────────────────────────────────────────

def _aoe_value(battle: BattleState, hero: Hero, spell: Spell,
               friendly: List[Unit], enemies: List[Unit], s: AIState
               ) -> Tuple[float, Optional[Unit]]:
    pattern = spell.aoe_pattern
    base_dmg = spell_damage(spell, hero.power)

    if pattern in ("all_tagged", "all_units"):
        # Army-wide: sum value over all matching units.
        value = 0.0
        for unit in battle.alive():
            if spell.target_tags:
                if not all(unit.has_tag(t) for t in spell.target_tags):
                    continue
            if spell.exclude_tags:
                if any(unit.has_tag(t) for t in spell.exclude_tags):
                    continue
            if unit.is_immune_to_spells:
                continue
            dmg = base_dmg
            if unit.team == s.my_team:
                value -= _damage_heuristic(unit, dmg, s.my_army, s.my_avg_speed)
            else:
                value += _damage_heuristic(unit, dmg, s.enemy_army, s.enemy_avg_speed)
        # Pick first enemy as placeholder target
        best_target = enemies[0] if enemies else None
        return max(value, 0.0), best_target

    if pattern == "chain":
        return _chain_lightning_value(battle, hero, spell, enemies, s)

    # Ring-based AOE (ring1 / ring2 / ring_outer): evaluate per enemy as center.
    # Simplified: use the target's position as center and evaluate splash.
    best_v, best_t = 0.0, None
    for e in enemies:
        center = e.pos
        cells = _aoe_cells(battle, center, pattern)
        value = 0.0
        for cell in cells:
            u = battle.unit_at(cell)
            if u is None or not u.is_alive:
                continue
            if u.is_immune_to_spells:
                continue
            if u.team == s.my_team:
                value -= _damage_heuristic(u, base_dmg, s.my_army, s.my_avg_speed)
            else:
                value += _damage_heuristic(u, base_dmg, s.enemy_army, s.enemy_avg_speed)
        if value > best_v:
            best_v, best_t = value, e
    return max(best_v, 0.0), best_t


def _chain_lightning_value(battle: BattleState, hero: Hero, spell: Spell,
                           enemies: List[Unit], s: AIState
                           ) -> Tuple[float, Optional[Unit]]:
    """Chain Lightning: simulates 4 bounces with halving damage."""
    base_dmg = spell_damage(spell, hero.power)
    best_v, best_t = 0.0, None

    for start in enemies:
        if start.is_immune_to_spells:
            continue
        value = 0.0
        dmg = base_dmg
        hit: list = []
        current = start
        for _ in range(4):
            if current is None or not current.is_alive or current in hit:
                break
            hit.append(current)
            if current.team == s.my_team:
                value -= _damage_heuristic(current, dmg, s.my_army, s.my_avg_speed)
            else:
                value += _damage_heuristic(current, dmg, s.enemy_army, s.enemy_avg_speed)
            # Find nearest alive unit for next bounce.
            candidates = [u for u in battle.alive() if u not in hit]
            if candidates:
                current = min(candidates,
                              key=lambda u: (abs(u.col - current.col)
                                             + abs(u.row - current.row)))
            else:
                break
            dmg = max(1, dmg // 2)
        if value > best_v:
            best_v, best_t = value, start
    return max(best_v, 0.0), best_t


def _aoe_cells(battle: BattleState, center: tuple, pattern: str) -> set:
    """Return cells affected by a ring pattern."""
    if pattern == "ring1":
        cells = {center}
        cells.update(battle.grid.neighbors(*center))
        return cells
    if pattern == "ring2":
        cells = {center}
        ring1 = set(battle.grid.neighbors(*center))
        cells.update(ring1)
        for c in ring1:
            cells.update(battle.grid.neighbors(*c))
        return cells
    if pattern == "ring_outer":
        return set(battle.grid.neighbors(*center))
    return set()


# ── buff / debuff / control ──────────────────────────────────────

def _effect_value(battle: BattleState, hero: Hero, spell: Spell,
                  friendly: List[Unit], enemies: List[Unit], s: AIState
                  ) -> Tuple[float, Optional[Unit]]:
    """Evaluate a BUFF / DEBUFF / CONTROL spell (single or mass)."""
    targets = friendly if spell.side_friendly else enemies
    is_mass = spell.is_mass

    total_v, best_t = 0.0, None
    best_single = 0.0
    for t in targets:
        if t.is_immune_to_spells:
            continue
        if t.has_effect(spell.name):
            continue  # don't re-cast
        # Tag exclusion
        if spell.exclude_tags:
            if any(t.has_tag(tag) for tag in spell.exclude_tags):
                continue
        ratio = _effect_ratio(spell, t, enemies, s, battle)
        if ratio <= 0:
            continue
        # A2 §3.5-#35/#36: multiply by spellDurationMultiplier.
        # fheroes2: target.GetStrength() * ratio * spellDurationMultiplier(target)
        v = t.strength * ratio * _spell_duration_multiplier(hero, t)
        if v > best_single:
            best_single = v
            best_t = t
        total_v += v

    if is_mass:
        return max(total_v, 0.0), best_t
    return max(best_single, 0.0), best_t


def _effect_ratio(spell: Spell, target: Unit,
                  enemies: List[Unit], s: AIState, battle=None) -> float:
    """Per-spell value ratio (spellEffectValue switch in the original)."""
    name = spell.name

    if name == "Slow" or name == "Mass Slow":
        grid = battle.grid if battle is not None else None
        return _slow_ratio(target, s, grid)
    if name == "Haste" or name == "Mass Haste":
        return _haste_ratio(target, s)
    if name in ("Bless", "Mass Bless"):
        # Useless if damage is already fixed
        if target.damage_min == target.damage_max:
            return 0.0
        ratio = 0.15
        if target.has_effect("Curse"):
            ratio *= 2  # extra value dispelling Curse
        return ratio
    if name in ("Curse", "Mass Curse"):
        if target.damage_min == target.damage_max:
            return 0.0
        ratio = 0.15
        if target.has_effect("Bless"):
            ratio *= 2  # extra value dispelling Bless
        return ratio
    if name == "Bloodlust":
        return 0.1   # bloodLustRatio in original
    if name == "Stone Skin":
        return 0.1
    if name == "Steel Skin":
        return 0.2
    if name == "Shield" or name == "Mass Shield":
        if s.enemy_shooters <= 0:
            return 0.0
        ratio = s.enemy_shooters / max(s.enemy_army, 1.0) * 0.3
        if target.is_archer:
            ratio *= 1.25
        return ratio
    if name == "Anti-Magic":
        # Scaled by enemy spell threat (simplified: enemy army * 0.1)
        spell_threat = s.enemy_army * 0.1
        return min(spell_threat / 200.0 * 0.036, 0.9)
    if name == "Disrupting Ray":
        return _disrupting_ray_ratio(target, s)
    if name == "Dragon Slayer":
        return _dragon_slayer_ratio(target, enemies, s)
    if name == "Blind":
        return _blind_ratio(target, enemies, s)
    if name == "Paralyze":
        return _paralyze_ratio(target, enemies, s)
    return 0.0


# ── individual ratio functions ───────────────────────────────────

def _slow_ratio(target: Unit, s: AIState, grid=None) -> float:
    # Slow is useless against archers or troops defending castle.
    if target.is_archer or s.attacking_castle:
        return 0.01
    # A2 §3.3-#11: dynamic speed loss — Speed::getSlowSpeedFromSpell.
    current_speed = target.speed
    new_speed = max(1, current_speed - 2)
    lost = current_speed - new_speed  # usually 2
    ratio = 0.1 * lost
    if current_speed < s.my_avg_speed:
        ratio /= 2  # already slower than our army
    if target.has_effect("Haste"):
        ratio *= 2
    # A2 §3.3-#14: distance reduction for non-flying, non-Haste targets.
    # fheroes2: else if (!target.isFlying()) ratio /= ReduceEffectivenessByDistance(target)
    elif not target.is_flying and grid is not None:
        ratio /= _distance_from_starting_edge(target, grid)
    return ratio


def _haste_ratio(target: Unit, s: AIState) -> float:
    # A2 §3.3-#15: dynamic speed gain — Speed::getHasteSpeedFromSpell.
    current_speed = target.speed
    new_speed = min(SPEED_INSTANT, current_speed + 2)
    gained = new_speed - current_speed  # usually 2
    ratio = 0.05 * gained
    if target.speed < s.enemy_avg_speed:
        ratio *= 2  # very useful if slower than the enemy army
    if target.has_effect("Slow"):
        ratio *= 2
    elif target.is_archer or s.defensive:
        ratio /= 2  # no need to move
    return ratio


def _disrupting_ray_ratio(target: Unit, s: AIState) -> float:
    # Original: getSpellDisruptingRayRatio
    if target.effective_defense <= 1:
        return 0.0  # already minimum
    ratio = 0.2
    if s.my_army < target.strength:
        ratio *= s.my_army / target.strength
    return ratio


def _dragon_slayer_ratio(target: Unit, enemies: List[Unit], s: AIState) -> float:
    # Only valuable if enemies have dragons.
    dragon_str = sum(e.strength for e in enemies if e.has_tag("dragon"))
    enemy_str = max(sum(e.strength for e in enemies), 1.0)
    if dragon_str <= 0:
        return 0.0
    # Scaled from Bloodlust ratio: 0.1 * (5/3) * dragon_proportion
    return 0.1 * (5.0 / 3.0) * dragon_str / enemy_str


def _blind_ratio(target: Unit, enemies: List[Unit], s: AIState) -> float:
    # Original: ai_battle_spell.cpp:384-409
    if len(enemies) == 1:
        # Last enemy: reduced ratio
        if target.has_ability("unlimited_retaliation"):
            return 0.0
        if target.retaliated:
            return 0.0
        return 0.4
    return 0.8


def _paralyze_ratio(target: Unit, enemies: List[Unit], s: AIState) -> float:
    # Original: ai_battle_spell.cpp:432-464
    if len(enemies) == 1:
        if target.has_ability("unlimited_retaliation"):
            return 0.0
        if target.retaliated:
            return 0.0
        return 0.5
    return 0.85


# ── dispel ───────────────────────────────────────────────────────

def _dispel_value(battle: BattleState, hero: Hero, spell: Spell,
                  friendly: List[Unit], enemies: List[Unit], s: AIState
                  ) -> Tuple[float, Optional[Unit]]:
    """Evaluate Dispel / Mass Dispel — remove effects from targets."""
    best_v, best_t = 0.0, None
    is_mass = spell.is_mass

    # Check friendly units: value = sum of negative effects removed
    for unit in friendly:
        if not unit.effects:
            continue
        if unit.is_immune_to_spells:
            continue
        unit_v = 0.0
        for e in unit.effects:
            if not e.is_positive:
                # Value of removing this debuff
                ratio = _effect_ratio_for_removal(e, unit, enemies, s, battle.grid if battle else None)
                unit_v += unit.strength * ratio
        if is_mass:
            best_v += unit_v
        elif unit_v > best_v:
            best_v, best_t = unit_v, unit

    # Check enemy units: value = sum of positive effects removed
    if spell.name == "Dispel Magic":  # single dispel can target anyone
        for unit in enemies:
            if not unit.effects:
                continue
            if unit.is_immune_to_spells:
                continue
            unit_v = 0.0
            for e in unit.effects:
                if e.is_positive:
                    ratio = _effect_ratio_for_removal(e, unit, enemies, s, battle.grid if battle else None)
                    unit_v += unit.strength * ratio
            if is_mass:
                best_v += unit_v
            elif unit_v > best_v:
                best_v, best_t = unit_v, unit

    return max(best_v, 0.0), best_t


def _effect_ratio_for_removal(effect, unit: Unit,
                               enemies: List[Unit], s: AIState,
                               grid=None) -> float:
    """Estimate the value of removing an effect (for Dispel evaluation)."""
    name = effect.name
    if name in ("Slow", "Mass Slow"):
        return _slow_ratio(unit, s, grid)
    if name in ("Haste", "Mass Haste"):
        return _haste_ratio(unit, s)
    if name in ("Bless", "Mass Bless", "Curse", "Mass Curse"):
        return 0.15
    if name == "Bloodlust":
        return 0.1
    if name in ("Stone Skin", "Steel Skin"):
        return 0.15
    if name in ("Blind", "Paralyze", "Petrify"):
        return 0.8  # very valuable to remove control
    return 0.1  # default


# ── cure ─────────────────────────────────────────────────────────

def _cure_value(battle: BattleState, hero: Hero, spell: Spell,
                friendly: List[Unit], s: AIState
                ) -> Tuple[float, Optional[Unit]]:
    """Evaluate Cure / Mass Cure — remove debuffs + heal."""
    heal_amount = spell.heal_base * hero.power
    best_v, best_t = 0.0, None
    is_mass = spell.is_mass
    total_v = 0.0

    for unit in friendly:
        if unit.is_immune_to_spells:
            continue
        unit_v = 0.0
        # Value of removing negative effects
        for e in unit.effects:
            if not e.is_positive:
                unit_v += unit.strength * 0.15
        # Value of healing
        missing = unit.count * unit.max_hp - unit._total_hp
        healed = min(missing, heal_amount)
        if healed > 0:
            unit_v += healed * unit.monster_strength / max(unit.max_hp, 1)
        if is_mass:
            total_v += unit_v
        elif unit_v > best_v:
            best_v, best_t = unit_v, unit

    if is_mass:
        return max(total_v, 0.0), friendly[0] if friendly else None
    return max(best_v, 0.0), best_t


# ── utility ──────────────────────────────────────────────────────

def _utility_value(battle: BattleState, hero: Hero, spell: Spell,
                   friendly: List[Unit], enemies: List[Unit], s: AIState
                   ) -> Tuple[float, Optional[Unit]]:
    if spell.name == "Teleport":
        return _teleport_value(battle, hero, spell, friendly, enemies, s)
    if spell.name == "Earthquake":
        return _earthquake_value(battle, hero, spell, friendly, s)
    return 0.0, None


def _teleport_value(battle: BattleState, hero: Hero, spell: Spell,
                    friendly: List[Unit], enemies: List[Unit], s: AIState
                    ) -> Tuple[float, Optional[Unit]]:
    """Teleport: useful for melee units that can't reach enemies."""
    if s.defensive:
        return 0.0, None
    best_v, best_t = 0.0, None
    for unit in friendly:
        if unit.is_flying or unit.is_archer:
            continue  # already mobile
        if unit.is_immune_to_spells:
            continue
        # Check if unit can reach any enemy
        can_reach = False
        for e in enemies:
            if battle.grid.distance(unit.pos, e.pos) <= unit.speed:
                can_reach = True
                break
        if can_reach:
            continue  # doesn't need teleport
        v = unit.strength * 0.1  # bloodLustRatio in original
        if v > best_v:
            best_v, best_t = v, unit
    return best_v, best_t


def _earthquake_value(battle: BattleState, hero: Hero, spell: Spell,
                      friendly: List[Unit], s: AIState
                      ) -> Tuple[float, Optional[Unit]]:
    """Earthquake: valuable when attacking a castle with melee units."""
    if not s.attacking_castle:
        return 0.0, None
    # Count melee strength (non-flyer, non-archer)
    melee_str = sum(u.strength for u in friendly
                    if not u.is_flying and not u.is_archer)
    if melee_str <= 0:
        return 0.0, None
    ratio = melee_str / max(s.my_army, 1.0)
    enemy_shooter_ratio = s.enemy_shooters / max(s.enemy_army, 1.0)
    v = melee_str * ratio * enemy_shooter_ratio * 0.2
    return v, friendly[0] if friendly else None

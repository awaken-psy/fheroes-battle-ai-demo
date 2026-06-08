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
from engine.spells import Spell, DAMAGE, BUFF, DEBUFF

from .evaluation import AIState


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
        if retreating and spell.kind != DAMAGE:
            continue  # farewell cast considers only damage spells
        value, target = _spell_value(battle, hero, spell, friendly, enemies, s)
        if target is None or value <= 0:
            continue
        if retreating:
            # No cost discount, ignore the threshold — just deal max damage.
            if value > best_value:
                best_value, best = value, (spell, target)
            continue
        # Diminish by spell-point cost: sqrt so high-level spells aren't linear.
        spv = value / math.sqrt(spell.cost / 3.0)
        if spv > best_value and spv > threshold:
            best_value, best = spv, (spell, target)
    return best


def _spell_value(battle: BattleState, hero: Hero, spell: Spell,
                 friendly: List[Unit], enemies: List[Unit], s: AIState
                 ) -> Tuple[float, Optional[Unit]]:
    if spell.kind == DAMAGE:
        return _damage_value(battle, hero, spell, enemies, s)
    targets = friendly if spell.side_friendly else enemies
    best_v, best_t = 0.0, None
    for t in targets:
        if t.has_effect(spell.name):
            continue  # don't re-cast the same effect (isSpellcastUselessForUnit)
        ratio = _effect_ratio(spell, t, s)
        v = t.strength * ratio
        if v > best_v:
            best_v, best_t = v, t
    return best_v, best_t


def _damage_value(battle: BattleState, hero: Hero, spell: Spell,
                  enemies: List[Unit], s: AIState
                  ) -> Tuple[float, Optional[Unit]]:
    """damageHeuristic over each enemy — kill bonus or fraction of strength lost."""
    from engine.spells import spell_damage
    damage = spell_damage(spell, hero.power)
    best_v, best_t = 0.0, None
    for e in enemies:
        hp = e._total_hp
        if damage >= hp:
            # Full kill: full strength plus a bonus for removing the stack.
            bonus = 0.07 if e.speed > s.enemy_avg_speed else 0.035
            v = e.strength + s.enemy_army * bonus
        else:
            v = min(damage / hp, 1.0) * e.strength
        if v > best_v:
            best_v, best_t = v, e
    return best_v, best_t


def _effect_ratio(spell: Spell, target: Unit, s: AIState) -> float:
    """Per-spell value ratio (spellEffectValue switch in the original)."""
    name = spell.name
    if name == "Slow":
        return _slow_ratio(target, s)
    if name == "Haste":
        return _haste_ratio(target, s)
    if name in ("Bless", "Curse"):
        return 0.15
    return 0.0


def _slow_ratio(target: Unit, s: AIState) -> float:
    # Slow is useless against archers (they don't need to move).
    if target.is_archer:
        return 0.01
    lost = 2  # Haste/Slow change speed by 2
    ratio = 0.1 * lost
    if target.speed < s.my_avg_speed:
        ratio /= 2  # already slower than our army
    if target.has_effect("Haste"):
        ratio *= 2
    return ratio


def _haste_ratio(target: Unit, s: AIState) -> float:
    gained = 2
    ratio = 0.05 * gained
    if target.speed < s.enemy_avg_speed:
        ratio *= 2  # very useful if slower than the enemy army
    if target.has_effect("Slow"):
        ratio *= 2
    elif target.is_archer or s.defensive:
        ratio /= 2  # no need to move
    return ratio

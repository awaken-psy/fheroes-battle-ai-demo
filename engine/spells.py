"""Spell definitions and timed status effects.

Reimplemented from fheroes2's spell system (the M3 subset). Damage spells deal
``base * power`` damage; buffs/debuffs attach a timed Effect that modifies a
unit's speed or damage and expires after ``power`` rounds.

Spell base values match the originals (spell.cpp data table):
  Magic Arrow   cost 3, damage 10
  Lightning Bolt cost 7, damage 25
  Haste/Slow    cost 3, speed +-2
  Bless/Curse   cost 3, max/min damage (approximated as +-20%)
"""

from dataclasses import dataclass, field
from typing import Optional

# Spell "kinds" decide who the AI targets and how the value is scored.
DAMAGE = "damage"   # targets an enemy stack, deals damage
BUFF = "buff"       # targets a friendly stack
DEBUFF = "debuff"   # targets an enemy stack


@dataclass(frozen=True)
class Spell:
    name: str
    kind: str
    cost: int                     # spell points
    base_damage: int = 0          # for DAMAGE spells: damage = base_damage * power
    speed_delta: int = 0          # for effects: change to speed
    damage_mult: float = 1.0      # for effects: multiplier on dealt damage
    side_friendly: bool = False   # buffs apply to own army, others to the enemy


SPELLS = {
    "Magic Arrow":    Spell("Magic Arrow", DAMAGE, cost=3, base_damage=10),
    "Lightning Bolt": Spell("Lightning Bolt", DAMAGE, cost=7, base_damage=25),
    "Haste":          Spell("Haste", BUFF, cost=3, speed_delta=2, side_friendly=True),
    "Bless":          Spell("Bless", BUFF, cost=3, damage_mult=1.2, side_friendly=True),
    "Slow":           Spell("Slow", DEBUFF, cost=3, speed_delta=-2),
    "Curse":          Spell("Curse", DEBUFF, cost=3, damage_mult=0.8),
}

# The default spellbook for a configurable hero.
DEFAULT_SPELLBOOK = list(SPELLS.keys())


@dataclass
class Effect:
    """A timed status on a unit (one spell's lingering effect)."""
    name: str
    remaining: int               # rounds left before it expires
    speed_delta: int = 0
    damage_mult: float = 1.0
    skip_turn: bool = False          # unit skips its action while active
    break_on_damage: bool = False    # removed when unit takes damage


def spell_damage(spell: Spell, power: int) -> int:
    """Damage dealt by a DAMAGE spell cast at the given hero power."""
    return spell.base_damage * power


def make_effect(spell: Spell, power: int) -> Optional[Effect]:
    """Build the timed Effect for a buff/debuff spell, lasting `power` rounds."""
    if spell.kind == DAMAGE:
        return None
    return Effect(name=spell.name, remaining=power,
                  speed_delta=spell.speed_delta, damage_mult=spell.damage_mult)


# ── spell_caster combat ability effects ───────────────────────
# These are applied by monster abilities (not hero spellcasting).
# Blind/Paralyze/Petrify skip the unit's turn; broken when damaged.
# Curse is the same effect as the hero spell.
# Dispel removes all existing effects instead of adding one.

_CONTROL_EFFECTS = {
    "blind":    lambda: Effect("Blind",    remaining=100, skip_turn=True,  break_on_damage=True),
    "paralyze": lambda: Effect("Paralyze", remaining=100, skip_turn=True,  break_on_damage=True),
    "petrify":  lambda: Effect("Petrify",  remaining=100, skip_turn=True,  break_on_damage=True),
    "curse":    lambda: Effect("Curse",    remaining=3,   damage_mult=0.8),
    # "dispel" is handled specially: remove all effects from target.
}


def make_spell_caster_effect(spell_name: str):
    """Return an Effect for a spell_caster combat ability, or None for dispel."""
    factory = _CONTROL_EFFECTS.get(spell_name)
    if factory is not None:
        return factory()
    return None

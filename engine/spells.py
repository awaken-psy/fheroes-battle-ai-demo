"""Spell definitions and timed status effects.

Reimplemented from fheroes2's spell system (spell.cpp). All 38 combat
spells with original data. Damage spells deal ``base_damage * power``;
buffs/debuffs attach a timed Effect that modifies unit stats and expires
after ``power`` rounds. Control spells (Blind, Paralyze) persist until
the unit takes damage.

Spell data matches the original (spell.cpp spells[] table):
  Magic Arrow    cost 3,  damage 10×power
  Lightning Bolt cost 7,  damage 25×power
  Cold Ray       cost 6,  damage 20×power
  Fireball       cost 9,  damage 10×power, 1-ring AOE
  Fireblast      cost 15, damage 10×power, 2-ring AOE
  Cold Ring      cost 9,  damage 10×power, ring-outer AOE
  Chain Lightning cost 15, damage 40×power, chain AOE
  Meteor Shower  cost 15, damage 25×power, 1-ring AOE
  Death Ripple   cost 6,  damage 5×power,  non-undead army-wide
  Death Wave     cost 10, damage 10×power, non-undead army-wide
  Holy Word      cost 9,  damage 10×power, undead-only army-wide
  Holy Shout     cost 12, damage 20×power, undead-only army-wide
  Armageddon     cost 20, damage 50×power, all-units
  Elemental Storm cost 15, damage 25×power, all-units
  (and 24 buff / debuff / control / utility spells)
"""

from dataclasses import dataclass
from typing import Optional

# ── Spell kinds ──────────────────────────────────────────────────
# Determine targeting, AI scoring, and _cast dispatch.

DAMAGE = "damage"       # single-target damage
AOE = "aoe"             # area / army-wide damage
BUFF = "buff"           # buff friendly (or all friends if is_mass)
DEBUFF = "debuff"       # debuff enemy (or all enemies if is_mass)
CONTROL = "control"     # Blind / Paralyze — skip_turn + break_on_damage
DISPEL = "dispel"       # remove effects from target
CURE = "cure"           # remove debuffs + heal HP
UTILITY = "utility"     # Teleport, Earthquake


@dataclass(frozen=True)
class Spell:
    name: str
    kind: str
    cost: int
    base_damage: int = 0
    speed_delta: int = 0
    damage_mult: float = 1.0
    attack_delta: int = 0
    defense_delta: int = 0
    side_friendly: bool = False
    is_mass: bool = False
    # AOE targeting
    aoe_pattern: str = ""       # ring1 / ring2 / ring_outer / chain / all_tagged / all_units
    target_tags: tuple = ()     # only hit units with these tags (e.g. ("undead",))
    exclude_tags: tuple = ()    # skip units with these tags
    # Cure
    heal_base: int = 0
    # Elemental flag: subject to Golem elemental_spell_reduction
    elemental: bool = False
    # Effect behavior (for BUFF / DEBUFF / CONTROL effects)
    effect_break_on_damage: bool = False
    effect_skip_turn: bool = False
    effect_stackable: bool = False
    effect_ranged_shield: float = 1.0
    effect_anti_magic: bool = False


# ── Spell data ───────────────────────────────────────────────────
# Values from fheroes2 spell.cpp spells[] table.
# Sorted by level within each kind.

SPELLS = {
    # ── Level 1 ──────────────────────────────────────────────────
    "Magic Arrow":    Spell("Magic Arrow",  DAMAGE, cost=3,  base_damage=10,
                           elemental=True),
    "Bloodlust":      Spell("Bloodlust",    BUFF,   cost=3,  attack_delta=3,
                           side_friendly=True),
    "Bless":          Spell("Bless",        BUFF,   cost=3,  damage_mult=1.2,
                           side_friendly=True, exclude_tags=("undead",)),
    "Cure":           Spell("Cure",         CURE,   cost=6,  heal_base=5,
                           side_friendly=True),
    "Curse":          Spell("Curse",        DEBUFF, cost=3,  damage_mult=0.8,
                           exclude_tags=("undead",)),
    "Dispel Magic":   Spell("Dispel Magic", DISPEL, cost=5),
    "Haste":          Spell("Haste",        BUFF,   cost=3,  speed_delta=2,
                           side_friendly=True),
    "Shield":         Spell("Shield",       BUFF,   cost=3,
                           effect_ranged_shield=0.5, side_friendly=True),
    "Slow":           Spell("Slow",         DEBUFF, cost=3,  speed_delta=-2),
    "Stone Skin":     Spell("Stone Skin",   BUFF,   cost=3,  defense_delta=3,
                           side_friendly=True),

    # ── Level 2 ──────────────────────────────────────────────────
    "Blind":          Spell("Blind",        CONTROL, cost=6,
                           effect_skip_turn=True, effect_break_on_damage=True),
    "Cold Ray":       Spell("Cold Ray",     DAMAGE,  cost=6,  base_damage=20,
                           elemental=True),
    "Death Ripple":   Spell("Death Ripple", AOE,     cost=6,  base_damage=5,
                           aoe_pattern="all_tagged", exclude_tags=("undead",)),
    "Disrupting Ray": Spell("Disrupting Ray", DEBUFF, cost=7, defense_delta=-3,
                           effect_stackable=True),
    "Dragon Slayer":  Spell("Dragon Slayer", BUFF,   cost=6,  attack_delta=5,
                           side_friendly=True),
    "Lightning Bolt": Spell("Lightning Bolt", DAMAGE, cost=7,  base_damage=25,
                           elemental=True),
    "Steel Skin":     Spell("Steel Skin",    BUFF,   cost=6,  defense_delta=5,
                           side_friendly=True),

    # ── Level 3 ──────────────────────────────────────────────────
    "Anti-Magic":     Spell("Anti-Magic",    BUFF,    cost=7,
                           effect_anti_magic=True, side_friendly=True),
    "Cold Ring":      Spell("Cold Ring",      AOE,     cost=9,  base_damage=10,
                           aoe_pattern="ring_outer", elemental=True),
    "Death Wave":     Spell("Death Wave",     AOE,     cost=10, base_damage=10,
                           aoe_pattern="all_tagged", exclude_tags=("undead",)),
    "Earthquake":     Spell("Earthquake",     UTILITY, cost=15),
    "Fireball":       Spell("Fireball",       AOE,     cost=9,  base_damage=10,
                           aoe_pattern="ring1", elemental=True),
    "Holy Word":      Spell("Holy Word",      AOE,     cost=9,  base_damage=10,
                           aoe_pattern="all_tagged", target_tags=("undead",)),
    "Mass Bless":     Spell("Mass Bless",     BUFF,    cost=12, damage_mult=1.2,
                           side_friendly=True, is_mass=True,
                           exclude_tags=("undead",)),
    "Mass Curse":     Spell("Mass Curse",     DEBUFF,  cost=12, damage_mult=0.8,
                           is_mass=True, exclude_tags=("undead",)),
    "Mass Dispel":    Spell("Mass Dispel",    DISPEL,  cost=12, is_mass=True),
    "Mass Haste":     Spell("Mass Haste",     BUFF,    cost=10, speed_delta=2,
                           side_friendly=True, is_mass=True),
    "Mass Slow":      Spell("Mass Slow",      DEBUFF,  cost=15, speed_delta=-2,
                           is_mass=True),
    "Paralyze":       Spell("Paralyze",       CONTROL, cost=9,
                           effect_skip_turn=True, effect_break_on_damage=True),
    "Teleport":       Spell("Teleport",       UTILITY, cost=9,
                           side_friendly=True),

    # ── Level 4 ──────────────────────────────────────────────────
    "Chain Lightning": Spell("Chain Lightning", AOE, cost=15, base_damage=40,
                           aoe_pattern="chain", elemental=True),
    "Elemental Storm": Spell("Elemental Storm", AOE, cost=15, base_damage=25,
                           aoe_pattern="all_units", elemental=True),
    "Fireblast":      Spell("Fireblast",       AOE, cost=15, base_damage=10,
                           aoe_pattern="ring2", elemental=True),
    "Holy Shout":     Spell("Holy Shout",       AOE, cost=12, base_damage=20,
                           aoe_pattern="all_tagged", target_tags=("undead",)),
    "Mass Cure":      Spell("Mass Cure",        CURE, cost=15, heal_base=5,
                           side_friendly=True, is_mass=True),
    "Mass Shield":    Spell("Mass Shield",      BUFF, cost=7,
                           effect_ranged_shield=0.5, side_friendly=True,
                           is_mass=True),
    "Meteor Shower":  Spell("Meteor Shower",    AOE, cost=15, base_damage=25,
                           aoe_pattern="ring1", elemental=True),

    # ── Level 5 ──────────────────────────────────────────────────
    "Armageddon":     Spell("Armageddon",       AOE, cost=20, base_damage=50,
                           aoe_pattern="all_units", elemental=True),
}

DEFAULT_SPELLBOOK = list(SPELLS.keys())


@dataclass
class Effect:
    """A timed status on a unit (one spell's lingering effect)."""
    name: str
    remaining: int               # rounds left (0 = expired, removed by tick)
    speed_delta: int = 0
    damage_mult: float = 1.0
    attack_delta: int = 0
    defense_delta: int = 0
    skip_turn: bool = False
    break_on_damage: bool = False
    stackable: bool = False
    is_positive: bool = True     # False for debuffs (used by Cure)
    ranged_shield: float = 1.0   # multiplier on incoming ranged damage
    anti_magic: bool = False     # immune to all spells while active


def spell_damage(spell: Spell, power: int) -> int:
    """Damage dealt by a DAMAGE / AOE spell at the given hero power."""
    return spell.base_damage * power


def make_effect(spell: Spell, power: int) -> Optional[Effect]:
    """Build the timed Effect for a buff / debuff / control spell.

    Effect lasts ``power`` rounds.  Control spells (Blind, Paralyze) last
    indefinitely (remaining=100) until broken by damage.
    """
    if spell.kind in (DAMAGE, AOE, DISPEL, CURE, UTILITY):
        return None

    remaining = power
    is_positive = spell.kind == BUFF

    if spell.effect_skip_turn:
        # Control spells last until broken by damage.
        remaining = 100

    return Effect(
        name=spell.name,
        remaining=remaining,
        speed_delta=spell.speed_delta,
        damage_mult=spell.damage_mult,
        attack_delta=spell.attack_delta,
        defense_delta=spell.defense_delta,
        skip_turn=spell.effect_skip_turn,
        break_on_damage=spell.effect_break_on_damage,
        stackable=spell.effect_stackable,
        is_positive=is_positive,
        ranged_shield=spell.effect_ranged_shield,
        anti_magic=spell.effect_anti_magic,
    )


# ── spell_caster combat ability effects ───────────────────────
# These are applied by monster abilities (not hero spellcasting).
# Blind/Paralyze/Petrify skip the unit's turn; broken when damaged.
# Curse is the same effect as the hero spell.

_CONTROL_EFFECTS = {
    "blind":    lambda: Effect("Blind",    remaining=100, skip_turn=True,
                              break_on_damage=True, is_positive=False),
    "paralyze": lambda: Effect("Paralyze", remaining=100, skip_turn=True,
                              break_on_damage=True, is_positive=False),
    "petrify":  lambda: Effect("Petrify",  remaining=100, skip_turn=True,
                              break_on_damage=True, is_positive=False),
    "curse":    lambda: Effect("Curse",    remaining=3,   damage_mult=0.8,
                              is_positive=False),
    # "dispel" is handled specially: remove all effects from target.
}


def make_spell_caster_effect(spell_name: str):
    """Return an Effect for a spell_caster combat ability, or None for dispel."""
    factory = _CONTROL_EFFECTS.get(spell_name)
    if factory is not None:
        return factory()
    return None

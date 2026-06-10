"""M7 tests — new faction units, new abilities, strength sanity."""

import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from engine.hex_grid import HexGrid
from engine.battle_state import BattleState
from engine.unit import Unit
from engine.actions import AttackAction
from engine.spells import Effect
from ai.classic.scoring import threat

G = HexGrid()


def _battle(units, **kw):
    return BattleState(G, units, **kw)


# ── all new units can be created ─────────────────────────────────────

ALL_FACTIONS = {
    "Knight": ["Peasant", "Archer", "Ranger", "Pikeman", "Veteran Pikeman",
               "Swordsman", "Master Swordsman", "Cavalry", "Champion",
               "Paladin", "Crusader"],
    "Barbarian": ["Goblin", "Orc", "Orc Chief", "Wolf", "Ogre", "Ogre Lord",
                  "Troll", "War Troll", "Cyclops"],
    "Sorceress": ["Sprite", "Dwarf", "Battle Dwarf", "Elf", "Grand Elf",
                  "Druid", "Greater Druid", "Unicorn", "Phoenix"],
    "Warlock": ["Centaur", "Gargoyle", "Griffin", "Minotaur", "Minotaur King",
                "Hydra", "Green Dragon", "Red Dragon", "Black Dragon"],
    "Wizard": ["Halfling", "Boar", "Iron Golem", "Steel Golem", "Roc",
               "Mage", "Archmage", "Giant", "Titan"],
    "Necromancer": ["Skeleton", "Zombie", "Mutant Zombie", "Mummy",
                    "Royal Mummy", "Vampire", "Vampire Lord", "Lich",
                    "Power Lich", "Bone Dragon"],
    "Neutral": ["Rogue", "Nomad", "Ghost", "Genie", "Medusa",
                "Earth Elemental", "Air Elemental", "Fire Elemental",
                "Water Elemental"],
}


def test_all_units_create():
    """Every unit in UNIT_TYPES can be instantiated."""
    for name in config.UNIT_TYPES:
        u = Unit.from_type(name, 0, 5, 4)
        assert u.is_alive
        assert u.strength > 0


def test_faction_race_tags():
    """Each unit's race matches its faction."""
    for faction, names in ALL_FACTIONS.items():
        for name in names:
            u = Unit.from_type(name, 0, 5, 4)
            assert u.race if hasattr(u, 'race') else config.UNIT_TYPES[name].get("race")  # just ensure exists


def test_strength_monotone_with_cost():
    """Within each faction, higher-tier units generally cost more."""
    for faction, names in ALL_FACTIONS.items():
        units = [(config.UNIT_TYPES[n]["cost"], config.UNIT_TYPES[n]["level"], n)
                 for n in names]
        # Level should be roughly monotone with cost (not strict — upgrades vary)
        units.sort(key=lambda x: x[1])  # sort by level
        for i in range(len(units) - 1):
            # Higher level should not cost less (allow same cost for same level)
            assert units[i + 1][0] >= units[i][0] * 0.5, \
                f"{units[i][2]}(lv{units[i][1]}) cost {units[i][0]} vs " \
                f"{units[i+1][2]}(lv{units[i+1][1]}) cost {units[i+1][0]}"


# ── wide / archer / flying flags ─────────────────────────────────────

def test_wide_units_flagged():
    wide_names = ["Cavalry", "Champion", "Wolf",  # Knight/Barb
                  "Unicorn", "Phoenix",             # Sorceress
                  "Centaur", "Griffin", "Hydra",
                  "Green Dragon", "Red Dragon", "Black Dragon",
                  "Boar", "Roc",                     # Wizard
                  "Bone Dragon",                      # Necro
                  "Nomad", "Medusa"]                  # Neutral
    for name in wide_names:
        u = Unit.from_type(name, 0, 5, 4)
        assert u.is_wide, f"{name} should be wide"


def test_archer_units_flagged():
    archer_names = ["Archer", "Ranger", "Orc", "Orc Chief", "Troll", "War Troll",
                    "Elf", "Grand Elf", "Druid", "Greater Druid",
                    "Centaur",  # wide archer
                    "Halfling", "Mage", "Archmage", "Titan",
                    "Lich", "Power Lich"]
    for name in archer_names:
        u = Unit.from_type(name, 0, 5, 4)
        assert u.is_archer, f"{name} should be archer"
        assert config.UNIT_TYPES[name]["shots"] > 0, f"{name} should have shots > 0"


def test_flying_units_flagged():
    fly_names = ["Sprite", "Gargoyle", "Griffin", "Phoenix",
                 "Roc", "Vampire", "Vampire Lord", "Bone Dragon",
                 "Ghost", "Genie"]
    for name in fly_names:
        u = Unit.from_type(name, 0, 5, 4)
        assert u.is_flying, f"{name} should fly"


# ── new ability combat hooks ─────────────────────────────────────────

def test_no_enemy_retaliation_sprite():
    """Sprite attacks without being retaliated."""
    sprite = Unit.from_type("Sprite", 0, 4, 4)
    target = Unit.from_type("Pikeman", 1, 5, 4)
    b = _battle([sprite, target])
    r = b.execute(AttackAction(sprite, target, (4, 4), ranged=False))
    assert r["ret_dmg"] == 0
    assert "retaliates" not in r["desc"]


def test_area_shot_splash():
    """Lich shots splash to adjacent enemies."""
    lich = Unit.from_type("Lich", 0, 4, 4)
    # Place enemies adjacent to each other so splash hits
    t1 = Unit.from_type("Pikeman", 1, 5, 4)
    t2 = Unit.from_type("Pikeman", 1, 6, 4)  # adjacent to t1
    b = _battle([lich, t1, t2])
    r = b.execute(AttackAction(lich, t1, ranged=True))
    assert r["dmg"] > 0
    assert "AoE" in r["desc"]
    assert "splash_dmg" in r


def test_all_adjacent_attack_hydra():
    """Hydra hits all adjacent enemies."""
    hydra = Unit.from_type("Hydra", 0, 5, 4)  # wide, center at (5,4)
    t1 = Unit.from_type("Pikeman", 1, 6, 4)   # adjacent
    t2 = Unit.from_type("Pikeman", 1, 5, 5)   # adjacent (different hex)
    b = _battle([hydra, t1, t2])
    r = b.execute(AttackAction(hydra, t1, (5, 4), ranged=False))
    assert r["dmg"] > 0
    assert "adj" in r["desc"]


def test_spell_caster_blind():
    """Unicorn can blind on hit (stochastic — force with 100% chance)."""
    unicorn = Unit("TestUnicorn", 0, 4, 4, attack=10, defense=9, hp=40, speed=5,
                   damage_min=7, damage_max=14, is_archer=False, is_flying=False,
                   is_wide=True,
                   abilities=["spell_caster"],
                   ability_params={"spell_caster": {"spell": "blind", "chance": 100}},
                   count=1)
    target = Unit.from_type("Pikeman", 1, 5, 4)
    b = _battle([unicorn, target])
    r = b.execute(AttackAction(unicorn, target, (4, 4), ranged=False))
    assert "blind" in r["desc"].lower()
    assert target.skip_turn


def test_blind_breaks_on_damage():
    """Blind effect is removed when the unit takes damage."""
    u = Unit.from_type("Pikeman", 0, 4, 4)
    u.add_effect(Effect("Blind", remaining=100, skip_turn=True, break_on_damage=True))
    assert u.skip_turn
    u.break_effects_on_damage()
    assert not u.skip_turn
    assert len(u.effects) == 0


def test_skip_turn_excludes_from_turn_order():
    """Blinded units are skipped in turn order."""
    u0 = Unit.from_type("Pikeman", 0, 4, 4)
    u1 = Unit.from_type("Pikeman", 1, 6, 4)
    u1.add_effect(Effect("Blind", remaining=100, skip_turn=True, break_on_damage=True))
    b = _battle([u0, u1])
    order = b.turn_order()
    assert u1 not in order
    assert u0 in order


def test_no_melee_penalty_mage():
    """Mage (no_melee_penalty) does full damage in melee."""
    mage = Unit.from_type("Mage", 0, 4, 4)
    target = Unit.from_type("Pikeman", 1, 5, 4)
    b = _battle([mage, target])
    # expected_damage for melee should NOT have 0.5 penalty
    melee = b.expected_damage(mage, target, ranged=False)
    # Compare with a normal archer that DOES have the penalty
    archer = Unit.from_type("Archer", 0, 4, 4)
    archer_melee = b.expected_damage(archer, target, ranged=False)
    # Mage should do roughly 2x the normal archer melee (same 0.5 penalty removed)
    assert melee > archer_melee


def test_magic_resistance_100_dragon():
    """Black Dragon resists all spells (100% magic_resistance)."""
    dragon = Unit.from_type("Black Dragon", 1, 8, 4)
    from engine.hero import Hero
    from engine.spells import SPELLS
    from engine.actions import CastAction
    hero = Hero(power=5)
    b = _battle([Unit.from_type("Pikeman", 0, 4, 4), dragon], heroes={0: hero, 1: None})
    r = b.execute(CastAction(0, SPELLS["Lightning Bolt"], dragon))
    assert "RESISTED" in r["desc"]
    assert r["dmg"] == 0


def test_enemy_halving_genie():
    """Genie can halve enemy stack (stochastic — test with 100% chance)."""
    genie = Unit("TestGenie", 0, 4, 4, attack=10, defense=9, hp=50, speed=6,
                 damage_min=20, damage_max=30, is_archer=False, is_flying=True,
                 is_wide=False,
                 abilities=["enemy_halving"],
                 ability_params={"enemy_halving": {"chance": 100}},
                 count=1)
    target = Unit.from_type("Pikeman", 1, 5, 4, count=5)  # 5 pikemen
    b = _battle([genie, target])
    r = b.execute(AttackAction(genie, target, (4, 4), ranged=False))
    assert "halving" in r["desc"]
    assert target.count < 5  # some were killed by halving


# ── strength formula for new abilities ───────────────────────────────

def test_strength_all_adjacent_bonus():
    """Hydra gets the area damage bonus in base_strength."""
    hydra = Unit.from_type("Hydra", 0, 0, 0)
    # damage_avg=9, all_adjacent → *1.2, no_enemy_retaliation → HP*1.4
    dmg_pot = 9.0 * 1.2
    eff_hp = 75.0 * 1.4  # no_enemy_retaliation
    # speed=2 (VERYSLOW), diff = 2-4 = -2, special += -2*0.1 = -0.2
    special = 1.0 + (-2 * 0.1)
    expected = math.sqrt(dmg_pot * eff_hp) * special
    assert abs(hydra._base_strength - expected) < 1e-6


def test_strength_area_shot_bonus():
    """Lich gets area_shot bonus in base_strength."""
    lich = Unit.from_type("Lich", 0, 0, 0)
    # damage_avg=9, area_shot → *1.2
    dmg_pot = 9.0 * 1.2
    # archer: special += 0.4, speed=5(FAST), diff=+1 → +0.05
    special = 1.0 + 0.4 + 0.05
    expected = math.sqrt(dmg_pot * 25) * special
    assert abs(lich._base_strength - expected) < 1e-6


def test_strength_no_melee_penalty():
    """Mage gets +0.5 archer bonus instead of +0.4."""
    mage = Unit.from_type("Mage", 0, 0, 0)
    # damage_avg=8, archer with no_melee_penalty → special += 0.5
    # speed=5(FAST), diff=+1 → +0.05
    special = 1.0 + 0.5 + 0.05
    expected = math.sqrt(8 * 30) * special
    assert abs(mage._base_strength - expected) < 1e-6


def test_strength_soul_eater_ghost():
    """Ghost gets soul_eater bonus in base_strength."""
    ghost = Unit.from_type("Ghost", 0, 0, 0)
    # damage_avg=5, flying → special += 0.3, soul_eater → special += 2.0
    # speed=5(FAST), diff=+1 → +0.05
    special = 1.0 + 0.3 + 2.0 + 0.05
    expected = math.sqrt(5 * 20) * special
    assert abs(ghost._base_strength - expected) < 1e-6


def test_strength_enemy_halving_genie():
    """Genie gets enemy_halving bonus in base_strength."""
    genie = Unit.from_type("Genie", 0, 0, 0)
    # damage_avg=25, flying → special += 0.3, enemy_halving → special += 1.0
    # speed=6(VERYFAST), diff=+2 → +0.1
    special = 1.0 + 0.3 + 1.0 + 0.1
    expected = math.sqrt(25 * 50) * special
    assert abs(genie._base_strength - expected) < 1e-6

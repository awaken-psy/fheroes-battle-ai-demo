"""M6a tests — Knight+Barbarian unit data, new abilities, and geometry helpers."""

import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.hex_grid import HexGrid
from engine.battle_state import BattleState
from engine.unit import Unit
from engine.actions import AttackAction
import config

G = HexGrid()


def _battle(units):
    return BattleState(G, units)


# ── unit creation & data sanity ──────────────────────────────────────

KNIGHT_UNITS = [
    "Peasant", "Archer", "Ranger", "Pikeman", "Veteran Pikeman",
    "Swordsman", "Master Swordsman", "Cavalry", "Champion",
    "Paladin", "Crusader",
]

BARBARIAN_UNITS = [
    "Goblin", "Orc", "Orc Chief", "Wolf", "Ogre", "Ogre Lord",
    "Troll", "War Troll", "Cyclops",
]


def test_all_knight_barbarian_units_can_be_created():
    for name in KNIGHT_UNITS + BARBARIAN_UNITS:
        u = Unit.from_type(name, 0, 5, 4)
        assert u.is_alive
        assert u.count > 0
        assert u.max_hp > 0
        assert u.damage_min > 0
        assert u.damage_max >= u.damage_min


def test_from_type_count_override():
    u = Unit.from_type("Swordsman", 0, 0, 0, count=100)
    assert u.count == 100


def test_damage_min_max_model():
    """New units use damage_min/damage_max; old units fall back to single damage."""
    # New unit (Knight Swordsman) has a damage spread.
    sw = Unit.from_type("Swordsman", 0, 0, 0)
    assert sw.damage_min == 4 and sw.damage_max == 6
    assert sw.damage_avg == 5.0

    # Griffin now has original min/max range (3-5).
    gr = Unit.from_type("Griffin", 0, 0, 0)
    assert gr.damage_min == 3 and gr.damage_max == 5


# ── faction / flag consistency ───────────────────────────────────────

def test_wide_units_marked_correctly():
    for name in ("Cavalry", "Champion", "Wolf"):
        u = Unit.from_type(name, 0, 5, 4)
        assert u.is_wide, f"{name} should be wide"

    # Non-wide units
    for name in ("Swordsman", "Ogre", "Cyclops"):
        u = Unit.from_type(name, 0, 5, 4)
        assert not u.is_wide, f"{name} should NOT be wide"


def test_archer_flag_matches_shots():
    for name in KNIGHT_UNITS + BARBARIAN_UNITS:
        t = config.UNIT_TYPES[name]
        if t["shots"] > 0:
            assert t["is_archer"], f"{name} has shots>0 but is_archer=False"
        else:
            assert not t["is_archer"], f"{name} has shots=0 but is_archer=True"


def test_ability_flags():
    assert "double_shooting" in Unit.from_type("Ranger", 0, 0, 0).abilities
    assert "double_melee" in Unit.from_type("Paladin", 0, 0, 0).abilities
    assert "double_melee" in Unit.from_type("Crusader", 0, 0, 0).abilities
    assert "double_damage_to_undead" in Unit.from_type("Crusader", 0, 0, 0).abilities
    assert "double_melee" in Unit.from_type("Wolf", 0, 0, 0).abilities
    assert "self_heal" in Unit.from_type("Troll", 0, 0, 0).abilities
    assert "self_heal" in Unit.from_type("War Troll", 0, 0, 0).abilities
    assert "two_cell_melee" in Unit.from_type("Cyclops", 0, 0, 0).abilities


# ── strength monotonicity (cost → strength) ──────────────────────────

def test_strength_monotonically_increases_with_cost():
    """Within each faction, strictly-higher-cost units have higher monster_strength."""
    for faction_units in (KNIGHT_UNITS, BARBARIAN_UNITS):
        pairs = []
        for name in faction_units:
            u = Unit.from_type(name, 0, 0, 0)
            t = config.UNIT_TYPES[name]
            pairs.append((t["cost"], u.monster_strength, name))
        pairs.sort(key=lambda x: x[0])
        for i in range(len(pairs) - 1):
            cost_lo, str_lo, name_lo = pairs[i]
            cost_hi, str_hi, name_hi = pairs[i + 1]
            if cost_lo < cost_hi:
                assert str_lo <= str_hi, (
                    f"{name_lo} (cost {cost_lo}, str {str_lo:.1f}) > "
                    f"{name_hi} (cost {cost_hi}, str {str_hi:.1f})")


# ── strength formula: new ability terms ──────────────────────────────

def test_base_strength_double_shooting_doubles_damage():
    """Ranger (double_shooting) has exactly 2× damagePotential in base_strength."""
    ranger = Unit.from_type("Ranger", 0, 0, 0)
    # Compute what a hypothetical unit without double_shooting would have.
    archer = Unit.from_type("Archer", 0, 0, 0)
    # Same damage_avg, same hp, but Ranger has speed 4 vs Archer speed 2.
    # Ranger: damagePotential = 2.5 * 2 = 5.0
    # Archer: damagePotential = 2.5 (no multiplier)
    ranger_dmg_potential = 2.5 * 2  # double_shooting
    ranger_special = 1.0 + 0.4 + (4 - 4) * 0.05  # archer, speed diff 0
    expected = math.sqrt(ranger_dmg_potential * 10) * ranger_special
    assert abs(ranger._base_strength - expected) < 1e-6


def test_base_strength_double_melee():
    """Paladin gets damagePotential *= 1.75 (no no_enemy_retaliation)."""
    paladin = Unit.from_type("Paladin", 0, 0, 0)
    dmg_potential = 15.0 * 1.75  # damage_avg * double_melee
    special = 1.0 + (5 - 4) * 0.05  # speed diff +1
    expected = math.sqrt(dmg_potential * 50) * special
    assert abs(paladin._base_strength - expected) < 1e-6


def test_base_strength_two_cell_melee():
    """Cyclops gets damagePotential *= 1.2."""
    cyclops = Unit.from_type("Cyclops", 0, 0, 0)
    dmg_potential = 18.0 * 1.2  # damage_avg * two_cell
    special = 1.0 + (5 - 4) * 0.05
    expected = math.sqrt(dmg_potential * 80) * special
    assert abs(cyclops._base_strength - expected) < 1e-6


# ── combat hooks: double_shooting ────────────────────────────────────

def test_double_shooting_fires_twice():
    ranger = Unit.from_type("Ranger", 0, 4, 4)
    target = Unit.from_type("Goblin", 1, 8, 4, count=100)
    b = _battle([ranger, target])
    # Seed for reproducibility.
    random.seed(42)
    r = b.execute(AttackAction(ranger, target, (4, 4), ranged=True))
    assert "+2nd shot" in r["desc"]
    # Two shots: damage should be substantially more than one shot.
    one_shot = b.expected_damage(ranger, target, ranged=True) // 2
    assert r["dmg"] > one_shot


def test_expected_damage_doubled_for_double_shooting():
    ranger = Unit.from_type("Ranger", 0, 4, 4)
    target = Unit.from_type("Goblin", 1, 8, 4)
    b = _battle([ranger, target])
    ed = b.expected_damage(ranger, target, ranged=True)
    # Without double, a normal archer with same stats.
    archer = Unit.from_type("Archer", 0, 4, 4)
    ea = _battle([archer, target]).expected_damage(archer, target, ranged=True)
    # Ranger expected should be roughly 2× Archer (same damage, similar speed bonus).
    assert abs(ed / ea - 2.0) < 0.15  # allow speed-difference margin


# ── combat hooks: double_melee ───────────────────────────────────────

def test_double_melee_attacks_twice():
    paladin = Unit.from_type("Paladin", 0, 4, 4)
    target = Unit.from_type("Goblin", 1, 5, 4, count=100)
    target.retaliated = True  # skip retaliation to isolate double hit
    b = _battle([paladin, target])
    random.seed(0)
    r = b.execute(AttackAction(paladin, target, (4, 4), ranged=False))
    assert "+2nd hit" in r["desc"]
    assert r["dmg"] > paladin.count * paladin.damage_avg  # more than one hit


def test_double_melee_happens_after_retaliation():
    wolf = Unit.from_type("Wolf", 0, 4, 4)          # double_melee, wide
    target = Unit("Tgt", 1, 5, 4, attack=5, defense=5, hp=500, speed=4,
                  damage=50, count=1, is_archer=False, is_flying=False)
    b = _battle([wolf, target])
    r = b.execute(AttackAction(wolf, target, (4, 4), ranged=False))
    # Target retaliates (ret_dmg > 0), then wolf hits second time.
    assert r["ret_dmg"] > 0, "target should retaliate between hits"
    assert "+2nd hit" in r["desc"]


def test_expected_damage_175x_for_double_melee():
    paladin = Unit.from_type("Paladin", 0, 4, 4)
    target = Unit.from_type("Goblin", 1, 5, 4)
    b = _battle([paladin, target])
    ed = b.expected_damage(paladin, target, ranged=False)
    # A non-double unit with same attack/defense would give base damage.
    base = paladin.count * paladin.damage_avg * BattleState._damage_mult(paladin, target)
    assert abs(ed / base - 1.75) < 0.01


# ── combat hooks: two_cell_melee ─────────────────────────────────────

def test_two_cell_melee_splashes_behind_target():
    cyclops = Unit.from_type("Cyclops", 0, 3, 4)     # adjacent to target
    target = Unit.from_type("Goblin", 1, 4, 4, count=100)
    target.retaliated = True  # skip retaliation so Cyclops survives
    behind_unit = Unit.from_type("Goblin", 1, 5, 4, count=100)
    b = _battle([cyclops, target, behind_unit])
    r = b.execute(AttackAction(cyclops, target, (3, 4), ranged=False))
    assert "splash" in r["desc"]
    assert r.get("splash_dmg", 0) > 0
    assert behind_unit._total_hp < behind_unit.count * behind_unit.max_hp


def test_two_cell_melee_no_friendly_fire():
    cyclops = Unit.from_type("Cyclops", 0, 2, 4)
    target = Unit.from_type("Goblin", 1, 4, 4, count=50)
    friendly_behind = Unit.from_type("Goblin", 0, 6, 4, count=50)  # same team as cyclops
    b = _battle([cyclops, target, friendly_behind])
    r = b.execute(AttackAction(cyclops, target, (2, 4), ranged=False))
    assert "splash" not in r["desc"]
    assert friendly_behind._total_hp == friendly_behind.count * friendly_behind.max_hp


# ── cell_behind geometry ─────────────────────────────────────────────

def test_cell_behind_east():
    assert G.cell_behind((0, 4), (1, 4)) == (2, 4)


def test_cell_behind_diagonal():
    behind = G.cell_behind((3, 3), (4, 4))
    assert behind is not None
    assert G.distance((4, 4), behind) == 1


def test_cell_behind_at_edge_returns_none():
    # From (9,4) eastward, behind (10,4) would be (11,4) — off-grid.
    assert G.cell_behind((8, 4), (10, 4)) is None

"""Spell engine tests — damage, timed effects, hero, cast execution."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.hex_grid import HexGrid
from engine.battle_state import BattleState
from engine.unit import Unit
from engine.hero import Hero
from engine.actions import CastAction
from engine.spells import SPELLS, spell_damage, make_effect


def test_spell_damage_scales_with_power():
    assert spell_damage(SPELLS["Lightning Bolt"], 3) == 75   # 25 * 3
    assert spell_damage(SPELLS["Magic Arrow"], 4) == 40      # 10 * 4


def test_make_effect_only_for_non_damage():
    assert make_effect(SPELLS["Magic Arrow"], 3) is None
    eff = make_effect(SPELLS["Haste"], 3)
    assert eff.remaining == 3 and eff.speed_delta == 2


# ── unit effects ────────────────────────────────────────────────────

def test_haste_and_slow_change_effective_speed():
    u = Unit.from_type("Swordsman", 0, 0, 0)  # base speed 4
    u.add_effect(make_effect(SPELLS["Haste"], 3))
    assert u.speed == 6
    u.add_effect(make_effect(SPELLS["Slow"], 3))   # net 4 + 2 - 2
    assert u.speed == 4


def test_speed_floored_at_one():
    u = Unit.from_type("Archer", 0, 0, 0)  # base speed 3
    for _ in range(3):
        # stack slow shouldn't matter (one effect per name), but force low base
        u.add_effect(make_effect(SPELLS["Slow"], 3))
    assert u.speed == max(1, 3 - 2)


def test_bless_and_curse_change_damage_factor():
    u = Unit.from_type("Swordsman", 0, 0, 0)
    u.add_effect(make_effect(SPELLS["Bless"], 3))
    assert abs(u.damage_factor - 1.2) < 1e-9
    u.add_effect(make_effect(SPELLS["Curse"], 3))
    assert abs(u.damage_factor - 1.2 * 0.8) < 1e-9


def test_effects_expire_after_duration():
    u = Unit.from_type("Swordsman", 0, 0, 0)
    u.add_effect(make_effect(SPELLS["Haste"], 2))
    u.tick_effects(); assert u.has_effect("Haste")   # 2 -> 1
    u.tick_effects(); assert not u.has_effect("Haste")  # 1 -> 0, removed
    assert u.speed == 4


def test_add_effect_refreshes_same_name():
    u = Unit.from_type("Swordsman", 0, 0, 0)
    u.add_effect(make_effect(SPELLS["Haste"], 1))
    u.add_effect(make_effect(SPELLS["Haste"], 5))
    assert sum(1 for e in u.effects if e.name == "Haste") == 1


# ── hero ────────────────────────────────────────────────────────────

def test_hero_can_cast_once_per_round():
    h = Hero(power=3, max_spell_points=15)
    arrow = SPELLS["Magic Arrow"]
    assert h.can_cast(arrow)
    h.cast(arrow)
    assert not h.can_cast(arrow)        # already cast this round
    h.reset_round()
    assert h.can_cast(arrow)
    assert h.spell_points == 15 - 3     # cost deducted


def test_hero_cannot_cast_without_points():
    h = Hero(power=3, max_spell_points=2)
    assert not h.can_cast(SPELLS["Magic Arrow"])  # cost 3 > 2


# ── cast execution ──────────────────────────────────────────────────

def _battle(units, hero0=None):
    return BattleState(HexGrid(), units, heroes={0: hero0, 1: None})


def test_damage_cast_reduces_hp_and_spends_points():
    h = Hero(power=3)
    target = Unit.from_type("Archer", 1, 8, 4)
    b = _battle([Unit.from_type("Swordsman", 0, 1, 4), target], hero0=h)
    before = target._total_hp
    r = b.execute(CastAction(0, SPELLS["Lightning Bolt"], target))
    assert r["dmg"] == 75
    assert target._total_hp == before - 75
    assert h.spell_points == 15 - 7


def test_buff_cast_applies_effect():
    h = Hero(power=3)
    friend = Unit.from_type("Swordsman", 0, 1, 4)
    b = _battle([friend, Unit.from_type("Archer", 1, 8, 4)], hero0=h)
    b.execute(CastAction(0, SPELLS["Haste"], friend))
    assert friend.has_effect("Haste")
    assert friend.speed == 6


def test_start_round_ticks_effects_and_resets_hero():
    h = Hero(power=3)
    friend = Unit.from_type("Swordsman", 0, 1, 4)
    b = _battle([friend, Unit.from_type("Archer", 1, 8, 4)], hero0=h)
    friend.add_effect(make_effect(SPELLS["Haste"], 1))
    h.cast(SPELLS["Magic Arrow"])
    b.start_round()
    assert not friend.has_effect("Haste")   # ticked to 0
    assert h.can_cast(SPELLS["Magic Arrow"])  # round reset

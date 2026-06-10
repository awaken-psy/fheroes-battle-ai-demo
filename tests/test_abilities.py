"""Special ability tests — retaliation, drain, spell_caster, scoring, new abilities."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.hex_grid import HexGrid
from engine.battle_state import BattleState
from engine.unit import Unit
from engine.actions import AttackAction
from ai.classic.scoring import threat

G = HexGrid()


def _battle(units):
    return BattleState(G, units)


# ── unlimited retaliation ───────────────────────────────────────────

def test_unlimited_retaliation_strikes_every_attacker():
    griffin = Unit.from_type("Griffin", 1, 5, 4)          # unlimited_retaliation
    a1 = Unit.from_type("Archer", 0, 4, 4)                # weak melee, won't kill
    a2 = Unit.from_type("Archer", 0, 5, 3)
    b = _battle([griffin, a1, a2])
    r1 = b.execute(AttackAction(a1, griffin, (4, 4), ranged=False))
    r2 = b.execute(AttackAction(a2, griffin, (5, 3), ranged=False))
    assert griffin.is_alive
    assert r1["ret_dmg"] > 0 and r2["ret_dmg"] > 0


def test_normal_unit_retaliates_only_once_per_round():
    sword = Unit.from_type("Swordsman", 1, 5, 4)
    a1 = Unit.from_type("Archer", 0, 4, 4)
    a2 = Unit.from_type("Archer", 0, 5, 3)
    b = _battle([sword, a1, a2])
    r1 = b.execute(AttackAction(a1, sword, (4, 4), ranged=False))
    r2 = b.execute(AttackAction(a2, sword, (5, 3), ranged=False))
    assert r1["ret_dmg"] > 0 and r2["ret_dmg"] == 0


# ── hp drain ────────────────────────────────────────────────────────

def test_hp_drain_heals_attacker():
    # Vampire Lord (not Vampire) has hp_drain in the original.
    vampire = Unit.from_type("Vampire Lord", 0, 4, 4)
    vampire.take_damage(45)            # leave a wounded creature to heal
    target = Unit.from_type("Pikeman", 1, 5, 4)
    target.retaliated = True           # isolate: no retaliation muddying HP
    b = _battle([vampire, target])
    before = vampire._total_hp
    r = b.execute(AttackAction(vampire, target, (4, 4), ranged=False))
    assert vampire._total_hp > before
    assert "drains" in r["desc"]


def test_no_enemy_retaliation_skips_counter():
    # Vampire (no hp_drain) has no_enemy_retaliation — no counterattack.
    vampire = Unit.from_type("Vampire", 0, 4, 4)
    target = Unit.from_type("Pikeman", 1, 5, 4)
    b = _battle([vampire, target])
    r = b.execute(AttackAction(vampire, target, (4, 4), ranged=False))
    assert r["ret_dmg"] == 0


def test_heal_never_resurrects():
    u = Unit.from_type("Troll", 0, 0, 0)
    u.take_damage(10_000)
    assert not u.is_alive
    assert u.heal(1000) == 0


# ── death gaze (legacy, kept for backward compat) ────────────────

def test_death_gaze_kills_extra():
    # Create a unit with the legacy death_gaze ability for this test.
    gazer = Unit("Gazer", 0, 4, 4, attack=8, defense=9, hp=35, speed=4,
                 damage_min=6, damage_max=10, is_archer=False, is_flying=False,
                 abilities=["death_gaze"], count=5)
    target = Unit.from_type("Pikeman", 1, 5, 4)
    b = _battle([gazer, target])
    r = b.execute(AttackAction(gazer, target, (4, 4), ranged=False))
    assert "gaze" in r["desc"]
    assert r["killed"] >= 1


# ── self heal (regeneration) ────────────────────────────────────────

def test_self_heal_regenerates_at_round_start():
    troll = Unit.from_type("Troll", 0, 1, 4)
    troll.take_damage(15)              # wounded
    before = troll._total_hp
    b = _battle([troll, Unit.from_type("Archer", 1, 9, 4)])
    b.start_round()
    assert troll._total_hp > before


# ── scoring reflects abilities ──────────────────────────────────────

def test_base_strength_includes_ability_terms():
    # Griffin: unlimited_retaliation → damage *1.25, flying → special += 0.3
    import math
    griffin = Unit.from_type("Griffin", 0, 0, 0)
    # damage_avg = 4, hp = 25, unlimited_retal → *1.25
    dmg_pot = 4.0 * 1.25
    # special: flying += 0.3, speed=4 (AVERAGE) → no speed diff
    special = 1.0 + 0.3
    expected = math.sqrt(dmg_pot * 25) * special
    assert abs(griffin._base_strength - expected) < 1e-6


def test_threat_scaled_by_attacker_abilities():
    # Genie has enemy_halving → threat x2 (same multiplier as old death_gaze)
    genie = Unit.from_type("Genie", 0, 4, 4)
    target = Unit.from_type("Pikeman", 1, 5, 4)  # adjacent -> dist_mod 1
    b = _battle([genie, target])
    raw = b.expected_damage(genie, target, ranged=False)
    assert abs(threat(b, genie, target) - raw * 2) < 1e-6

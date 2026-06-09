"""Damage formula tests — the deterministic multiplier and expected/roll split.

Mirrors fheroes2's combat math:
  attack > defense -> +10% per point, capped at 3.0x
  attack < defense -> -5%  per point, floored at 0.3x
  archer attacking in melee -> additional 0.5x penalty
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.hex_grid import HexGrid
from engine.battle_state import BattleState
from engine.unit import Unit


def _unit(name="U", team=0, col=0, row=0, **stats):
    base = dict(attack=5, defense=5, hp=10, speed=3, damage=2, count=10,
                is_archer=False, is_flying=False)
    base.update(stats)
    return Unit(name, team, col, row, **base)


def _battle(a, b):
    return BattleState(HexGrid(), [a, b])


# ── deterministic multiplier ──────────────────────────────────────

def test_attacker_advantage_plus_10_percent():
    a = _unit("A", 0, 0, 0, attack=7)
    d = _unit("D", 1, 1, 0, defense=5)
    # diff 2 -> 1 + 0.1*2 = 1.2
    assert BattleState._damage_mult(a, d) == 1.2


def test_defender_advantage_minus_5_percent():
    a = _unit("A", 0, 0, 0, attack=5)
    d = _unit("D", 1, 1, 0, defense=8)
    # diff 3 -> 1 - 0.05*3 = 0.85
    assert abs(BattleState._damage_mult(a, d) - 0.85) < 1e-9


def test_multiplier_capped_at_3x():
    a = _unit("A", 0, 0, 0, attack=40)
    d = _unit("D", 1, 1, 0, defense=5)
    # diff 35 -> 1 + 3.5 = 4.5, capped to 3.0
    assert BattleState._damage_mult(a, d) == 3.0


def test_multiplier_floored_at_0_3x():
    a = _unit("A", 0, 0, 0, attack=1)
    d = _unit("D", 1, 1, 0, defense=40)
    # diff 39 -> 1 - 1.95 = -0.95, floored to 0.3
    assert BattleState._damage_mult(a, d) == 0.3


def test_archer_melee_penalty_half():
    a = _unit("Arc", 0, 0, 0, attack=5, is_archer=True)
    d = _unit("D", 1, 1, 0, defense=5)  # equal -> base mult 1.0
    assert BattleState._damage_mult(a, d, ranged=True) == 1.0
    assert BattleState._damage_mult(a, d, ranged=False) == 0.5


# ── expected_damage (deterministic) ────────────────────────────────

def test_expected_damage_is_base_times_mult():
    a = _unit("A", 0, 0, 0, attack=7, damage=3, count=20)  # base 60
    d = _unit("D", 1, 1, 0, defense=5)                      # mult 1.2
    b = _battle(a, d)
    assert b.expected_damage(a, d) == int(60 * 1.2)  # 72


def test_expected_damage_is_deterministic():
    a = _unit("A", 0, 0, 0, attack=7, damage=3, count=20)
    d = _unit("D", 1, 1, 0, defense=5)
    b = _battle(a, d)
    assert b.expected_damage(a, d) == b.expected_damage(a, d)


def test_expected_damage_min_one():
    a = _unit("A", 0, 0, 0, attack=1, damage=1, count=1)
    d = _unit("D", 1, 1, 0, defense=40)
    b = _battle(a, d)
    assert b.expected_damage(a, d) >= 1


# ── roll_damage (per-creature min/max spread) ───────────────────────

def test_roll_damage_no_spread_when_min_equals_max():
    # A single ``damage`` value means min == max, so no random spread.
    a = _unit("A", 0, 0, 0, attack=7, damage=3, count=20)  # base 60, mult 1.2
    d = _unit("D", 1, 1, 0, defense=5)
    b = _battle(a, d)
    for i in range(200):
        random.seed(i)
        assert b.roll_damage(a, d) == int(60 * 1.2)


def test_roll_damage_respects_min_max_range():
    # attack == defense -> mult 1.0; 10 creatures each roll in [2, 8].
    a = _unit("A", 0, 0, 0, attack=5, damage_min=2, damage_max=8, count=10)
    d = _unit("D", 1, 1, 0, defense=5)
    b = _battle(a, d)
    vals = set()
    for i in range(300):
        random.seed(i)
        vals.add(b.roll_damage(a, d))
    assert len(vals) > 1                       # spread is present
    assert min(vals) >= 10 * 2 and max(vals) <= 10 * 8


def test_roll_damage_reproducible_with_seed():
    a = _unit("A", 0, 0, 0, attack=7, damage=3, count=20)
    d = _unit("D", 1, 1, 0, defense=5)
    b = _battle(a, d)
    random.seed(42); first = b.roll_damage(a, d)
    random.seed(42); second = b.roll_damage(a, d)
    assert first == second


def test_calc_damage_is_roll_alias():
    assert BattleState.calc_damage is BattleState.roll_damage


# ── strength formula (fheroes2 GetMonsterStrength / GetStrength) ─────

def test_base_strength_plain_unit():
    # damage 2, hp 8 -> sqrt(16)=4; speed=AVERAGE, not archer/flyer -> special 1.0
    u = _unit("P", 0, 0, 0, damage=2, hp=8, speed=4, is_archer=False, is_flying=False)
    assert abs(u._base_strength - 4.0) < 1e-9


def test_base_strength_archer_and_speed_remap():
    # damage 2, hp 10 -> sqrt(20); archer +0.4; speed 3 (diff -1) -> -0.1 => special 1.3
    import math
    u = _unit("Arc", 0, 0, 0, damage=2, hp=10, speed=3, is_archer=True)
    assert abs(u._base_strength - math.sqrt(20) * 1.3) < 1e-9


def test_base_strength_flyer_bonus_and_fast():
    # flying +0.3; speed 6 (diff +2) -> +0.10 => special 1.4
    import math
    u = _unit("Gr", 0, 0, 0, damage=3, hp=12, speed=6, is_flying=True)
    assert abs(u._base_strength - math.sqrt(36) * 1.4) < 1e-9


def test_monster_strength_applies_attack_defense():
    # base 4.0 (as above) ; atk 10 def 0 -> (1 + 1.0 + 0) * 4 = 8.0
    u = _unit("P", 0, 0, 0, damage=2, hp=8, speed=4, attack=10, defense=0)
    assert abs(u.monster_strength - 8.0) < 1e-9


def test_stack_strength_scales_with_count():
    u = _unit("P", 0, 0, 0, damage=2, hp=8, speed=4, attack=10, defense=0, count=5)
    assert abs(u.strength - 8.0 * 5) < 1e-9


def test_dead_unit_has_zero_strength():
    u = _unit("P", 0, 0, 0, count=3)
    u.take_damage(10_000)
    assert not u.is_alive
    assert u.strength == 0

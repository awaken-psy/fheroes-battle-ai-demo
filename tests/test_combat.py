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


# ── roll_damage (random spread) ─────────────────────────────────────

def test_roll_damage_within_spread():
    a = _unit("A", 0, 0, 0, attack=7, damage=3, count=20)  # base 60, mult 1.2
    d = _unit("D", 1, 1, 0, defense=5)
    b = _battle(a, d)
    lo = int(60 * 1.2 * 0.85)
    hi = int(60 * 1.2 * 1.15) + 1
    for i in range(200):
        random.seed(i)
        dmg = b.roll_damage(a, d)
        assert lo <= dmg <= hi


def test_roll_damage_reproducible_with_seed():
    a = _unit("A", 0, 0, 0, attack=7, damage=3, count=20)
    d = _unit("D", 1, 1, 0, defense=5)
    b = _battle(a, d)
    random.seed(42); first = b.roll_damage(a, d)
    random.seed(42); second = b.roll_damage(a, d)
    assert first == second


def test_calc_damage_is_roll_alias():
    assert BattleState.calc_damage is BattleState.roll_damage

"""Morale / luck tests — engine-only effects the AI must NOT evaluate."""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.hex_grid import HexGrid
from engine.battle_state import BattleState
from engine.unit import Unit
from ai.planner import BattleAI


def _battle(**kw):
    a = Unit.from_type("Swordsman", 0, 1, 4)
    d = Unit.from_type("Archer", 1, 2, 4)
    return BattleState(HexGrid(), [a, d], **kw), a, d


# ── luck ────────────────────────────────────────────────────────────

def test_no_luck_keeps_damage_in_normal_spread():
    b, a, d = _battle()  # luck 0
    base = b.expected_damage(a, d) / b._damage_mult(a, d)  # count*damage*factor
    hi = int(base * b._damage_mult(a, d) * 1.15) + 1
    for s in range(300):
        random.seed(s)
        assert b.roll_damage(a, d) <= hi


def test_good_luck_can_double_damage():
    b, a, d = _battle(luck={0: 3, 1: 0})
    normal_hi = b.expected_damage(a, d) * 1.15
    doubled = any((random.seed(s) or b.roll_damage(a, d)) > normal_hi * 1.5
                  for s in range(500))
    assert doubled


def test_bad_luck_can_halve_damage():
    b, a, d = _battle(luck={0: -3, 1: 0})
    normal_lo = b.expected_damage(a, d) * 0.85
    halved = any((random.seed(s) or b.roll_damage(a, d)) < normal_lo * 0.7
                 for s in range(500))
    assert halved


def test_expected_damage_ignores_luck():
    b0, a0, d0 = _battle(luck={0: 0, 1: 0})
    b3, a3, d3 = _battle(luck={0: 3, 1: 0})
    assert b0.expected_damage(a0, d0) == b3.expected_damage(a3, d3)


# ── morale ──────────────────────────────────────────────────────────

def test_morale_sign_bounds():
    b_pos, _, _ = _battle(morale={0: 3, 1: 0})
    b_neg, _, _ = _battle(morale={0: -3, 1: 0})
    b_zero, _, _ = _battle()
    for s in range(300):
        random.seed(s); assert b_pos.roll_morale(0) in (0, 1)
        random.seed(s); assert b_neg.roll_morale(0) in (0, -1)
        random.seed(s); assert b_zero.roll_morale(0) == 0


def test_good_and_bad_morale_actually_trigger():
    b_pos, _, _ = _battle(morale={0: 3, 1: 0})
    b_neg, _, _ = _battle(morale={0: -3, 1: 0})
    assert any((random.seed(s) or b_pos.roll_morale(0)) == 1 for s in range(200))
    assert any((random.seed(s) or b_neg.roll_morale(0)) == -1 for s in range(200))


# ── the AI must not react to morale / luck ──────────────────────────

def test_ai_decision_unaffected_by_morale_and_luck():
    ai = BattleAI()
    units_a = [Unit.from_type("Swordsman", 0, 1, 4), Unit.from_type("Archer", 1, 8, 4)]
    units_b = [Unit.from_type("Swordsman", 0, 1, 4), Unit.from_type("Archer", 1, 8, 4)]
    plain = BattleState(HexGrid(), units_a)
    juiced = BattleState(HexGrid(), units_b, morale={0: 3, 1: 3}, luck={0: 3, 1: 3})
    assert ai.decide(plain, plain.alive(0)[0])[1] == ai.decide(juiced, juiced.alive(0)[0])[1]

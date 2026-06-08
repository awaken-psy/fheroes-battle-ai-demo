"""Threat scoring tests — expected damage discounted by distance.

Core of fheroes2's evaluateThreatForUnit: threat = expected_damage / distMod,
where distMod is 1.0 for shooters/flyers or melee units within reach, and
1.5*distance/speed for melee units that cannot strike this turn.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.hex_grid import HexGrid
from engine.battle_state import BattleState
from engine.unit import Unit
from ai.classic.scoring import threat


def _battle(units):
    return BattleState(HexGrid(), units)


def test_archer_threat_ignores_distance():
    near = Unit.from_type("Swordsman", 1, 1, 4)
    far = Unit.from_type("Swordsman", 1, 10, 4)
    arc = Unit.from_type("Archer", 0, 0, 4)
    b = _battle([arc, near, far])
    # shooters reach anywhere -> distMod 1.0 -> threat is the raw expected (ranged) damage
    assert threat(b, arc, near) == threat(b, arc, far)
    assert threat(b, arc, near) == b.expected_damage(arc, near, ranged=True)


def test_melee_threat_in_range_is_undiscounted():
    sw = Unit.from_type("Swordsman", 0, 4, 4)   # speed 4 -> range 5
    enemy = Unit.from_type("Archer", 1, 6, 4)   # within range
    b = _battle([sw, enemy])
    assert b.grid.distance(sw.pos, enemy.pos) <= sw.speed + 1
    assert threat(b, sw, enemy) == b.expected_damage(sw, enemy, ranged=False)


def test_melee_threat_out_of_range_is_discounted():
    sw = Unit.from_type("Swordsman", 0, 0, 4)    # speed 4
    enemy = Unit.from_type("Archer", 1, 10, 4)
    b = _battle([sw, enemy])
    dist = b.grid.distance(sw.pos, enemy.pos)
    assert dist > sw.speed + 1
    dist_mod = 1.5 * dist / sw.speed
    expected = b.expected_damage(sw, enemy, ranged=False) / dist_mod
    assert abs(threat(b, sw, enemy) - expected) < 1e-9
    # and a closer-but-still-out-of-range target is rated higher
    closer = Unit.from_type("Archer", 1, 7, 4)
    b2 = _battle([sw, closer])
    assert threat(b2, sw, closer) > threat(b, sw, enemy)

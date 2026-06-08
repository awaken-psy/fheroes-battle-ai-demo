"""Berserk tests — attack/move toward the nearest unit, friend or foe."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.hex_grid import HexGrid
from engine.battle_state import BattleState
from engine.unit import Unit
from engine.spells import Effect
from engine.actions import AttackAction, MoveAction
from ai.classic import BattleAI

AI = BattleAI()


def _berserk(units, actor):
    b = BattleState(HexGrid(), units)
    actor.add_effect(Effect("Berserk", 3))
    return AI.decide(b, actor)


def test_berserk_attacks_nearest_even_if_friendly():
    actor = Unit.from_type("Swordsman", 0, 5, 4)
    friend = Unit.from_type("Archer", 0, 6, 4)    # adjacent friendly
    enemy = Unit.from_type("Cavalry", 1, 1, 4)    # distant enemy
    action, desc = _berserk([actor, friend, enemy], actor)
    assert isinstance(action, AttackAction)
    assert action.target is friend   # nearest unit, ignoring allegiance
    assert "BERSERK" in desc


def test_berserk_archer_shoots_nearest():
    actor = Unit.from_type("Archer", 0, 5, 4)
    enemy_near = Unit.from_type("Cavalry", 1, 7, 4)
    friend_far = Unit.from_type("Swordsman", 0, 0, 4)
    action, _ = _berserk([actor, enemy_near, friend_far], actor)
    assert isinstance(action, AttackAction)
    assert action.ranged is True
    assert action.target is enemy_near


def test_berserk_moves_toward_nearest_when_out_of_reach():
    actor = Unit.from_type("Swordsman", 0, 1, 4)   # speed 4
    far = Unit.from_type("Cavalry", 1, 10, 4)
    action, desc = _berserk([actor, far], actor)
    assert isinstance(action, MoveAction)
    assert action.path[-1][0] > action.path[0][0]   # advances toward target

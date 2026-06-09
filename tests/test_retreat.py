"""Retreat tests — threshold, farewell spell, and battle termination."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.hex_grid import HexGrid
from engine.battle_state import BattleState
from engine.unit import Unit
from engine.hero import Hero
from engine.actions import RetreatAction, CastAction
from ai.classic.evaluation import AIState, analyze
from ai.classic import BattleAI
from ai.classic.retreat import should_retreat, retreat_ratio

AI = BattleAI()


def _battle(units, **kw):
    return BattleState(HexGrid(), units, **kw)


def _state(battle, team):
    return analyze(battle, battle.alive(team)[0])


# ── threshold ───────────────────────────────────────────────────────

def test_retreat_ratio_by_difficulty():
    assert retreat_ratio("Normal") == 100.0 / 7.5
    assert retreat_ratio("Impossible") == 100.0 / 10.0
    assert retreat_ratio("???") == retreat_ratio("Normal")  # default


def test_even_armies_do_not_retreat():
    b = _battle([Unit.from_type("Swordsman", 0, 1, 4),
                 Unit.from_type("Swordsman", 1, 9, 4)])
    assert not should_retreat(_state(b, 0))


def test_hopeless_army_retreats():
    # Retreat only when the enemy is many times stronger (ratio ~13.3 on Normal).
    s = AIState()
    s.my_army, s.enemy_army, s.enemy_shooters = 100.0, 2000.0, 0.0
    assert should_retreat(s)             # 100 * 13.3 < 2000
    s.enemy_army = 1000.0
    assert not should_retreat(s)         # 100 * 13.3 > 1000


def test_retreat_threshold_tracks_difficulty():
    s = AIState()
    s.my_army, s.enemy_army, s.enemy_shooters = 100.0, 1100.0, 0.0
    # Easy ratio 16.7 -> continue; Impossible ratio 10 -> retreat
    assert not should_retreat(s, "Easy")
    assert should_retreat(s, "Impossible")


# ── RetreatAction ends the battle ───────────────────────────────────

def test_retreat_action_ends_battle_with_loser():
    b = _battle([Unit.from_type("Swordsman", 0, 1, 4),
                 Unit.from_type("Swordsman", 1, 9, 4)])
    assert not b.is_over()
    b.execute(RetreatAction(0))
    assert b.is_over()
    assert b.winner() == 1   # team 0 fled


# ── check_retreat (planner) ─────────────────────────────────────────

def test_check_retreat_needs_a_hero():
    units = [Unit.from_type("Archer", 0, 0, 4)]
    units += [Unit.from_type("Cavalry", 1, 8, r) for r in (1, 3, 5, 7)]
    b = _battle(units)  # no heroes
    assert AI.check_retreat(b, b.alive(0)[0]) is None


def test_check_retreat_returns_farewell_and_retreat():
    weak = Unit.from_type("Archer", 0, 0, 4)
    weak.take_damage(71)    # down to 1 creature (80 total hp → 9 left) → hopeless
    units = [weak] + [Unit.from_type("Cavalry", 1, 8, r) for r in (1, 3, 5, 7)]
    hero = Hero(power=3, spells=["Lightning Bolt"])
    b = _battle(units, heroes={0: hero, 1: None})
    result = AI.check_retreat(b, b.alive(0)[0])
    assert result is not None
    farewell, retreat_action = result
    assert isinstance(retreat_action, RetreatAction)
    # farewell is a damage spell cast on an enemy
    assert farewell is not None
    cast = farewell[0]
    assert isinstance(cast, CastAction)
    assert cast.spell.name == "Lightning Bolt"
    assert cast.target.team == 1


def test_strong_army_does_not_trigger_retreat():
    units = [Unit.from_type("Cavalry", 0, 1, r) for r in (1, 3, 5)]
    units += [Unit.from_type("Archer", 1, 9, 4)]
    hero = Hero()
    b = _battle(units, heroes={0: hero, 1: None})
    assert AI.check_retreat(b, b.alive(0)[0]) is None

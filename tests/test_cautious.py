"""Cautious positioning tests — advance to the safest reachable cell."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.hex_grid import HexGrid
from engine.battle_state import BattleState
from engine.unit import Unit
from engine.actions import MoveAction
from ai.classic import BattleAI
from ai.classic.evaluation import analyze

AI = BattleAI()
G = HexGrid()


def _slow_melee(col, row, team=1):
    """A slow melee unit (speed 3) for cautious-pathfinding tests."""
    return Unit("SlowEnemy", team, col, row,
                attack=5, defense=5, hp=10, speed=3, damage=2, count=10,
                is_archer=False, is_flying=False)


def test_state_is_cautious_without_enemy_shooters():
    b = BattleState(G, [Unit.from_type("Swordsman", 0, 0, 4),
                        _slow_melee(6, 4)])
    assert analyze(b, b.alive(0)[0]).cautious


def test_safest_step_stops_outside_slow_enemy_reach():
    sw = Unit.from_type("Swordsman", 0, 0, 4)        # speed 4
    enemies = [_slow_melee(6, r) for r in (2, 4, 6)]  # reach 3
    b = BattleState(G, [sw] + enemies)
    occ = b.occupied(exclude=sw)
    tc = G.nearest_cell_next_to(sw.pos, enemies[0].pos, occ, False, sw.speed * 3)
    seg = G.find_path(sw.pos, tc, occ, False, sw.speed * 3)[:sw.speed + 1]

    naive = seg[-1]
    safe = AI._safest_step_on_path(b, sw, seg, enemies)[-1]
    assert AI._cell_threat(b, sw, safe, enemies) < AI._cell_threat(b, sw, naive, enemies)
    assert AI._cell_threat(b, sw, safe, enemies) == 0.0


def test_cautious_decide_does_not_charge_into_reach():
    sw = Unit.from_type("Swordsman", 0, 0, 4)
    enemies = [_slow_melee(6, r) for r in (2, 4, 6)]
    b = BattleState(G, [sw] + enemies)
    action, _ = AI.decide(b, sw)
    assert isinstance(action, MoveAction)
    # lands on a cell no slow enemy can reach next turn
    assert AI._cell_threat(b, sw, action.path[-1], enemies) == 0.0


def test_non_cautious_unit_advances_further():
    # Give the enemy archers so the side is NOT cautious; the chaser presses on.
    sw = Unit.from_type("Swordsman", 0, 0, 4)
    enemies = [Unit.from_type("Archer", 1, 6, r) for r in (2, 4, 6)]
    b = BattleState(G, [sw] + enemies)
    assert not analyze(b, sw).cautious
    action, _ = AI.decide(b, sw)
    assert isinstance(action, MoveAction)
    assert action.path[-1][0] == 4   # full 4-step advance, no safety truncation

"""Planner decision regression tests.

Fixed battlefield snapshots -> assert the AI picks the expected action.
Locks in behaviour that has been aligned with fheroes2 so later refactors
(e.g. the M2 strength/threat rework) can't silently regress it.

Decisions are deterministic because the planner reasons about
``expected_damage`` (the random spread only happens at execution time).
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.hex_grid import HexGrid
from engine.battle_state import BattleState
from engine.unit import Unit
from engine.actions import MoveAction, AttackAction, SkipAction
from ai.planner import BattleAI

GRID = HexGrid()
AI = BattleAI()


def U(type_name, team, col, row):
    return Unit.from_type(type_name, team, col, row)


def decide(units, actor):
    return AI.decide(BattleState(GRID, units), actor)


# ── archer behaviour ───────────────────────────────────────────────

def test_free_archer_shoots_only_enemy():
    arc = U("Archer", 0, 0, 4)
    enemy = U("Swordsman", 1, 8, 4)
    action, _ = decide([arc, enemy], arc)
    assert isinstance(action, AttackAction)
    assert action.ranged is True
    assert action.target is enemy


def test_blocked_archer_attacks_in_melee():
    arc = U("Archer", 0, 4, 4)
    cav = U("Cavalry", 1, 5, 4)  # adjacent, far too fast to flee from
    action, desc = decide([arc, cav], arc)
    assert isinstance(action, AttackAction)
    assert action.ranged is False
    assert action.target is cav
    assert "blocked" in desc


# ── melee offense ──────────────────────────────────────────────────

def test_melee_attacks_adjacent_target():
    sw = U("Swordsman", 0, 4, 4)
    archer = U("Archer", 1, 5, 4)  # adjacent
    action, desc = decide([sw, archer], sw)
    assert isinstance(action, AttackAction)
    assert action.ranged is False
    assert action.target is archer
    assert "[ME]" in desc


def test_melee_chases_distant_archer():
    sw = U("Swordsman", 0, 1, 4)
    archer = U("Archer", 1, 9, 4)  # far away
    action, desc = decide([sw, archer], sw)
    assert isinstance(action, MoveAction)
    # path must advance toward the enemy (higher column)
    assert action.path[-1][0] > action.path[0][0]
    assert "chasing" in desc


# ── melee defense (cover an archer) ────────────────────────────────

def test_melee_covers_friendly_archer():
    guard = U("Swordsman", 0, 1, 2)
    friend_archer = U("Archer", 0, 0, 4)
    enemy = U("Swordsman", 1, 9, 2)  # distant -> defensive posture
    action, desc = decide([guard, friend_archer, enemy], guard)
    assert isinstance(action, MoveAction)
    assert "[DEF]" in desc
    assert "covers" in desc


# ── decoupling guard ────────────────────────────────────────────────

def test_engine_and_ai_import_without_pygame():
    """engine + ai must import even when pygame is unavailable."""
    code = (
        "import sys\n"
        "class B:\n"
        "    def find_module(self, name, path=None):\n"
        "        return self if name == 'pygame' or name.startswith('pygame.') else None\n"
        "    def load_module(self, name):\n"
        "        raise ImportError('pygame imported by engine/ai')\n"
        "sys.meta_path.insert(0, B())\n"
        "import engine.hex_grid, engine.battle_state, engine.unit, engine.actions\n"
        "import ai.planner, ai.scoring, ai.evaluation\n"
    )
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    proc = subprocess.run([sys.executable, "-c", code], cwd=repo_root,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

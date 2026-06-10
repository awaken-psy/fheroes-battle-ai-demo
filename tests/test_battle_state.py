"""Battle state tests — turn order interleaving and anti-stalemate retreat."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.hex_grid import HexGrid
from engine.battle_state import BattleState
from engine.unit import Unit


def _battle(units, **kw):
    return BattleState(HexGrid(), units, **kw)


# ── turn order (faithful merge of two speed-sorted queues) ──────────

def test_equal_speed_units_alternate_between_teams():
    units = [
        Unit.from_type("Swordsman", 0, 0, 1),
        Unit.from_type("Swordsman", 0, 0, 3),
        Unit.from_type("Swordsman", 1, 10, 1),
        Unit.from_type("Swordsman", 1, 10, 3),
    ]
    b = _battle(units, first_team=0)
    teams = [u.team for u in b.turn_order()]
    assert teams == [0, 1, 0, 1]


def test_first_team_breaks_the_initiative_tie():
    units = [
        Unit.from_type("Swordsman", 0, 0, 1),
        Unit.from_type("Swordsman", 1, 10, 1),
    ]
    assert _battle(units, first_team=0).turn_order()[0].team == 0
    assert _battle(units, first_team=1).turn_order()[0].team == 1


def test_faster_unit_acts_first_regardless_of_team():
    slow = Unit.from_type("Pikeman", 0, 0, 1)    # speed 4
    fast = Unit.from_type("Champion", 1, 10, 1)  # speed 7
    b = _battle([slow, fast], first_team=0)
    assert b.turn_order()[0] is fast


# ── anti-stalemate retreat ──────────────────────────────────────────

def test_stalemate_forces_attacker_to_retreat():
    units = [Unit.from_type("Pikeman", 0, 0, 0),
             Unit.from_type("Pikeman", 1, 10, 8)]
    b = _battle(units, attacker_team=0)
    assert not b.is_over()
    # advance death-free rounds (no attacks executed -> no deaths)
    for _ in range(BattleState.MAX_TURNS_WITHOUT_DEATHS + 1):
        b.start_round()
        b.deaths_this_round = 0
    assert b.is_stalemate()
    assert b.is_over()
    assert b.winner() == 1  # attacker (team 0) retreats -> team 1 wins


def test_a_death_resets_the_stalemate_counter():
    units = [Unit.from_type("Pikeman", 0, 0, 0),
             Unit.from_type("Pikeman", 1, 10, 8)]
    b = _battle(units, attacker_team=0)
    for _ in range(40):
        b.start_round()
        b.deaths_this_round = 0
    b.start_round()
    b.deaths_this_round = 1  # a death happens this round
    b.start_round()          # next round start sees the death -> reset
    assert b._stale_rounds == 0
    assert not b.is_stalemate()

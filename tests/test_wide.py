"""Wide-unit (two-hex) tests — M5b.

Covers the head/tail footprint model, wide-aware pathfinding, footprint-based
occupancy/adjacency, and a full wide-vs-wide battle. fheroes2: a wide unit's
head faces the enemy and the tail trails behind on the same row.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from engine.hex_grid import HexGrid
from engine.battle_state import BattleState
from engine.unit import Unit
from engine.actions import AttackAction
from ai import create_ai
from ai.classic.planner import ClassicAI
from headless import _take_unit_turn

G = HexGrid()


# ── footprint model ─────────────────────────────────────────────────

def test_single_hex_footprint_is_just_pos():
    u = Unit.from_type("Swordsman", 0, 3, 4)
    assert not u.is_wide
    assert u.tail_cell is None
    assert u.occupied_cells() == {(3, 4)}


def test_wide_team0_tail_is_left_of_head():
    # Team 0 faces right, so the tail trails to the left.
    k = Unit.from_type("Champion", 0, 5, 4)
    assert k.is_wide
    assert k.pos == (5, 4)
    assert k.tail_cell == (4, 4)
    assert k.occupied_cells() == {(5, 4), (4, 4)}


def test_wide_team1_tail_is_right_of_head():
    # Team 1 faces left, so the tail trails to the right.
    k = Unit.from_type("Champion", 1, 5, 4)
    assert k.tail_cell == (6, 4)
    assert k.occupied_cells() == {(5, 4), (6, 4)}


def test_moving_head_carries_tail():
    k = Unit.from_type("Champion", 0, 5, 4)
    k.pos = (6, 4)
    assert k.occupied_cells() == {(6, 4), (5, 4)}


# ── wide-aware pathfinding ──────────────────────────────────────────

def test_reachable_excludes_heads_whose_tail_leaves_grid():
    # Team 0 wide head at col 0 would put the tail at col -1 (off-board).
    reach = G.reachable((5, 4), 3, occupied=set(), flying=False, tail_dir=-1)
    assert all(c != 0 for (c, r) in reach)


def test_reachable_excludes_heads_whose_tail_is_blocked():
    # Block the cell that would become the tail of head (5,4): col 4.
    occ = {(4, 4)}
    reach = G.reachable((6, 4), 2, occupied=occ, flying=False, tail_dir=-1)
    assert (5, 4) not in reach          # its tail (4,4) is occupied
    free = G.reachable((6, 4), 2, occupied=set(), flying=False, tail_dir=-1)
    assert (5, 4) in free               # free when nothing blocks the tail


def test_find_path_respects_tail():
    path = G.find_path((6, 4), (2, 4), occupied=set(), flying=False, tail_dir=-1)
    assert path is not None and path[0] == (6, 4) and path[-1] == (2, 4)


# ── footprint occupancy / adjacency ─────────────────────────────────

def test_unit_at_covers_both_cells():
    k = Unit.from_type("Champion", 0, 5, 4)
    b = BattleState(G, [k])
    assert b.unit_at((5, 4)) is k        # head
    assert b.unit_at((4, 4)) is k        # tail
    assert b.occupied() == {(5, 4), (4, 4)}


def test_occupied_excludes_full_footprint_of_self():
    k = Unit.from_type("Champion", 0, 5, 4)
    other = Unit.from_type("Swordsman", 1, 8, 4)
    b = BattleState(G, [k, other])
    occ = b.occupied(exclude=k)
    assert (5, 4) not in occ and (4, 4) not in occ
    assert (8, 4) in occ


def test_enemy_adjacent_to_tail_counts_as_blocking():
    # An enemy touching only the tail is still in melee range of a wide unit.
    k = Unit.from_type("Champion", 0, 5, 4)          # tail at (4,4)
    foe = Unit.from_type("Swordsman", 1, 3, 4)       # adjacent to tail (4,4)
    assert ClassicAI._dist(G, k, foe) == 1
    # the attack-from cells around a wide target include tail neighbours
    cells = ClassicAI._attack_cells(G, k)
    assert (3, 4) in cells


# ── full battle ─────────────────────────────────────────────────────

def test_wide_clash_battle_completes():
    for seed in range(5):
        random.seed(seed)
        units = []
        for team, placements in config.PRESETS["Wide Clash"].items():
            for name, col, row in placements:
                units.append(Unit.from_type(name, team, col, row))
        battle = BattleState(G, units)
        ai = create_ai("classic")
        guard = 0
        while not battle.is_over() and guard < 300:
            order = battle.turn_order()
            if not order:
                break
            battle.start_round()
            for u in order:
                if not u.is_alive or battle.is_over():
                    continue
                _take_unit_turn(battle, ai, u)
            if battle._retreated is not None:
                break
            guard += 1
        assert battle.is_over()
        assert battle.winner() in (0, 1)


def test_wide_units_never_overlap_during_battle():
    # Invariant: no two living units ever share a cell (footprint collisions).
    random.seed(0)
    units = []
    for team, placements in config.PRESETS["Wide Clash"].items():
        for name, col, row in placements:
            units.append(Unit.from_type(name, team, col, row))
    battle = BattleState(G, units)
    ai = create_ai("classic")
    guard = 0
    while not battle.is_over() and guard < 300:
        order = battle.turn_order()
        if not order:
            break
        battle.start_round()
        for u in order:
            if not u.is_alive or battle.is_over():
                continue
            _take_unit_turn(battle, ai, u)
            # after every activation, all footprints must be disjoint
            seen = set()
            for v in battle.alive():
                cells = v.occupied_cells()
                assert seen.isdisjoint(cells), f"overlap at {cells & seen}"
                seen |= cells
        if battle._retreated is not None:
            break
        guard += 1

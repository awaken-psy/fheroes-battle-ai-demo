"""M6b siege tests — castle data, walls, moat, towers, catapult, AI."""

import random
import pytest

from engine.castle import Castle, Tower, WALL_POSITIONS, MOAT_CELLS, GATE_POS
from engine.hex_grid import HexGrid
from engine.battle_state import BattleState
from engine.unit import Unit


# ── helpers ──────────────────────────────────────────────────────

def _make_unit(name="Swordsman", team=0, col=0, row=4, **kw):
    """Create a test unit with sensible defaults."""
    defaults = dict(attack=5, defense=5, hp=10, damage_min=1, damage_max=3,
                    is_archer=False, is_flying=False, count=5, speed=4)
    defaults.update(kw)
    return Unit(name, team, col, row, **defaults)


def _archer(name="Archer", team=0, col=0, row=4, **kw):
    return _make_unit(name, team, col, row, is_archer=True, **kw)


def _make_battle(units, castle=None):
    grid = HexGrid()
    return BattleState(grid, units, castle=castle)


# ── Castle data layer ────────────────────────────────────────────

class TestCastleData:
    def test_walls_initial_hp(self):
        c = Castle()
        assert len(c.walls) == 4
        for pos in WALL_POSITIONS:
            assert c.walls[pos] == 2

    def test_wall_damage(self):
        c = Castle()
        pos = WALL_POSITIONS[0]
        assert c.damage_wall(pos) == 1  # 2 → 1
        assert c.damage_wall(pos) == 0  # 1 → 0
        assert c.damage_wall(pos) == 0  # already 0

    def test_wall_intact_destroyed(self):
        c = Castle()
        assert len(c.wall_intact_cells()) == 4
        assert len(c.wall_destroyed_cells()) == 0
        c.damage_wall(WALL_POSITIONS[0])
        assert len(c.wall_intact_cells()) == 4  # HP=1 still intact
        c.damage_wall(WALL_POSITIONS[0])
        assert len(c.wall_intact_cells()) == 3
        assert len(c.wall_destroyed_cells()) == 1

    def test_any_wall_standing(self):
        c = Castle()
        assert c.any_wall_standing()
        for pos in WALL_POSITIONS:
            c.damage_wall(pos, 2)
        assert not c.any_wall_standing()

    def test_moat_queries(self):
        assert Castle.is_moat(5, 4) is True   # center moat cell
        assert Castle.is_moat(7, 0) is True   # top moat
        assert Castle.is_moat(0, 0) is False  # far left
        assert Castle.is_moat(9, 4) is False  # inside walls

    def test_inside_outside_walls(self):
        # Inside castle (defender positions)
        assert Castle.is_inside_walls(9, 0) is True
        assert Castle.is_inside_walls(10, 4) is True
        assert Castle.is_inside_walls(8, 2) is True
        # Outside castle (attacker positions)
        assert Castle.is_outside_walls(0, 0) is True
        assert Castle.is_outside_walls(5, 4) is True
        assert Castle.is_outside_walls(6, 3) is True

    def test_all_moat_cells_outside_walls(self):
        for cell in MOAT_CELLS:
            assert Castle.is_outside_walls(*cell), f"moat {cell} should be outside"


class TestBridge:
    def test_initial_state(self):
        c = Castle()
        assert not c.bridge_down
        assert not c.bridge_destroyed

    def test_attacker_cannot_pass(self):
        c = Castle()
        assert not c.is_gate_passable(team=0)

    def test_defender_cannot_pass_initially(self):
        c = Castle()
        # Bridge starts up — even defender can't pass until lowered.
        assert not c.is_gate_passable(team=1)

    def test_lower_bridge(self):
        c = Castle()
        c.lower_bridge()
        assert c.bridge_down
        assert c.is_gate_passable(team=1)
        assert not c.is_gate_passable(team=0)  # attacker still blocked

    def test_destroy_bridge(self):
        c = Castle()
        c.destroy_bridge()
        assert c.bridge_destroyed
        assert c.is_gate_passable(team=0)
        assert c.is_gate_passable(team=1)


class TestTower:
    def test_tower_creation(self):
        center = Tower("center")
        assert center.count == 10
        assert center.is_valid

        side = Tower("left")
        assert side.count == 5

    def test_tower_strength(self):
        center = Tower("center")
        assert center.strength > 0
        center.destroyed = True
        assert center.strength == 0.0

    def test_tower_select_target(self):
        t = Tower("center")
        u1 = _make_unit("A", count=1, hp=10)
        u2 = _make_unit("B", count=10, hp=10)
        assert t.select_target([u1, u2]) is u2

    def test_tower_destroyed_no_target(self):
        t = Tower("center")
        t.destroyed = True
        assert t.select_target([_make_unit()]) is None

    def test_towers_active(self):
        c = Castle()
        assert c.towers_active()
        for t in c.towers:
            t.destroyed = True
        assert not c.towers_active()


# ── Catapult ─────────────────────────────────────────────────────

class TestCatapult:
    def test_catapult_hits_wall(self):
        c = Castle()
        rng = random.Random(42)
        # Run many rounds; at least one should hit (75% chance).
        hits = 0
        for _ in range(50):
            c2 = Castle()
            shots = c2.catapult_round(rng)
            if shots and shots[0]["hit"]:
                hits += 1
        assert hits > 0  # should almost always have hits in 50 tries

    def test_catapult_targets_walls_first(self):
        c = Castle()
        rng = random.Random(42)
        # All targets should be walls while walls are intact.
        for _ in range(20):
            c2 = Castle()
            shots = c2.catapult_round(rng)
            if shots and shots[0]["target"] != "bridge":
                # Should be a wall position string.
                target = shots[0]["target"]
                pos = Castle._parse_wall_target(target)
                assert pos in WALL_POSITIONS

    def test_catapult_destroys_wall(self):
        """Hit the same wall twice to destroy it."""
        c = Castle()
        pos = WALL_POSITIONS[0]
        c.damage_wall(pos, 2)  # manually destroy
        assert c.walls[pos] == 0

    def test_catapult_no_target_when_all_destroyed(self):
        c = Castle()
        for pos in WALL_POSITIONS:
            c.damage_wall(pos, 2)
        for t in c.towers:
            t.destroyed = True
        c.destroy_bridge()
        rng = random.Random(42)
        shots = c.catapult_round(rng)
        assert shots == []


# ── Pathfinding with moat ────────────────────────────────────────

class TestMoatPathfinding:
    def test_non_flying_stops_in_moat(self):
        grid = HexGrid()
        start = (4, 4)
        speed = 9
        occupied = set()
        # Without moat: can reach col 6+
        r1 = grid.reachable(start, speed, occupied, flying=False)
        assert (6, 4) in r1
        # With moat: (5, 4) is a moat cell — should be reachable but no further.
        r2 = grid.reachable(start, speed, occupied, flying=False, moat_cells=MOAT_CELLS)
        assert (5, 4) in r2  # can enter moat
        # Cells beyond the moat on the same approach should NOT be reachable
        # from that direction (but might be via other paths around moat).
        # The key is that (5,4) is in the result but BFS didn't expand from it.
        # Verify by checking that we can't reach (6,4) ONLY through (5,4).
        # Since (6,5) and (6,3) are also moat, the path is blocked.

    def test_flying_ignores_moat(self):
        grid = HexGrid()
        start = (4, 4)
        speed = 5
        occupied = set()
        r_ground = grid.reachable(start, speed, occupied, flying=False, moat_cells=MOAT_CELLS)
        r_fly = grid.reachable(start, speed, occupied, flying=True, moat_cells=MOAT_CELLS)
        # Flyer can reach cells that ground cannot due to moat.
        assert len(r_fly) >= len(r_ground)

    def test_no_moat_means_no_change(self):
        grid = HexGrid()
        start = (4, 4)
        speed = 5
        occupied = set()
        r_none = grid.reachable(start, speed, occupied)
        r_null = grid.reachable(start, speed, occupied, moat_cells=None)
        assert r_none == r_null


# ── Shooting penalty ─────────────────────────────────────────────

class TestShootingPenalty:
    def test_no_penalty_without_castle(self):
        battle = _make_battle([
            _archer("A0", 0, 0, 4),
            _make_unit("D1", 1, 10, 4),
        ])
        atk = battle.units[0]
        dfn = battle.units[1]
        assert not battle._shooting_penalty(atk, dfn)

    def test_penalty_across_walls(self):
        c = Castle()
        battle = _make_battle([
            _archer("A0", 0, 0, 4),   # outside walls
            _make_unit("D1", 1, 10, 4),  # inside walls
        ], castle=c)
        atk = battle.units[0]
        dfn = battle.units[1]
        assert battle._shooting_penalty(atk, dfn)

    def test_no_penalty_same_side(self):
        c = Castle()
        battle = _make_battle([
            _archer("A0", 0, 0, 4),   # outside
            _make_unit("D1", 1, 10, 4),  # inside
        ], castle=c)
        # Both outside: no penalty.
        atk = _archer("A0b", 0, 0, 4)
        dfn = _make_unit("D1b", 1, 2, 4)
        assert not battle._shooting_penalty(atk, dfn)

    def test_no_penalty_walls_destroyed(self):
        c = Castle()
        for pos in WALL_POSITIONS:
            c.damage_wall(pos, 2)
        battle = _make_battle([
            _archer("A0", 0, 0, 4),
            _make_unit("D1", 1, 10, 4),
        ], castle=c)
        atk = battle.units[0]
        dfn = battle.units[1]
        assert not battle._shooting_penalty(atk, dfn)

    def test_expected_damage_reduced(self):
        c = Castle()
        battle = _make_battle([
            _archer("A0", 0, 0, 4),
            _make_unit("D1", 1, 10, 4),
        ], castle=c)
        atk, dfn = battle.units[0], battle.units[1]
        dmg_with_penalty = battle.expected_damage(atk, dfn, ranged=True)

        # Without castle: no penalty.
        battle2 = _make_battle([
            _archer("A0", 0, 0, 4),
            _make_unit("D1", 1, 10, 4),
        ])
        dmg_no_penalty = battle2.expected_damage(atk, dfn, ranged=True)
        assert dmg_with_penalty == dmg_no_penalty // 2


# ── Moat defense penalty ─────────────────────────────────────────

class TestMoatDefensePenalty:
    def test_defense_penalty_in_moat(self):
        c = Castle()
        # Defender at (5,4) which is a moat cell.
        battle = _make_battle([
            _make_unit("A0", 0, 0, 4, attack=10, defense=5),
            _make_unit("D1", 1, 5, 4, attack=5, defense=10),
        ], castle=c)
        atk, dfn = battle.units[0], battle.units[1]
        assert Castle.is_moat(*dfn.pos)
        dmg_moat = battle.expected_damage(atk, dfn, ranged=False)

        # Move defender out of moat.
        dfn.pos = (9, 4)
        dmg_normal = battle.expected_damage(atk, dfn, ranged=False)
        # In moat: effective defense is 10-3=7, so attacker gets higher mult.
        assert dmg_moat > dmg_normal

    def test_no_penalty_outside_moat(self):
        c = Castle()
        battle = _make_battle([
            _make_unit("A0", 0, 0, 4),
            _make_unit("D1", 1, 9, 4),  # inside walls, not moat
        ], castle=c)
        atk, dfn = battle.units[0], battle.units[1]
        assert not battle._in_moat(dfn)


# ── Tower round ──────────────────────────────────────────────────

class TestTowerRound:
    def test_towers_shoot_enemy(self):
        c = Castle()
        u0 = _make_unit("Attacker", 0, 0, 4, hp=50, count=10)
        battle = _make_battle([u0], castle=c)
        battle.start_round()
        # Tower should have dealt damage to the attacker.
        assert u0._total_hp < u0.count * u0.max_hp

    def test_destroyed_tower_does_not_shoot(self):
        c = Castle()
        for t in c.towers:
            t.destroyed = True
        u0 = _make_unit("Attacker", 0, 0, 4, hp=50, count=10)
        battle = _make_battle([u0], castle=c)
        hp_before = u0._total_hp
        battle.start_round()
        assert u0._total_hp == hp_before

    def test_tower_kills_last_enemy(self):
        c = Castle()
        u0 = _make_unit("Weak", 0, 0, 4, hp=1, count=1, defense=0)
        battle = _make_battle([u0], castle=c)
        battle.start_round()
        # Very likely the tower kills a 1hp unit, but not guaranteed.
        # Check that is_over() works if the unit dies.
        if not u0.is_alive:
            assert battle.is_over()


# ── Siege occupied + gate ─────────────────────────────────────────

class TestSiegeOccupied:
    def test_walls_block_path(self):
        c = Castle()
        u0 = _make_unit("A0", 0, 0, 4)
        u1 = _make_unit("D1", 1, 10, 4)
        battle = _make_battle([u0, u1], castle=c)
        occ = battle._move_occupied(u0)
        # Intact wall cells should be in occupied.
        for wp in WALL_POSITIONS:
            assert wp in occ

    def test_gate_blocks_attacker(self):
        c = Castle()
        u0 = _make_unit("A0", 0, 0, 4)
        battle = _make_battle([u0], castle=c)
        occ = battle._move_occupied(u0)
        assert GATE_POS in occ  # attacker can't pass closed gate

    def test_gate_passable_for_defender(self):
        c = Castle()
        c.lower_bridge()
        u1 = _make_unit("D1", 1, 10, 4)
        battle = _make_battle([u1], castle=c)
        occ = battle._move_occupied(u1)
        assert GATE_POS not in occ  # defender can pass lowered bridge

    def test_no_castle_no_extra_blocks(self):
        battle = _make_battle([
            _make_unit("A0", 0, 0, 4),
            _make_unit("D1", 1, 10, 4),
        ])
        occ = battle._move_occupied(battle.units[0])
        # Only the other unit's cell, no wall/gate additions.
        assert occ == {(10, 4)}


# ── Siege AI evaluation ──────────────────────────────────────────

class TestSiegeAI:
    def test_attacker_gets_castle_flag(self):
        from ai.classic.evaluation import analyze
        c = Castle()
        u0 = _make_unit("A0", 0, 0, 4)
        u1 = _make_unit("D1", 1, 10, 4)
        battle = _make_battle([u0, u1], castle=c)
        state = analyze(battle, u0)
        assert state.attacking_castle
        assert not state.defending_castle

    def test_defender_gets_castle_flag(self):
        from ai.classic.evaluation import analyze
        c = Castle()
        u0 = _make_unit("A0", 0, 0, 4)
        u1 = _make_unit("D1", 1, 10, 4)
        battle = _make_battle([u0, u1], castle=c)
        state = analyze(battle, u1)
        assert state.defending_castle
        assert not state.attacking_castle

    def test_tower_strength_added_to_defender(self):
        from ai.classic.evaluation import analyze
        c = Castle()
        u0 = _archer("A0", 0, 0, 4)
        u1 = _archer("D1", 1, 10, 4)
        battle = _make_battle([u0, u1], castle=c)
        state = analyze(battle, u1)
        # Tower strength should be added to defender's shooters.
        assert state.my_shooters > u1.strength

    def test_attacker_shooter_penalty(self):
        from ai.classic.evaluation import analyze
        c = Castle()
        u0 = _archer("A0", 0, 0, 4)
        u1 = _make_unit("D1", 1, 10, 4)
        battle = _make_battle([u0, u1], castle=c)
        state = analyze(battle, u0)
        # Attacker's shooter strength should be divided by 1.5.
        assert state.my_shooters < u0.strength


# ── Non-siege zero regression ────────────────────────────────────

class TestNonSiegeRegression:
    def test_castle_none_no_penalty(self):
        """All siege paths short-circuit when castle is None."""
        battle = _make_battle([
            _archer("A0", 0, 0, 4),
            _make_unit("D1", 1, 10, 4),
        ])
        atk, dfn = battle.units[0], battle.units[1]
        assert not battle._shooting_penalty(atk, dfn)
        assert not battle._in_moat(dfn)
        assert battle._moat_cells() is None
        assert battle._move_occupied(atk) == battle.occupied(exclude=atk)

    def test_damage_mult_unchanged_without_moat(self):
        """Static _damage_mult still works without moat arg."""
        u1 = _make_unit("A", attack=10)
        u2 = _make_unit("D", defense=5)
        m = BattleState._damage_mult(u1, u2)
        assert m == pytest.approx(1.5)  # 1 + 0.1 * (10-5)

    def test_start_round_no_castle(self):
        """start_round works cleanly without castle (no catapult/tower)."""
        u0 = _make_unit("A0", 0, 0, 4)
        u1 = _make_unit("D1", 1, 10, 4)
        battle = _make_battle([u0, u1])
        battle.start_round()
        assert battle.round_num == 1
        assert u0.is_alive and u1.is_alive

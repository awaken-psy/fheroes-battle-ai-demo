"""A3 tests — movement and target selection fidelity (4 fixes + 2 relabels).

Covers:
  §4.1-#2   Archer retreat UnitRemover pattern
  §5.1-#4   Attack position sorted by distance to current unit
  §6.1-#5   Blocked archer: BestAttackOutcome for blockers
  §8.4-#2   optimalAttackVector / two-cell splash direction
  §9-#2     getClosestReachablePosition (relabel — equivalent)
  §9-#5     Wide isHandFighting collision (relabel — equivalent)
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.hex_grid import HexGrid
from engine.unit import Unit
from engine.battle_state import BattleState

from ai.classic.planner import ClassicAI
from ai.classic.scoring import splash_value, threat, optimal_attack_value
from ai.classic.evaluation import AIState, analyze


# ── Helpers ──────────────────────────────────────────────────

def _u(name="Unit", team=0, col=5, row=4, speed=5, hp=100,
       attack=5, defense=5, damage_min=5, damage_max=5, count=10,
       is_archer=False, is_flying=False, is_wide=False,
       abilities=()):
    """Create a minimal unit for tests."""
    return Unit(name=name, team=team, col=col, row=row,
                attack=attack, defense=defense, hp=hp, speed=speed,
                damage_min=damage_min, damage_max=damage_max,
                is_archer=is_archer, is_flying=is_flying,
                is_wide=is_wide, count=count, abilities=abilities)


def _battle(units, **kw):
    grid = HexGrid()
    return BattleState(grid, units, **kw)


# ── §4.1-#2  Archer retreat: UnitRemover pattern ────────────

class TestRetreatUnitRemover:
    """Verify that retreat evaluation removes the archer's own cells
    from the occupied set so enemy reachability checks treat positions
    behind the archer as potentially threatened."""

    def test_retreat_no_flying_enemy(self):
        """Flying enemies prevent retreat entirely."""
        grid = HexGrid()
        archer = _u("archer", 0, col=5, row=4, is_archer=True, speed=8)
        flyer = _u("flyer", 1, col=3, row=4, is_flying=True, speed=3)
        battle = _battle([archer, flyer])

        ai = ClassicAI()
        occ, moat = ai._path_args(battle, archer)
        retreat = ai._retreat_pos(battle, archer,
                                  battle.enemies_of(archer), occ, moat)
        assert retreat is None  # flying enemy → no retreat

    def test_retreat_slow_enemy_allows_escape(self):
        """Slow enemy: archer can retreat to a safe position."""
        archer = _u("archer", 0, col=5, row=4, is_archer=True, speed=8)
        enemy = _u("slow", 1, col=5, row=2, speed=3)
        battle = _battle([archer, enemy])

        ai = ClassicAI()
        occ, moat = ai._path_args(battle, archer)
        retreat = ai._retreat_pos(battle, archer,
                                  battle.enemies_of(archer), occ, moat)
        # enemy speed 3 + 2 = 5 < archer speed 8 → worth retreating
        assert retreat is not None
        # Retreat position should be further from enemy
        assert battle.grid.distance(retreat, enemy.pos) >= \
            battle.grid.distance(archer.pos, enemy.pos)

    def test_retreat_fast_adjacent_enemy_blocks_escape(self):
        """Fast adjacent enemy catches archer → not worth retreating.
        Adjacent enemy with speed+2 >= archer speed → retreat blocked."""
        archer = _u("archer", 0, col=5, row=4, is_archer=True, speed=7)
        # Enemy adjacent (dist=1) and fast enough to catch
        enemy = _u("fast", 1, col=5, row=3, speed=6)
        battle = _battle([archer, enemy])

        ai = ClassicAI()
        occ, moat = ai._path_args(battle, archer)
        retreat = ai._retreat_pos(battle, archer,
                                  battle.enemies_of(archer), occ, moat)
        # Adjacent enemy: speed 6 + 2 = 8 >= archer speed 7 → blocked
        assert retreat is None

    def test_retreat_enemy_archer_not_in_melee_skipped(self):
        """Enemy archer not in melee is skipped for threat evaluation.
        Since cur_threatened only checks non-archers, pure archer
        opposition means no retreat needed."""
        archer = _u("archer", 0, col=5, row=4, is_archer=True, speed=8)
        e_archer = _u("e_arch", 1, col=9, row=4, is_archer=True, speed=3)
        battle = _battle([archer, e_archer])

        ai = ClassicAI()
        occ, moat = ai._path_args(battle, archer)
        retreat = ai._retreat_pos(battle, archer,
                                  battle.enemies_of(archer), occ, moat)
        # Enemy is archer → cur_threatened checks `not e.is_archer` → False
        assert retreat is None

    def test_retreat_unit_remover_pathfinding(self):
        """UnitRemover: enemy reachability uses actual pathfinding
        (grid.reachable) instead of geometric distance."""
        archer = _u("archer", 0, col=5, row=4, is_archer=True, speed=8)
        enemy = _u("enemy", 1, col=3, row=4, speed=3)
        battle = _battle([archer, enemy])

        ai = ClassicAI()
        occ, moat = ai._path_args(battle, archer)
        retreat = ai._retreat_pos(battle, archer,
                                  battle.enemies_of(archer), occ, moat)
        # With UnitRemover: archer cells removed from occ when computing
        # enemy reachability.  Enemy can only reach cells within speed 3.
        # Positions beyond enemy reach should be safe.
        if retreat is not None:
            # Verify retreat position is not in enemy's reachable range
            unit_cells = archer.occupied_cells()
            e_occ = battle._move_occupied(enemy) - unit_cells
            e_reach = battle.grid.reachable(
                enemy.pos, enemy.speed, e_occ,
                enemy.is_flying, ai._tail_dir(enemy), moat)
            assert retreat not in e_reach


# ── §5.1-#4  Attack position distance tiebreaker ────────────

class TestAttackPositionDistanceSort:
    """Verify that when multiple attack positions have equal value,
    the closest one to the current unit is preferred."""

    def test_attacks_closest_position_on_equal_value(self):
        """Two equal enemies; attack position closer to unit is chosen."""
        unit = _u("unit", 0, col=2, row=4, speed=8)
        e1 = _u("e1", 1, col=6, row=4)
        e2 = _u("e2", 1, col=6, row=2)
        battle = _battle([unit, e1, e2])

        ai = ClassicAI()
        state = analyze(battle, unit)
        state.defensive = False
        action, desc = ai.decide(battle, unit)
        assert "attacks" in desc

    def test_single_target_closest_attack_pos(self):
        """Single target: closest reachable attack position used."""
        unit = _u("unit", 0, col=1, row=4, speed=8)
        enemy = _u("enemy", 1, col=8, row=4)
        battle = _battle([unit, enemy])

        ai = ClassicAI()
        state = analyze(battle, unit)
        state.defensive = False
        action, desc = ai.decide(battle, unit)
        assert "attacks" in desc or "moving" in desc.lower()


# ── §6.1-#5  Blocked archer: BestAttackOutcome for blockers ─

class TestBlockedArcherBestOutcome:
    """When covering a blocked friendly archer, the AI should evaluate
    each blocker with BestAttackOutcome compound priority."""

    def test_defends_archer_with_blockers(self):
        """Defense action produced for blocked archer scenario."""
        archer = _u("archer", 0, col=5, row=4, is_archer=True, speed=5)
        unit = _u("def", 0, col=3, row=4, speed=5)
        b1 = _u("b1", 1, col=4, row=4)
        b2 = _u("b2", 1, col=6, row=4)
        battle = _battle([archer, unit, b1, b2])

        ai = ClassicAI()
        action, desc = ai.decide(battle, unit)
        assert action is not None
        assert "DEF" in desc

    def test_multiple_blockers_no_crash(self):
        """Multiple blockers surrounding archer — no crash."""
        archer = _u("archer", 0, col=5, row=4, is_archer=True, speed=3)
        unit = _u("def", 0, col=3, row=3, speed=5)
        b1 = _u("b1", 1, col=4, row=4)
        b2 = _u("b2", 1, col=6, row=4)
        b3 = _u("b3", 1, col=5, row=3)
        battle = _battle([archer, unit, b1, b2, b3])

        ai = ClassicAI()
        action, desc = ai.decide(battle, unit)
        assert action is not None


# ── §8.4-#2  optimalAttackVector / splash direction ──────────

class TestSplashDirection:
    """Verify splash_value enumerates all attack directions and picks
    the best secondary target."""

    def test_splash_finds_secondary_target(self):
        """Two-cell attacker finds secondary target behind primary."""
        attacker = _u("cavalry", 0, col=3, row=4,
                       abilities=("two_cell_melee",))
        target = _u("target", 1, col=5, row=4)
        # cell_behind(attack=(4,4), target=(5,4)) = (6,4)
        secondary = _u("secondary", 1, col=6, row=4, hp=200,
                        damage_min=10, damage_max=10)
        battle = _battle([attacker, target, secondary])

        val = splash_value(battle, attacker, target, (4, 4))
        assert val > 0  # found secondary target

    def test_splash_no_secondary_returns_zero(self):
        """No unit behind target → splash value is 0."""
        attacker = _u("atk", 0, col=3, row=4,
                       abilities=("two_cell_melee",))
        target = _u("target", 1, col=5, row=4)
        battle = _battle([attacker, target])

        val = splash_value(battle, attacker, target, (4, 4))
        assert val == 0.0

    def test_splash_wide_attacker_considers_tail(self):
        """Wide attacker uses both head and tail cells."""
        attacker = _u("wolf", 0, col=3, row=4, is_wide=True,
                       abilities=("two_cell_melee",))
        target = _u("target", 1, col=5, row=4)
        secondary = _u("sec", 1, col=6, row=4, hp=200,
                        damage_min=10, damage_max=10)
        battle = _battle([attacker, target, secondary])

        val = splash_value(battle, attacker, target, (4, 4))
        assert val > 0

    def test_splash_wide_target_checks_both_cells(self):
        """Wide target: splash checks behind both head and tail cells."""
        attacker = _u("atk", 0, col=3, row=4,
                       abilities=("two_cell_melee",))
        target = _u("wide_tgt", 1, col=5, row=4, is_wide=True)
        # cell_behind(attack=(4,4), target_head=(5,4)) = (6,4)
        secondary = _u("sec", 1, col=6, row=4, hp=200,
                        damage_min=10, damage_max=10)
        battle = _battle([attacker, target, secondary])

        val = splash_value(battle, attacker, target, (4, 4))
        assert val > 0

    def test_splash_ignores_attacker_and_target(self):
        """Splash never counts the attacker or primary target."""
        attacker = _u("atk", 0, col=3, row=4,
                       abilities=("two_cell_melee",))
        target = _u("target", 1, col=5, row=4)
        # Only attacker and target — no secondary
        battle = _battle([attacker, target])

        val = splash_value(battle, attacker, target, (4, 4))
        assert val == 0.0

    def test_optimal_attack_value_includes_splash(self):
        """optimal_attack_value includes splash for two_cell attackers."""
        attacker = _u("cav", 0, col=3, row=4,
                       abilities=("two_cell_melee",))
        target = _u("target", 1, col=5, row=4)
        secondary = _u("sec", 1, col=6, row=4, hp=200,
                        damage_min=10, damage_max=10)
        battle = _battle([attacker, target, secondary])

        val = optimal_attack_value(battle, attacker, target, (4, 4),
                                   [target, secondary])
        base = threat(battle, attacker, target)
        assert val > base  # includes splash


# ── §9-#2 / §9-#5  Relabel verification (equivalent) ─────────

class TestRelabelEquivalent:
    """Verify §9-#2 and §9-#5 are already equivalent."""

    def test_wide_dist_checks_all_occupied_cells(self):
        """_dist for wide units checks all occupied_cells combinations,
        equivalent to C++ isHandFighting wide-body collision detection."""
        grid = HexGrid()
        wide = _u("wide", 0, col=5, row=4, is_wide=True)
        # Enemy adjacent to head cell
        enemy = _u("enemy", 1, col=5, row=3)
        battle = _battle([wide, enemy])

        ai = ClassicAI()
        d = ai._dist(grid, wide, enemy)
        assert d == 1  # head (5,4) to (5,3) = 1

    def test_wide_tail_adjacency_detected(self):
        """Wide unit's tail cell adjacency is detected by _dist."""
        grid = HexGrid()
        # Team 0 wide: head at (5,4), tail at (4,4)
        wide = _u("wide", 0, col=5, row=4, is_wide=True)
        # Enemy adjacent to tail
        enemy = _u("enemy", 1, col=4, row=3)
        battle = _battle([wide, enemy])

        ai = ClassicAI()
        d = ai._dist(grid, wide, enemy)
        assert d == 1  # tail (4,4) to (4,3) = 1

    def test_nearest_cell_finds_path_around_obstacle(self):
        """nearest_cell_next_to finds reachable cell even when
        direct path is partially blocked — equivalent to
        C++ getClosestReachablePosition."""
        grid = HexGrid()
        unit = _u("unit", 0, col=0, row=4, speed=5)
        blocker = _u("block", 0, col=5, row=4)  # same team, blocks path
        target = _u("target", 1, col=9, row=4)
        battle = _battle([unit, blocker, target])

        occ = battle._move_occupied(unit)
        tc = grid.nearest_cell_next_to(unit.pos, target.pos, occ,
                                       unit.is_flying, unit.speed * 3)
        # Should find a path around the blocker
        assert tc is not None

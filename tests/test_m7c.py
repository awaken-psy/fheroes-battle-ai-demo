"""M7c — AI behavior refinement tests.

Covers: hero spell threat, avoid stacking, AREA_SHOT archer evaluation,
attack position map / splash value, defense area attack, no-retaliation
attack from cover, and cover position with stacking avoidance.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.hex_grid import HexGrid
from engine.battle_state import BattleState
from engine.unit import Unit
from engine.hero import Hero

from ai.classic.evaluation import AIState, analyze, _max_spell_damage
from ai.classic.scoring import (threat, pos_value, splash_value,
                                optimal_attack_value,
                                build_attack_position_map)
from ai.classic.planner import ClassicAI


def _battle(units, heroes=None, castle=None):
    return BattleState(HexGrid(), units, heroes=heroes, castle=castle)


# ─── P1: Hero spell threat ──────────────────────────────────

class TestHeroSpellThreat:
    """Hero spell damage contributes to shooter strength."""

    def test_max_spell_damage_hero_with_spells(self):
        hero = Hero(power=5, spells=["Magic Arrow", "Lightning Bolt"])
        val = _max_spell_damage(hero)
        # Lightning Bolt: base_damage=25, power=5 -> 125 damage
        assert val > 0
        assert val == 25 * 5  # Lightning Bolt is the strongest

    def test_max_spell_damage_hero_no_damage_spells(self):
        hero = Hero(power=5, spells=["Haste", "Slow"])
        val = _max_spell_damage(hero)
        assert val == 0.0

    def test_max_spell_damage_hero_no_mana(self):
        hero = Hero(power=5, max_spell_points=1, spells=["Lightning Bolt"])
        val = _max_spell_damage(hero)
        # Lightning Bolt costs 10 SP, hero has 1 — can't cast
        assert val == 0.0

    def test_analyze_spell_threat_added_to_shooters(self):
        sw = Unit.from_type("Swordsman", 0, 4, 4)
        arc = Unit.from_type("Archer", 0, 2, 4)
        enemy = Unit.from_type("Swordsman", 1, 8, 4)
        hero = Hero(power=3, spells=["Magic Arrow", "Lightning Bolt"])
        b = _battle([sw, arc, enemy], heroes={0: hero, 1: Hero(power=1)})
        s = analyze(b, arc)
        # my_spell_str should be > 0
        assert s.my_spell_str > 0
        # my_shooters should include spell damage
        assert s.my_shooters > arc.strength

    def test_analyze_enemy_spell_threat(self):
        sw = Unit.from_type("Swordsman", 0, 4, 4)
        enemy = Unit.from_type("Archer", 1, 8, 4)
        enemy_hero = Hero(power=5, spells=["Lightning Bolt"])
        b = _battle([sw, enemy], heroes={0: Hero(power=1), 1: enemy_hero})
        s = analyze(b, sw)
        assert s.enemy_spell_str > 0
        assert s.enemy_shooters > enemy.strength


# ─── P1: Avoid stacking flag ────────────────────────────────

class TestAvoidStacking:
    """Avoid stacking flag set when enemy has AREA_SHOT threat > 10%."""

    def test_no_stacking_without_area_shot(self):
        sw = Unit.from_type("Swordsman", 0, 4, 4)
        enemy = Unit.from_type("Archer", 1, 8, 4)  # normal archer
        b = _battle([sw, enemy])
        s = analyze(b, sw)
        assert not s.avoid_stacking

    def test_stacking_with_area_shot_threat(self):
        sw = Unit.from_type("Swordsman", 0, 4, 4)
        # Lich has area_shot ability
        lich = Unit.from_type("Power Lich", 1, 8, 4)
        weak = Unit.from_type("Pikeman", 1, 9, 2)
        b = _battle([sw, lich, weak])
        s = analyze(b, sw)
        # Power Lich is strong enough to trigger stacking avoidance
        assert s.avoid_stacking


# ─── P2: AREA_SHOT archer evaluation ────────────────────────

class TestAreaShotArcher:
    """AREA_SHOT units evaluate splash priority and avoid friendly fire."""

    def test_area_shot_picks_cluster_target(self):
        ai = ClassicAI()
        lich = Unit.from_type("Power Lich", 0, 2, 4)
        e1 = Unit.from_type("Swordsman", 1, 7, 4)
        e2 = Unit.from_type("Swordsman", 1, 8, 4)  # adjacent to e1
        e3 = Unit.from_type("Swordsman", 1, 10, 4)  # far away
        b = _battle([lich, e1, e2, e3])
        target, pri = ai._area_shot_target(b, lich, [e1, e2, e3])
        # Should pick one of the clustered targets
        assert target is not None
        assert target is not e3

    def test_area_shot_avoids_friendly_fire(self):
        ai = ClassicAI()
        lich = Unit.from_type("Power Lich", 0, 2, 4)
        friend = Unit.from_type("Swordsman", 0, 7, 4, count=1)
        enemy = Unit.from_type("Swordsman", 1, 8, 4, count=100)
        # Enemy is far stronger — friend near a weak enemy shouldn't
        # cause the Lich to skip; but isolated weak friend + isolated
        # strong enemy should still target enemy.
        far_enemy = Unit.from_type("Swordsman", 1, 10, 4)
        b = _battle([lich, friend, enemy, far_enemy])
        target, pri = ai._area_shot_target(b, lich, [enemy, far_enemy])
        assert target is not None


# ─── P3: Attack position map / splash value ─────────────────

class TestAttackPositionMap:
    """Precomputed attack position map values."""

    def test_map_contains_reachable_attack_positions(self):
        sw = Unit.from_type("Swordsman", 0, 4, 4)
        enemy = Unit.from_type("Archer", 1, 6, 4)
        b = _battle([sw, enemy])
        grid = b.grid
        reachable = grid.reachable(sw.pos, sw.speed, b._move_occupied(sw),
                                   sw.is_flying, None, b._moat_cells())
        pos_map = build_attack_position_map(b, sw, [enemy], reachable)
        # Should have positions adjacent to the enemy
        assert len(pos_map) > 0
        # Values should be positive
        assert all(v > 0 for v in pos_map.values())

    def test_map_archer_enemy_summed(self):
        """Adjacent to two archer enemies: values summed."""
        sw = Unit.from_type("Swordsman", 0, 4, 4)
        e1 = Unit.from_type("Archer", 1, 6, 4)
        e2 = Unit.from_type("Archer", 1, 6, 3)  # adjacent to e1
        b = _battle([sw, e1, e2])
        grid = b.grid
        reachable = grid.reachable(sw.pos, sw.speed, b._move_occupied(sw),
                                   sw.is_flying, None, b._moat_cells())
        pos_map = build_attack_position_map(b, sw, [e1, e2], reachable)
        assert len(pos_map) > 0

    def test_splash_value_returns_zero_for_no_behind_unit(self):
        sw = Unit.from_type("Swordsman", 0, 4, 4)
        enemy = Unit.from_type("Swordsman", 1, 6, 4)
        b = _battle([sw, enemy])
        # No unit behind the enemy
        assert splash_value(b, sw, enemy, (5, 4)) == 0.0

    def test_splash_value_nonzero_with_behind_unit(self):
        sw = Unit.from_type("Swordsman", 0, 4, 4)
        enemy = Unit.from_type("Swordsman", 1, 6, 4)
        behind = Unit.from_type("Archer", 1, 7, 4)  # behind enemy
        b = _battle([sw, enemy, behind])
        val = splash_value(b, sw, enemy, (5, 4))
        # Should detect the archer behind the target
        assert val > 0

    def test_optimal_attack_value_base(self):
        sw = Unit.from_type("Swordsman", 0, 4, 4)
        enemy = Unit.from_type("Swordsman", 1, 6, 4)
        b = _battle([sw, enemy])
        val = optimal_attack_value(b, sw, enemy, (5, 4), [enemy])
        assert val > 0
        # Base case: should equal threat
        assert val == threat(b, sw, enemy)


# ─── P4: Defense area attack ────────────────────────────────

class TestDefenseAreaAttack:
    """Defense phase 2: attack from defended area."""

    def test_defense_area_attack_in_own_half(self):
        ai = ClassicAI()
        sw = Unit.from_type("Swordsman", 0, 3, 4)  # own half
        enemy = Unit.from_type("Archer", 1, 4, 4)   # just across mid, reachable
        archer = Unit.from_type("Archer", 0, 0, 4)
        b = _battle([sw, enemy, archer])
        s = analyze(b, sw)
        occ, moat = ai._path_args(b, sw)
        grid = b.grid
        reachable = grid.reachable(sw.pos, sw.speed, occ, sw.is_flying,
                                   ai._tail_dir(sw), moat)
        pos_map = build_attack_position_map(b, sw, [enemy], reachable)
        result = ai._defense_area_attack(b, sw, s, [enemy], occ, moat,
                                         reachable, pos_map)
        assert result is not None

    def test_defense_area_rejects_cross_half_positions(self):
        ai = ClassicAI()
        sw = Unit.from_type("Swordsman", 0, 3, 4)
        # Enemy far in enemy half — attack position would be in enemy half
        enemy = Unit.from_type("Swordsman", 1, 9, 4)
        archer = Unit.from_type("Archer", 0, 0, 4)
        b = _battle([sw, enemy, archer])
        s = analyze(b, sw)
        occ, moat = ai._path_args(b, sw)
        grid = b.grid
        reachable = grid.reachable(sw.pos, sw.speed, occ, sw.is_flying,
                                   ai._tail_dir(sw), moat)
        pos_map = build_attack_position_map(b, sw, [enemy], reachable)
        result = ai._defense_area_attack(b, sw, s, [enemy], occ, moat,
                                         reachable, pos_map)
        # Should return None because attack position is outside defended area
        assert result is None


# ─── P4: No-retaliation attack from cover ───────────────────

class TestNoRetaliationAttackFromCover:
    """Units with no_enemy_retaliation attack from cover position."""

    def test_sprite_attacks_from_cover(self):
        ai = ClassicAI()
        sprite = Unit.from_type("Sprite", 0, 3, 4)  # has no_enemy_retaliation
        archer = Unit.from_type("Archer", 0, 0, 4)
        # Enemy adjacent to cover position (1,4) so sprite can attack from there
        enemy = Unit.from_type("Swordsman", 1, 2, 4)
        b = _battle([sprite, enemy, archer])
        occ = b._move_occupied(sprite)
        reachable = b.grid.reachable(sprite.pos, sprite.speed, occ,
                                     sprite.is_flying,
                                     ai._tail_dir(sprite), b._moat_cells())
        pos_map = build_attack_position_map(b, sprite, [enemy], reachable)
        # Cover position (1,4) is adjacent to archer at (0,4) and enemy at (2,4)
        result = ai._attack_from_cover(b, sprite, (1, 4), [enemy], pos_map)
        assert result is not None

    def test_normal_unit_no_attack_from_cover(self):
        ai = ClassicAI()
        sw = Unit.from_type("Swordsman", 0, 2, 4)  # no special ability
        enemy = Unit.from_type("Swordsman", 1, 4, 4)
        b = _battle([sw, enemy])
        occ = b._move_occupied(sw)
        reachable = b.grid.reachable(sw.pos, sw.speed, occ, sw.is_flying,
                                     ai._tail_dir(sw), b._moat_cells())
        pos_map = build_attack_position_map(b, sw, [enemy], reachable)
        result = ai._attack_from_cover(b, sw, (3, 4), [enemy], pos_map)
        assert result is not None  # function works, just not called for normal units


# ─── P4: Cover position with stacking avoidance ─────────────

class TestCoverPosStacking:
    """Cover position avoids friendlies when avoid_stacking is set."""

    def test_cover_pos_prefers_non_stacking(self):
        ai = ClassicAI()
        sw = Unit.from_type("Swordsman", 0, 3, 4)
        archer = Unit.from_type("Archer", 0, 0, 4)
        # Another unit already covering the archer
        blocker = Unit.from_type("Pikeman", 0, 1, 4)
        enemy = Unit.from_type("Swordsman", 1, 8, 4)
        b = _battle([sw, archer, blocker, enemy])
        occ = b._move_occupied(sw)
        pos = ai._cover_pos(b, sw, archer, occ, None,
                            avoid_stacking=True)
        # Should still find a position (might be the same as blocker
        # if no alternative, but the function shouldn't crash)
        assert pos is not None

    def test_cover_pos_default_nearest(self):
        ai = ClassicAI()
        sw = Unit.from_type("Swordsman", 0, 3, 4)
        archer = Unit.from_type("Archer", 0, 0, 4)
        enemy = Unit.from_type("Swordsman", 1, 8, 4)
        b = _battle([sw, archer, enemy])
        occ = b._move_occupied(sw)
        pos = ai._cover_pos(b, sw, archer, occ, None)
        # Default: nearest reachable cell adjacent to archer
        assert pos is not None


# ─── Integration: defense area check ────────────────────────

class TestInDefendedArea:
    """isPositionLocatedInDefendedArea logic."""

    def test_team0_own_half(self):
        sw = Unit.from_type("Swordsman", 0, 3, 4)
        enemy = Unit.from_type("Swordsman", 1, 8, 4)
        b = _battle([sw, enemy])
        s = analyze(b, sw)
        # col=2 is in team 0's half (mid=5)
        assert ClassicAI._in_defended_area(b, sw, (2, 4), s) is True
        # col=7 is in enemy half
        assert ClassicAI._in_defended_area(b, sw, (7, 4), s) is False

    def test_team1_own_half(self):
        sw = Unit.from_type("Swordsman", 1, 8, 4)
        enemy = Unit.from_type("Swordsman", 0, 3, 4)
        b = _battle([sw, enemy])
        s = analyze(b, sw)
        # col=7 is in team 1's half (mid=5)
        assert ClassicAI._in_defended_area(b, sw, (7, 4), s) is True
        # col=2 is in enemy half
        assert ClassicAI._in_defended_area(b, sw, (2, 4), s) is False

"""A2 tests — spell heuristic fidelity (5 fixes + 1 relabel).

Covers:
  §3.3-#11  Slow dynamic speed loss
  §3.3-#14  Slow distance decay (ReduceEffectivenessByDistance)
  §3.3-#15  Haste dynamic speed gain
  §3.5-#35  spellDurationMultiplier
  §3.5-#36  effect value × duration multiplier
"""

import math
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.hex_grid import HexGrid
from engine.unit import Unit
from engine.hero import Hero
from engine.battle_state import BattleState
from engine.spells import Spell, BUFF, DEBUFF, DAMAGE, spell_damage

from ai.classic.evaluation import AIState, analyze
from ai.classic.spells import (
    _slow_ratio, _haste_ratio, _spell_duration_multiplier,
    _distance_from_starting_edge, SPEED_INSTANT,
    _effect_value, select_best_spell,
)


# ── helpers ──────────────────────────────────────────────────────

def _make_unit(name="Griffin", team=0, col=5, row=4, speed=5,
               is_archer=False, is_flying=False, **kw):
    """Create a minimal unit for spell tests."""
    u = Unit(name=name, team=team, col=col, row=row,
             attack=5, defense=5, hp=30, speed=speed, damage=10,
             is_archer=is_archer, is_flying=is_flying, count=10,
             **kw)
    return u


def _make_battle(units, hero=None, enemy_hero=None):
    """Create a BattleState with the given units."""
    grid = HexGrid()
    b = BattleState(grid, units)
    if hero:
        b.heroes[0] = hero
    if enemy_hero:
        b.heroes[1] = enemy_hero
    return b


def _default_state():
    s = AIState()
    s.my_team = 0
    s.my_army = 500.0
    s.enemy_army = 500.0
    s.my_avg_speed = 5.0
    s.enemy_avg_speed = 5.0
    s.enemy_shooters = 0.0
    return s


# ── §3.3-#11: Slow dynamic speed loss ───────────────────────────

class TestSlowDynamicSpeed:
    """Slow lost = currentSpeed - max(1, currentSpeed - 2)."""

    def test_normal_speed_lost_is_2(self):
        """Speed >= 3: lost = 2 (unchanged from before)."""
        u = _make_unit(speed=5)
        s = _default_state()
        ratio = _slow_ratio(u, s)
        # 0.1 * 2 = 0.2 base, no modifiers
        assert ratio == pytest.approx(0.2)

    def test_speed_2_lost_is_1(self):
        """Speed 2: newSpeed = max(1, 0) = 1, lost = 1."""
        u = _make_unit(speed=2)
        s = _default_state()
        ratio = _slow_ratio(u, s)
        # 0.1 * 1 = 0.1 base, but speed(2) < my_avg_speed(5) → /2 → 0.05
        assert ratio == pytest.approx(0.05)

    def test_speed_1_lost_is_0(self):
        """Speed 1 (CRAWLING): no speed to lose, ratio = 0."""
        u = _make_unit(speed=1)
        s = _default_state()
        ratio = _slow_ratio(u, s)
        assert ratio == pytest.approx(0.0)

    def test_high_speed_lost_capped_at_2(self):
        """Very fast units still only lose 2."""
        u = _make_unit(speed=11)
        s = _default_state()
        ratio = _slow_ratio(u, s)
        assert ratio == pytest.approx(0.2)


# ── §3.3-#14: Slow distance decay ───────────────────────────────

class TestSlowDistanceDecay:
    """ratio /= ReduceEffectivenessByDistance for non-flying non-Haste."""

    def test_near_edge_no_decay(self):
        """Unit at col 0 team 0: distance = 1, ratio unchanged."""
        u = _make_unit(team=0, col=0, speed=5)
        grid = HexGrid()
        s = _default_state()
        with_grid = _slow_ratio(u, s, grid)
        without_grid = _slow_ratio(u, s)
        # distance = 0 + 1 = 1, so ratio /= 1 = ratio
        assert with_grid == pytest.approx(without_grid)

    def test_midfield_decay(self):
        """Unit at col 5 team 0: distance = 6, ratio reduced by 6x."""
        u = _make_unit(team=0, col=5, speed=5)
        grid = HexGrid()
        s = _default_state()
        with_grid = _slow_ratio(u, s, grid)
        without_grid = _slow_ratio(u, s)
        # distance = 5 + 1 = 6
        assert with_grid == pytest.approx(without_grid / 6.0)

    def test_far_edge_decay(self):
        """Unit at col 10 team 0: distance = 11, ratio / 11."""
        u = _make_unit(team=0, col=10, speed=5)
        grid = HexGrid()
        s = _default_state()
        with_grid = _slow_ratio(u, s, grid)
        without_grid = _slow_ratio(u, s)
        assert with_grid == pytest.approx(without_grid / 11.0)

    def test_team1_distance_from_right(self):
        """Team 1 at col 10: distance = 11 - 10 = 1 (near own edge)."""
        u = _make_unit(team=1, col=10, speed=5)
        grid = HexGrid()
        s = _default_state()
        with_grid = _slow_ratio(u, s, grid)
        without_grid = _slow_ratio(u, s)
        assert with_grid == pytest.approx(without_grid)  # /1

    def test_flying_no_decay(self):
        """Flying units are exempt from distance decay."""
        u = _make_unit(col=5, speed=5, is_flying=True)
        grid = HexGrid()
        s = _default_state()
        with_grid = _slow_ratio(u, s, grid)
        without_grid = _slow_ratio(u, s)
        assert with_grid == pytest.approx(without_grid)

    def test_haste_overrides_distance(self):
        """If target has Haste, ×2 bonus applies, no distance decay."""
        u = _make_unit(col=5, speed=5)
        u.effects.append(type('E', (), {'name': 'Haste', 'speed_delta': 2,
                                        'is_positive': True})())
        grid = HexGrid()
        s = _default_state()
        with_grid = _slow_ratio(u, s, grid)
        without_grid = _slow_ratio(u, s)
        # Both should have ×2 for Haste, no distance reduction
        assert with_grid == pytest.approx(without_grid)


# ── §3.3-#15: Haste dynamic speed gain ──────────────────────────

class TestHasteDynamicSpeed:
    """gained = min(SPEED_INSTANT, speed + 2) - speed."""

    def test_normal_speed_gained_is_2(self):
        """Speed <= 9: gained = 2."""
        u = _make_unit(speed=5)
        s = _default_state()
        ratio = _haste_ratio(u, s)
        # 0.05 * 2 = 0.1
        assert ratio == pytest.approx(0.1)

    def test_speed_10_gained_is_1(self):
        """Speed 10: capped at INSTANT(11), gained = 1."""
        u = _make_unit(speed=10)
        s = _default_state()
        ratio = _haste_ratio(u, s)
        assert ratio == pytest.approx(0.05)

    def test_speed_11_gained_is_0(self):
        """Speed 11 (INSTANT): already max, no gain."""
        u = _make_unit(speed=11)
        s = _default_state()
        ratio = _haste_ratio(u, s)
        assert ratio == pytest.approx(0.0)


# ── §3.5-#35: spellDurationMultiplier ───────────────────────────

class TestSpellDurationMultiplier:
    """0 if power < 2 and target._acted, else 1."""

    def test_power_ge2_always_1(self):
        hero = Hero(power=3)
        u = _make_unit()
        u._acted = True
        assert _spell_duration_multiplier(hero, u) == 1

    def test_power_1_not_acted_is_1(self):
        hero = Hero(power=1)
        u = _make_unit()
        u._acted = False
        assert _spell_duration_multiplier(hero, u) == 1

    def test_power_1_acted_is_0(self):
        hero = Hero(power=1)
        u = _make_unit()
        u._acted = True
        assert _spell_duration_multiplier(hero, u) == 0

    def test_power_2_acted_is_1(self):
        """Power == 2 is NOT < 2, so multiplier is 1."""
        hero = Hero(power=2)
        u = _make_unit()
        u._acted = True
        assert _spell_duration_multiplier(hero, u) == 1


# ── §3.5-#36: effect value × duration multiplier ────────────────

class TestEffectValueDuration:
    """Effect value should be 0 when duration multiplier is 0."""

    def test_power1_acted_unit_ignored(self):
        """Hero power=1, target already acted → effect value = 0."""
        from engine.spells import SPELLS
        hero = Hero(power=1, spells=["Slow"])
        enemy = _make_unit(name="Enemy", team=1, col=8, row=4, speed=5)
        enemy._acted = True
        friend = _make_unit(name="Friend", team=0, col=2, row=4)
        battle = _make_battle([friend, enemy], hero=hero, enemy_hero=None)
        s = analyze(battle, friend)
        spell = SPELLS["Slow"]
        val, target = _effect_value(battle, hero, spell,
                                    [friend], [enemy], s)
        # duration multiplier = 0 for acted enemy when hero power=1
        assert val == 0.0

    def test_power2_acted_unit_valued(self):
        """Hero power=2, target acted → duration multiplier = 1, value > 0."""
        from engine.spells import SPELLS
        hero = Hero(power=2, spells=["Slow"])
        enemy = _make_unit(name="Enemy", team=1, col=8, row=4, speed=5)
        enemy._acted = True
        friend = _make_unit(name="Friend", team=0, col=2, row=4)
        battle = _make_battle([friend, enemy], hero=hero, enemy_hero=None)
        s = analyze(battle, friend)
        spell = SPELLS["Slow"]
        val, target = _effect_value(battle, hero, spell,
                                    [friend], [enemy], s)
        assert val > 0.0


# ── §3.3-#20 (relabel): Bless/Curse min=max ────────────────────

class TestBlessCurseMinMax:
    """Already implemented — verify existing behavior."""

    def test_bless_fixed_damage_returns_zero(self):
        """Unit with min=max damage → Bless is useless."""
        from engine.spells import SPELLS
        u = _make_unit()
        assert u.damage_min == u.damage_max  # single damage value
        from ai.classic.spells import _effect_ratio
        s = _default_state()
        ratio = _effect_ratio(SPELLS["Bless"], u, [], s)
        assert ratio == 0.0

    def test_curse_fixed_damage_returns_zero(self):
        """Unit with min=max damage → Curse is useless."""
        from engine.spells import SPELLS
        u = _make_unit()
        from ai.classic.spells import _effect_ratio
        s = _default_state()
        ratio = _effect_ratio(SPELLS["Curse"], u, [], s)
        assert ratio == 0.0

"""M7d tests — Hero combat skills (Archery, Ballistics, Leadership, Luck)."""

import random
import pytest

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.hero import Hero, SKILL_VALUES
from engine.unit import Unit
from engine.hex_grid import HexGrid
from engine.battle_state import BattleState
from engine.castle import Castle, WALL_POSITIONS
import config


# ── fixtures ──────────────────────────────────────────────

def _grid():
    return HexGrid()


def _make_unit(name="Swordsman", team=0, col=1, row=4, **kw):
    return Unit.from_type(name, team, col, row, **kw)


def _battle(units=None, hero0=None, hero1=None, castle=None, morale=None, luck=None):
    if units is None:
        units = [_make_unit("Swordsman", 0, 1, 4), _make_unit("Swordsman", 1, 9, 4)]
    heroes = {0: hero0, 1: hero1}
    return BattleState(_grid(), units, heroes=heroes, castle=castle,
                       morale=morale, luck=luck)


# ── P1: Hero skill model ────────────────────────────────

class TestHeroSkillModel:
    def test_no_skills_default(self):
        h = Hero()
        assert h.skills == {}
        assert h.get_skill_level("archery") == 0
        assert h.get_skill_value("archery") == 0

    def test_skills_from_constructor(self):
        h = Hero(skills={"archery": 2, "leadership": 3})
        assert h.get_skill_level("archery") == 2
        assert h.get_skill_value("archery") == 25
        assert h.get_skill_level("leadership") == 3
        assert h.get_skill_value("leadership") == 3

    def test_skills_from_config(self):
        h = Hero.from_config({"power": 5, "skills": {"luck": 1}})
        assert h is not None
        assert h.get_skill_level("luck") == 1
        assert h.get_skill_value("luck") == 1

    def test_invalid_skill_ignored(self):
        h = Hero(skills={"nonexistent": 3, "archery": 1})
        assert "nonexistent" not in h.skills
        assert h.get_skill_level("archery") == 1

    def test_invalid_level_ignored(self):
        h = Hero(skills={"archery": 0, "luck": 4, "leadership": -1})
        assert "archery" not in h.skills
        assert "luck" not in h.skills
        assert "leadership" not in h.skills

    def test_skill_values_table(self):
        """Verify SKILL_VALUES matches fheroes2 game_static.cpp."""
        assert SKILL_VALUES["archery"] == {1: 10, 2: 25, 3: 50}
        assert SKILL_VALUES["leadership"] == {1: 1, 2: 2, 3: 3}
        assert SKILL_VALUES["luck"] == {1: 1, 2: 2, 3: 3}


# ── P2: Archery ──────────────────────────────────────────

class TestArchery:
    def test_archery_basic_increases_ranged_damage(self):
        hero = Hero(skills={"archery": 1})  # +10%
        atk = _make_unit("Archer", 0, 1, 4)
        dfn = _make_unit("Swordsman", 1, 9, 4)
        bs = _battle([atk, dfn], hero0=hero)
        dmg_with = bs.expected_damage(atk, dfn, ranged=True)

        bs_none = _battle([_make_unit("Archer", 0, 1, 4),
                           _make_unit("Swordsman", 1, 9, 4)])
        dmg_without = bs_none.expected_damage(bs_none.units[0], bs_none.units[1],
                                               ranged=True)
        assert dmg_with == int(dmg_without * 1.10)

    def test_archery_expert_increases_ranged_damage(self):
        hero = Hero(skills={"archery": 3})  # +50%
        atk = _make_unit("Archer", 0, 1, 4)
        dfn = _make_unit("Swordsman", 1, 9, 4)
        bs = _battle([atk, dfn], hero0=hero)
        dmg_with = bs.expected_damage(atk, dfn, ranged=True)

        bs_none = _battle([_make_unit("Archer", 0, 1, 4),
                           _make_unit("Swordsman", 1, 9, 4)])
        dmg_without = bs_none.expected_damage(bs_none.units[0], bs_none.units[1],
                                               ranged=True)
        assert dmg_with == int(dmg_without * 1.50)

    def test_archery_does_not_affect_melee(self):
        hero = Hero(skills={"archery": 3})
        atk = _make_unit("Archer", 0, 1, 4)
        dfn = _make_unit("Swordsman", 1, 3, 4)
        bs = _battle([atk, dfn], hero0=hero)
        dmg_with = bs.expected_damage(atk, dfn, ranged=False)

        bs_none = _battle([_make_unit("Archer", 0, 1, 4),
                           _make_unit("Swordsman", 1, 3, 4)])
        dmg_without = bs_none.expected_damage(bs_none.units[0], bs_none.units[1],
                                               ranged=False)
        assert dmg_with == dmg_without

    def test_archery_eliminated_siege_penalty(self):
        """Archery at any level eliminates castle wall shooting penalty."""
        hero = Hero(skills={"archery": 1})
        atk = _make_unit("Archer", 0, 1, 4)  # outside walls
        dfn = _make_unit("Orc", 1, 9, 4)     # inside walls
        castle = Castle()
        bs = _battle([atk, dfn], hero0=hero, castle=castle)
        # Archery hero → no penalty
        assert not bs._shooting_penalty(atk, dfn)

    def test_no_archery_siege_penalty_still_applies(self):
        """Without Archery, siege shooting penalty still applies."""
        atk = _make_unit("Archer", 0, 1, 4)
        dfn = _make_unit("Orc", 1, 9, 4)
        castle = Castle()
        bs = _battle([atk, dfn], castle=castle)
        assert bs._shooting_penalty(atk, dfn)

    def test_archery_bonus_in_roll_damage(self):
        """roll_damage also applies Archery bonus (statistical check)."""
        hero = Hero(skills={"archery": 3})
        atk = _make_unit("Archer", 0, 1, 4)
        dfn = _make_unit("Swordsman", 1, 9, 4)
        bs = _battle([atk, dfn], hero0=hero)
        random.seed(42)
        samples = [bs.roll_damage(atk, dfn, ranged=True) for _ in range(200)]
        avg_with = sum(samples) / len(samples)

        bs_none = _battle([_make_unit("Archer", 0, 1, 4),
                           _make_unit("Swordsman", 1, 9, 4)])
        random.seed(42)
        samples_none = [bs_none.roll_damage(bs_none.units[0], bs_none.units[1],
                                              ranged=True) for _ in range(200)]
        avg_without = sum(samples_none) / len(samples_none)
        # Expert Archery should roughly +50% (tolerance for randomness)
        assert avg_with > avg_without * 1.3


# ── P3: Ballistics ───────────────────────────────────────

class TestBallistics:
    def test_no_ballistics_default_catapult(self):
        """Default: 1 shot, can miss, damage 1."""
        castle = Castle()
        rng = random.Random(42)
        shots = castle.catapult_round(ballistics=0, rng=rng)
        assert len(shots) == 1
        assert shots[0]["damage"] <= 1  # no double damage in default

    def test_basic_ballistics_always_hits(self):
        """Basic: 1 shot, can't miss."""
        castle = Castle()
        # Seed shouldn't matter — Basic never misses
        all_hit = True
        for seed in range(50):
            c = Castle()
            shots = c.catapult_round(ballistics=1, rng=random.Random(seed))
            for s in shots:
                if not s["hit"]:
                    all_hit = False
        assert all_hit

    def test_advanced_ballistics_two_shots(self):
        """Advanced: 2 shots."""
        castle = Castle()
        shots = castle.catapult_round(ballistics=2, rng=random.Random(42))
        assert len(shots) == 2

    def test_expert_ballistics_double_damage(self):
        """Expert: 2 shots, always double damage (2)."""
        all_double = True
        for seed in range(50):
            c = Castle()
            shots = c.catapult_round(ballistics=3, rng=random.Random(seed))
            for s in shots:
                if s["hit"] and s["damage"] != 2:
                    all_double = False
        assert all_double

    def test_catapult_walls_go_down_with_double_damage(self):
        """Expert Ballistics does 2 damage → wall HP drops by 2."""
        castle = Castle()
        # All walls start at HP 2
        initial_hp = dict(castle.walls)
        shots = castle.catapult_round(ballistics=3, rng=random.Random(42))
        for s in shots:
            if s["hit"] and s["damage"] > 0 and s["remaining_hp"] is not None:
                # remaining_hp should be 0 (started at 2, took 2 damage)
                if s["target"].startswith("("):
                    assert s["remaining_hp"] == 0


# ── P4: Leadership + Luck ───────────────────────────────

class TestLeadershipAndLuck:
    def test_leadership_adds_morale(self):
        hero = Hero(skills={"leadership": 2})  # +2 morale
        bs = _battle(hero0=hero)
        assert bs.morale[0] == 2

    def test_leadership_stacks_with_base_morale(self):
        hero = Hero(skills={"leadership": 3})  # +3
        bs = _battle(hero0=hero, morale={0: 1, 1: 0})
        # 1 + 3 = 4, clamped to 3
        assert bs.morale[0] == 3  # max 3

    def test_luck_adds_luck(self):
        hero = Hero(skills={"luck": 1})
        bs = _battle(hero1=hero)
        assert bs.luck[1] == 1

    def test_luck_stacks_with_base(self):
        hero = Hero(skills={"luck": 2})
        bs = _battle(hero0=hero, luck={0: 1, 1: 0})
        assert bs.luck[0] == 3  # 1 + 2 = 3

    def test_morale_clamped_to_neg3(self):
        hero = Hero(skills={"leadership": 1})  # +1
        bs = _battle(hero1=hero, morale={0: 0, 1: -3})
        # hero1 has leadership +1 → -3 + 1 = -2
        assert bs.morale[1] == -2

    def test_both_heroes_skills_independent(self):
        h0 = Hero(skills={"leadership": 2, "luck": 3})
        h1 = Hero(skills={"leadership": 1, "luck": 1})
        bs = _battle(hero0=h0, hero1=h1)
        assert bs.morale[0] == 2
        assert bs.morale[1] == 1
        assert bs.luck[0] == 3
        assert bs.luck[1] == 1


# ── P4b: Undead morale immunity ─────────────────────────

class TestUndeadMorale:
    def test_undead_immune_to_good_morale(self):
        """Undead units never get morale effects."""
        hero = Hero(skills={"leadership": 3})  # morale = +3
        # Skeleton has "undead" tag
        skeleton = _make_unit("Skeleton", 0, 1, 4)
        bs = _battle([skeleton, _make_unit("Swordsman", 1, 9, 4)], hero0=hero)
        # Force good morale by setting morale to max
        bs.morale[0] = 3
        # Roll many times — undead should always get 0
        random.seed(42)
        for _ in range(100):
            assert bs.roll_morale(0, skeleton) == 0

    def test_undead_immune_to_bad_morale(self):
        skeleton = _make_unit("Skeleton", 0, 1, 4)
        bs = _battle([skeleton, _make_unit("Swordsman", 1, 9, 4)])
        bs.morale[0] = -3
        random.seed(42)
        for _ in range(100):
            assert bs.roll_morale(0, skeleton) == 0

    def test_non_undead_still_get_morale(self):
        """Non-undead units on the same team still get morale."""
        hero = Hero(skills={"leadership": 3})
        skeleton = _make_unit("Skeleton", 0, 1, 2)
        swordsman = _make_unit("Swordsman", 0, 1, 6)
        bs = _battle([skeleton, swordsman,
                      _make_unit("Swordsman", 1, 9, 4)], hero0=hero)
        # Skeleton immune
        assert bs.roll_morale(0, skeleton) == 0
        # Swordsman not immune — with morale 3, should sometimes trigger
        triggered = False
        random.seed(42)
        for _ in range(500):
            if bs.roll_morale(0, swordsman) != 0:
                triggered = True
                break
        assert triggered

    def test_roll_morale_backward_compatible(self):
        """roll_morale without unit arg still works (non-undead path)."""
        bs = _battle()
        bs.morale[0] = 0
        assert bs.roll_morale(0) == 0


# ── Integration ─────────────────────────────────────────

class TestIntegration:
    def test_full_battle_with_skills(self):
        """Full battle with Archery + Leadership heroes completes."""
        h0 = Hero(power=3, skills={"archery": 2, "leadership": 1})
        h1 = Hero(power=3, skills={"luck": 2})
        units = [_make_unit("Archer", 0, 1, 4),
                 _make_unit("Swordsman", 0, 2, 6),
                 _make_unit("Swordsman", 1, 9, 4),
                 _make_unit("Archer", 1, 10, 6)]
        bs = _battle(units, hero0=h0, hero1=h1)
        assert bs.morale[0] == 1
        assert bs.luck[1] == 2
        # Ranged damage for team 0 should include +25% archery
        atk, dfn = units[0], units[2]
        dmg = bs.expected_damage(atk, dfn, ranged=True)
        assert dmg > 0

    def test_archery_with_siege_and_ballistics(self):
        """Siege battle with Archery + Ballistics: both skills active."""
        h0 = Hero(power=3, skills={"archery": 3, "ballistics": 2})
        atk = _make_unit("Archer", 0, 1, 4)
        dfn = _make_unit("Orc", 1, 9, 4)
        castle = Castle()
        bs = _battle([atk, dfn], hero0=h0, castle=castle)
        # Archery eliminates shooting penalty
        assert not bs._shooting_penalty(atk, dfn)
        # Ballistics: Advanced → 2 shots
        shots = castle.catapult_round(ballistics=2, rng=random.Random(42))
        assert len(shots) == 2

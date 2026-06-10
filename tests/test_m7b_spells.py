"""M7b spell expansion tests — effect properties, AOE, control, utility."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from engine.hex_grid import HexGrid
from engine.unit import Unit
from engine.battle_state import BattleState
from engine.hero import Hero
from engine.spells import (Spell, SPELLS, Effect, DAMAGE, AOE, BUFF, DEBUFF,
                            CONTROL, DISPEL, CURE, UTILITY, spell_damage,
                            make_effect)
from engine.actions import CastAction


# ── helpers ──────────────────────────────────────────────────────

def _make_battle(**kw):
    grid = HexGrid(11, 9)
    u0 = Unit("Test0", 0, 1, 4, attack=5, defense=5, hp=30, speed=4,
              damage=4, is_archer=False, is_flying=False, is_wide=False,
              count=10, tags=[])
    u1 = Unit("Test1", 1, 9, 4, attack=5, defense=5, hp=30, speed=4,
              damage=4, is_archer=False, is_flying=False, is_wide=False,
              count=10, tags=[])
    return BattleState(grid, [u0, u1], **kw)


def _make_battle_with_units(units, **kw):
    grid = HexGrid(11, 9)
    return BattleState(grid, units, **kw)


# ── 1. Effect property tests ────────────────────────────────────

class TestEffectProperties:
    """New Effect attributes: attack_delta, defense_delta, ranged_shield."""

    def test_attack_delta_from_bloodlust(self):
        u = Unit("Bloodlusted", 0, 1, 4, attack=5, defense=5, hp=30, speed=4,
                 damage=4, is_archer=False, is_flying=False, is_wide=False,
                 count=10)
        assert u.effective_attack == 5
        u.add_effect(Effect("Bloodlust", remaining=3, attack_delta=3))
        assert u.effective_attack == 8

    def test_defense_delta_from_stone_skin(self):
        u = Unit("Armored", 0, 1, 4, attack=5, defense=5, hp=30, speed=4,
                 damage=4, is_archer=False, is_flying=False, is_wide=False,
                 count=10)
        assert u.effective_defense == 5
        u.add_effect(Effect("Stone Skin", remaining=3, defense_delta=3))
        assert u.effective_defense == 8

    def test_disrupting_ray_stacks(self):
        u = Unit("Weakened", 0, 1, 4, attack=5, defense=9, hp=30, speed=4,
                 damage=4, is_archer=False, is_flying=False, is_wide=False,
                 count=10)
        u.add_effect(Effect("Disrupting Ray", remaining=3,
                            defense_delta=-3, stackable=True))
        assert u.effective_defense == 6
        u.add_effect(Effect("Disrupting Ray", remaining=3,
                            defense_delta=-3, stackable=True))
        assert u.effective_defense == 3

    def test_disrupting_ray_cant_go_below_zero(self):
        u = Unit("Weak", 0, 1, 4, attack=5, defense=2, hp=30, speed=4,
                 damage=4, is_archer=False, is_flying=False, is_wide=False,
                 count=10)
        u.add_effect(Effect("Disrupting Ray", remaining=3,
                            defense_delta=-3, stackable=True))
        u.add_effect(Effect("Disrupting Ray", remaining=3,
                            defense_delta=-3, stackable=True))
        assert u.effective_defense == 0

    def test_shield_reduces_ranged_damage(self):
        battle = _make_battle()
        u0, u1 = battle.units
        # u1 is defender
        dmg_before = battle.expected_damage(u0, u1, ranged=True)
        u1.add_effect(Effect("Shield", remaining=3, ranged_shield=0.5))
        dmg_after = battle.expected_damage(u0, u1, ranged=True)
        assert dmg_after == max(1, dmg_before // 2)

    def test_anti_magic_blocks_spells(self):
        u = Unit("Protected", 0, 1, 4, attack=5, defense=5, hp=30, speed=4,
                 damage=4, is_archer=False, is_flying=False, is_wide=False,
                 count=10)
        assert not u.is_immune_to_spells
        u.add_effect(Effect("Anti-Magic", remaining=5, anti_magic=True))
        assert u.is_immune_to_spells

    def test_non_stackable_replaces_same_effect(self):
        u = Unit("Blessed", 0, 1, 4, attack=5, defense=5, hp=30, speed=4,
                 damage=4, is_archer=False, is_flying=False, is_wide=False,
                 count=10)
        u.add_effect(Effect("Bless", remaining=2, damage_mult=1.2))
        u.add_effect(Effect("Bless", remaining=5, damage_mult=1.2))
        assert len([e for e in u.effects if e.name == "Bless"]) == 1
        assert u.effects[0].remaining == 5

    def test_effect_is_positive_flag(self):
        buff = make_effect(SPELLS["Bloodlust"], 3)
        assert buff.is_positive is True
        debuff = make_effect(SPELLS["Slow"], 3)
        assert debuff.is_positive is False
        control = make_effect(SPELLS["Blind"], 3)
        assert control.is_positive is False


# ── 2. _cast dispatch tests ─────────────────────────────────────

class TestCastDispatch:
    """Verify _cast handles all new spell kinds."""

    def test_cold_ray_damage(self):
        battle = _make_battle()
        u0, u1 = battle.units
        hero = Hero(power=5, spells=["Cold Ray"])
        battle.heroes[0] = hero
        hp_before = u1._total_hp
        spell = SPELLS["Cold Ray"]
        battle.execute(CastAction(0, spell, u1))
        assert u1._total_hp < hp_before
        assert u1._total_hp == hp_before - 20 * 5  # 100 damage

    def test_bloodlust_increases_attack(self):
        battle = _make_battle()
        u0, u1 = battle.units
        hero = Hero(power=3, spells=["Bloodlust"])
        battle.heroes[0] = hero
        assert u0.effective_attack == 5
        battle.execute(CastAction(0, SPELLS["Bloodlust"], u0))
        assert u0.effective_attack == 8

    def test_stone_skin_increases_defense(self):
        battle = _make_battle()
        u0, u1 = battle.units
        hero = Hero(power=3, spells=["Stone Skin"])
        battle.heroes[0] = hero
        assert u0.effective_defense == 5
        battle.execute(CastAction(0, SPELLS["Stone Skin"], u0))
        assert u0.effective_defense == 8

    def test_dispel_removes_all_effects(self):
        battle = _make_battle()
        u0, u1 = battle.units
        hero = Hero(power=3, spells=["Dispel Magic", "Haste", "Slow"])
        battle.heroes[0] = hero
        u0.add_effect(Effect("Haste", remaining=3, speed_delta=2))
        u1.add_effect(Effect("Slow", remaining=3, speed_delta=-2))
        assert u0.has_effect("Haste")
        battle.execute(CastAction(0, SPELLS["Dispel Magic"], u0))
        assert not u0.has_effect("Haste")
        # u1 still has Slow (not targeted)
        assert u1.has_effect("Slow")

    def test_cure_removes_debuffs_and_heals(self):
        battle = _make_battle()
        u0, u1 = battle.units
        hero = Hero(power=5, spells=["Cure", "Slow"])
        battle.heroes[0] = hero
        u0.add_effect(Effect("Slow", remaining=3, speed_delta=-2,
                              is_positive=False))
        u0.take_damage(10)  # lose some HP
        hp_before = u0._total_hp
        battle.execute(CastAction(0, SPELLS["Cure"], u0))
        assert not u0.has_effect("Slow")
        assert u0._total_hp > hp_before  # healed 5*5=25

    def test_blind_skips_turn(self):
        battle = _make_battle()
        u0, u1 = battle.units
        hero = Hero(power=3, spells=["Blind"])
        battle.heroes[0] = hero
        assert not u1.skip_turn
        battle.execute(CastAction(0, SPELLS["Blind"], u1))
        assert u1.skip_turn
        assert u1.has_effect("Blind")

    def test_paralyze_skips_turn(self):
        battle = _make_battle()
        u0, u1 = battle.units
        hero = Hero(power=3, spells=["Paralyze"])
        battle.heroes[0] = hero
        battle.execute(CastAction(0, SPELLS["Paralyze"], u1))
        assert u1.skip_turn

    def test_teleport_moves_unit(self):
        battle = _make_battle()
        u0, u1 = battle.units
        hero = Hero(power=3, spells=["Teleport"])
        battle.heroes[0] = hero
        old_pos = u0.pos
        battle.execute(CastAction(0, SPELLS["Teleport"], u0,
                                  destination=(5, 4)))
        assert u0.pos == (5, 4)
        assert u0.pos != old_pos


# ── 3. AOE spell tests ──────────────────────────────────────────

class TestAOESpells:
    """Area-of-effect damage spells."""

    def test_fireball_splash(self):
        """Fireball hits center + 6 neighbors."""
        grid = HexGrid(11, 9)
        # Place enemies in a cluster around (5,4)
        enemies = [
            Unit("E1", 1, 5, 4, attack=5, defense=5, hp=20, speed=4,
                 damage=3, is_archer=False, is_flying=False, is_wide=False,
                 count=1, tags=[]),
            Unit("E2", 1, 6, 4, attack=5, defense=5, hp=20, speed=4,
                 damage=3, is_archer=False, is_flying=False, is_wide=False,
                 count=1, tags=[]),
        ]
        friend = Unit("F1", 0, 1, 4, attack=5, defense=5, hp=30, speed=4,
                      damage=4, is_archer=False, is_flying=False, is_wide=False,
                      count=10, tags=[])
        battle = BattleState(grid, [friend] + enemies)
        hero = Hero(power=3, spells=["Fireball"])
        battle.heroes[0] = hero
        battle.execute(CastAction(0, SPELLS["Fireball"], enemies[0],
                                  cell=(5, 4)))
        # Both enemies should be damaged (base 10 * power 3 = 30 > 20 hp)
        assert not enemies[0].is_alive or enemies[0]._total_hp < 20
        # E2 at (6,4) is a neighbor of (5,4), should also be hit
        assert enemies[1]._total_hp < 20

    def test_cold_ring_does_not_hit_center(self):
        """Cold Ring hits neighbors but NOT the center unit."""
        grid = HexGrid(11, 9)
        center = Unit("Center", 1, 5, 4, attack=5, defense=5, hp=100, speed=4,
                      damage=3, is_archer=False, is_flying=False, is_wide=False,
                      count=1, tags=[])
        neighbor = Unit("Neighbor", 1, 6, 4, attack=5, defense=5, hp=20,
                        speed=4, damage=3, is_archer=False, is_flying=False,
                        is_wide=False, count=1, tags=[])
        friend = Unit("F1", 0, 1, 4, attack=5, defense=5, hp=30, speed=4,
                      damage=4, is_archer=False, is_flying=False, is_wide=False,
                      count=10, tags=[])
        battle = BattleState(grid, [friend, center, neighbor])
        hero = Hero(power=3, spells=["Cold Ring"])
        battle.heroes[0] = hero
        center_hp = center._total_hp
        battle.execute(CastAction(0, SPELLS["Cold Ring"], center,
                                  cell=(5, 4)))
        # Center should NOT be damaged
        assert center._total_hp == center_hp
        # Neighbor should be damaged
        assert neighbor._total_hp < 20

    def test_death_ripple_skips_undead(self):
        """Death Ripple damages non-undead but not undead units."""
        grid = HexGrid(11, 9)
        living = Unit("Living", 1, 9, 4, attack=5, defense=5, hp=30, speed=4,
                      damage=4, is_archer=False, is_flying=False, is_wide=False,
                      count=5, tags=["undead"])
        # Actually Death Ripple damages NON-undead, so let's fix:
        # living enemy with no undead tag should be hit
        human = Unit("Human", 1, 8, 4, attack=5, defense=5, hp=20, speed=4,
                     damage=4, is_archer=False, is_flying=False, is_wide=False,
                     count=2, tags=[])
        skeleton = Unit("Skeleton", 1, 7, 4, attack=4, defense=3, hp=4, speed=4,
                        damage=2, is_archer=False, is_flying=False, is_wide=False,
                        count=5, tags=["undead"])
        friend = Unit("F1", 0, 1, 4, attack=5, defense=5, hp=30, speed=4,
                      damage=4, is_archer=False, is_flying=False, is_wide=False,
                      count=10, tags=[])
        battle = BattleState(grid, [friend, human, skeleton])
        hero = Hero(power=5, spells=["Death Ripple"])
        battle.heroes[0] = hero
        human_hp = human._total_hp
        skel_hp = skeleton._total_hp
        battle.execute(CastAction(0, SPELLS["Death Ripple"], human))
        # Human (non-undead) should be damaged
        assert human._total_hp < human_hp
        # Skeleton (undead) should NOT be damaged
        assert skeleton._total_hp == skel_hp

    def test_holy_word_hits_undead_only(self):
        """Holy Word damages only undead units."""
        grid = HexGrid(11, 9)
        human = Unit("Human", 1, 9, 4, attack=5, defense=5, hp=20, speed=4,
                     damage=4, is_archer=False, is_flying=False, is_wide=False,
                     count=2, tags=[])
        skeleton = Unit("Skeleton", 1, 7, 4, attack=4, defense=3, hp=4, speed=4,
                        damage=2, is_archer=False, is_flying=False, is_wide=False,
                        count=5, tags=["undead"])
        friend = Unit("F1", 0, 1, 4, attack=5, defense=5, hp=30, speed=4,
                      damage=4, is_archer=False, is_flying=False, is_wide=False,
                      count=10, tags=[])
        battle = BattleState(grid, [friend, human, skeleton])
        hero = Hero(power=5, spells=["Holy Word"])
        battle.heroes[0] = hero
        human_hp = human._total_hp
        skel_hp = skeleton._total_hp
        battle.execute(CastAction(0, SPELLS["Holy Word"], skeleton))
        # Human (non-undead) should NOT be damaged
        assert human._total_hp == human_hp
        # Skeleton (undead) should be damaged
        assert skeleton._total_hp < skel_hp

    def test_armageddon_hits_everyone(self):
        """Armageddon damages ALL units on both sides."""
        grid = HexGrid(11, 9)
        u0 = Unit("Friend", 0, 1, 4, attack=5, defense=5, hp=100, speed=4,
                   damage=4, is_archer=False, is_flying=False, is_wide=False,
                   count=1, tags=[])
        u1 = Unit("Enemy", 1, 9, 4, attack=5, defense=5, hp=100, speed=4,
                   damage=4, is_archer=False, is_flying=False, is_wide=False,
                   count=1, tags=[])
        battle = BattleState(grid, [u0, u1])
        hero = Hero(power=2, spells=["Armageddon"])
        battle.heroes[0] = hero
        u0_hp = u0._total_hp
        u1_hp = u1._total_hp
        battle.execute(CastAction(0, SPELLS["Armageddon"], u1))
        assert u0._total_hp < u0_hp  # friendly fire
        assert u1._total_hp < u1_hp


# ── 4. Mass spell tests ─────────────────────────────────────────

class TestMassSpells:

    def test_mass_haste_affects_all_friendlies(self):
        grid = HexGrid(11, 9)
        f1 = Unit("F1", 0, 1, 3, attack=5, defense=5, hp=30, speed=4,
                   damage=4, is_archer=False, is_flying=False, is_wide=False,
                   count=5, tags=[])
        f2 = Unit("F2", 0, 1, 5, attack=5, defense=5, hp=30, speed=4,
                   damage=4, is_archer=False, is_flying=False, is_wide=False,
                   count=5, tags=[])
        e1 = Unit("E1", 1, 9, 4, attack=5, defense=5, hp=30, speed=4,
                   damage=4, is_archer=False, is_flying=False, is_wide=False,
                   count=5, tags=[])
        battle = BattleState(grid, [f1, f2, e1])
        hero = Hero(power=3, spells=["Mass Haste"])
        battle.heroes[0] = hero
        assert f1.speed == 4
        assert f2.speed == 4
        battle.execute(CastAction(0, SPELLS["Mass Haste"], f1))
        assert f1.speed == 6
        assert f2.speed == 6
        assert e1.speed == 4  # enemies unaffected

    def test_mass_slow_affects_all_enemies(self):
        grid = HexGrid(11, 9)
        f1 = Unit("F1", 0, 1, 4, attack=5, defense=5, hp=30, speed=4,
                   damage=4, is_archer=False, is_flying=False, is_wide=False,
                   count=5, tags=[])
        e1 = Unit("E1", 1, 9, 3, attack=5, defense=5, hp=30, speed=5,
                   damage=4, is_archer=False, is_flying=False, is_wide=False,
                   count=5, tags=[])
        e2 = Unit("E2", 1, 9, 5, attack=5, defense=5, hp=30, speed=6,
                   damage=4, is_archer=False, is_flying=False, is_wide=False,
                   count=5, tags=[])
        battle = BattleState(grid, [f1, e1, e2])
        hero = Hero(power=3, spells=["Mass Slow"])
        battle.heroes[0] = hero
        battle.execute(CastAction(0, SPELLS["Mass Slow"], e1))
        assert e1.speed == 3
        assert e2.speed == 4
        assert f1.speed == 4

    def test_mass_dispel_clears_everything(self):
        grid = HexGrid(11, 9)
        f1 = Unit("F1", 0, 1, 4, attack=5, defense=5, hp=30, speed=4,
                   damage=4, is_archer=False, is_flying=False, is_wide=False,
                   count=5, tags=[])
        e1 = Unit("E1", 1, 9, 4, attack=5, defense=5, hp=30, speed=4,
                   damage=4, is_archer=False, is_flying=False, is_wide=False,
                   count=5, tags=[])
        battle = BattleState(grid, [f1, e1])
        hero = Hero(power=3, spells=["Mass Dispel"])
        battle.heroes[0] = hero
        f1.add_effect(Effect("Haste", remaining=3, speed_delta=2))
        e1.add_effect(Effect("Slow", remaining=3, speed_delta=-2,
                              is_positive=False))
        battle.execute(CastAction(0, SPELLS["Mass Dispel"], f1))
        assert not f1.has_effect("Haste")
        assert not e1.has_effect("Slow")


# ── 5. Unit tags test ───────────────────────────────────────────

class TestUnitTags:

    def test_skeleton_is_undead(self):
        u = Unit.from_type("Skeleton", 0, 1, 4)
        assert u.has_tag("undead")

    def test_dragon_tag(self):
        u = Unit.from_type("Green Dragon", 0, 1, 4)
        assert u.has_tag("dragon")
        assert not u.has_tag("undead")

    def test_bone_dragon_is_undead_and_dragon(self):
        u = Unit.from_type("Bone Dragon", 0, 1, 4)
        assert u.has_tag("undead")
        assert u.has_tag("dragon")

    def test_elemental_tag(self):
        u = Unit.from_type("Fire Elemental", 0, 1, 4)
        assert u.has_tag("elemental")

    def test_peasant_has_no_tags(self):
        u = Unit.from_type("Peasant", 0, 1, 4)
        assert not u.has_tag("undead")
        assert not u.has_tag("dragon")
        assert not u.has_tag("elemental")


# ── 6. AI evaluation tests ──────────────────────────────────────

class TestSpellAI:
    """Verify AI selects appropriate spells."""

    def test_ai_picks_cold_ray_over_arrow_for_high_damage(self):
        """With high power, Cold Ray (20×power) beats Magic Arrow (10×power)."""
        from ai.classic.spells import select_best_spell
        from ai.classic.evaluation import analyze
        grid = HexGrid(11, 9)
        u0 = Unit("F1", 0, 1, 4, attack=5, defense=5, hp=30, speed=4,
                   damage=4, is_archer=False, is_flying=False, is_wide=False,
                   count=5, tags=[])
        u1 = Unit("E1", 1, 9, 4, attack=5, defense=5, hp=30, speed=4,
                   damage=4, is_archer=False, is_flying=False, is_wide=False,
                   count=5, tags=[])
        battle = BattleState(grid, [u0, u1])
        hero = Hero(power=5, max_spell_points=100,
                    spells=["Magic Arrow", "Cold Ray"])
        battle.heroes[0] = hero
        s = analyze(battle, u0)
        result = select_best_spell(battle, 0, s)
        assert result is not None
        assert result[0].name == "Cold Ray"

    def test_ai_blind_ratio_high_for_multi_enemy(self):
        """Blind ratio should be 0.8 when multiple enemies exist."""
        from ai.classic.spells import _blind_ratio
        from ai.classic.evaluation import AIState
        target = Unit("E", 1, 9, 4, attack=5, defense=5, hp=30, speed=4,
                      damage=4, is_archer=False, is_flying=False, is_wide=False,
                      count=5, tags=[])
        s = AIState()
        ratio = _blind_ratio(target, [target, target], s)
        assert ratio == 0.8

    def test_ai_blind_ratio_low_for_last_enemy(self):
        """Blind ratio drops to 0.4 for last enemy."""
        from ai.classic.spells import _blind_ratio
        from ai.classic.evaluation import AIState
        target = Unit("E", 1, 9, 4, attack=5, defense=5, hp=30, speed=4,
                      damage=4, is_archer=False, is_flying=False, is_wide=False,
                      count=5, tags=[])
        s = AIState()
        ratio = _blind_ratio(target, [target], s)
        assert ratio == 0.4

    def test_dragon_slayer_zero_without_dragons(self):
        """Dragon Slayer has 0 ratio when no enemy dragons."""
        from ai.classic.spells import _dragon_slayer_ratio
        from ai.classic.evaluation import AIState
        friend = Unit("F", 0, 1, 4, attack=5, defense=5, hp=30, speed=4,
                      damage=4, is_archer=False, is_flying=False, is_wide=False,
                      count=5, tags=[])
        enemy = Unit("E", 1, 9, 4, attack=5, defense=5, hp=30, speed=4,
                     damage=4, is_archer=False, is_flying=False, is_wide=False,
                     count=5, tags=[])
        s = AIState()
        assert _dragon_slayer_ratio(friend, [enemy], s) == 0.0

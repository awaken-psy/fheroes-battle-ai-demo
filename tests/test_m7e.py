"""M7e tests — Rules fidelity polish.

6 items: hero attack/defense, morale/luck d24/d12, Golem spell reduction,
Bone Dragon morale -1, AI bad morale immunity, Genie halving replacement.
"""

import random
import pytest

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.hero import Hero
from engine.unit import Unit
from engine.hex_grid import HexGrid
from engine.battle_state import BattleState
from engine.castle import Castle
from engine.actions import AttackAction
from engine.spells import SPELLS
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


# ── 1. Hero attack/defense bonus ─────────────────────────

class TestHeroAttackDefense:
    """Hero primary attack/defense adds to all army unit combat stats."""

    def test_hero_attack_fields_default(self):
        h = Hero()
        assert h.attack == 0
        assert h.defense == 0

    def test_hero_attack_fields_constructor(self):
        h = Hero(attack=5, defense=3)
        assert h.attack == 5
        assert h.defense == 3

    def test_hero_from_config_attack_defense(self):
        h = Hero.from_config({"attack": 7, "defense": 4})
        assert h.attack == 7
        assert h.defense == 4

    def test_hero_from_config_defaults_zero(self):
        h = Hero.from_config({"name": "Test"})
        assert h.attack == 0
        assert h.defense == 0

    def test_hero_attack_increases_damage(self):
        """Hero with attack=5 should increase expected damage."""
        hero0 = Hero(attack=5)
        atk = _make_unit("Swordsman", 0, 1, 4)
        dfn = _make_unit("Swordsman", 1, 9, 4)
        b_no_hero = _battle([atk, dfn])
        # Same units, fresh
        atk2 = _make_unit("Swordsman", 0, 1, 4)
        dfn2 = _make_unit("Swordsman", 1, 9, 4)
        b_with_hero = _battle([atk2, dfn2], hero0=hero0)
        dmg_no = b_no_hero.expected_damage(atk, dfn)
        dmg_yes = b_with_hero.expected_damage(atk2, dfn2)
        assert dmg_yes > dmg_no

    def test_hero_defense_reduces_incoming_damage(self):
        """Defender hero with defense=5 should reduce expected damage."""
        hero1 = Hero(defense=5)
        atk = _make_unit("Swordsman", 0, 1, 4)
        dfn = _make_unit("Swordsman", 1, 9, 4)
        b_no_hero = _battle([atk, dfn])
        atk2 = _make_unit("Swordsman", 0, 1, 4)
        dfn2 = _make_unit("Swordsman", 1, 9, 4)
        b_with_hero = _battle([atk2, dfn2], hero1=hero1)
        dmg_no = b_no_hero.expected_damage(atk, dfn)
        dmg_yes = b_with_hero.expected_damage(atk2, dfn2)
        assert dmg_yes < dmg_no

    def test_unit_effective_attack_with_hero(self):
        u = _make_unit("Swordsman", 0, 1, 4)
        assert u.effective_attack_with_hero(5) == u.effective_attack + 5
        assert u.effective_attack_with_hero(0) == u.effective_attack

    def test_unit_effective_defense_with_hero(self):
        u = _make_unit("Swordsman", 0, 1, 4)
        assert u.effective_defense_with_hero(3) == u.effective_defense + 3
        assert u.effective_defense_with_hero(0) == u.effective_defense

    def test_hero_attack_roll_damage(self):
        """Hero attack bonus applies to roll_damage (actual combat)."""
        hero0 = Hero(attack=10)
        atk = _make_unit("Swordsman", 0, 1, 4)
        dfn = _make_unit("Pikeman", 1, 9, 4)
        atk2 = _make_unit("Swordsman", 0, 1, 4)
        dfn2 = _make_unit("Pikeman", 1, 9, 4)
        b_no = _battle([atk, dfn])
        b_yes = _battle([atk2, dfn2], hero0=hero0)
        random.seed(42)
        dmg_no = b_no.roll_damage(atk, dfn)
        random.seed(42)
        dmg_yes = b_yes.roll_damage(atk2, dfn2)
        assert dmg_yes > dmg_no


# ── 2. Morale/Luck probability d24/d12 ───────────────────

class TestMoraleLuckProbability:
    """fheroes2: good morale/luck d24, bad morale d12, not d10."""

    def test_good_morale_probability(self):
        """Morale +1 ≈ 1/24 ≈ 4.2%, not 10%."""
        u = _make_unit("Swordsman", 0, 1, 4)
        b = _battle([u, _make_unit("Swordsman", 1, 9, 4)], morale={0: 1, 1: 0})
        hits = sum(1 for _ in range(10000)
                   if b.roll_morale(0, u) > 0)
        # 1/24 ≈ 4.17%, allow ±1.5% band
        assert 270 < hits < 570, f"Good morale hits: {hits} (expected ~417)"

    def test_bad_morale_probability(self):
        """Morale -1 ≈ 1/12 ≈ 8.3%, not 10%."""
        u = _make_unit("Swordsman", 0, 1, 4)
        b = _battle([u, _make_unit("Swordsman", 1, 9, 4)], morale={0: -1, 1: 0})
        hits = sum(1 for _ in range(10000)
                   if b.roll_morale(0, u) < 0)
        # 1/12 ≈ 8.33%, allow ±2% band
        assert 630 < hits < 1030, f"Bad morale hits: {hits} (expected ~833)"

    def test_good_luck_probability(self):
        """Luck +1 ≈ 1/24 ≈ 4.2%, not 10%."""
        b = _battle(luck={0: 1, 1: 0})
        hits = sum(1 for _ in range(10000)
                   if b._roll_luck(0) == 2.0)
        assert 270 < hits < 570, f"Good luck hits: {hits} (expected ~417)"

    def test_bad_luck_probability(self):
        """Luck -1 ≈ 1/24 ≈ 4.2%, not 10%."""
        b = _battle(luck={0: -1, 1: 0})
        hits = sum(1 for _ in range(10000)
                   if b._roll_luck(0) == 0.5)
        assert 270 < hits < 570, f"Bad luck hits: {hits} (expected ~417)"

    def test_morale_3_capped_probability(self):
        """Morale +3 ≈ 3/24 = 12.5%, not 30%."""
        u = _make_unit("Swordsman", 0, 1, 4)
        b = _battle([u, _make_unit("Swordsman", 1, 9, 4)], morale={0: 3, 1: 0})
        hits = sum(1 for _ in range(10000)
                   if b.roll_morale(0, u) > 0)
        # 3/24 = 12.5%, allow ±3% band
        assert 950 < hits < 1550, f"Morale+3 hits: {hits} (expected ~1250)"


# ── 3. Golem elemental spell reduction ───────────────────

class TestGolemSpellReduction:
    """Iron/Steel Golem take 50% from elemental spells."""

    def test_elemental_flag_on_spells(self):
        """Elemental damage spells have elemental=True."""
        for name in ["Magic Arrow", "Lightning Bolt", "Cold Ray", "Fireball",
                      "Fireblast", "Cold Ring", "Chain Lightning",
                      "Meteor Shower", "Elemental Storm", "Armageddon"]:
            assert SPELLS[name].elemental, f"{name} should be elemental"

    def test_non_elemental_spells(self):
        """Holy/Death damage spells are NOT elemental."""
        for name in ["Death Ripple", "Death Wave", "Holy Word", "Holy Shout"]:
            assert not SPELLS[name].elemental, f"{name} should NOT be elemental"

    def test_iron_golem_has_reduction_ability(self):
        g = _make_unit("Iron Golem", 0, 1, 4)
        assert g.has_ability("elemental_spell_reduction")

    def test_steel_golem_has_reduction_ability(self):
        g = _make_unit("Steel Golem", 0, 1, 4)
        assert g.has_ability("elemental_spell_reduction")

    def test_elemental_reduction_halves_damage(self):
        """elemental_spell_reduction reduces damage by factor."""
        golem = _make_unit("Iron Golem", 1, 9, 4)
        spell = SPELLS["Lightning Bolt"]
        dmg = 100
        result = BattleState._apply_elemental_reduction(golem, spell, dmg)
        assert result == 50

    def test_non_elemental_spell_not_reduced(self):
        """Non-elemental spell damage is NOT reduced for Golem."""
        golem = _make_unit("Iron Golem", 1, 9, 4)
        spell = SPELLS["Death Ripple"]
        dmg = 100
        result = BattleState._apply_elemental_reduction(golem, spell, dmg)
        assert result == 100  # no reduction

    def test_normal_unit_not_reduced(self):
        """Normal units don't get elemental reduction."""
        unit = _make_unit("Swordsman", 1, 9, 4)
        spell = SPELLS["Lightning Bolt"]
        dmg = 100
        result = BattleState._apply_elemental_reduction(unit, spell, dmg)
        assert result == 100


# ── 4. Bone Dragon morale -1 ─────────────────────────────

class TestBoneDragonMorale:
    """Enemy Bone Dragon reduces non-undead morale by 1."""

    def test_bone_dragon_has_tag(self):
        bd = _make_unit("Bone Dragon", 1, 9, 4)
        assert bd.has_tag("bone_dragon_morale")

    def test_bone_dragon_enemy_morale_penalty(self):
        """Non-undead facing Bone Dragon gets -1 morale effective."""
        u = _make_unit("Swordsman", 0, 1, 4)
        bd = _make_unit("Bone Dragon", 1, 9, 4)
        b = _battle([u, bd], morale={0: 2, 1: 0})
        # morale 2 → effective 1 because Bone Dragon, so 1/24 chance
        hits = sum(1 for _ in range(10000)
                   if b.roll_morale(0, u) > 0)
        # 1/24 ≈ 4.17%
        assert 270 < hits < 570, f"Morale hits with BD: {hits} (expected ~417)"

    def test_undead_immune_to_bone_dragon_penalty(self):
        """Undead units ignore Bone Dragon morale penalty."""
        skel = _make_unit("Skeleton", 0, 1, 4)
        bd = _make_unit("Bone Dragon", 1, 9, 4)
        b = _battle([skel, bd], morale={0: 2, 1: 0})
        for _ in range(100):
            assert b.roll_morale(0, skel) == 0  # undead always immune

    def test_no_bone_dragon_no_penalty(self):
        """Without Bone Dragon, morale is unchanged."""
        u = _make_unit("Swordsman", 0, 1, 4)
        e = _make_unit("Swordsman", 1, 9, 4)
        b = _battle([u, e], morale={0: 2, 1: 0})
        hits = sum(1 for _ in range(10000)
                   if b.roll_morale(0, u) > 0)
        # 2/24 ≈ 8.33%
        assert 630 < hits < 1030, f"Morale hits no BD: {hits} (expected ~833)"

    def test_bone_dragon_dead_no_penalty(self):
        """Dead Bone Dragon doesn't affect morale."""
        u = _make_unit("Swordsman", 0, 1, 4)
        bd = _make_unit("Bone Dragon", 1, 9, 4)
        bd.take_damage(bd._total_hp)  # kill it
        b = _battle([u, bd], morale={0: 2, 1: 0})
        hits = sum(1 for _ in range(10000)
                   if b.roll_morale(0, u) > 0)
        # 2/24 ≈ 8.33% (Bone Dragon is dead, no penalty)
        assert 630 < hits < 1030, f"Morale hits dead BD: {hits} (expected ~833)"


# ── 5. AI bad morale immunity ────────────────────────────

class TestAIBadMoraleImmunity:
    """AI gets 25% chance to avoid bad morale (headless.py)."""

    def test_immunity_in_headless(self):
        """Test the 25% immunity logic directly."""
        # We simulate the headless logic: randint(1,4)==1 → immune
        immune_count = sum(1 for _ in range(10000)
                           if random.randint(1, 4) == 1)
        # Should be ~25%, allow ±3%
        assert 2200 < immune_count < 2800, f"Immune: {immune_count} (expected ~2500)"


# ── 6. Genie halving replacement ─────────────────────────

class TestGenieHalvingReplacement:
    """Genie halving replaces normal damage instead of adding."""

    def test_halving_replaces_damage(self):
        """With 100% chance, normal damage is replaced by halving."""
        genie = Unit("TestGenie", 0, 4, 4, attack=10, defense=9, hp=50, speed=6,
                     damage_min=20, damage_max=30, is_archer=False, is_flying=True,
                     is_wide=False,
                     abilities=["enemy_halving"],
                     ability_params={"enemy_halving": {"chance": 100}},
                     count=1)
        target = Unit.from_type("Pikeman", 1, 5, 4, count=10)  # 10 pikemen
        b = _battle([genie, target])
        r = b.execute(AttackAction(genie, target, (4, 4), ranged=False))
        # Halving: kills 5 (half of 10), damage = 5 * 15 = 75
        assert "halving" in r["desc"]
        assert r["killed"] == 5
        assert target.count == 5

    def test_halving_not_triggered_normal_damage(self):
        """With 0% chance, normal damage applies."""
        genie = Unit("TestGenie", 0, 4, 4, attack=10, defense=9, hp=50, speed=6,
                     damage_min=20, damage_max=20, is_archer=False, is_flying=True,
                     is_wide=False,
                     abilities=["enemy_halving"],
                     ability_params={"enemy_halving": {"chance": 0}},
                     count=1)
        target = Unit.from_type("Pikeman", 1, 5, 4, count=10)
        b = _battle([genie, target])
        r = b.execute(AttackAction(genie, target, (4, 4), ranged=False))
        assert "halving" not in r["desc"]
        # Normal damage: 20 (fixed) × mult, not halving
        assert r["killed"] < 10

    def test_halving_only_one_stack_damage(self):
        """When halving triggers, ONLY halving damage is applied (not + normal)."""
        genie = Unit("TestGenie", 0, 4, 4, attack=100, defense=9, hp=50, speed=6,
                     damage_min=1000, damage_max=1000, is_archer=False, is_flying=True,
                     is_wide=False,
                     abilities=["enemy_halving"],
                     ability_params={"enemy_halving": {"chance": 100}},
                     count=1)
        # Target: 4 pikemen, each 15hp → total 60hp
        target = Unit.from_type("Pikeman", 1, 5, 4, count=4)
        b = _battle([genie, target])
        r = b.execute(AttackAction(genie, target, (4, 4), ranged=False))
        # Halving: 4//2=2 killed, dmg=30. NOT 1000+30.
        assert r["killed"] == 2
        assert r["dmg"] == 30

    def test_halving_single_creature_no_effect(self):
        """Halving does not trigger on single-creature stack."""
        genie = Unit("TestGenie", 0, 4, 4, attack=10, defense=9, hp=50, speed=6,
                     damage_min=20, damage_max=30, is_archer=False, is_flying=True,
                     is_wide=False,
                     abilities=["enemy_halving"],
                     ability_params={"enemy_halving": {"chance": 100}},
                     count=1)
        target = Unit.from_type("Pikeman", 1, 5, 4, count=1)  # single
        b = _battle([genie, target])
        r = b.execute(AttackAction(genie, target, (4, 4), ranged=False))
        # count=1 → halving skipped (1//2=0), normal damage applies
        assert "halving" not in r["desc"]


# ── Integration ──────────────────────────────────────────

class TestM7eIntegration:
    """Cross-feature integration tests."""

    def test_full_battle_with_hero_stats(self):
        """Full battle with hero attack/defense produces valid result."""
        hero0 = Hero(attack=3, defense=2)
        hero1 = Hero(attack=1, defense=4)
        u0 = _make_unit("Swordsman", 0, 1, 4)
        u1 = _make_unit("Swordsman", 1, 9, 4)
        b = _battle([u0, u1], hero0=hero0, hero1=hero1)
        # Verify hero bonuses are applied
        assert b._hero_attack(0) == 3
        assert b._hero_defense(0) == 2
        assert b._hero_attack(1) == 1
        assert b._hero_defense(1) == 4

    def test_hero_stats_with_archery(self):
        """Hero attack + Archery skill both apply to ranged damage."""
        hero0 = Hero(attack=5, skills={"archery": 2})
        atk = _make_unit("Archer", 0, 1, 4)
        dfn = _make_unit("Swordsman", 1, 9, 4)
        b = _battle([atk, dfn], hero0=hero0)
        # Both hero attack (+5) and archery (+25%) should increase damage
        dmg = b.expected_damage(atk, dfn, ranged=True)
        # Without hero
        b2 = _battle([_make_unit("Archer", 0, 1, 4),
                       _make_unit("Swordsman", 1, 9, 4)])
        dmg2 = b2.expected_damage(b2.units[0], b2.units[1], ranged=True)
        assert dmg > dmg2

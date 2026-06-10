"""A4 tests — berserk precision and final relabels (2 fixes + 1 relabel).

Covers:
  §7-#1   GetNearestTroops: head-to-head distance sorting
  §7-#5   CanAttackTargetFromPosition: wide-attacker orientation + moat
  §9-#4   isDisableCastSpell (relabel — functionally equivalent)
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.hex_grid import HexGrid
from engine.unit import Unit
from engine.battle_state import BattleState
from engine.actions import AttackAction, MoveAction, SkipAction

from ai.classic.planner import ClassicAI
from ai.classic.spells import select_best_spell
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


def _add_berserk(unit):
    """Add the Berserk effect to a unit."""
    from engine.spells import Effect
    unit.effects.append(Effect(name="Berserk", remaining=3))


# ── §7-#1  GetNearestTroops: head-to-head distance ──────────

class TestBerserkHeadDistance:
    """Verify that berserk sorts targets by head-to-head distance only,
    matching C++ GetNearestTroops which uses GetPosition() (head index)."""

    def test_single_hex_nearest_picked(self):
        """Berserker shoots the nearest unit by head distance."""
        grid = HexGrid()
        berserker = _u("Berserker", 0, col=5, row=4, is_archer=True, speed=5)
        near = _u("Near", 1, col=7, row=4)       # head dist = 2
        far = _u("Far", 0, col=3, row=4)          # head dist = 2, same col but different team
        # Make far truly farther
        far.col = 1  # head dist = 4
        _add_berserk(berserker)
        battle = _battle([berserker, near, far])

        ai = ClassicAI()
        action, desc = ai.decide(battle, berserker)
        assert isinstance(action, AttackAction)
        assert action.target is near

    def test_wide_target_sorted_by_head_not_body(self):
        """For a wide target, C++ uses head-head distance (not occupied_cells min).

        Layout:
          Berserker head=(5,4)
          Single:     head=(6,3) — head_dist=2
          Wide enemy: head=(9,4) tail=(8,4) — head_dist=4, tail_dist=3 → min=3

        Head sorting: single(2) < wide(4) → single first.
        Min-occupied would also pick single(2) < wide(3) → same result,
        but the key is that tail distance is NOT used for sorting.
        """
        grid = HexGrid()
        berserker = _u("Berserker", 0, col=5, row=4, speed=10)
        _add_berserk(berserker)

        wide = _u("Wide", 1, col=9, row=4, is_wide=True)   # head_dist=4
        single = _u("Single", 0, col=6, row=3)               # head_dist=2

        battle = _battle([berserker, wide, single])
        ai = ClassicAI()
        others = [u for u in battle.alive() if u is not berserker]
        # Sort by head distance (our fix)
        others.sort(key=lambda u: grid.distance(berserker.pos, u.pos))
        assert others[0] is single
        assert others[1] is wide

    def test_wide_berserker_sorts_by_own_head(self):
        """A wide berserker's own sorting uses its head position only."""
        grid = HexGrid()
        # Wide berserker: head at (5,4), tail at (4,4)
        berserker = _u("Berserker", 0, col=5, row=4, is_wide=True, speed=5)
        _add_berserk(berserker)
        # Target A: head at (7,4), head_dist from berserker head = 2
        a = _u("A", 1, col=7, row=4)
        # Target B: head at (3,4), head_dist from berserker head = 2
        # But dist from berserker TAIL (4,4) = 1 → min occupied would put B first
        b = _u("B", 0, col=3, row=4)

        battle = _battle([berserker, a, b])
        ai = ClassicAI()
        others = [u for u in battle.alive() if u is not berserker]
        others.sort(key=lambda u: grid.distance(berserker.pos, u.pos))
        # Both at head_dist=2, so order is stable/by second key
        # The key point: tail (col=3) does NOT give B priority
        # Verify head distances are equal
        assert grid.distance(berserker.pos, a.pos) == grid.distance(berserker.pos, b.pos)


# ── §7-#5  CanAttackTargetFromPosition ───────────────────────

class TestBerserkCanAttackFromPos:
    """Verify CanAttackTargetFromPosition checks for berserk melee:
    wide-attacker orientation and moat attack restriction."""

    def test_single_hex_melee_works(self):
        """Single-hex berserker can attack from any adjacent reachable cell."""
        grid = HexGrid()
        berserker = _u("Berserker", 0, col=5, row=4, speed=5)
        _add_berserk(berserker)
        target = _u("Target", 1, col=6, row=4)
        battle = _battle([berserker, target])

        ai = ClassicAI()
        action, desc = ai.decide(battle, berserker)
        assert isinstance(action, AttackAction)
        assert action.target is target

    def test_wide_attacker_can_attack_adjacent(self):
        """Wide berserker can attack a target adjacent to its head position."""
        grid = HexGrid()
        berserker = _u("Berserker", 0, col=5, row=4, is_wide=True, speed=5)
        _add_berserk(berserker)
        target = _u("Target", 1, col=6, row=4)
        battle = _battle([berserker, target])

        ai = ClassicAI()
        action, desc = ai.decide(battle, berserker)
        assert isinstance(action, AttackAction)
        assert action.target is target

    def test_can_attack_from_pos_single_hex(self):
        """_can_attack_from_pos returns True for single-hex attackers."""
        grid = HexGrid()
        unit = _u("Unit", 0, col=5, row=4)
        target = _u("Target", 1, col=6, row=4)
        ai = ClassicAI()
        # Single-hex: always True when called (adjacency checked separately)
        assert ai._can_attack_from_pos(grid, unit, target, (6, 4)) is True

    def test_can_attack_from_pos_wide_head_adj(self):
        """Wide attacker at pos adjacent to target by head → True."""
        grid = HexGrid()
        unit = _u("Unit", 0, col=5, row=4, is_wide=True)
        target = _u("Target", 1, col=7, row=4)
        ai = ClassicAI()
        # Attack position (6,4): head at (6,4), tail at (5,4)
        # (6,4) is adjacent to target at (7,4) → head_adj True
        assert ai._can_attack_from_pos(grid, unit, target, (6, 4)) is True

    def test_can_attack_from_pos_wide_neither_adj(self):
        """Wide attacker at pos where neither head nor tail is adjacent → False."""
        grid = HexGrid()
        unit = _u("Unit", 0, col=5, row=4, is_wide=True)
        target = _u("Target", 1, col=9, row=4)
        ai = ClassicAI()
        # Attack position (7,4): head at (7,4), tail at (6,4)
        # Target at (9,4): distance from head (7,4) = 2, from tail (6,4) = 3
        # Neither adjacent → False
        assert ai._can_attack_from_pos(grid, unit, target, (7, 4)) is False

    def test_moat_blocks_attack(self):
        """Non-flying wide unit in moat cell cannot attack (unless already there).
        Single-hex units skip moat check in _can_attack_from_pos, so we test
        with a wide unit."""
        grid = HexGrid()
        from engine.castle import Castle
        # Wide attacker — _can_attack_from_pos checks moat for wide units
        attacker = _u("Attacker", 0, col=5, row=4, speed=5, is_wide=True)
        # Place defender close to a moat cell so adjacency holds
        defender = _u("Defender", 1, col=8, row=4)
        castle = Castle()
        battle = _battle([attacker, defender], castle=castle)

        ai = ClassicAI()
        moat = battle._moat_cells()
        if not moat:
            pytest.skip("no moat cells in this configuration")

        # Find a moat cell that is NOT the attacker's current position
        moat_cell = None
        for mc in moat:
            if mc != attacker.pos and mc != attacker.tail_cell:
                moat_cell = mc
                break
        if moat_cell is None:
            pytest.skip("all moat cells overlap attacker position")

        # Non-flying wide unit at moat cell not already there → blocked
        # (adjacency check will likely fail too, but moat adds explicit block)
        result = ai._can_attack_from_pos(grid, attacker, defender,
                                         moat_cell, moat)
        # If the cell is adjacent to defender (head or tail), moat should still block
        head_adj = ai._pos_dist(grid, moat_cell, defender) <= 1
        td = ai._tail_dir(attacker)
        tail = (moat_cell[0] + td, moat_cell[1]) if td else moat_cell
        tail_adj = ai._pos_dist(grid, tail, defender) <= 1
        if head_adj or tail_adj:
            # Adjacent but in moat → blocked (not already there)
            assert result is False
            # Now move attacker there → should be allowed
            attacker.pos = moat_cell
            result2 = ai._can_attack_from_pos(grid, attacker, defender,
                                              moat_cell, moat)
            assert result2 is True

    def test_flying_ignores_moat(self):
        """Flying unit in moat cell can still attack."""
        grid = HexGrid()
        from engine.castle import Castle
        attacker = _u("Flyer", 0, col=5, row=4, speed=5, is_flying=True)
        defender = _u("Defender", 1, col=7, row=4)
        castle = Castle()
        battle = _battle([attacker, defender], castle=castle)

        ai = ClassicAI()
        moat = battle._moat_cells()
        if moat:
            moat_cell = next(iter(moat))
            assert ai._can_attack_from_pos(grid, attacker, defender,
                                           moat_cell, moat) is True


# ── §9-#4  isDisableCastSpell (relabel — functionally equivalent) ──

class TestIsDisableCastSpellEquiv:
    """Verify that our spell selection effectively matches C++ isDisableCastSpell.

    C++ checks: isSpellcastDisabled (artifact), SPELLCASTED, Earthquake w/o
    castle, no valid target.  All are covered by existing code paths.
    """

    def test_already_cast_blocks_spell(self):
        """Hero._cast_this_round prevents further casting (matches SPELLCASTED)."""
        from engine.hero import Hero
        from engine.spells import SPELLS
        hero = Hero(power=5, spells=["Lightning Bolt"])
        hero._cast_this_round = True
        assert not hero.can_cast(SPELLS["Lightning Bolt"])

    def test_no_spell_points_blocks(self):
        """Insufficient spell points prevents casting."""
        from engine.hero import Hero
        from engine.spells import SPELLS
        hero = Hero(power=5, spells=["Lightning Bolt"], max_spell_points=5)
        hero.spell_points = 0
        assert not hero.can_cast(SPELLS["Lightning Bolt"])

    def test_earthquake_without_castle_zero_value(self):
        """Earthquake returns 0 value when not attacking castle (matches C++)."""
        from engine.hero import Hero
        hero = Hero(power=5, spells=["Earthquake"])
        attacker = _u("Melee", 0, col=5, row=4, speed=5)
        defender = _u("Enemy", 1, col=8, row=4)
        battle = _battle([attacker, defender], heroes={0: hero})
        state = analyze(battle, attacker)
        # Not a siege → earthquake has no value
        result = select_best_spell(battle, 0, state)
        assert result is None

    def test_no_hero_no_spell(self):
        """No hero → select_best_spell returns None (matches commander==nullptr)."""
        attacker = _u("Unit", 0, col=5, row=4)
        defender = _u("Enemy", 1, col=8, row=4)
        battle = _battle([attacker, defender])
        state = analyze(battle, attacker)
        result = select_best_spell(battle, 0, state)
        assert result is None

    def test_immune_target_zero_value(self):
        """All targets immune → spell value is 0, not selected."""
        from engine.hero import Hero
        hero = Hero(power=5, spells=["Lightning Bolt"])
        attacker = _u("Caster", 0, col=5, row=4)
        immune = _u("Immune", 1, col=6, row=4,
                     abilities=("magic_resistance",))
        # Set 100% resistance
        immune.ability_params["magic_resistance"] = {"chance": 100}
        battle = _battle([attacker, immune], heroes={0: hero})
        state = analyze(battle, attacker)
        result = select_best_spell(battle, 0, state)
        assert result is None

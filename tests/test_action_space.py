"""Tests for R3 action space — encoding, decoding, legality mask."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

from engine.hex_grid import HexGrid
from engine.unit import Unit
from engine.hero import Hero
from engine.battle_state import BattleState
from engine.castle import Castle
from engine.actions import (MoveAction, AttackAction, SkipAction,
                            CastAction, RetreatAction)
from engine.spells import SPELLS, DAMAGE, AOE, BUFF, DEBUFF, CONTROL

from ai.action_space import (
    ACTION_DIM, GRID_CELLS, NUM_SPELLS,
    MAX_ENEMIES, MAX_ATTACK_POSITIONS,
    WAIT_IDX, DEFEND_IDX, RETREAT_IDX,
    MOVE_START, MOVE_END,
    ATTACK_START, ATTACK_END,
    CAST_START, CAST_END,
    cell_to_index, index_to_cell,
    action_to_index, index_to_action,
    legal_mask, enumerate_legal, action_type_label,
    _SPELL_ORDER, _SPELL_INDEX,
)


# ── Fixtures ───────────────────────────────────────────────────

def _grid():
    return HexGrid()


def _unit(name, team, col, row, **kw):
    return Unit.from_type(name, team, col, row, **kw)


def _battle(units, heroes=None, castle=None, **kw):
    grid = _grid()
    return BattleState(grid, units, heroes=heroes, castle=castle, **kw)


def _simple_battle():
    """Pikeman vs Pikeman — simplest possible battle."""
    u0 = _unit("Pikeman", 0, 0, 4)
    u1 = _unit("Pikeman", 1, 10, 4)
    return _battle([u0, u1])


def _archer_battle():
    """Archer + Pikeman (team 0) vs Swordsman (team 1)."""
    archer = _unit("Archer", 0, 1, 2)
    pike = _unit("Pikeman", 0, 0, 4)
    sword = _unit("Swordsman", 1, 10, 4)
    return _battle([archer, pike, sword])


def _hero_battle(spells=None):
    """Battle with heroes that can cast spells."""
    u0 = _unit("Pikeman", 0, 0, 4)
    u1 = _unit("Swordsman", 1, 10, 4)
    heroes = {
        0: Hero(power=5, max_spell_points=50, spells=spells),
        1: Hero(power=5, max_spell_points=50, spells=spells),
    }
    return _battle([u0, u1], heroes=heroes)


# ── Cell indexing ──────────────────────────────────────────────

class TestCellIndexing:

    def test_round_trip(self):
        for row in range(9):
            for col in range(11):
                idx = cell_to_index(col, row)
                c, r = index_to_cell(idx)
                assert (c, r) == (col, row)

    def test_range(self):
        assert cell_to_index(0, 0) == 0
        assert cell_to_index(10, 8) == 98

    def test_known_values(self):
        assert cell_to_index(0, 0) == 0
        assert cell_to_index(1, 0) == 1
        assert cell_to_index(0, 1) == 11
        assert cell_to_index(5, 4) == 4 * 11 + 5


# ── Constants ─────────────────────────────────────────────────

class TestConstants:

    def test_action_dim(self):
        expected = 2 + GRID_CELLS + MAX_ENEMIES * MAX_ATTACK_POSITIONS + NUM_SPELLS * GRID_CELLS + 1
        assert ACTION_DIM == expected

    def test_spell_count(self):
        # 38 total - 1 Teleport = 37
        assert NUM_SPELLS == 37
        assert "Teleport" not in _SPELL_ORDER

    def test_boundaries(self):
        assert WAIT_IDX == 0
        assert DEFEND_IDX == 1
        assert MOVE_START == 2
        assert MOVE_END == 100
        assert ATTACK_START == 101
        assert ATTACK_END == 156
        assert CAST_START == 157
        assert CAST_END == 3819
        assert RETREAT_IDX == 3820
        assert ACTION_DIM == 3821

    def test_all_spells_have_index(self):
        for name in _SPELL_ORDER:
            assert name in _SPELL_INDEX
        assert "Teleport" not in _SPELL_INDEX


# ── action_to_index / index_to_action round-trips ─────────────

class TestRoundTrip:

    def test_skip_action(self):
        b = _simple_battle()
        u = b.alive(0)[0]
        action = SkipAction(u)
        idx = action_to_index(action, b, u)
        assert idx == WAIT_IDX
        decoded = index_to_action(idx, b, u)
        assert isinstance(decoded, SkipAction)

    def test_defend_index(self):
        b = _simple_battle()
        u = b.alive(0)[0]
        decoded = index_to_action(DEFEND_IDX, b, u)
        assert isinstance(decoded, SkipAction)

    def test_retreat_action(self):
        b = _hero_battle()
        u = b.alive(0)[0]
        action = RetreatAction(0)
        idx = action_to_index(action, b, u)
        assert idx == RETREAT_IDX
        decoded = index_to_action(idx, b, u)
        assert isinstance(decoded, RetreatAction)
        assert decoded.team == 0

    def test_move_round_trip(self):
        b = _simple_battle()
        u = b.alive(0)[0]
        # Move to a reachable cell
        occ = b._move_occupied(u)
        reachable = b.grid.reachable(u.pos, u.speed, occ)
        dest = None
        for cell in reachable:
            if cell != u.pos:
                dest = cell
                break
        assert dest is not None

        path = b.grid.find_path(u.pos, dest, occ, max_len=u.speed)
        action = MoveAction(u, path)
        idx = action_to_index(action, b, u)
        assert MOVE_START <= idx <= MOVE_END
        assert idx == MOVE_START + cell_to_index(*dest)

    def test_melee_attack_round_trip(self):
        b = _simple_battle()
        u0 = b.alive(0)[0]
        u1 = b.alive(1)[0]
        # Move u0 adjacent to u1 first
        u0.pos = (9, 4)
        action = AttackAction(u0, u1, from_pos=(9, 4), ranged=False)
        idx = action_to_index(action, b, u0)
        assert ATTACK_START <= idx <= ATTACK_END
        decoded = index_to_action(idx, b, u0)
        assert isinstance(decoded, AttackAction)
        assert decoded.target is u1
        assert decoded.from_pos == (9, 4)
        assert not decoded.ranged

    def test_ranged_attack_round_trip(self):
        b = _archer_battle()
        archer = b.alive(0)[0]  # Archer
        assert archer.is_archer
        enemy = b.alive(1)[0]   # Swordsman
        action = AttackAction(archer, enemy, ranged=True)
        idx = action_to_index(action, b, archer)
        assert ATTACK_START <= idx <= ATTACK_END
        decoded = index_to_action(idx, b, archer)
        assert isinstance(decoded, AttackAction)
        assert decoded.target is enemy
        assert decoded.ranged is True

    def test_cast_round_trip(self):
        b = _hero_battle(spells=["Magic Arrow"])
        u0 = b.alive(0)[0]
        u1 = b.alive(1)[0]
        spell = SPELLS["Magic Arrow"]
        action = CastAction(0, spell, u1)
        idx = action_to_index(action, b, u0)
        assert CAST_START <= idx <= CAST_END
        decoded = index_to_action(idx, b, u0)
        assert isinstance(decoded, CastAction)
        assert decoded.spell.name == "Magic Arrow"
        assert decoded.target is u1


# ── legal_mask basic ───────────────────────────────────────────

class TestLegalMask:

    def test_shape_and_dtype(self):
        b = _simple_battle()
        u = b.alive(0)[0]
        m = legal_mask(b, u)
        assert m.shape == (ACTION_DIM,)
        assert m.dtype == np.float32

    def test_always_legal_actions(self):
        b = _simple_battle()
        u = b.alive(0)[0]
        m = legal_mask(b, u)
        assert m[WAIT_IDX] == 1.0
        assert m[DEFEND_IDX] == 1.0

    def test_retreat_no_hero(self):
        b = _simple_battle()
        u = b.alive(0)[0]
        m = legal_mask(b, u)
        assert m[RETREAT_IDX] == 0.0

    def test_retreat_with_hero(self):
        b = _hero_battle()
        u = b.alive(0)[0]
        m = legal_mask(b, u)
        assert m[RETREAT_IDX] == 1.0

    def test_move_legal_are_reachable(self):
        b = _simple_battle()
        u = b.alive(0)[0]
        m = legal_mask(b, u)
        occ = b._move_occupied(u)
        reachable = b.grid.reachable(u.pos, u.speed, occ)
        for cell in reachable:
            if cell == u.pos:
                continue
            idx = MOVE_START + cell_to_index(*cell)
            assert m[idx] == 1.0, f"Reachable cell {cell} not marked legal"

    def test_move_not_at_current_pos(self):
        b = _simple_battle()
        u = b.alive(0)[0]
        m = legal_mask(b, u)
        current_idx = MOVE_START + cell_to_index(*u.pos)
        assert m[current_idx] == 0.0

    def test_ranged_legal_for_archer(self):
        b = _archer_battle()
        archer = b.alive(0)[0]
        enemy = b.alive(1)[0]
        m = legal_mask(b, archer)
        enemies = b.enemies_of(archer)
        enemy_idx = enemies.index(enemy)
        # Position 0 = ranged
        attack_idx = ATTACK_START + enemy_idx * MAX_ATTACK_POSITIONS + 0
        assert m[attack_idx] == 1.0

    def test_no_ranged_for_non_archer(self):
        b = _simple_battle()
        u0 = b.alive(0)[0]
        u1 = b.alive(1)[0]
        m = legal_mask(b, u0)
        enemies = b.enemies_of(u0)
        enemy_idx = enemies.index(u1)
        # Position 0 = ranged, should NOT be legal for non-archer
        attack_idx = ATTACK_START + enemy_idx * MAX_ATTACK_POSITIONS + 0
        if b.grid.distance(u0.pos, u1.pos) > 1:
            assert m[attack_idx] == 0.0

    def test_melee_legal_when_adjacent(self):
        """If unit is adjacent to enemy, melee from current pos is legal."""
        b = _simple_battle()
        u0 = b.alive(0)[0]
        u1 = b.alive(1)[0]
        # Place adjacent
        u0.pos = (9, 4)
        m = legal_mask(b, u0)
        enemies = b.enemies_of(u0)
        enemy_idx = enemies.index(u1)
        # Check if any melee position is legal
        from ai.action_space import _attack_positions
        positions = _attack_positions(b.grid, u0, u1)
        found_legal = False
        for pos_idx in range(1, min(len(positions), MAX_ATTACK_POSITIONS)):
            if m[ATTACK_START + enemy_idx * MAX_ATTACK_POSITIONS + pos_idx] == 1.0:
                found_legal = True
                break
        assert found_legal, "Expected at least one legal melee position"

    def test_mask_values_binary(self):
        b = _archer_battle()
        u = b.alive(0)[0]
        m = legal_mask(b, u)
        assert set(m).issubset({0.0, 1.0})

    def test_at_least_one_legal(self):
        """Mask must always have at least Wait."""
        b = _simple_battle()
        u = b.alive(0)[0]
        m = legal_mask(b, u)
        assert m.sum() >= 1


# ── Spell legality ─────────────────────────────────────────────

class TestSpellLegality:

    def test_no_hero_no_cast(self):
        b = _simple_battle()
        u = b.alive(0)[0]
        m = legal_mask(b, u)
        for i in range(CAST_START, CAST_END + 1):
            assert m[i] == 0.0

    def test_cast_with_hero(self):
        b = _hero_battle(spells=["Magic Arrow"])
        u0 = b.alive(0)[0]
        u1 = b.alive(1)[0]
        m = legal_mask(b, u0)
        spell = SPELLS["Magic Arrow"]
        slot = _SPELL_INDEX["Magic Arrow"]
        base = CAST_START + slot * GRID_CELLS
        # Magic Arrow targets enemies — u1 must be marked
        tgt_idx = cell_to_index(*u1.pos)
        assert m[base + tgt_idx] == 1.0
        # Friendly not targeted
        own_idx = cell_to_index(*u0.pos)
        assert m[base + own_idx] == 0.0

    def test_buff_targets_friendly(self):
        b = _hero_battle(spells=["Bloodlust"])
        u0 = b.alive(0)[0]
        m = legal_mask(b, u0)
        slot = _SPELL_INDEX["Bloodlust"]
        base = CAST_START + slot * GRID_CELLS
        # Bloodlust targets friendly
        tgt_idx = cell_to_index(*u0.pos)
        assert m[base + tgt_idx] == 1.0

    def test_already_has_buff_not_legal(self):
        """Unit that already has the buff should not be a legal target."""
        b = _hero_battle(spells=["Bloodlust"])
        u0 = b.alive(0)[0]
        # Apply Bloodlust effect manually
        from engine.spells import make_effect
        spell = SPELLS["Bloodlust"]
        u0.add_effect(make_effect(spell, 5))
        m = legal_mask(b, u0)
        slot = _SPELL_INDEX["Bloodlust"]
        base = CAST_START + slot * GRID_CELLS
        tgt_idx = cell_to_index(*u0.pos)
        assert m[base + tgt_idx] == 0.0

    def test_mass_spell_all_hexes_legal(self):
        b = _hero_battle(spells=["Mass Haste"])
        u0 = b.alive(0)[0]
        m = legal_mask(b, u0)
        slot = _SPELL_INDEX["Mass Haste"]
        base = CAST_START + slot * GRID_CELLS
        for i in range(GRID_CELLS):
            assert m[base + i] == 1.0

    def test_armywide_aoe_all_hexes_legal(self):
        b = _hero_battle(spells=["Death Ripple"])
        u0 = b.alive(0)[0]
        m = legal_mask(b, u0)
        slot = _SPELL_INDEX["Death Ripple"]
        base = CAST_START + slot * GRID_CELLS
        for i in range(GRID_CELLS):
            assert m[base + i] == 1.0

    def test_ring_aoe_all_hexes_legal(self):
        b = _hero_battle(spells=["Fireball"])
        u0 = b.alive(0)[0]
        m = legal_mask(b, u0)
        slot = _SPELL_INDEX["Fireball"]
        base = CAST_START + slot * GRID_CELLS
        for i in range(GRID_CELLS):
            assert m[base + i] == 1.0

    def test_chain_lightning_targets_enemies(self):
        b = _hero_battle(spells=["Chain Lightning"])
        u0 = b.alive(0)[0]
        u1 = b.alive(1)[0]
        m = legal_mask(b, u0)
        slot = _SPELL_INDEX["Chain Lightning"]
        base = CAST_START + slot * GRID_CELLS
        # Enemy should be legal
        assert m[base + cell_to_index(*u1.pos)] == 1.0
        # Friendly should not
        assert m[base + cell_to_index(*u0.pos)] == 0.0

    def test_immune_unit_not_targetable(self):
        """Anti-Magic makes unit immune to targeted spells."""
        b = _hero_battle(spells=["Lightning Bolt"])
        u0 = b.alive(0)[0]
        u1 = b.alive(1)[0]
        # Give u1 Anti-Magic
        from engine.spells import make_effect
        am_spell = SPELLS["Anti-Magic"]
        u1.add_effect(make_effect(am_spell, 5))
        m = legal_mask(b, u0)
        slot = _SPELL_INDEX["Lightning Bolt"]
        base = CAST_START + slot * GRID_CELLS
        assert m[base + cell_to_index(*u1.pos)] == 0.0

    def test_no_cast_after_casting(self):
        """Hero can only cast once per round."""
        b = _hero_battle(spells=["Magic Arrow"])
        u0 = b.alive(0)[0]
        hero = b.heroes[0]
        hero._cast_this_round = True
        m = legal_mask(b, u0)
        for i in range(CAST_START, CAST_END + 1):
            assert m[i] == 0.0

    def test_no_cast_if_insufficient_mana(self):
        b = _hero_battle(spells=["Armageddon"])
        u0 = b.alive(0)[0]
        hero = b.heroes[0]
        hero.spell_points = 5  # Armageddon costs 20
        m = legal_mask(b, u0)
        slot = _SPELL_INDEX["Armageddon"]
        base = CAST_START + slot * GRID_CELLS
        for i in range(GRID_CELLS):
            assert m[base + i] == 0.0

    def test_spell_not_in_spellbook(self):
        """Hero without the spell can't cast it."""
        b = _hero_battle(spells=["Magic Arrow"])
        u0 = b.alive(0)[0]
        m = legal_mask(b, u0)
        slot = _SPELL_INDEX["Armageddon"]
        base = CAST_START + slot * GRID_CELLS
        for i in range(GRID_CELLS):
            assert m[base + i] == 0.0


# ── enumerate_legal ───────────────────────────────────────────

class TestEnumerateLegal:

    def test_sorted(self):
        b = _simple_battle()
        u = b.alive(0)[0]
        legal = enumerate_legal(b, u)
        assert legal == sorted(legal)

    def test_contains_wait_defend(self):
        b = _simple_battle()
        u = b.alive(0)[0]
        legal = enumerate_legal(b, u)
        assert WAIT_IDX in legal
        assert DEFEND_IDX in legal

    def test_matches_mask(self):
        b = _archer_battle()
        u = b.alive(0)[0]
        m = legal_mask(b, u)
        legal = enumerate_legal(b, u)
        mask_legal = sorted(int(i) for i in np.nonzero(m)[0])
        assert legal == mask_legal


# ── Wide units ─────────────────────────────────────────────────

class TestWideUnits:

    def _wide_battle(self):
        """Cavalry (wide, team 0) vs Pikeman (team 1)."""
        cav = _unit("Cavalry", 0, 1, 4)
        pike = _unit("Pikeman", 1, 10, 4)
        return _battle([cav, pike])

    def test_wide_move_legal(self):
        b = self._wide_battle()
        cav = b.alive(0)[0]
        m = legal_mask(b, cav)
        occ = b._move_occupied(cav)
        td = -1 if cav.team == 0 else 1
        reachable = b.grid.reachable(cav.pos, cav.speed, occ,
                                     cav.is_flying, td)
        for cell in reachable:
            if cell == cav.pos:
                continue
            idx = MOVE_START + cell_to_index(*cell)
            assert m[idx] == 1.0

    def test_wide_melee_attack(self):
        """Wide unit can melee from adjacent position."""
        b = self._wide_battle()
        cav = b.alive(0)[0]
        pike = b.alive(1)[0]
        cav.pos = (9, 4)  # adjacent to pike at (10,4)
        m = legal_mask(b, cav)
        enemies = b.enemies_of(cav)
        enemy_idx = enemies.index(pike)
        # Check that some melee attack position is legal
        has_melee = False
        for pos_idx in range(1, MAX_ATTACK_POSITIONS):
            idx = ATTACK_START + enemy_idx * MAX_ATTACK_POSITIONS + pos_idx
            if idx < ACTION_DIM and m[idx] == 1.0:
                has_melee = True
                break
        assert has_melee


# ── action_type_label ─────────────────────────────────────────

class TestActionTypeLabel:

    def test_wait(self):
        assert action_type_label(0) == "Wait"

    def test_defend(self):
        assert action_type_label(1) == "Defend"

    def test_move(self):
        label = action_type_label(MOVE_START + cell_to_index(5, 3))
        assert "Move(5,3)" in label

    def test_retreat(self):
        assert action_type_label(RETREAT_IDX) == "Retreat"


# ── Edge cases ─────────────────────────────────────────────────

class TestEdgeCases:

    def test_out_of_range_index(self):
        """Index beyond ACTION_DIM returns SkipAction."""
        b = _simple_battle()
        u = b.alive(0)[0]
        decoded = index_to_action(ACTION_DIM + 100, b, u)
        assert isinstance(decoded, SkipAction)

    def test_move_unreachable_cell(self):
        """Move to an unreachable cell returns SkipAction."""
        b = _simple_battle()
        u = b.alive(0)[0]
        # Top-left corner (0,0) for a unit at (0,4) — actually reachable
        # Let's use a cell occupied by the enemy
        u1 = b.alive(1)[0]
        unreachable_idx = MOVE_START + cell_to_index(*u1.pos)
        decoded = index_to_action(unreachable_idx, b, u)
        assert isinstance(decoded, SkipAction)

    def test_attack_no_target_at_hex(self):
        """Attack a hex with no unit → SkipAction."""
        b = _simple_battle()
        u = b.alive(0)[0]
        # Target an empty hex
        empty_hex = (5, 0)
        pos_idx = cell_to_index(*u.pos)
        tgt_idx = cell_to_index(*empty_hex)
        idx = ATTACK_START + pos_idx * GRID_CELLS + tgt_idx
        decoded = index_to_action(idx, b, u)
        assert isinstance(decoded, SkipAction)

    def test_cast_without_hero(self):
        """Cast action when no hero exists → SkipAction."""
        b = _simple_battle()
        u = b.alive(0)[0]
        slot = _SPELL_INDEX["Magic Arrow"]
        hex_idx = cell_to_index(10, 4)
        idx = CAST_START + slot * GRID_CELLS + hex_idx
        decoded = index_to_action(idx, b, u)
        assert isinstance(decoded, SkipAction)

    def test_flying_unit_has_more_moves(self):
        """Flying units can reach more cells than ground units."""
        fly = _unit("Phoenix", 0, 0, 4)
        walk = _unit("Pikeman", 0, 2, 4)
        enemy = _unit("Pikeman", 1, 10, 4)
        b = _battle([fly, walk, enemy])
        m_fly = legal_mask(b, fly)
        m_walk = legal_mask(b, walk)
        fly_moves = sum(1 for i in range(MOVE_START, MOVE_END + 1) if m_fly[i])
        walk_moves = sum(1 for i in range(MOVE_START, MOVE_END + 1) if m_walk[i])
        assert fly_moves > walk_moves

    def test_teleport_excluded_from_index(self):
        """Teleport is not in _SPELL_INDEX."""
        assert "Teleport" not in _SPELL_INDEX
        # All spell names in _SPELL_ORDER should be in SPELLS
        for name in _SPELL_ORDER:
            assert name in SPELLS

    def test_mask_no_nan_or_inf(self):
        b = _archer_battle()
        u = b.alive(0)[0]
        m = legal_mask(b, u)
        assert np.all(np.isfinite(m))

"""Tests for R2/T8 observation encoder — ai/observation.py.

Covers shape, player-relative encoding, unit placement, attribute normalisation,
wide units, status effects, siege structures, global vector, unit-type encoding,
and edge cases.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from ai.observation import (
    encode_observation, GLOBAL_DIM, GRID_COLS, GRID_ROWS, NUM_GRID_CHANNELS,
    _CH_MY_TYPE, _CH_ENEMY_TYPE, _MAX_TYPE_INDEX,
)
from config.units import UNIT_TYPE_INDEX, NUM_UNIT_TYPES
from engine.battle_state import BattleState
from engine.castle import Castle, MOAT_CELLS, WALL_POSITIONS
from engine.hero import Hero
from engine.hex_grid import HexGrid
from engine.spells import make_effect, SPELLS
from engine.unit import Unit


# ── Helpers ────────────────────────────────────────────────────


def _battle(units, **kw):
    return BattleState(HexGrid(), units, **kw)


def _enc(battle, unit):
    return encode_observation(battle, unit)


# ── Shape & dtype ──────────────────────────────────────────────


def test_output_shapes():
    u0 = Unit.from_type("Swordsman", 0, 2, 4)
    u1 = Unit.from_type("Swordsman", 1, 8, 4)
    b = _battle([u0, u1])
    grid, gvec = _enc(b, u0)
    assert grid.shape == (NUM_GRID_CHANNELS, GRID_ROWS, GRID_COLS)
    assert grid.dtype == np.float32
    assert gvec.shape == (GLOBAL_DIM,)
    assert gvec.dtype == np.float32


def test_num_grid_channels():
    """T8: 35 channels (33 original + 2 type-index)."""
    assert NUM_GRID_CHANNELS == 35


# ── Player-relative encoding ──────────────────────────────────


def test_player_relative_my_vs_enemy():
    """Team 0 acting → ch0–9 are team 0, ch10–19 are team 1.  Team 1 flips."""
    u0 = Unit.from_type("Swordsman", 0, 2, 4)
    u1 = Unit.from_type("Swordsman", 1, 8, 4)
    b = _battle([u0, u1])

    # From team 0's perspective
    g0, _ = _enc(b, u0)
    assert g0[0, 4, 2] == 1.0   # my existence at (2,4)
    assert g0[10, 4, 8] == 1.0  # enemy existence at (8,4)
    assert g0[0, 4, 8] == 0.0   # team 1 NOT in my channels
    assert g0[10, 4, 2] == 0.0  # team 0 NOT in enemy channels

    # From team 1's perspective → flipped
    g1, _ = _enc(b, u1)
    assert g1[0, 4, 8] == 1.0   # my (team 1) at (8,4)
    assert g1[10, 4, 2] == 1.0  # enemy (team 0) at (2,4)


def test_player_relative_attributes():
    """Attribute channels follow the correct team mapping."""
    u0 = Unit.from_type("Mage", 0, 2, 4, count=5)       # archer
    u1 = Unit.from_type("Pikeman", 1, 8, 4, count=10)   # not archer
    b = _battle([u0, u1])

    g, _ = _enc(b, u0)
    # Mage (team 0) is "my" — ch6 (archer) should be 1 at (2,4)
    assert g[6, 4, 2] == 1.0
    # Pikeman (team 1) is "enemy" — ch16 (enemy archer) should be 0
    assert g[16, 4, 8] == 0.0


# ── Unit placement ─────────────────────────────────────────────


def test_unit_at_correct_position():
    u = Unit.from_type("Swordsman", 0, 3, 5)
    b = _battle([u])
    g, _ = _enc(b, u)
    assert g[0, 5, 3] == 1.0   # existence
    assert g[0, 0, 3] == 0.0   # not at row 0


def test_dead_units_not_encoded():
    u0 = Unit.from_type("Swordsman", 0, 2, 4)
    u1 = Unit.from_type("Swordsman", 1, 8, 4)
    b = _battle([u0, u1])
    u1._total_hp = 0
    u1.count = 0
    u1.is_alive = False

    g, _ = _enc(b, u0)
    # Only u0 should appear
    assert g[0, 4, 2] == 1.0   # u0 alive
    assert g[10, 4, 8] == 0.0  # u1 dead → not in enemy channels


# ── Attribute normalisation ───────────────────────────────────


def test_hp_and_count_ratios():
    u = Unit.from_type("Swordsman", 0, 5, 4, count=10)
    b = _battle([u])
    # Take 50% damage
    u._total_hp = u._total_hp // 2
    u.count = (u._total_hp + u.max_hp - 1) // u.max_hp

    g, _ = _enc(b, u)
    hp_ratio = g[1, 4, 5]
    count_ratio = g[2, 4, 5]
    assert 0.0 < hp_ratio < 1.0
    assert 0.0 < count_ratio <= 1.0


def test_full_health_unit_has_ratio_one():
    u = Unit.from_type("Swordsman", 0, 5, 4, count=10)
    b = _battle([u])
    g, _ = _enc(b, u)
    assert g[1, 4, 5] == 1.0   # HP ratio
    assert g[2, 4, 5] == 1.0   # count ratio


def test_speed_normalised():
    u = Unit.from_type("Champion", 0, 5, 4)  # speed 7
    b = _battle([u])
    g, _ = _enc(b, u)
    assert g[5, 4, 5] == pytest.approx(7 / 10)


# ── Wide units ─────────────────────────────────────────────────


def test_wide_unit_head_and_tail():
    """Wide unit: head gets full attributes, tail gets existence + wide-tail only."""
    u = Unit.from_type("Cavalry", 0, 5, 4)  # is_wide=True
    b = _battle([u])

    g, _ = _enc(b, u)
    # Head at (5,4)
    assert g[0, 4, 5] == 1.0   # existence
    assert g[8, 4, 5] == 0.0   # wide-tail NOT on head
    assert g[3, 4, 5] > 0.0    # attack value on head

    # Tail at (4,4) for team 0 (tail_dir = -1)
    assert g[0, 4, 4] == 1.0   # existence on tail
    assert g[8, 4, 4] == 1.0   # wide-tail marker on tail
    assert g[3, 4, 4] == 0.0   # attack NOT on tail


def test_wide_team1_tail_direction():
    """Team 1 wide unit: tail is at col+1, encoded in 'my' channels."""
    u0 = Unit.from_type("Swordsman", 0, 2, 4)
    u1 = Unit.from_type("Cavalry", 1, 5, 4)
    b = _battle([u0, u1])

    # From team 1's perspective: team 1 is "my" (base=0)
    g, _ = _enc(b, u1)
    # Head at (5,4) → my channels
    assert g[0, 4, 5] == 1.0
    # Tail at (6,4) → team 1 tail_dir = +1
    assert g[0, 4, 6] == 1.0
    assert g[8, 4, 6] == 1.0   # wide-tail marker

    # From team 0's perspective: team 1 is "enemy" (base=10)
    g0, _ = _enc(b, u0)
    assert g0[10, 4, 5] == 1.0   # enemy head
    assert g0[10, 4, 6] == 1.0   # enemy tail
    assert g0[18, 4, 6] == 1.0   # enemy wide-tail marker


# ── Status effects ─────────────────────────────────────────────


def test_haste_effect():
    u = Unit.from_type("Swordsman", 0, 5, 4)
    b = _battle([u])
    u.add_effect(make_effect(SPELLS["Haste"], power=3))
    g, _ = _enc(b, u)
    assert g[20, 4, 5] == 1.0  # Haste channel


def test_slow_effect():
    u = Unit.from_type("Swordsman", 0, 5, 4)
    b = _battle([u])
    u.add_effect(make_effect(SPELLS["Slow"], power=3))
    g, _ = _enc(b, u)
    assert g[21, 4, 5] == 1.0  # Slow channel


def test_bless_effect():
    u = Unit.from_type("Swordsman", 0, 5, 4)
    b = _battle([u])
    u.add_effect(make_effect(SPELLS["Bless"], power=3))
    g, _ = _enc(b, u)
    assert g[22, 4, 5] == 1.0  # Bless channel


def test_curse_effect():
    u = Unit.from_type("Swordsman", 0, 5, 4)
    b = _battle([u])
    u.add_effect(make_effect(SPELLS["Curse"], power=3))
    g, _ = _enc(b, u)
    assert g[23, 4, 5] == 1.0  # Curse channel


def test_blind_effect():
    u = Unit.from_type("Swordsman", 0, 5, 4)
    b = _battle([u])
    u.add_effect(make_effect(SPELLS["Blind"], power=3))
    g, _ = _enc(b, u)
    assert g[24, 4, 5] == 1.0  # Blind/Paralyze channel


def test_bloodlust_effect():
    u = Unit.from_type("Swordsman", 0, 5, 4)
    b = _battle([u])
    u.add_effect(make_effect(SPELLS["Bloodlust"], power=3))
    g, _ = _enc(b, u)
    assert g[25, 4, 5] == 1.0  # attack buff channel


def test_stone_skin_effect():
    u = Unit.from_type("Swordsman", 0, 5, 4)
    b = _battle([u])
    u.add_effect(make_effect(SPELLS["Stone Skin"], power=3))
    g, _ = _enc(b, u)
    assert g[26, 4, 5] == 1.0  # defense buff channel


def test_shield_effect():
    u = Unit.from_type("Swordsman", 0, 5, 4)
    b = _battle([u])
    u.add_effect(make_effect(SPELLS["Shield"], power=3))
    g, _ = _enc(b, u)
    assert g[27, 4, 5] == 1.0  # Shield channel


def test_anti_magic_effect():
    u = Unit.from_type("Swordsman", 0, 5, 4)
    b = _battle([u])
    u.add_effect(make_effect(SPELLS["Anti-Magic"], power=3))
    g, _ = _enc(b, u)
    assert g[28, 4, 5] == 1.0  # Anti-Magic channel


def test_disrupting_ray_stacks():
    u = Unit.from_type("Swordsman", 0, 5, 4)
    b = _battle([u])
    # Stack 3 Disrupting Ray effects
    for _ in range(3):
        u.add_effect(make_effect(SPELLS["Disrupting Ray"], power=3))
    g, _ = _enc(b, u)
    assert g[29, 4, 5] == pytest.approx(3 / 5)


def test_effects_on_wide_unit_both_cells():
    """Effects mark ALL occupied cells (head + tail)."""
    u = Unit.from_type("Cavalry", 0, 5, 4)  # wide
    b = _battle([u])
    u.add_effect(make_effect(SPELLS["Haste"], power=3))
    g, _ = _enc(b, u)
    assert g[20, 4, 5] == 1.0  # head
    assert g[20, 4, 4] == 1.0  # tail (team 0 tail at col-1)


def test_multiple_effects_stack_channels():
    """Unit with both Haste and Bloodlust sets both channels."""
    u = Unit.from_type("Swordsman", 0, 5, 4)
    b = _battle([u])
    u.add_effect(make_effect(SPELLS["Haste"], power=3))
    u.add_effect(make_effect(SPELLS["Bloodlust"], power=3))
    g, _ = _enc(b, u)
    assert g[20, 4, 5] == 1.0  # Haste
    assert g[25, 4, 5] == 1.0  # Bloodlust


# ── Siege channels ─────────────────────────────────────────────


def test_siege_wall_hp():
    u0 = Unit.from_type("Swordsman", 0, 2, 4)
    u1 = Unit.from_type("Swordsman", 1, 9, 4)
    castle = Castle()
    castle.walls[WALL_POSITIONS[0]] = 1  # damaged
    b = _battle([u0, u1], castle=castle)
    g, _ = _enc(b, u0)

    c, r = WALL_POSITIONS[0]
    assert g[30, r, c] == pytest.approx(0.5)  # damaged wall = 1/2

    # Intact wall
    c2, r2 = WALL_POSITIONS[1]
    assert g[30, r2, c2] == pytest.approx(1.0)  # HP=2 → 2/2=1.0


def test_siege_moat():
    u0 = Unit.from_type("Swordsman", 0, 2, 4)
    u1 = Unit.from_type("Swordsman", 1, 9, 4)
    b = _battle([u0, u1], castle=Castle())
    g, _ = _enc(b, u0)

    # Check a known moat cell
    moat_cell = next(iter(MOAT_CELLS))
    assert g[31, moat_cell[1], moat_cell[0]] == 1.0


def test_siege_towers():
    u0 = Unit.from_type("Swordsman", 0, 2, 4)
    u1 = Unit.from_type("Swordsman", 1, 9, 4)
    castle = Castle()
    b = _battle([u0, u1], castle=castle)
    g, _ = _enc(b, u0)

    # Left tower at (8,1), right at (8,7) — both active
    assert g[32, 1, 8] == 1.0
    assert g[32, 7, 8] == 1.0

    # Destroy left tower → channel clears
    castle.towers[0].destroyed = True
    g2, _ = _enc(b, u0)
    assert g2[32, 1, 8] == 0.0
    assert g2[32, 7, 8] == 1.0


def test_no_siege_all_zeros():
    u = Unit.from_type("Swordsman", 0, 5, 4)
    b = _battle([u])
    g, _ = _enc(b, u)
    assert g[30:33].sum() == 0.0


# ── Global vector ──────────────────────────────────────────────


def test_global_round():
    u = Unit.from_type("Swordsman", 0, 5, 4)
    b = _battle([u])
    b.round_num = 50
    _, g = _enc(b, u)
    assert g[0] == pytest.approx(50 / 200)


def test_global_attacker_team():
    u = Unit.from_type("Swordsman", 0, 5, 4)
    b = _battle([u], attacker_team=1)
    _, g = _enc(b, u)
    assert g[1] == 1.0


def test_global_unit_counts():
    u0 = Unit.from_type("Swordsman", 0, 2, 4)
    u1 = Unit.from_type("Pikeman", 0, 3, 4)
    u2 = Unit.from_type("Swordsman", 1, 8, 4)
    u3 = Unit.from_type("Swordsman", 1, 9, 4)
    b = _battle([u0, u1, u2, u3])
    _, g = _enc(b, u0)
    assert g[2] == pytest.approx(2 / 7)   # my team (0): 2 alive
    assert g[3] == pytest.approx(2 / 7)   # enemy team (1): 2 alive


def test_global_hp_ratio():
    u0 = Unit.from_type("Swordsman", 0, 2, 4, count=10)
    u1 = Unit.from_type("Swordsman", 1, 8, 4, count=10)
    b = _battle([u0, u1])
    _, g = _enc(b, u0)
    assert g[4] == pytest.approx(1.0)  # my team full HP
    assert g[5] == pytest.approx(1.0)  # enemy full HP

    # Damage my unit
    u0.take_damage(u0.max_hp * 5)
    _, g2 = _enc(b, u0)
    assert g2[4] < 1.0
    assert g2[5] == pytest.approx(1.0)  # enemy unchanged


def test_global_hero_fields():
    hero0 = Hero(power=8, max_spell_points=30, attack=5, defense=3, name="Wizard")
    u0 = Unit.from_type("Swordsman", 0, 2, 4)
    u1 = Unit.from_type("Swordsman", 1, 8, 4)
    b = _battle([u0, u1], heroes={0: hero0, 1: None})
    _, g = _enc(b, u0)

    assert g[6] == pytest.approx(30 / 100)  # my SP
    assert g[7] == 0.0                       # no enemy hero
    assert g[8] == pytest.approx(8 / 15)    # my power
    assert g[9] == 0.0                       # no enemy hero
    assert g[10] == pytest.approx(5 / 15)   # my attack
    assert g[11] == 0.0
    assert g[12] == pytest.approx(3 / 15)   # my defense
    assert g[13] == 0.0


def test_global_no_hero():
    u = Unit.from_type("Swordsman", 0, 5, 4)
    b = _battle([u])
    _, g = _enc(b, u)
    for i in (6, 7, 8, 9, 10, 11, 12, 13):
        assert g[i] == 0.0


def test_global_siege_flags():
    u0 = Unit.from_type("Swordsman", 0, 2, 4)
    u1 = Unit.from_type("Swordsman", 1, 9, 4)
    b = _battle([u0, u1], castle=Castle())
    _, g = _enc(b, u0)

    assert g[14] == 1.0          # is siege
    assert g[15] == pytest.approx(3 / 3)  # all towers active
    assert g[16] == pytest.approx(4 / 4)  # all walls intact


def test_global_morale_luck():
    u = Unit.from_type("Swordsman", 0, 5, 4)
    b = _battle([u], morale={0: 2, 1: -1}, luck={0: -3, 1: 1})
    _, g = _enc(b, u)
    assert g[17] == pytest.approx(2 / 3)   # my morale
    assert g[18] == pytest.approx(-3 / 3)  # my luck


def test_global_current_unit_index():
    u0 = Unit.from_type("Champion", 0, 2, 4)  # speed 7
    u1 = Unit.from_type("Pikeman", 0, 3, 4)   # speed 4
    u2 = Unit.from_type("Swordsman", 1, 8, 4) # speed 5
    b = _battle([u0, u1, u2])
    # Turn order: Champion(7), Swordsman(5), Pikeman(4)
    _, g = _enc(b, u0)
    assert g[19] == pytest.approx(0 / 14)  # Champion is first

    _, g = _enc(b, u1)
    # Pikeman is last (index 2)
    assert g[19] == pytest.approx(2 / 14)


# ── Normalisation bounds ──────────────────────────────────────


def test_all_grid_values_in_valid_range():
    """All grid values must be in [0, 1] (unit/effect/type channels) or [0, 1] (siege)."""
    u0 = Unit.from_type("Champion", 0, 2, 4, count=20)
    u1 = Unit.from_type("Pikeman", 1, 8, 4, count=15)
    hero0 = Hero(power=10, max_spell_points=50, attack=12, defense=8)
    hero1 = Hero(power=6, max_spell_points=20, attack=3, defense=2)
    b = _battle([u0, u1], heroes={0: hero0, 1: hero1}, castle=Castle(),
                morale={0: 2, 1: -1}, luck={0: 1, 1: 0})

    u0.add_effect(make_effect(SPELLS["Haste"], power=3))
    u0.add_effect(make_effect(SPELLS["Bloodlust"], power=3))
    u1.add_effect(make_effect(SPELLS["Disrupting Ray"], power=3))

    g, gvec = _enc(b, u0)

    # Grid: all channels should be in [0, 1]
    assert g.min() >= 0.0
    assert g.max() <= 1.0

    # Global vector: most dims in [0, 1], morale/luck in [-1, 1]
    assert gvec.min() >= -1.0
    assert gvec.max() <= 1.0


# ── Acted flag ─────────────────────────────────────────────────


def test_acted_flag():
    u = Unit.from_type("Swordsman", 0, 5, 4)
    b = _battle([u])
    g, _ = _enc(b, u)
    assert g[9, 4, 5] == 0.0   # not acted yet

    u._acted = True
    g2, _ = _enc(b, u)
    assert g2[9, 4, 5] == 1.0


# ── T8: Unit-type encoding ────────────────────────────────────


def test_unit_type_encoding_on_head():
    """Different unit types should produce different type-index values."""
    u0 = Unit.from_type("Swordsman", 0, 2, 4)
    u1 = Unit.from_type("Black Dragon", 1, 8, 4)
    b = _battle([u0, u1])

    g, _ = _enc(b, u0)

    # My type channel at (2,4)
    swordsman_norm = UNIT_TYPE_INDEX["Swordsman"] / _MAX_TYPE_INDEX
    assert g[_CH_MY_TYPE, 4, 2] == pytest.approx(swordsman_norm)

    # Enemy type channel at (8,4)
    dragon_norm = UNIT_TYPE_INDEX["Black Dragon"] / _MAX_TYPE_INDEX
    assert g[_CH_ENEMY_TYPE, 4, 8] == pytest.approx(dragon_norm)

    # Different types → different values
    assert g[_CH_MY_TYPE, 4, 2] != g[_CH_ENEMY_TYPE, 4, 8]


def test_empty_cell_type_zero():
    """Cells with no unit should have type index 0.0 (= padding)."""
    u0 = Unit.from_type("Swordsman", 0, 2, 4)
    u1 = Unit.from_type("Swordsman", 1, 8, 4)
    b = _battle([u0, u1])

    g, _ = _enc(b, u0)

    # (5,5) is empty
    assert g[_CH_MY_TYPE, 5, 5] == 0.0
    assert g[_CH_ENEMY_TYPE, 5, 5] == 0.0


def test_wide_unit_type_on_both_cells():
    """Wide unit: type index should appear on both head and tail cells."""
    u0 = Unit.from_type("Swordsman", 0, 2, 4)
    u1 = Unit.from_type("Cavalry", 1, 5, 4)  # wide, team 1
    b = _battle([u0, u1])

    # From team 0's perspective: Cavalry is enemy
    g, _ = _enc(b, u0)

    cavalry_norm = UNIT_TYPE_INDEX["Cavalry"] / _MAX_TYPE_INDEX

    # Head at (5,4)
    assert g[_CH_ENEMY_TYPE, 4, 5] == pytest.approx(cavalry_norm)
    # Tail at (6,4) — team 1 tail_dir = +1
    assert g[_CH_ENEMY_TYPE, 4, 6] == pytest.approx(cavalry_norm)


def test_player_relative_type_channels():
    """Type channels flip with player-relative perspective."""
    u0 = Unit.from_type("Swordsman", 0, 2, 4)
    u1 = Unit.from_type("Black Dragon", 1, 8, 4)
    b = _battle([u0, u1])

    # From team 0's view
    g0, _ = _enc(b, u0)
    # Swordsman → my type, Dragon → enemy type
    assert g0[_CH_MY_TYPE, 4, 2] > 0.0
    assert g0[_CH_ENEMY_TYPE, 4, 8] > 0.0
    assert g0[_CH_MY_TYPE, 4, 8] == 0.0   # Dragon is NOT my type
    assert g0[_CH_ENEMY_TYPE, 4, 2] == 0.0  # Swordsman is NOT enemy type

    # From team 1's view → flipped
    g1, _ = _enc(b, u1)
    assert g1[_CH_MY_TYPE, 4, 8] > 0.0     # Dragon is now my type
    assert g1[_CH_ENEMY_TYPE, 4, 2] > 0.0   # Swordsman is now enemy type


def test_same_unit_type_same_value():
    """Two units of the same type should have the same normalised value."""
    u0 = Unit.from_type("Swordsman", 0, 2, 4)
    u1 = Unit.from_type("Swordsman", 1, 8, 4)
    u2 = Unit.from_type("Pikeman", 0, 3, 4)
    b = _battle([u0, u1, u2])

    g, _ = _enc(b, u0)

    # Both Swordsmen should have same type value (different sides)
    my_type = g[_CH_MY_TYPE, 4, 2]
    enemy_type = g[_CH_ENEMY_TYPE, 4, 8]
    assert my_type == pytest.approx(enemy_type)

    # But Pikeman should differ
    pikeman_type = g[_CH_MY_TYPE, 4, 3]
    assert pikeman_type != pytest.approx(my_type)


# ── Comprehensive scenario ────────────────────────────────────


def test_complex_battle_observation():
    """Multi-unit battle with heroes, effects, and siege."""
    u0 = Unit.from_type("Champion", 0, 2, 4, count=5)   # wide, speed 7
    u1 = Unit.from_type("Mage", 0, 1, 2, count=3)       # archer
    u2 = Unit.from_type("Pikeman", 1, 9, 4, count=10)
    hero0 = Hero(power=5, max_spell_points=20, attack=3, defense=2)
    castle = Castle()
    b = _battle([u0, u1, u2], heroes={0: hero0, 1: None}, castle=castle,
                morale={0: 1, 1: 0}, luck={0: 0, 1: -1})

    # Apply effects
    u0.add_effect(make_effect(SPELLS["Haste"], power=5))
    u2.add_effect(make_effect(SPELLS["Slow"], power=3))

    # Damage pikeman
    u2.take_damage(u2.max_hp * 3)

    g, gvec = _enc(b, u0)

    # Champion (team 0) is "my" — base=0
    assert g[0, 4, 2] == 1.0   # head existence
    assert g[0, 4, 1] == 1.0   # tail existence (team 0, tail at col-1)
    assert g[6, 4, 2] == 0.0   # Champion is not archer

    # Mage is "my" archer
    assert g[6, 2, 1] == 1.0   # archer channel

    # Pikeman is "enemy" — base=10
    assert g[10, 4, 9] == 1.0  # existence
    assert g[14, 4, 9] > 0.0   # defense > 0

    # Effects
    assert g[20, 4, 2] == 1.0  # Haste on Champion head
    assert g[20, 4, 1] == 1.0  # Haste on Champion tail
    assert g[21, 4, 9] == 1.0  # Slow on Pikeman

    # Siege
    assert g[30].sum() > 0     # walls present
    assert g[31].sum() > 0     # moat present

    # Global
    assert gvec[2] == pytest.approx(2 / 7)   # 2 alive my units
    assert gvec[3] == pytest.approx(1 / 7)   # 1 alive enemy
    assert gvec[6] > 0          # my hero has SP
    assert gvec[7] == 0.0       # no enemy hero
    assert gvec[14] == 1.0      # siege
    assert gvec[17] == pytest.approx(1 / 3)  # morale +1

    # Type channels (T8)
    champion_norm = UNIT_TYPE_INDEX["Champion"] / _MAX_TYPE_INDEX
    mage_norm = UNIT_TYPE_INDEX["Mage"] / _MAX_TYPE_INDEX
    pikeman_norm = UNIT_TYPE_INDEX["Pikeman"] / _MAX_TYPE_INDEX

    # Champion head + tail
    assert g[_CH_MY_TYPE, 4, 2] == pytest.approx(champion_norm)
    assert g[_CH_MY_TYPE, 4, 1] == pytest.approx(champion_norm)  # tail too
    # Mage
    assert g[_CH_MY_TYPE, 2, 1] == pytest.approx(mage_norm)
    # Pikeman (enemy)
    assert g[_CH_ENEMY_TYPE, 4, 9] == pytest.approx(pikeman_norm)
    # Empty cell
    assert g[_CH_MY_TYPE, 0, 0] == 0.0

"""R3 action space — flat discrete encoding + legality mask.

Encodes every possible battle action into a single integer index and provides
a binary mask indicating which actions are legal in the current state.

Action layout (3 825 total):
  Index 0              → Wait
  Index 1              → Defend
  Indices 2–100        → Move(hex[0..98])
  Indices 101–156      → Attack(enemy_index[0..6] × position[0..7])
  Indices 157–3823     → Cast(spell[0..36], hex[0..98])
  Index 3824           → Retreat

Hex indexing: row-major  = row × 11 + col  (0–98).

Attack uses a compact enemy_index × position encoding instead of the
old pos × target (9 801 dim) scheme.  This reduces ACTION_DIM by 72%
and makes the policy head much smaller.

Teleport is excluded (needs two hexes; niche spell).
"""

from typing import List, Optional, Set, Tuple

import numpy as np

from engine.actions import (Action, MoveAction, AttackAction, SkipAction,
                            CastAction, RetreatAction)
from engine.battle_state import BattleState
from engine.unit import Unit
from engine.spells import (Spell, SPELLS,
                            DAMAGE, AOE, BUFF, DEBUFF, CONTROL,
                            DISPEL, CURE, UTILITY)

# ── Grid constants ──────────────────────────────────────────────

GRID_ROWS = 9
GRID_COLS = 11
GRID_CELLS = GRID_ROWS * GRID_COLS  # 99


def cell_to_index(col: int, row: int) -> int:
    """(col, row) → flat index 0–98."""
    return row * GRID_COLS + col


def index_to_cell(idx: int) -> Tuple[int, int]:
    """Flat index 0–98 → (col, row)."""
    return idx % GRID_COLS, idx // GRID_COLS


# ── Spell ordering (alphabetical, excluding Teleport) ──────────

_SPELL_ORDER: List[str] = sorted(n for n in SPELLS if n != "Teleport")
_SPELL_INDEX: dict = {n: i for i, n in enumerate(_SPELL_ORDER)}  # 0–36
NUM_SPELLS = len(_SPELL_ORDER)  # 37

# ── Attack sub-space constants ─────────────────────────────────

MAX_ENEMIES = 7          # max alive enemy units in a standard army
MAX_ATTACK_POSITIONS = 8  # 6 hex neighbors + current pos + ranged marker

# ── Action-range boundaries ────────────────────────────────────

WAIT_IDX    = 0
DEFEND_IDX  = 1
MOVE_START  = 2
MOVE_END    = MOVE_START + GRID_CELLS - 1              # 100
ATTACK_START = MOVE_END + 1                             # 101
ATTACK_DIM   = MAX_ENEMIES * MAX_ATTACK_POSITIONS      # 56
ATTACK_END   = ATTACK_START + ATTACK_DIM - 1            # 156
CAST_START   = ATTACK_END + 1                           # 157
CAST_END     = CAST_START + NUM_SPELLS * GRID_CELLS - 1 # 3823
RETREAT_IDX  = CAST_END + 1                             # 3824
ACTION_DIM   = RETREAT_IDX + 1                          # 3825


# ── Wide-unit geometry helpers (mirrors ClassicAI) ─────────────

def _tail_dir(unit: Unit) -> Optional[int]:
    """Column offset of a wide unit's tail; None for single-hex units."""
    if not unit.is_wide:
        return None
    return -1 if unit.team == 0 else 1


def _attack_cells(grid, target: Unit) -> List[Tuple[int, int]]:
    """Cells from which a melee attacker can strike *target*."""
    cells = list(grid.neighbors(*target.pos))
    if target.is_wide:
        body = target.occupied_cells()
        for tc in grid.neighbors(*target.tail_cell):
            if tc not in cells and tc not in body:
                cells.append(tc)
    return cells


def _can_attack_from_pos(grid, unit: Unit, target: Unit,
                         pos: Tuple[int, int], moat=None) -> bool:
    """Validate that a melee attacker at *pos* can strike *target*.

    Handles wide-unit orientation and moat attack restriction.
    Mirrors ClassicAI._can_attack_from_pos.
    """
    if not unit.is_wide:
        return True  # single-hex adjacency already guaranteed

    td = _tail_dir(unit)
    tail = (pos[0] + td, pos[1]) if td is not None else pos

    # At least one of head / tail must be adjacent to target's body
    head_adj = _pos_dist(grid, pos, target) <= 1
    tail_adj = _pos_dist(grid, tail, target) <= 1
    if not head_adj and not tail_adj:
        return False

    # Moat restriction: non-flying units can't attack from moat unless
    # they are already standing there.
    if moat and pos in moat and not unit.is_flying:
        if pos != unit.pos and (not unit.is_wide or pos != unit.tail_cell):
            return False

    return True


def _pos_dist(grid, pos: Tuple[int, int], unit: Unit) -> int:
    """Min distance from a bare cell to a unit's body."""
    if not unit.is_wide:
        return grid.distance(pos, unit.pos)
    return min(grid.distance(pos, cb) for cb in unit.occupied_cells())


# ── Spell legality helpers ─────────────────────────────────────

def _is_mass_or_armywide(spell: Spell) -> bool:
    """True if the spell doesn't need an individual target hex."""
    return (spell.is_mass
            or spell.aoe_pattern in ("all_tagged", "all_units")
            or (spell.kind == UTILITY and spell.name == "Earthquake"))


def _is_ring_aoe(spell: Spell) -> bool:
    """True for AOE spells that take a center cell."""
    return spell.aoe_pattern in ("ring1", "ring2", "ring_outer")


# ── Attack position encoding ───────────────────────────────────

def _enemy_list(battle: BattleState, current_unit: Unit) -> List[Unit]:
    """Return the ordered list of alive enemy units (for enemy_index)."""
    return battle.enemies_of(current_unit)


def _attack_positions(grid, unit: Unit, target: Unit) -> List[Tuple[int, int]]:
    """Return the ordered list of positions from which *unit* can melee *target*.

    Position index 0 is always the attacker's current position (for ranged).
    Positions 1-7 are melee attack cells (neighbors of target, or neighbors
    of target's tail if wide).
    """
    cells = _attack_cells(grid, target)
    # Current position first (position index 0 = ranged)
    result = [unit.pos]
    for ac in cells:
        if ac != unit.pos:
            result.append(ac)
    return result[:MAX_ATTACK_POSITIONS]


# ── Core API ───────────────────────────────────────────────────

def action_to_index(action: Action, battle: BattleState,
                    current_unit: Unit) -> int:
    """Convert an Action object to its flat index.

    SkipAction maps to WAIT_IDX.
    AttackAction uses enemy_index × position encoding.
    """
    if isinstance(action, SkipAction):
        return WAIT_IDX
    if isinstance(action, RetreatAction):
        return RETREAT_IDX
    if isinstance(action, MoveAction):
        return MOVE_START + cell_to_index(*action.path[-1])
    if isinstance(action, AttackAction):
        enemies = _enemy_list(battle, current_unit)
        try:
            enemy_idx = enemies.index(action.target)
        except ValueError:
            raise ValueError(
                f"Target {action.target.name} not in enemy list")
        positions = _attack_positions(battle.grid, current_unit, action.target)
        if action.ranged:
            pos_idx = 0  # position 0 = ranged (current position)
        else:
            from_cell = action.from_pos if action.from_pos else current_unit.pos
            try:
                pos_idx = positions.index(from_cell)
            except ValueError:
                pos_idx = 0
        return ATTACK_START + enemy_idx * MAX_ATTACK_POSITIONS + pos_idx
    if isinstance(action, CastAction):
        slot = _SPELL_INDEX.get(action.spell.name)
        if slot is None:
            raise ValueError(f"Spell '{action.spell.name}' not in action space "
                             "(Teleport is excluded)")
        if action.cell is not None:
            hex_idx = cell_to_index(*action.cell)
        elif action.target is not None:
            hex_idx = cell_to_index(*action.target.pos)
        else:
            hex_idx = 0
        return CAST_START + slot * GRID_CELLS + hex_idx
    raise ValueError(f"Unknown action type: {type(action).__name__}")


def index_to_action(index: int, battle: BattleState,
                    current_unit: Unit) -> Action:
    """Convert a flat index back to an Action object.

    Returns SkipAction as fallback for invalid/out-of-range indices.
    """
    # ── Wait / Defend ──
    if index == WAIT_IDX or index == DEFEND_IDX:
        return SkipAction(current_unit)

    # ── Retreat ──
    if index == RETREAT_IDX:
        return RetreatAction(current_unit.team)

    # ── Move ──
    if MOVE_START <= index <= MOVE_END:
        hex_idx = index - MOVE_START
        col, row = index_to_cell(hex_idx)
        dest = (col, row)
        occ = battle._move_occupied(current_unit)
        moat = battle._moat_cells()
        td = _tail_dir(current_unit)
        path = battle.grid.find_path(current_unit.pos, dest, occ,
                                     current_unit.is_flying, current_unit.speed,
                                     td, moat)
        if path is None:
            return SkipAction(current_unit)
        return MoveAction(current_unit, path[:current_unit.speed + 1])

    # ── Attack ──
    if ATTACK_START <= index <= ATTACK_END:
        offset = index - ATTACK_START
        enemy_idx = offset // MAX_ATTACK_POSITIONS
        pos_idx = offset % MAX_ATTACK_POSITIONS
        enemies = _enemy_list(battle, current_unit)
        if enemy_idx >= len(enemies):
            return SkipAction(current_unit)
        target = enemies[enemy_idx]
        if pos_idx == 0 and current_unit.is_archer:
            return AttackAction(current_unit, target, ranged=True)
        positions = _attack_positions(battle.grid, current_unit, target)
        if pos_idx >= len(positions):
            return SkipAction(current_unit)
        from_pos = positions[pos_idx]
        return AttackAction(current_unit, target,
                            from_pos=from_pos, ranged=False)

    # ── Cast ──
    if CAST_START <= index <= CAST_END:
        offset = index - CAST_START
        spell_slot = offset // GRID_CELLS
        hex_idx = offset % GRID_CELLS
        hex_col, hex_row = index_to_cell(hex_idx)
        spell_name = _SPELL_ORDER[spell_slot]
        spell = SPELLS[spell_name]
        team = current_unit.team
        hero = battle.heroes.get(team)

        if hero is None:
            return SkipAction(current_unit)

        # Determine target unit and optional cell/destination
        target = battle.unit_at((hex_col, hex_row))
        cell = None

        if _is_ring_aoe(spell):
            cell = (hex_col, hex_row)
            if target is None:
                alive = battle.alive(1 - team) or battle.alive(team)
                target = alive[0] if alive else None
        elif spell.aoe_pattern == "chain":
            if target is None:
                enemies = battle.alive(1 - team)
                target = enemies[0] if enemies else None
        elif _is_mass_or_armywide(spell):
            if target is None:
                side = team if spell.side_friendly else (1 - team)
                if spell.name == "Earthquake":
                    side = 0  # placeholder
                candidates = battle.alive(side)
                target = candidates[0] if candidates else None

        if target is None:
            return SkipAction(current_unit)

        return CastAction(team, spell, target, cell=cell)

    # Out of range → fallback
    return SkipAction(current_unit)


# ── Legal mask ─────────────────────────────────────────────────

def legal_mask(battle: BattleState, current_unit: Unit) -> np.ndarray:
    """Return a float32 binary mask of shape (ACTION_DIM,).

    1.0 = legal, 0.0 = illegal.  Always non-empty (Wait is always legal).
    """
    mask = np.zeros(ACTION_DIM, dtype=np.float32)

    # Wait / Defend — always legal
    mask[WAIT_IDX] = 1.0
    mask[DEFEND_IDX] = 1.0

    # Retreat — legal if that side has a hero
    hero = battle.heroes.get(current_unit.team)
    if hero is not None:
        mask[RETREAT_IDX] = 1.0

    # ── Move ──
    occ = battle._move_occupied(current_unit)
    moat = battle._moat_cells()
    td = _tail_dir(current_unit)
    reachable = battle.grid.reachable(
        current_unit.pos, current_unit.speed, occ,
        current_unit.is_flying, td, moat)
    for cell in reachable:
        if cell == current_unit.pos:
            continue  # staying put is not a "move"
        mask[MOVE_START + cell_to_index(*cell)] = 1.0

    # ── Attack ──
    _mark_attack_legal(mask, battle, current_unit, reachable, occ, moat)

    # ── Cast ──
    if hero is not None and not hero._cast_this_round:
        _mark_cast_legal(mask, battle, current_unit, hero)

    return mask


def _mark_attack_legal(mask: np.ndarray, battle: BattleState,
                        unit: Unit, reachable: Set[Tuple[int, int]],
                        occ: Set[Tuple[int, int]], moat) -> None:
    """Mark legal melee and ranged attack actions."""
    grid = battle.grid
    enemies = battle.enemies_of(unit)

    for enemy_idx, enemy in enumerate(enemies):
        if enemy_idx >= MAX_ENEMIES:
            break

        positions = _attack_positions(grid, unit, enemy)

        # ── Ranged (archer only, position 0 = current pos) ──
        if unit.is_archer:
            base = ATTACK_START + enemy_idx * MAX_ATTACK_POSITIONS
            mask[base + 0] = 1.0  # ranged attack

        # ── Melee ──
        for pos_idx, ac in enumerate(positions):
            if pos_idx == 0:
                continue  # position 0 reserved for ranged
            if ac in occ:
                continue
            if ac not in reachable and ac != unit.pos:
                continue
            if not _can_attack_from_pos(grid, unit, enemy, ac, moat):
                continue
            mask[ATTACK_START + enemy_idx * MAX_ATTACK_POSITIONS + pos_idx] = 1.0


def _mark_cast_legal(mask: np.ndarray, battle: BattleState,
                      unit: Unit, hero) -> None:
    """Mark legal spell-casting actions for the hero of *unit*'s team."""
    team = unit.team
    friendly = battle.alive(team)
    enemies = battle.alive(1 - team)

    hero_spell_names = {s.name for s in hero.spellbook}

    for spell_slot, spell_name in enumerate(_SPELL_ORDER):
        spell = SPELLS[spell_name]

        if spell_name not in hero_spell_names:
            continue
        if not hero.can_cast(spell):
            continue

        base = CAST_START + spell_slot * GRID_CELLS

        if _is_mass_or_armywide(spell):
            mask[base:base + GRID_CELLS] = 1.0
            continue

        if _is_ring_aoe(spell):
            mask[base:base + GRID_CELLS] = 1.0
            continue

        if spell.aoe_pattern == "chain":
            for e in enemies:
                if e.is_immune_to_spells:
                    continue
                idx = cell_to_index(*e.pos)
                mask[base + idx] = 1.0
            continue

        _mark_single_target_spell(mask, base, battle, spell, team,
                                   friendly, enemies)


def _mark_single_target_spell(mask: np.ndarray, base: int,
                                battle: BattleState, spell: Spell,
                                team: int,
                                friendly: list, enemies: list) -> None:
    """Mark legal hexes for a single-target spell."""
    if spell.side_friendly or spell.kind in (BUFF, CURE):
        candidates = friendly
    elif spell.kind == DISPEL:
        candidates = friendly + enemies
    else:
        candidates = enemies

    for unit in candidates:
        if not unit.is_alive:
            continue
        if unit.is_immune_to_spells:
            continue
        if spell.kind in (BUFF, DEBUFF, CONTROL) and unit.has_effect(spell.name):
            continue
        if spell.exclude_tags:
            if any(unit.has_tag(t) for t in spell.exclude_tags):
                continue
        idx = cell_to_index(*unit.pos)
        mask[base + idx] = 1.0


# ── Convenience ────────────────────────────────────────────────

def enumerate_legal(battle: BattleState, current_unit: Unit) -> List[int]:
    """Return sorted list of all legal action indices."""
    m = legal_mask(battle, current_unit)
    return sorted(int(i) for i in np.nonzero(m)[0])


def action_type_label(index: int) -> str:
    """Human-readable label for an action index (for debugging)."""
    if index == WAIT_IDX:
        return "Wait"
    if index == DEFEND_IDX:
        return "Defend"
    if MOVE_START <= index <= MOVE_END:
        col, row = index_to_cell(index - MOVE_START)
        return f"Move({col},{row})"
    if ATTACK_START <= index <= ATTACK_END:
        offset = index - ATTACK_START
        enemy_idx = offset // MAX_ATTACK_POSITIONS
        pos_idx = offset % MAX_ATTACK_POSITIONS
        return f"Attack(enemy={enemy_idx},pos={pos_idx})"
    if CAST_START <= index <= CAST_END:
        offset = index - CAST_START
        slot = offset // GRID_CELLS
        hex_idx = offset % GRID_CELLS
        hc, hr = index_to_cell(hex_idx)
        return f"Cast({_SPELL_ORDER[slot]}@({hc},{hr}))"
    if index == RETREAT_IDX:
        return "Retreat"
    return f"Unknown({index})"

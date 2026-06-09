"""Siege data: walls, moat, towers, bridge, catapult.

Pure data + geometry — no rendering or battle-state mutation.
All coordinates use our (col, row) system on the 11x9 board.

Board layout (team 0 = attacker outside, team 1 = defender inside):

   col:  0  1  2  3  4  5  6  7  8  9 10
row 0:   .  .  .  .  .  .  . [M] [W] I  I
row 1:   .  .  .  .  .  .  . [M] [T] I  I
row 2:   .  .  .  .  .  . [M] [W]  I  I  I
row 3:   .  .  .  .  .  . [M] [G]  I  I  I   G = gate tower (non-shooting)
row 4:   .  .  .  .  . [M] [==] [G]  I  I  I   == = gate/bridge
row 5:   .  .  .  .  .  . [M] [G]  I  I  I
row 6:   .  .  .  .  .  . [M] [W]  I  I  I
row 7:   .  .  .  .  .  .  . [M] [T] I  I   T = archer tower (shooting)
row 8:   .  .  .  .  .  .  . [M] [W] I  I

Index mapping: our (col, row) = fheroes2 flat index  row*11 + col.

Simplifications vs fheroes2 (no hero skills / artifacts / castle buildings):
  - Wall HP fixed at 2 (no fortification 3-HP variant)
  - Catapult: 1 shot/round, 75% hit, 1 damage (no Ballistics skill)
  - Shooting penalty: fixed 50% (no Golden Bow / Archery exemption)
  - 3 towers always present (no build prerequisites)
"""

import math
import random
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

# ── Board geometry constants ─────────────────────────────────────

# 4 wall segments: HP 2 (intact) → 1 (damaged) → 0 (destroyed, passable).
WALL_POSITIONS: List[Tuple[int, int]] = [(8, 0), (7, 2), (7, 6), (8, 8)]

# 9 moat cells (outside the wall line, attacker side).
MOAT_CELLS: FrozenSet[Tuple[int, int]] = frozenset({
    (7, 0), (7, 1), (6, 2), (6, 3), (5, 4),
    (6, 5), (6, 6), (7, 7), (7, 8),
})

# Gate / drawbridge position.
GATE_POS: Tuple[int, int] = (6, 4)

# Gate towers (non-shooting, impassable, damageable by earthquake only).
GATE_TOWER_POSITIONS: List[Tuple[int, int]] = [(7, 3), (7, 5)]

# Archer tower positions (shooting, damageable by catapult).
ARCHER_TOWER_POSITIONS: List[Tuple[int, int]] = [(8, 1), (8, 7)]

# First "inside-walls" column per row (0-indexed).
# A cell (c, r) is inside the castle if c >= _INSIDE_COL[r].
_INSIDE_COL = [9, 9, 8, 8, 7, 8, 8, 9, 9]

# Catapult placement (far left, attacker side).
CATAPULT_POS: Tuple[int, int] = (0, 7)


# ── Tower ────────────────────────────────────────────────────────

class Tower:
    """Virtual archer tower — shoots once per round.

    Modelled after fheroes2's Battle::Tower (inherits from Unit as a
    pseudo-Archer).  Simplified: fixed count, no mage-guild attack bonus.
    """

    # Base Archer stats (from fheroes2 monster_info.cpp)
    _ARCHER_ATTACK = 5
    _ARCHER_DAMAGE_MIN = 1
    _ARCHER_DAMAGE_MAX = 3
    _ARCHER_HP = 10

    def __init__(self, kind: str):
        """kind: "center", "left", or "right"."""
        assert kind in ("center", "left", "right")
        self.kind = kind
        self.destroyed = False
        # Center tower (Ballista): count = 10.  Side turrets: count = 5.
        self.count = 10 if kind == "center" else 5
        self.attack = self._ARCHER_ATTACK

    @property
    def is_valid(self) -> bool:
        return not self.destroyed

    @property
    def damage_avg(self) -> float:
        return (self._ARCHER_DAMAGE_MIN + self._ARCHER_DAMAGE_MAX) / 2

    @property
    def strength(self) -> float:
        """Tower combat strength for AI evaluation.

        Mirrors the base_strength formula: sqrt(dmg_avg * hp) * count.
        """
        if self.destroyed:
            return 0.0
        base = math.sqrt(self.damage_avg * self._ARCHER_HP)
        return base * self.count

    def select_target(self, enemies):
        """Pick highest-strength enemy (fheroes2: highest evaluateThreatForUnit)."""
        if self.destroyed or not enemies:
            return None
        return max(enemies, key=lambda e: e.strength)

    def expected_damage(self) -> float:
        """Average damage per shot (AI evaluation)."""
        if self.destroyed:
            return 0.0
        return self.count * self.damage_avg

    def roll_damage(self) -> int:
        """Roll actual damage (combat execution)."""
        if self.destroyed:
            return 0
        return sum(random.randint(self._ARCHER_DAMAGE_MIN, self._ARCHER_DAMAGE_MAX)
                   for _ in range(self.count))


# ── Castle ───────────────────────────────────────────────────────

class Castle:
    """All siege structures for a single castle battle.

    Created once per siege battle and passed to BattleState.
    Non-siege battles simply have ``castle=None``.
    """

    def __init__(self):
        # Wall HP: position → {2=intact, 1=damaged, 0=destroyed}
        self.walls: Dict[Tuple[int, int], int] = {p: 2 for p in WALL_POSITIONS}

        # Bridge / gate state
        self.bridge_down: bool = False
        self.bridge_destroyed: bool = False

        # 3 towers: [left, center, right]
        self.towers: List[Tower] = [
            Tower("left"), Tower("center"), Tower("right"),
        ]

    # ── geometry queries ─────────────────────────────────────

    @staticmethod
    def is_moat(col: int, row: int) -> bool:
        return (col, row) in MOAT_CELLS

    @staticmethod
    def is_inside_walls(col: int, row: int) -> bool:
        """True if the cell is on the defender (castle interior) side."""
        if row < 0 or row >= len(_INSIDE_COL):
            return False
        return col >= _INSIDE_COL[row]

    @staticmethod
    def is_outside_walls(col: int, row: int) -> bool:
        return not Castle.is_inside_walls(col, row)

    # ── wall state ───────────────────────────────────────────

    def wall_intact_cells(self) -> Set[Tuple[int, int]]:
        """Wall cells that still block movement (HP > 0)."""
        return {p for p, hp in self.walls.items() if hp > 0}

    def wall_destroyed_cells(self) -> Set[Tuple[int, int]]:
        """Wall cells that no longer block movement (HP == 0)."""
        return {p for p, hp in self.walls.items() if hp == 0}

    def damage_wall(self, pos: Tuple[int, int], amount: int = 1) -> int:
        """Deal damage to a wall segment. Returns remaining HP."""
        hp = self.walls.get(pos, 0)
        hp = max(0, hp - amount)
        self.walls[pos] = hp
        return hp

    def any_wall_standing(self) -> bool:
        return any(hp > 0 for hp in self.walls.values())

    # ── bridge / gate ────────────────────────────────────────

    def is_gate_passable(self, team: int) -> bool:
        """Can *team* (0=attacker, 1=defender) walk through the gate cell?

        fheroes2 rules:
          - Defender: passable when bridge is down or destroyed.
          - Attacker: only when bridge is destroyed.
        """
        if self.bridge_destroyed:
            return True
        if team == 1 and self.bridge_down:
            return True
        return False

    def lower_bridge(self):
        """Defender lowers the drawbridge."""
        if not self.bridge_destroyed and not self.bridge_down:
            self.bridge_down = True

    def destroy_bridge(self):
        """Catapult destroys the bridge — permanently passable for everyone."""
        self.bridge_destroyed = True
        self.bridge_down = True  # destroyed implies permanently down

    @property
    def gate_block_cells(self) -> Set[Tuple[int, int]]:
        """Cells that block movement due to siege structures.

        Includes intact wall segments and the gate (when impassable).
        The caller should check team-dependent gate passability separately.
        """
        return self.wall_intact_cells()

    # ── tower helpers ────────────────────────────────────────

    def towers_active(self) -> bool:
        return any(t.is_valid for t in self.towers)

    def tower_strength(self) -> float:
        return sum(t.strength for t in self.towers)

    def damage_tower(self, index: int):
        """Destroy a tower by index (0=left, 1=center, 2=right)."""
        if 0 <= index < len(self.towers):
            self.towers[index].destroyed = True

    # ── catapult round ───────────────────────────────────────

    def catapult_round(self, rng: random.Random = None) -> List[dict]:
        """Execute one catapult firing round.

        Returns list of shot dicts: {target, hit, damage, remaining_hp}.
        Target priority (fheroes2): random intact wall → tower → bridge → center.
        """
        if rng is None:
            rng = random.Random()

        shots: List[dict] = []
        target = self._catapult_pick_target(rng)
        if target is None:
            return shots

        # 75% hit chance (fheroes2: canMiss, miss on <=5 out of 1..20)
        hit = rng.randint(1, 20) >= 6
        damage = 1  # fixed 1 damage (no Ballistics skill)

        remaining = 0
        if hit:
            if target == "bridge":
                self.destroy_bridge()
                remaining = 0
            elif target.startswith("tower_"):
                idx = int(target.split("_")[1])
                self.damage_tower(idx)
                remaining = 0  # towers are one-shot destroy
            else:
                # target is a wall position string like "(8, 0)"
                pos = self._parse_wall_target(target)
                remaining = self.damage_wall(pos, damage)
        else:
            remaining = self._target_hp(target)

        shots.append({
            "target": target,
            "hit": hit,
            "damage": damage if hit else 0,
            "remaining_hp": remaining,
        })
        return shots

    # ── catapult internals ───────────────────────────────────

    def _catapult_pick_target(self, rng: random.Random) -> Optional[str]:
        """Pick catapult target. Priority: walls → towers → bridge."""
        # 1. Intact walls (random among remaining)
        intact_walls = [str(p) for p, hp in self.walls.items() if hp > 0]
        if intact_walls:
            return rng.choice(intact_walls)

        # 2. Active towers
        active_towers = [f"tower_{i}" for i, t in enumerate(self.towers)
                         if t.is_valid]
        if active_towers:
            return rng.choice(active_towers)

        # 3. Bridge
        if not self.bridge_destroyed:
            return "bridge"

        # 4. Center tower (last resort)
        if self.towers[1].is_valid:
            return "tower_1"

        return None

    @staticmethod
    def _parse_wall_target(target: str) -> Tuple[int, int]:
        """Parse wall target string '(c, r)' back to tuple."""
        # target is like "(8, 0)"
        parts = target.strip("()").split(",")
        return (int(parts[0]), int(parts[1]))

    def _target_hp(self, target: str) -> int:
        if target == "bridge":
            return 0 if self.bridge_destroyed else 1
        if target.startswith("tower_"):
            idx = int(target.split("_")[1])
            return 1 if self.towers[idx].is_valid else 0
        pos = self._parse_wall_target(target)
        return self.walls.get(pos, 0)

"""Battle AI planner — top-level decision dispatch.

Reimplemented from fheroes2's ai_battle.cpp / ai_battle_spell.cpp.

Original source:
  src/fheroes2/ai/ai_battle.cpp        (2091 lines)
  src/fheroes2/ai/ai_battle_spell.cpp  (958 lines)
"""

from typing import Optional, Tuple, List, Set

from engine.battle_state import BattleState
from engine.actions import (Action, MoveAction, AttackAction, SkipAction,
                            CastAction, RetreatAction)
from engine.unit import Unit

from ai.base import AIPlayer
from .evaluation import AIState, analyze
from .scoring import threat, pos_value
from .spells import select_best_spell
from .retreat import should_retreat


class ClassicAI(AIPlayer):
    """Core tactical AI, faithful to fheroes2's decision logic."""

    def check_retreat(self, battle: BattleState, unit: Unit
                      ) -> Optional[Tuple[Optional[Tuple[CastAction, str]], RetreatAction]]:
        """Before a unit acts, decide whether its hero flees. — ai_battle.cpp Step 2

        Returns (farewell_cast_or_None, RetreatAction) when retreating, else None.
        The farewell cast is the best damage spell (selectBestSpell retreating).
        """
        team = unit.team
        hero = battle.heroes.get(team)
        if hero is None:
            return None
        state = analyze(battle, unit)
        if not should_retreat(state, battle.difficulty):
            return None
        farewell = None
        if not hero._cast_this_round:
            choice = select_best_spell(battle, team, state, retreating=True)
            if choice is not None:
                spell, target = choice
                farewell = (CastAction(team, spell, target),
                            f"[FAREWELL] {spell.name} -> {target.name}")
        return farewell, RetreatAction(team)

    def maybe_cast_spell(self, battle: BattleState, unit: Unit
                         ) -> Optional[Tuple[CastAction, str]]:
        """Before a unit acts, let its hero cast one spell this round.

        Returns (CastAction, description) or None. — ai_battle.cpp Step 3
        """
        team = unit.team
        hero = battle.heroes.get(team)
        if hero is None or hero._cast_this_round:
            return None
        state = analyze(battle, unit)
        choice = select_best_spell(battle, team, state)
        if choice is None:
            return None
        spell, target = choice
        return CastAction(team, spell, target), f"[CAST] {spell.name} -> {target.name}"

    def decide(self, battle: BattleState, unit: Unit) -> Tuple[Action, str]:
        """planUnitTurn() — ai_battle.cpp:689

        Returns (action, human-readable description).
        """
        # Berserk units act on instinct toward the nearest unit, friend or foe.
        if unit.has_effect("Berserk"):
            return self._berserk(battle, unit)

        state = analyze(battle, unit)

        # dispatch by unit type
        if unit.is_archer:
            action, detail = self._archer(battle, unit, state)
        elif state.defensive:
            action, detail = self._defense(battle, unit, state)
        else:
            action, detail = self._offense(battle, unit, state)

        tactic = "DEF" if state.defensive else ("CAUT" if state.cautious else "ATK")
        desc = (
            f"[{tactic}]  "
            f"Me:{state.my_army:.0f}([ARC]{state.my_shooters:.0f})  "
            f"En:{state.enemy_army:.0f}([ARC]{state.enemy_shooters:.0f})  "
            f"-> {detail}"
        )
        return action, desc

    # ================================================================
    #  wide-unit geometry helpers
    # ================================================================
    #
    # Each reduces to the original single-hex computation when neither unit is
    # wide, so single-hex battles stay byte-for-byte identical (fingerprint
    # 49b740ae1e7e90d3 guards this). ``tail_dir`` threads a wide mover's tail
    # offset through the grid pathfinding calls.

    @staticmethod
    def _tail_dir(unit: Unit) -> Optional[int]:
        """Column offset of a wide unit's tail; None for single-hex units."""
        if not unit.is_wide:
            return None
        return -1 if unit.team == 0 else 1

    @staticmethod
    def _path_args(battle: BattleState, unit: Unit):
        """Return (occupied, moat_cells) for siege-aware pathfinding."""
        return battle._move_occupied(unit), battle._moat_cells()

    @staticmethod
    def _dist(grid, a: Unit, b: Unit) -> int:
        """Min distance between two units' bodies (melee adjacency == 1)."""
        if not a.is_wide and not b.is_wide:
            return grid.distance(a.pos, b.pos)
        return min(grid.distance(ca, cb)
                   for ca in a.occupied_cells() for cb in b.occupied_cells())

    @staticmethod
    def _pos_dist(grid, pos: tuple, b: Unit) -> int:
        """Min distance from a bare cell to a unit's body."""
        if not b.is_wide:
            return grid.distance(pos, b.pos)
        return min(grid.distance(pos, cb) for cb in b.occupied_cells())

    @staticmethod
    def _attack_cells(grid, target: Unit) -> List[tuple]:
        """Cells a melee attacker can strike `target` from (head or tail adj)."""
        if not target.is_wide:
            return grid.neighbors(*target.pos)
        cells = list(grid.neighbors(*target.pos))
        body = target.occupied_cells()
        for tc in grid.neighbors(*target.tail_cell):
            if tc not in cells and tc not in body:
                cells.append(tc)
        return cells

    # ================================================================
    #  berserkTurn() — ai_battle.cpp:508
    # ================================================================

    def _berserk(self, battle: BattleState, unit: Unit) -> Tuple[Action, str]:
        """Attack / move toward the nearest unit, ignoring allegiance."""
        grid = battle.grid
        occ, moat = self._path_args(battle, unit)
        others = [u for u in battle.alive() if u is not unit]
        if not others:
            return SkipAction(unit), "[BERSERK] no target"
        td = self._tail_dir(unit)
        others.sort(key=lambda u: self._dist(grid, unit, u))

        blocked = any(self._dist(grid, unit, o) == 1 for o in others)
        if unit.is_archer and not blocked:
            tgt = others[0]
            return AttackAction(unit, tgt, ranged=True), f"[BERSERK] shoots {tgt.name}"

        # melee: attack the nearest unit reachable this turn
        reachable = grid.reachable(unit.pos, unit.speed, occ, unit.is_flying, td, moat)
        for tgt in others:
            for nb in self._attack_cells(grid, tgt):
                if nb in occ:
                    continue
                if nb in reachable or nb == unit.pos:
                    return (AttackAction(unit, tgt, nb, ranged=False),
                            f"[BERSERK] attacks {tgt.name}")

        # otherwise move toward the nearest unit
        tgt = others[0]
        tc = grid.nearest_cell_next_to(unit.pos, tgt.pos, occ,
                                       unit.is_flying, unit.speed * 3, td, moat)
        if tc:
            path = grid.find_path(unit.pos, tc, occ, unit.is_flying, unit.speed * 3, td, moat)
            if path and len(path) > 1:
                return (MoveAction(unit, path[:unit.speed + 1]),
                        f"[BERSERK] moves toward {tgt.name}")
        return SkipAction(unit), "[BERSERK] stuck"

    # ================================================================
    #  archerDecision() — ai_battle.cpp:1172
    # ================================================================

    def _archer(self, battle: BattleState, unit: Unit, s: AIState
                ) -> Tuple[Action, str]:
        enemies = battle.enemies_of(unit)
        occ, moat = self._path_args(battle, unit)
        td = self._tail_dir(unit)

        # blocked by melee?
        blocked = any(self._dist(battle.grid, unit, e) == 1 for e in enemies)

        if blocked:
            ret = self._retreat_pos(battle, unit, enemies, occ, moat)
            if ret:
                path = battle.grid.find_path(unit.pos, ret, occ, unit.is_flying, unit.speed, td, moat)
                if path and len(path) > 1:
                    return MoveAction(unit, path[:unit.speed + 1]), \
                        f"[ARC] retreat -> {ret}"

            best_e, best_d = None, float('-inf')
            for e in enemies:
                if self._dist(battle.grid, unit, e) != 1:
                    continue
                my_d = battle.expected_damage(unit, e, ranged=False)
                ret_d = battle.expected_damage(e, unit, ranged=False)
                diff = my_d - ret_d
                if diff > best_d:
                    best_d, best_e = diff, e
            if best_e:
                return AttackAction(unit, best_e, unit.pos, ranged=False), \
                    f"[ARC] blocked -> melee {best_e.name}"
            return SkipAction(unit), "[ARC] blocked, no target"

        # free to shoot
        best_e, best_t = None, float('-inf')
        for e in enemies:
            t = threat(battle, unit, e)
            if t > best_t:
                best_t, best_e = t, e
        if best_e:
            return AttackAction(unit, best_e, ranged=True), \
                f"[ARC] shoots {best_e.name} (threat {best_t:.0f})"
        return SkipAction(unit), "[ARC] no target"

    def _retreat_pos(self, battle: BattleState, unit: Unit,
                     enemies: List[Unit], occ: Set[tuple], moat=None
                     ) -> Optional[Tuple[int, int]]:
        """Archer retreat logic — ai_battle.cpp:1180-1379"""
        grid = battle.grid
        if any(e.is_flying for e in enemies):
            return None

        td = self._tail_dir(unit)
        reachable = grid.reachable(unit.pos, unit.speed, occ, unit.is_flying, td, moat)
        safe: list = []
        for pos in reachable:
            if pos in occ:
                continue
            is_threatened = False
            for e in enemies:
                d = self._pos_dist(grid, pos, e)
                if d == 1:
                    is_threatened = True; break
                if not e.is_archer and d <= e.speed + 1:
                    is_threatened = True; break
            if not is_threatened:
                safe.append(pos)

        cur_threatened = any(
            self._dist(grid, unit, e) <= e.speed + 1
            for e in enemies if not e.is_archer)
        if not cur_threatened:
            return None

        adj = [e for e in enemies if self._dist(grid, unit, e) == 1]
        if not all(e.speed + 2 < unit.speed for e in adj):
            return None

        if not safe:
            return None
        center = (grid.cols // 2, grid.rows // 2)
        return max(safe, key=lambda p: (
            min(self._pos_dist(grid, p, e) for e in enemies),
            -grid.distance(p, center)))

    # ================================================================
    #  meleeUnitOffense() — ai_battle.cpp:1568
    # ================================================================

    def _offense(self, battle: BattleState, unit: Unit, s: AIState
                 ) -> Tuple[Action, str]:
        enemies = battle.enemies_of(unit)
        occ, moat = self._path_args(battle, unit)
        grid = battle.grid
        td = self._tail_dir(unit)

        # tier 1: target in attack range
        reachable = grid.reachable(unit.pos, unit.speed, occ, unit.is_flying, td, moat)
        best_e, best_pos, best_val = None, None, float('-inf')
        for e in enemies:
            for nb in self._attack_cells(grid, e):
                if nb not in reachable and nb != unit.pos:
                    continue
                if nb in occ:
                    continue
                val = e.strength + pos_value(battle, unit, nb, enemies)
                if val > best_val:
                    best_val, best_e, best_pos = val, e, nb
        if best_e:
            return AttackAction(unit, best_e, best_pos, ranged=False), \
                f"[ME] attacks {best_e.name} (in range)"

        # tier 2: chase distant target
        result = self._chase(battle, unit, enemies, occ, moat, s,
                             lambda e: (e.is_archer
                                        or e.speed == 0
                                        or (not e.is_flying and e.speed < unit.speed)),
                             "chasing slow")
        if result:
            return result

        result = self._chase(battle, unit, enemies, occ, moat, s,
                             lambda e: True,
                             "chasing any")
        return result or (SkipAction(unit), "[ME] no target")

    def _chase(self, battle: BattleState, unit: Unit,
               enemies: List[Unit], occ: Set[tuple], moat, s: AIState,
               predicate, reason: str) -> Optional[Tuple[Action, str]]:
        grid = battle.grid
        td = self._tail_dir(unit)
        best_e, best_pri, best_path = None, float('-inf'), None
        for e in enemies:
            if not predicate(e):
                continue
            tgt_cell = grid.nearest_cell_next_to(unit.pos, e.pos, occ,
                                                  unit.is_flying, unit.speed * 3, td, moat)
            if not tgt_cell:
                continue
            path = grid.find_path(unit.pos, tgt_cell, occ,
                                  unit.is_flying, unit.speed * 3, td, moat)
            if not path:
                continue
            dist = len(path) - 1
            if dist <= 0:
                continue
            pri = threat(battle, unit, e) / dist
            if pri > best_pri:
                best_pri, best_e, best_path = pri, e, path

        if best_e and best_path:
            seg = best_path[:unit.speed + 1]
            final = seg[-1]
            if self._pos_dist(grid, final, best_e) == 1:
                return (AttackAction(unit, best_e, final, ranged=False),
                        f"[ME] {reason} {best_e.name}")
            # When playing cautiously, advance only to the safest cell on the
            # path instead of charging the full distance into enemy range.
            if s.cautious:
                seg = self._safest_step_on_path(battle, unit, seg, enemies)
            return (MoveAction(unit, seg),
                    f"[ME] {reason} {best_e.name}, moving closer")
        return None

    def _safest_step_on_path(self, battle: BattleState, unit: Unit,
                             seg: List[tuple], enemies: List[Unit]) -> List[tuple]:
        """findOptimalPositionForSubsequentAttack — pick the lowest-threat step.

        Among the cells reachable this turn (``seg[1:]``), return the path
        truncated at the one exposing the unit to the least enemy threat next
        turn; ties go to the cell that advances furthest.
        """
        if len(seg) <= 2:
            return seg
        best_idx, best_key = 1, None
        for idx in range(1, len(seg)):
            t = self._cell_threat(battle, unit, seg[idx], enemies)
            key = (t, -idx)   # least threat, then furthest progress
            if best_key is None or key < best_key:
                best_key, best_idx = key, idx
        return seg[:best_idx + 1]

    def _cell_threat(self, battle: BattleState, unit: Unit,
                     cell: tuple, enemies: List[Unit]) -> float:
        """Expected damage enemies could deal to `unit` if it stood at `cell`."""
        grid = battle.grid
        total = 0.0
        for e in enemies:
            if e.is_archer or e.is_flying:
                reachable = True   # shooters / flyers threaten anywhere
            else:
                reachable = self._pos_dist(grid, cell, e) <= e.speed + 1
            if reachable:
                total += battle.expected_damage(e, unit, ranged=e.is_archer)
        return total

    # ================================================================
    #  meleeUnitDefense() — ai_battle.cpp:1708
    # ================================================================

    def _defense(self, battle: BattleState, unit: Unit, s: AIState
                 ) -> Tuple[Action, str]:
        enemies = battle.enemies_of(unit)
        friends = battle.friends_of(unit)
        occ, moat = self._path_args(battle, unit)
        grid = battle.grid
        td = self._tail_dir(unit)

        archers = [f for f in friends if f.is_archer and f is not unit]
        if not archers:
            return self._offense(battle, unit, s)

        modifier = s.my_shooters / 15.0
        best_arch, best_val, best_cover = None, float('-inf'), None
        for a in archers:
            blockers = [e for e in enemies if self._dist(grid, a, e) == 1]
            cover = self._cover_pos(battle, unit, a, occ, moat)
            if not cover and not blockers:
                continue

            if cover:
                d = grid.distance(unit.pos, cover)
            else:
                d = min(self._dist(grid, unit, e) for e in blockers)
            val = a.strength - d * modifier
            if val > best_val:
                best_val, best_arch, best_cover = val, a, cover

                if blockers:
                    tgt = min(blockers, key=lambda e: self._dist(grid, unit, e))
                    tc = grid.nearest_cell_next_to(unit.pos, tgt.pos, occ,
                                                    unit.is_flying, unit.speed, td, moat)
                    if tc:
                        path = grid.find_path(unit.pos, tc, occ,
                                              unit.is_flying, unit.speed, td, moat)
                        if path:
                            return (AttackAction(unit, tgt, path[-1], ranged=False),
                                    f"[DEF] defends {a.name}, attacks {tgt.name}")

        if best_arch and best_cover:
            path = grid.find_path(unit.pos, best_cover, occ,
                                  unit.is_flying, unit.speed, td, moat)
            if path:
                seg = path[:unit.speed + 1]
                return (MoveAction(unit, seg),
                        f"[DEF] covers {best_arch.name}")

        return self._offense(battle, unit, s)

    def _cover_pos(self, battle: BattleState, unit: Unit,
                   archer: Unit, occ: Set[tuple], moat=None
                   ) -> Optional[Tuple[int, int]]:
        grid = battle.grid
        reachable = grid.reachable(unit.pos, unit.speed, occ, unit.is_flying,
                                   self._tail_dir(unit), moat)
        best, best_d = None, float('inf')
        for nb in grid.neighbors(*archer.pos):
            if nb in occ:
                continue
            if nb in reachable or nb == unit.pos:
                d = grid.distance(unit.pos, nb)
                if d < best_d:
                    best_d, best = d, nb
        return best

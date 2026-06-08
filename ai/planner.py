"""Battle AI planner — top-level decision dispatch.

Reimplemented from fheroes2's ai_battle.cpp / ai_battle_spell.cpp.

Original source:
  src/fheroes2/ai/ai_battle.cpp        (2091 lines)
  src/fheroes2/ai/ai_battle_spell.cpp  (958 lines)
"""

from typing import Optional, Tuple, List, Set

from engine.battle_state import BattleState
from engine.actions import Action, MoveAction, AttackAction, SkipAction
from engine.unit import Unit

from .evaluation import AIState, analyze
from .scoring import threat, pos_value


class BattleAI:
    """Core tactical AI, faithful to fheroes2's decision logic."""

    def decide(self, battle: BattleState, unit: Unit) -> Tuple[Action, str]:
        """planUnitTurn() — ai_battle.cpp:689

        Returns (action, human-readable description).
        """
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
    #  archerDecision() — ai_battle.cpp:1172
    # ================================================================

    def _archer(self, battle: BattleState, unit: Unit, s: AIState
                ) -> Tuple[Action, str]:
        enemies = battle.enemies_of(unit)
        occ = battle.occupied(exclude=unit)

        # blocked by melee?
        blocked = any(battle.grid.distance(unit.pos, e.pos) == 1 for e in enemies)

        if blocked:
            ret = self._retreat_pos(battle, unit, enemies, occ)
            if ret:
                path = battle.grid.find_path(unit.pos, ret, occ, unit.is_flying, unit.speed)
                if path and len(path) > 1:
                    return MoveAction(unit, path[:unit.speed + 1]), \
                        f"[ARC] retreat -> {ret}"

            best_e, best_d = None, float('-inf')
            for e in enemies:
                if battle.grid.distance(unit.pos, e.pos) != 1:
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
                     enemies: List[Unit], occ: Set[tuple]
                     ) -> Optional[Tuple[int, int]]:
        """Archer retreat logic — ai_battle.cpp:1180-1379"""
        grid = battle.grid
        if any(e.is_flying for e in enemies):
            return None

        reachable = grid.reachable(unit.pos, unit.speed, occ, unit.is_flying)
        safe: list = []
        for pos in reachable:
            if pos in occ:
                continue
            is_threatened = False
            for e in enemies:
                d = grid.distance(pos, e.pos)
                if d == 1:
                    is_threatened = True; break
                if not e.is_archer and d <= e.speed + 1:
                    is_threatened = True; break
            if not is_threatened:
                safe.append(pos)

        cur_threatened = any(
            grid.distance(unit.pos, e.pos) <= e.speed + 1
            for e in enemies if not e.is_archer)
        if not cur_threatened:
            return None

        adj = [e for e in enemies if grid.distance(unit.pos, e.pos) == 1]
        if not all(e.speed + 2 < unit.speed for e in adj):
            return None

        if not safe:
            return None
        center = (grid.cols // 2, grid.rows // 2)
        return max(safe, key=lambda p: (
            min(grid.distance(p, e.pos) for e in enemies),
            -grid.distance(p, center)))

    # ================================================================
    #  meleeUnitOffense() — ai_battle.cpp:1568
    # ================================================================

    def _offense(self, battle: BattleState, unit: Unit, s: AIState
                 ) -> Tuple[Action, str]:
        enemies = battle.enemies_of(unit)
        occ = battle.occupied(exclude=unit)
        grid = battle.grid

        # tier 1: target in attack range
        reachable = grid.reachable(unit.pos, unit.speed, occ, unit.is_flying)
        best_e, best_pos, best_val = None, None, float('-inf')
        for e in enemies:
            for nb in grid.neighbors(*e.pos):
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
        result = self._chase(battle, unit, enemies, occ, s,
                             lambda e: (e.is_archer
                                        or e.speed == 0
                                        or (not e.is_flying and e.speed < unit.speed)),
                             "chasing slow")
        if result:
            return result

        result = self._chase(battle, unit, enemies, occ, s,
                             lambda e: True,
                             "chasing any")
        return result or (SkipAction(unit), "[ME] no target")

    def _chase(self, battle: BattleState, unit: Unit,
               enemies: List[Unit], occ: Set[tuple], s: AIState,
               predicate, reason: str) -> Optional[Tuple[Action, str]]:
        grid = battle.grid
        best_e, best_pri, best_path = None, float('-inf'), None
        for e in enemies:
            if not predicate(e):
                continue
            tgt_cell = grid.nearest_cell_next_to(unit.pos, e.pos, occ,
                                                  unit.is_flying, unit.speed * 3)
            if not tgt_cell:
                continue
            path = grid.find_path(unit.pos, tgt_cell, occ,
                                  unit.is_flying, unit.speed * 3)
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
            if grid.distance(final, best_e.pos) == 1:
                return (AttackAction(unit, best_e, final, ranged=False),
                        f"[ME] {reason} {best_e.name}")
            return (MoveAction(unit, seg),
                    f"[ME] {reason} {best_e.name}, moving closer")
        return None

    # ================================================================
    #  meleeUnitDefense() — ai_battle.cpp:1708
    # ================================================================

    def _defense(self, battle: BattleState, unit: Unit, s: AIState
                 ) -> Tuple[Action, str]:
        enemies = battle.enemies_of(unit)
        friends = battle.friends_of(unit)
        occ = battle.occupied(exclude=unit)
        grid = battle.grid

        archers = [f for f in friends if f.is_archer and f is not unit]
        if not archers:
            return self._offense(battle, unit, s)

        modifier = s.my_shooters / 15.0
        best_arch, best_val, best_cover = None, float('-inf'), None
        for a in archers:
            blockers = [e for e in enemies if grid.distance(a.pos, e.pos) == 1]
            cover = self._cover_pos(battle, unit, a, occ)
            if not cover and not blockers:
                continue

            if cover:
                d = grid.distance(unit.pos, cover)
            else:
                d = min(grid.distance(unit.pos, e.pos) for e in blockers)
            val = a.strength - d * modifier
            if val > best_val:
                best_val, best_arch, best_cover = val, a, cover

                if blockers:
                    tgt = min(blockers, key=lambda e: grid.distance(unit.pos, e.pos))
                    tc = grid.nearest_cell_next_to(unit.pos, tgt.pos, occ,
                                                    unit.is_flying, unit.speed)
                    if tc:
                        path = grid.find_path(unit.pos, tc, occ,
                                              unit.is_flying, unit.speed)
                        if path:
                            return (AttackAction(unit, tgt, path[-1], ranged=False),
                                    f"[DEF] defends {a.name}, attacks {tgt.name}")

        if best_arch and best_cover:
            path = grid.find_path(unit.pos, best_cover, occ,
                                  unit.is_flying, unit.speed)
            if path:
                seg = path[:unit.speed + 1]
                return (MoveAction(unit, seg),
                        f"[DEF] covers {best_arch.name}")

        return self._offense(battle, unit, s)

    def _cover_pos(self, battle: BattleState, unit: Unit,
                   archer: Unit, occ: Set[tuple]) -> Optional[Tuple[int, int]]:
        grid = battle.grid
        reachable = grid.reachable(unit.pos, unit.speed, occ, unit.is_flying)
        best, best_d = None, float('inf')
        for nb in grid.neighbors(*archer.pos):
            if nb in occ:
                continue
            if nb in reachable or nb == unit.pos:
                d = grid.distance(unit.pos, nb)
                if d < best_d:
                    best_d, best = d, nb
        return best

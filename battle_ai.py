"""Battle AI — reimplemented from fheroes2's ai_battle.cpp / ai_battle_spell.cpp.

This is a simplified Python reimplementation of the core tactical AI.
Each method maps back to a specific function in the original C++ source.

Original source:
  src/fheroes2/ai/ai_battle.cpp        (2091 lines)
  src/fheroes2/ai/ai_battle_spell.cpp  (958 lines)

Learning guide:
  learn/ai决策/战斗AI学习指南.md
"""

from typing import Optional, Tuple, List, Set

from battle import Action, MoveAction, AttackAction, SkipAction, Unit, BattleState
from hex_grid import HexGrid


class AIState:
    """Temporary per-turn analysis (BattlePlanner member vars in C++)."""

    def __init__(self):
        self.my_team = 0
        self.my_army = 0.0
        self.enemy_army = 0.0
        self.my_shooters = 0.0
        self.enemy_shooters = 0.0
        self.my_avg_speed = 0.0
        self.enemy_avg_speed = 0.0
        self.defensive = False
        self.cautious = False


class BattleAI:
    """Core tactical AI, faithful to fheroes2's decision logic."""

    def decide(self, battle: BattleState, unit: Unit) -> Tuple[Action, str]:
        """planUnitTurn() — ai_battle.cpp:689

        Returns (action, human-readable description).
        """
        state = self._analyze(battle, unit)

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
            f"→ {detail}"
        )
        return action, desc

    # ================================================================
    #  analyzeBattleState() — ai_battle.cpp:949
    # ================================================================

    def _analyze(self, battle: BattleState, unit: Unit) -> AIState:
        s = AIState()
        s.my_team = unit.team
        enemies = battle.enemies_of(unit)
        friends = battle.friends_of(unit)
        if not enemies:
            return s

        # enemy stats
        e_sum = 0.0
        for e in enemies:
            v = e.strength
            s.enemy_army += v
            if e.is_archer:
                s.enemy_shooters += v
            s.enemy_avg_speed += e.speed * v
            e_sum += v
        if e_sum > 0:
            s.enemy_avg_speed /= e_sum

        # friendly stats
        f_sum = 0.0
        for f in friends:
            v = f.strength
            s.my_army += v
            if f.is_archer:
                s.my_shooters += v
            s.my_avg_speed += f.speed * v
            f_sum += v
        if f_sum > 0:
            s.my_avg_speed /= f_sum

        # ── tactical flags — ai_battle.cpp:1124-1164 ─────────
        s.defensive = self._should_defend(unit, s, battle)
        s.cautious = s.enemy_shooters / max(s.enemy_army, 1) < 0.15
        return s

    def _should_defend(self, unit: Unit, s: AIState, battle: BattleState) -> bool:
        """_defensiveTactics logic — ai_battle.cpp:1124"""
        grid = battle.grid
        # already past center → keep attacking
        if unit.col >= grid.cols // 2:
            return False
        # overwhelming power → no need to defend
        over = 6 if unit.is_flying else 10
        if s.my_army > s.enemy_army * over:
            return False
        # fewer shooters → attack
        if s.my_shooters < s.enemy_shooters:
            return False
        # too few archers → attack
        if s.my_shooters / max(s.my_army, 1) < 0.15:
            return False
        # enemy mostly shooters → rush them
        if s.enemy_shooters / max(s.enemy_army, 1) > 0.66:
            return False
        return True

    # ================================================================
    #  archerDecision() — ai_battle.cpp:1172
    # ================================================================

    def _archer(self, battle: BattleState, unit: Unit, s: AIState
                ) -> Tuple[Action, str]:
        enemies = battle.enemies_of(unit)
        occ = battle.occupied(exclude=unit)

        # ── blocked by melee? ─────────────────────────────────
        blocked = any(battle.grid.distance(unit.pos, e.pos) == 1 for e in enemies)

        if blocked:
            # try retreat — ai_battle.cpp:1180-1379
            ret = self._retreat_pos(battle, unit, enemies, occ)
            if ret:
                path = battle.grid.find_path(unit.pos, ret, occ, unit.is_flying, unit.speed)
                if path and len(path) > 1:
                    return MoveAction(unit, path[:unit.speed + 1]), \
                        f"[ARC] retreat → {ret}"

            # no retreat → melee the best adjacent target
            best_e, best_d = None, float('-inf')
            for e in enemies:
                if battle.grid.distance(unit.pos, e.pos) != 1:
                    continue
                my_d = battle.calc_damage(unit, e, ranged=False)
                ret_d = battle.calc_damage(e, unit, ranged=False)
                diff = my_d - ret_d
                if diff > best_d:
                    best_d, best_e = diff, e
            if best_e:
                return AttackAction(unit, best_e, unit.pos, ranged=False), \
                    f"[ARC] blocked → melee {best_e.name}"
            return SkipAction(unit), "[ARC] blocked, no target"

        # ── free to shoot ─────────────────────────────────────
        best_e, best_t = None, float('-inf')
        for e in enemies:
            t = self._threat(unit, e)
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
        # flyers can't be escaped
        if any(e.is_flying for e in enemies):
            return None

        reachable = grid.reachable(unit.pos, unit.speed, occ, unit.is_flying)
        # classify each reachable position
        safe: list = []
        for pos in reachable:
            if pos in occ:
                continue
            threatened = False
            for e in enemies:
                d = grid.distance(pos, e.pos)
                if d == 1:
                    threatened = True; break
                if not e.is_archer and d <= e.speed + 1:
                    threatened = True; break
            if not threatened:
                safe.append(pos)

        # current position safe?
        cur_threatened = any(
            grid.distance(unit.pos, e.pos) <= e.speed + 1
            for e in enemies if not e.is_archer
        )
        if not cur_threatened:
            return None

        # only retreat if we're faster than adjacent threats
        adj = [e for e in enemies if grid.distance(unit.pos, e.pos) == 1]
        if not all(e.speed + 2 < unit.speed for e in adj):
            return None

        if not safe:
            return None
        # pick farthest from enemy, prefer center
        center = (grid.cols // 2, grid.rows // 2)
        return max(safe, key=lambda p: (
            min(grid.distance(p, e.pos) for e in enemies),
            -grid.distance(p, center)
        ))

    # ================================================================
    #  meleeUnitOffense() — ai_battle.cpp:1568
    # ================================================================

    def _offense(self, battle: BattleState, unit: Unit, s: AIState
                 ) -> Tuple[Action, str]:
        enemies = battle.enemies_of(unit)
        occ = battle.occupied(exclude=unit)
        grid = battle.grid

        # ── tier 1: target in attack range ────────────────────
        reachable = grid.reachable(unit.pos, unit.speed, occ, unit.is_flying)
        best_e, best_pos, best_val = None, None, float('-inf')
        for e in enemies:
            for nb in grid.neighbors(*e.pos):
                if nb not in reachable and nb != unit.pos:
                    continue
                if nb in occ:
                    continue
                val = e.strength + self._pos_value(battle, unit, nb, enemies)
                if val > best_val:
                    best_val, best_e, best_pos = val, e, nb
        if best_e:
            return AttackAction(unit, best_e, best_pos, ranged=False), \
                f"[ME] attacks {best_e.name} (in range)"

        # ── tier 2: chase distant target ──────────────────────
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
            pri = self._threat(unit, e) / dist
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

        # evaluate each archer for protection priority
        modifier = s.my_shooters / 15.0
        best_arch, best_val, best_cover = None, float('-inf'), None
        for a in archers:
            # enemies blocking this archer
            blockers = [e for e in enemies if grid.distance(a.pos, e.pos) == 1]
            # cover position near archer
            cover = self._cover_pos(battle, unit, a, occ)
            if not cover and not blockers:
                continue

            # distance to archer or nearest blocker
            if cover:
                d = grid.distance(unit.pos, cover)
            else:
                d = min(grid.distance(unit.pos, e.pos) for e in blockers)
            val = a.strength - d * modifier
            if val > best_val:
                best_val, best_arch, best_cover = val, a, cover

                # if blockers exist, plan to attack one
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

    # ================================================================
    #  helpers
    # ================================================================

    def _threat(self, attacker: Unit, defender: Unit) -> float:
        """Simplified evaluateThreatForUnit()."""
        base = defender.strength
        if defender.is_archer:
            base *= 1.2
        if defender.speed > attacker.speed:
            base *= 1.1
        if defender.is_flying:
            base *= 1.3
        return base

    def _pos_value(self, battle: BattleState, unit: Unit,
                   pos: tuple, enemies: List[Unit]) -> float:
        """evaluatePotentialAttackPositions() — simplified."""
        val = 0.0
        for e in enemies:
            if battle.grid.distance(pos, e.pos) == 1:
                t = self._threat(unit, e)
                if e.is_archer:
                    val += t
                else:
                    val = max(val, t)
        return val

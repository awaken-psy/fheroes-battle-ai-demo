"""Battle state machine — turn order, damage, victory."""

import random
from typing import List, Optional, Set, Tuple

from .unit import Unit
from .actions import Action, MoveAction, AttackAction, SkipAction
from .hex_grid import HexGrid


class BattleState:
    def __init__(self, grid: HexGrid, units: List[Unit], first_team: int = 0,
                 attacker_team: int = 0):
        self.grid = grid
        self.units = units
        self.round_num = 0
        self.deaths_this_round = 0
        # Which team wins the initiative tie on equal speed. Arena flips this
        # per game to cancel any first-move advantage.
        self.first_team = first_team
        # The attacking side (fheroes2: the army that initiated the battle).
        # On a death-free stalemate the attacker is forced to retreat.
        self.attacker_team = attacker_team
        # Consecutive completed rounds in which no unit died (anti-stalemate).
        self._stale_rounds = 0

    def alive(self, team: Optional[int] = None) -> List[Unit]:
        u = [u for u in self.units if u.is_alive]
        if team is not None:
            u = [u for u in u if u.team == team]
        return u

    def enemies_of(self, unit: Unit) -> List[Unit]:
        return self.alive(1 - unit.team)

    def friends_of(self, unit: Unit) -> List[Unit]:
        return self.alive(unit.team)

    def occupied(self, exclude: Optional[Unit] = None) -> Set[Tuple[int, int]]:
        return {u.pos for u in self.alive() if u is not exclude}

    def unit_at(self, pos: Tuple[int, int]) -> Optional[Unit]:
        for u in self.alive():
            if u.pos == pos:
                return u
        return None

    def turn_order(self) -> List[Unit]:
        """Activation order for one round, faithful to fheroes2.

        Each army is speed-sorted, then the two queues are merged: the fastest
        available unit acts; on equal speed the "preferred" side goes, and the
        preference then flips to the other side. Equal-speed units therefore
        alternate between armies (A, B, A, B …) rather than one whole army
        acting before the other. (battle_arena.cpp GetCurrentUnit)
        """
        queues = {0: [], 1: []}
        for u in self.alive():
            queues[u.team].append(u)
        for team in queues:
            queues[team].sort(key=lambda u: (-u.speed, u.name))

        idx = {0: 0, 1: 0}
        preferred = self.first_team
        order: List[Unit] = []
        while idx[0] < len(queues[0]) or idx[1] < len(queues[1]):
            front0 = queues[0][idx[0]] if idx[0] < len(queues[0]) else None
            front1 = queues[1][idx[1]] if idx[1] < len(queues[1]) else None
            if front0 and front1:
                if front0.speed == front1.speed:
                    pick = preferred
                else:
                    pick = 0 if front0.speed > front1.speed else 1
            else:
                pick = 0 if front0 else 1
            order.append(queues[pick][idx[pick]])
            idx[pick] += 1
            preferred = 1 - pick  # next activation prefers the other army
        return order

    # ── damage ──────────────────────────────────────────────
    #
    # Split mirrors fheroes2: the AI reasons about *expected* (average)
    # damage — deterministic — while actual combat rolls a random spread.
    # Keeping these apart makes AI decisions and tests reproducible.

    @staticmethod
    def _damage_mult(atk: Unit, dfn: Unit, ranged: bool = False) -> float:
        """Deterministic damage multiplier (attack/defense + archer penalty)."""
        if atk.attack > dfn.defense:
            mult = min(1 + 0.1 * (atk.attack - dfn.defense), 3.0)
        else:
            mult = max(1 - 0.05 * (dfn.defense - atk.attack), 0.3)
        if atk.is_archer and not ranged:
            mult *= 0.5  # archer melee penalty
        return mult

    def expected_damage(self, atk: Unit, dfn: Unit, ranged: bool = False) -> int:
        """Average damage — used by the AI for decisions and by tests."""
        base = atk.count * atk.damage
        return max(1, int(base * self._damage_mult(atk, dfn, ranged)))

    def roll_damage(self, atk: Unit, dfn: Unit, ranged: bool = False) -> int:
        """Actual damage with random spread — used when executing an attack."""
        base = atk.count * atk.damage
        mult = self._damage_mult(atk, dfn, ranged) * random.uniform(0.85, 1.15)
        return max(1, int(base * mult))

    # Backwards-compatible alias: callers that want a real (rolled) hit.
    calc_damage = roll_damage

    # ── execute ─────────────────────────────────────────────

    def execute(self, action: Action) -> dict:
        """Execute an action, return result dict with damage details."""
        r = {'desc': '', 'dmg': 0, 'killed': 0,
             'ret_dmg': 0, 'ret_killed': 0,
             'target_alive': True, 'attacker_alive': True}

        if isinstance(action, MoveAction):
            action.unit.pos = action.path[-1]
            r['desc'] = f"{action.unit.name} moves to {action.path[-1]}"
            return r

        if isinstance(action, AttackAction):
            atk, tgt = action.attacker, action.target
            if not action.ranged and action.from_pos:
                atk.pos = action.from_pos

            dmg = self.roll_damage(atk, tgt, action.ranged)
            actual, killed = tgt.take_damage(dmg)
            r['dmg'] = actual
            r['killed'] = killed
            r['target_alive'] = tgt.is_alive

            verb = "shoots" if action.ranged else "attacks"
            desc = f"{atk.name} {verb} {tgt.name}: {actual} dmg"
            if killed > 0:
                desc += f" ({killed} killed)"
            if not tgt.is_alive:
                self.deaths_this_round += 1
                desc += " [DEAD]"

            # retaliation (melee only, once per round)
            if not action.ranged and not tgt.retaliated and tgt.is_alive:
                ret = self.roll_damage(tgt, atk)
                ret_actual, ret_killed = atk.take_damage(ret)
                tgt.retaliated = True
                r['ret_dmg'] = ret_actual
                r['ret_killed'] = ret_killed
                r['attacker_alive'] = atk.is_alive
                desc += f" -> {tgt.name} retaliates: {ret_actual}"
                if ret_killed > 0:
                    desc += f" ({ret_killed} killed)"
                if not atk.is_alive:
                    self.deaths_this_round += 1
                    desc += " [DEAD]"

            r['desc'] = desc
            return r

        if isinstance(action, SkipAction):
            r['desc'] = f"{action.unit.name} skips"
            return r

        return r

    # ── victory ─────────────────────────────────────────────

    # fheroes2 MAX_TURNS_WITHOUT_DEATHS: the attacker retreats after this many
    # death-free rounds, breaking stalemates. MAX_ROUNDS is an absolute backstop.
    MAX_TURNS_WITHOUT_DEATHS = 50
    MAX_ROUNDS = 200

    def is_stalemate(self) -> bool:
        return self._stale_rounds >= self.MAX_TURNS_WITHOUT_DEATHS

    def is_over(self) -> bool:
        return (len(self.alive(0)) == 0 or len(self.alive(1)) == 0
                or self.is_stalemate()
                or self.round_num >= self.MAX_ROUNDS)

    def winner(self) -> int:
        if not self.alive(0):
            return 1
        if not self.alive(1):
            return 0
        # Death-free stalemate: the attacking side gives up (fheroes2 retreat).
        if self.is_stalemate():
            return 1 - self.attacker_team
        # Absolute backstop reached — winner by remaining army strength.
        s0 = sum(u.strength for u in self.alive(0))
        s1 = sum(u.strength for u in self.alive(1))
        return 0 if s0 >= s1 else 1

    def start_round(self):
        # Update the death-free streak based on the round that just finished.
        if self.round_num >= 1:
            if self.deaths_this_round == 0:
                self._stale_rounds += 1
            else:
                self._stale_rounds = 0
        self.round_num += 1
        self.deaths_this_round = 0
        for u in self.alive():
            u.new_round()

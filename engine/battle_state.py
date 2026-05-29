"""Battle state machine — turn order, damage, victory."""

import random
from typing import List, Optional, Set, Tuple

from .unit import Unit
from .actions import Action, MoveAction, AttackAction, SkipAction
from .hex_grid import HexGrid


class BattleState:
    def __init__(self, grid: HexGrid, units: List[Unit]):
        self.grid = grid
        self.units = units
        self.round_num = 0
        self.deaths_this_round = 0

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
        return sorted(self.alive(), key=lambda u: (-u.speed, u.team, u.name))

    # ── damage ──────────────────────────────────────────────

    def calc_damage(self, atk: Unit, dfn: Unit, ranged: bool = False) -> int:
        base = atk.count * atk.damage
        if atk.attack > dfn.defense:
            mult = min(1 + 0.1 * (atk.attack - dfn.defense), 3.0)
        else:
            mult = max(1 - 0.05 * (dfn.defense - atk.attack), 0.3)
        if atk.is_archer and not ranged:
            mult *= 0.5  # archer melee penalty
        mult *= random.uniform(0.85, 1.15)
        return max(1, int(base * mult))

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

            dmg = self.calc_damage(atk, tgt, action.ranged)
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
                ret = self.calc_damage(tgt, atk)
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

    def is_over(self) -> bool:
        return len(self.alive(0)) == 0 or len(self.alive(1)) == 0

    def winner(self) -> int:
        if not self.alive(0):
            return 1
        if not self.alive(1):
            return 0
        return -1

    def start_round(self):
        self.round_num += 1
        self.deaths_this_round = 0
        for u in self.alive():
            u.new_round()

"""Battle mechanics: Unit, BattleState, damage calculation, turn execution."""

import random
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from hex_grid import HexGrid
import config


class Unit:
    """A stack of identical creatures on the battlefield."""

    def __init__(self, name: str, team: int, col: int, row: int, **kwargs):
        self.name = name
        self.team = team
        self.col = col
        self.row = row
        self.attack = kwargs["attack"]
        self.defense = kwargs["defense"]
        self.max_hp = kwargs["hp"]
        self.speed = kwargs["speed"]
        self.damage = kwargs["damage"]
        self.is_archer = kwargs["is_archer"]
        self.is_flying = kwargs["is_flying"]
        self.symbol = kwargs.get("symbol", name[0])

        self.count = kwargs["count"]
        self._total_hp = self.count * self.max_hp
        self.is_alive = True
        self.retaliated = False  # can retaliate once per round

    @staticmethod
    def from_type(type_name: str, team: int, col: int, row: int) -> "Unit":
        t = config.UNIT_TYPES[type_name]
        return Unit(type_name, team, col, row, **t)

    # ── properties ──────────────────────────────────────────

    @property
    def pos(self) -> Tuple[int, int]:
        return (self.col, self.row)

    @pos.setter
    def pos(self, value: Tuple[int, int]):
        self.col, self.row = value

    @property
    def hp(self) -> int:
        """HP of the top unit in the stack."""
        if self.count <= 0:
            return 0
        return self._total_hp - (self.count - 1) * self.max_hp

    @property
    def strength(self) -> float:
        """Approximate combat strength (used by AI)."""
        if not self.is_alive:
            return 0
        return (self.attack + self.defense) * self.count * self.damage * self.max_hp / 200.0

    # ── combat ──────────────────────────────────────────────

    def take_damage(self, dmg: int) -> tuple:
        """Apply damage, return (actual_damage, killed_count)."""
        old_count = self.count
        actual = min(dmg, self._total_hp)
        self._total_hp -= actual
        if self._total_hp <= 0:
            self.count = 0
            self.is_alive = False
        else:
            self.count = (self._total_hp + self.max_hp - 1) // self.max_hp
        return actual, old_count - self.count

    def new_round(self):
        self.retaliated = False


# ── Actions ─────────────────────────────────────────────────

class Action:
    pass

class MoveAction(Action):
    def __init__(self, unit: Unit, path: List[Tuple[int, int]]):
        self.unit = unit
        self.path = path

class AttackAction(Action):
    def __init__(self, attacker: Unit, target: Unit,
                 from_pos: Optional[Tuple[int, int]] = None,
                 ranged: bool = False):
        self.attacker = attacker
        self.target = target
        self.from_pos = from_pos  # position to attack from (melee)
        self.ranged = ranged

class SkipAction(Action):
    def __init__(self, unit: Unit):
        self.unit = unit


# ── Battle state ────────────────────────────────────────────

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
                desc += f" → {tgt.name} retaliates: {ret_actual}"
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

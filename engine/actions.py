"""Action types for the battle AI and animation engine."""

from typing import Optional, List, Tuple
from .unit import Unit


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

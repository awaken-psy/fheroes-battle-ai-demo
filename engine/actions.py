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


class CastAction(Action):
    """A hero casts one spell on a target unit (damage / buff / debuff).

    For AOE spells, ``cell`` is the center of the area.
    For Teleport, ``destination`` is where the unit moves to.
    """

    def __init__(self, team: int, spell, target: Unit,
                 cell: Optional[Tuple[int, int]] = None,
                 destination: Optional[Tuple[int, int]] = None):
        self.team = team
        self.spell = spell
        self.target = target
        self.cell = cell
        self.destination = destination


class RetreatAction(Action):
    """The hero of `team` flees — the battle ends and that side loses."""

    def __init__(self, team: int):
        self.team = team

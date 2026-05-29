"""Unit class — a stack of identical creatures on the battlefield."""

import config


class Unit:
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
        self._max_total_hp = self._total_hp
        self.is_alive = True
        self.retaliated = False  # can retaliate once per round

    @staticmethod
    def from_type(type_name: str, team: int, col: int, row: int) -> "Unit":
        t = config.UNIT_TYPES[type_name]
        return Unit(type_name, team, col, row, **t)

    # ── properties ──────────────────────────────────────────

    @property
    def pos(self) -> tuple:
        return (self.col, self.row)

    @pos.setter
    def pos(self, value: tuple):
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

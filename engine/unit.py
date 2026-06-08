"""Unit class — a stack of identical creatures on the battlefield."""

import math

import config

# fheroes2 Speed::AVERAGE — the pivot for the base-strength speed remap.
SPEED_AVERAGE = 4


class Unit:
    def __init__(self, name: str, team: int, col: int, row: int, **kwargs):
        self.name = name
        self.team = team
        self.col = col
        self.row = row
        self.attack = kwargs["attack"]
        self.defense = kwargs["defense"]
        self.max_hp = kwargs["hp"]
        self.base_speed = kwargs["speed"]
        self.damage = kwargs["damage"]
        self.is_archer = kwargs["is_archer"]
        self.is_flying = kwargs["is_flying"]
        self.symbol = kwargs.get("symbol", name[0])

        self.count = kwargs["count"]
        self._total_hp = self.count * self.max_hp
        self._max_total_hp = self._total_hp
        self.is_alive = True
        self.retaliated = False  # can retaliate once per round

        # Active timed spell effects (Haste / Slow / Bless / Curse …).
        self.effects = []

        # Per-type base strength (fheroes2 getMonsterBaseStrength), computed once.
        self._base_strength = self._compute_base_strength()

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
    def speed(self) -> int:
        """Effective speed = base plus active Haste/Slow effects (min 1)."""
        delta = sum(e.speed_delta for e in self.effects)
        return max(1, self.base_speed + delta)

    @property
    def damage_factor(self) -> float:
        """Combined damage multiplier from Bless / Curse effects."""
        factor = 1.0
        for e in self.effects:
            factor *= e.damage_mult
        return factor

    # ── spell effects ───────────────────────────────────────

    def has_effect(self, name: str) -> bool:
        return any(e.name == name for e in self.effects)

    def add_effect(self, effect) -> None:
        """Apply (or refresh) a timed effect; one of each name at a time."""
        self.effects = [e for e in self.effects if e.name != effect.name]
        self.effects.append(effect)

    def tick_effects(self) -> None:
        """Count down effects at the start of a round, dropping expired ones."""
        for e in self.effects:
            e.remaining -= 1
        self.effects = [e for e in self.effects if e.remaining > 0]

    def _compute_base_strength(self) -> float:
        """fheroes2 getMonsterBaseStrength() — depends only on the unit type.

        sqrt(damage * effectiveHP) * special, where special adds bonuses for
        being a shooter / flyer and a speed remap around Speed::AVERAGE.
        The demo has no special abilities, so those multipliers are omitted.
        """
        damage_potential = float(self.damage)
        effective_hp = float(self.max_hp)
        special = 1.0
        if self.is_archer:
            special += 0.4
        if self.is_flying:
            special += 0.3
        diff = self.base_speed - SPEED_AVERAGE
        special += diff * (0.1 if diff < 0 else 0.05)
        return math.sqrt(damage_potential * effective_hp) * special

    @property
    def monster_strength(self) -> float:
        """fheroes2 GetMonsterStrength() — single-creature strength."""
        return (1.0 + 0.1 * self.attack + 0.05 * self.defense) * self._base_strength

    @property
    def strength(self) -> float:
        """fheroes2 Troop::GetStrength() — stack strength (used by the AI)."""
        if not self.is_alive:
            return 0
        return self.monster_strength * self.count

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

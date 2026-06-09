"""Unit class — a stack of identical creatures on the battlefield."""

import math
from typing import Optional

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
        # fheroes2 damage is a per-creature [min, max] range rolled in combat.
        # Backward-compat: a single ``damage`` means min == max (no spread).
        if "damage_min" in kwargs and "damage_max" in kwargs:
            self.damage_min = kwargs["damage_min"]
            self.damage_max = kwargs["damage_max"]
        else:
            self.damage_min = self.damage_max = kwargs["damage"]
        self.is_archer = kwargs["is_archer"]
        self.is_flying = kwargs["is_flying"]
        # Wide units occupy two horizontally-adjacent cells (head + tail).
        # fheroes2: the head always faces the enemy; the tail trails behind.
        # We fix facing by team (team 0 faces right, team 1 faces left) rather
        # than reflecting on backward moves — our AI almost always advances.
        self.is_wide = kwargs.get("is_wide", False)
        self.abilities = set(kwargs.get("abilities", ()))
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
    def from_type(type_name: str, team: int, col: int, row: int,
                  count: int = None) -> "Unit":
        t = dict(config.UNIT_TYPES[type_name])
        if count is not None:
            t["count"] = count
        return Unit(type_name, team, col, row, **t)

    # ── properties ──────────────────────────────────────────

    @property
    def pos(self) -> tuple:
        """The head cell — the enemy-facing front of the unit."""
        return (self.col, self.row)

    @pos.setter
    def pos(self, value: tuple):
        self.col, self.row = value

    @property
    def tail_cell(self) -> Optional[tuple]:
        """The trailing cell for a wide unit, else None.

        Team 0 faces right so the tail sits to the left of the head; team 1
        faces left so the tail sits to the right. Same row in both cases.
        """
        if not self.is_wide:
            return None
        dc = -1 if self.team == 0 else 1
        return (self.col + dc, self.row)

    def occupied_cells(self) -> set:
        """All cells this unit's body occupies — {head} or {head, tail}.

        For single-hex units this is exactly {pos}, so occupancy / collision
        logic is byte-for-byte unchanged from before wide units existed.
        """
        if self.is_wide:
            return {self.pos, self.tail_cell}
        return {self.pos}

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
    def damage_avg(self) -> float:
        """Average per-creature damage — used by expected_damage / strength."""
        return (self.damage_min + self.damage_max) / 2.0

    @property
    def damage_factor(self) -> float:
        """Combined damage multiplier from Bless / Curse effects."""
        factor = 1.0
        for e in self.effects:
            factor *= e.damage_mult
        return factor

    # ── spell effects ───────────────────────────────────────

    def has_ability(self, name: str) -> bool:
        return name in self.abilities

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
        being a shooter / flyer, a speed remap around Speed::AVERAGE, and the
        special-ability terms from getMonsterBaseStrength.
        """
        damage_potential = float(self.damage_avg)
        effective_hp = float(self.max_hp)

        # fheroes2: NO_ENEMY_RETALIATION → effectiveHP *= 1.4
        if "no_enemy_retaliation" in self.abilities:
            effective_hp *= 1.4

        # fheroes2: double attack abilities (mutually exclusive).
        if "double_shooting" in self.abilities:
            damage_potential *= 2
        elif "double_melee" in self.abilities:
            damage_potential *= (2.0 if "no_enemy_retaliation" in self.abilities
                                 else 1.75)

        if "double_damage_to_undead" in self.abilities:
            damage_potential *= 1.15

        if "two_cell_melee" in self.abilities:
            damage_potential *= 1.2

        if "unlimited_retaliation" in self.abilities:
            damage_potential *= 1.25

        special = 1.0
        if self.is_archer:
            special += 0.4
        if self.is_flying:
            special += 0.3
        if "death_gaze" in self.abilities:  # enemy-halving
            special += 1.0
        if "hp_drain" in self.abilities:
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

    def heal(self, amount: int) -> int:
        """Top off living creatures by `amount` (no resurrection). Returns healed."""
        if not self.is_alive:
            return 0
        cap = self.count * self.max_hp          # living capacity, never resurrects
        healed = max(0, min(amount, cap - self._total_hp))
        self._total_hp += healed
        return healed

    def new_round(self):
        self.retaliated = False

"""Hero / commander — casts at most one spell per round.

A minimal stand-in for fheroes2's HeroBase: a spell power (scales damage and
effect duration), a spell-point pool, and a spellbook. Each team may have one.
"""

from typing import List, Optional

from .spells import SPELLS, DEFAULT_SPELLBOOK, Spell


class Hero:
    def __init__(self, power: int = 3, max_spell_points: int = 15,
                 spells: Optional[List[str]] = None, name: str = "Hero"):
        self.name = name
        self.power = power
        self.max_spell_points = max_spell_points
        self.spell_points = max_spell_points
        self.spells = list(spells) if spells is not None else list(DEFAULT_SPELLBOOK)
        self._cast_this_round = False

    @property
    def spellbook(self) -> List[Spell]:
        return [SPELLS[name] for name in self.spells if name in SPELLS]

    def can_cast(self, spell: Spell) -> bool:
        return (not self._cast_this_round
                and self.spell_points >= spell.cost)

    def cast(self, spell: Spell) -> None:
        self.spell_points -= spell.cost
        self._cast_this_round = True

    def reset_round(self) -> None:
        self._cast_this_round = False

    @staticmethod
    def from_config(data: Optional[dict]) -> Optional["Hero"]:
        """Build a Hero from a config dict, or None if absent."""
        if not data:
            return None
        return Hero(
            power=data.get("power", 3),
            max_spell_points=data.get("spell_points", 15),
            spells=data.get("spells"),
            name=data.get("name", "Hero"),
        )

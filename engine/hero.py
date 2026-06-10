"""Hero / commander — casts at most one spell per round.

A minimal stand-in for fheroes2's HeroBase: a spell power (scales damage and
effect duration), a spell-point pool, and a spellbook. Each team may have one.

M7d adds secondary-skill support for the five combat-relevant skills:
Archery, Ballistics, Leadership, Luck, and Wisdom (out-of-scope marker).
"""

from typing import Dict, List, Optional

from .spells import SPELLS, DEFAULT_SPELLBOOK, Spell

# fheroes2 game_static.cpp: secondarySkillValuesPerLevel
# {skill_name: {level: value}} — level 1=Basic, 2=Advanced, 3=Expert.
SKILL_VALUES: Dict[str, Dict[int, int]] = {
    "archery":    {1: 10, 2: 25, 3: 50},
    "ballistics": {1: 0,  2: 0,  3: 0},   # handled specially in catapult
    "leadership": {1: 1,  2: 2,  3: 3},
    "luck":       {1: 1,  2: 2,  3: 3},
    # Wisdom is out-of-scope (no spell-level restriction in demo).
    # Resistance does not exist in HoMM2 (it's a HoMM3 skill).
}


class Hero:
    def __init__(self, power: int = 3, max_spell_points: int = 15,
                 spells: Optional[List[str]] = None, name: str = "Hero",
                 skills: Optional[Dict[str, int]] = None):
        self.name = name
        self.power = power
        self.max_spell_points = max_spell_points
        self.spell_points = max_spell_points
        self.spells = list(spells) if spells is not None else list(DEFAULT_SPELLBOOK)
        self._cast_this_round = False
        # Secondary skills: {skill_name: level} where level 0–3
        # (0=NONE, 1=Basic, 2=Advanced, 3=Expert).
        self.skills: Dict[str, int] = {}
        if skills:
            for k, v in skills.items():
                if k in SKILL_VALUES and 1 <= v <= 3:
                    self.skills[k] = v

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

    # ── secondary skills ───────────────────────────────────

    def get_skill_level(self, skill: str) -> int:
        """Return skill level 0–3 (0 = not learned)."""
        return self.skills.get(skill, 0)

    def get_skill_value(self, skill: str) -> int:
        """Return the numeric value for *skill* at current level.

        Looks up ``SKILL_VALUES``; returns 0 if the hero doesn't have it.
        Ballistics returns 0 (its effects are handled in catapult logic,
        not as a numeric modifier).
        """
        level = self.get_skill_level(skill)
        if level == 0:
            return 0
        vals = SKILL_VALUES.get(skill)
        if vals is None:
            return 0
        return vals.get(level, 0)

    # ── factory ────────────────────────────────────────────

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
            skills=data.get("skills"),
        )

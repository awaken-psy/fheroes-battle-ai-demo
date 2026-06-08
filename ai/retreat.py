"""Retreat decision — flee a hopeless battle.

Reimplemented from fheroes2's retreat/surrender check (ai_battle.cpp Step 2).
The demo has no kingdom / gold / artifacts / hero primary skills, so only the
Retreat branch is modelled (Surrender depends on that economy). The hero
continues to fight while ``myStr * ratio >= enemyStr`` and otherwise retreats.
"""

from .evaluation import AIState

# Difficulty.getArmyStrengthRatioForAIRetreat — higher ratio = more reluctant
# to flee. The AI fights on unless the enemy is many times stronger.
RETREAT_RATIO = {
    "Easy": 100.0 / 6.0,
    "Normal": 100.0 / 7.5,
    "Hard": 100.0 / 8.5,
    "Expert": 100.0 / 8.5,
    "Impossible": 100.0 / 10.0,
}


def retreat_ratio(difficulty: str) -> float:
    return RETREAT_RATIO.get(difficulty, RETREAT_RATIO["Normal"])


def should_retreat(s: AIState, difficulty: str = "Normal") -> bool:
    """True if this army is so outmatched that the hero should flee."""
    if s.enemy_army <= 0:
        return False
    return s.my_army * retreat_ratio(difficulty) < s.enemy_army

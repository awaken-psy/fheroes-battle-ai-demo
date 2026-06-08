"""Classic rule-based battle AI — a faithful reimplementation of fheroes2.

This subpackage holds the original hand-written tactical AI. A future
deep-learning agent lives in ``ai/deep`` and implements the same
:class:`ai.base.AIPlayer` contract.
"""

from .planner import ClassicAI

# Back-compat alias: earlier code referred to the class as ``BattleAI``.
BattleAI = ClassicAI

__all__ = ["ClassicAI", "BattleAI"]

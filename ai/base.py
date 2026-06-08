"""AIPlayer — the pluggable battle-AI contract.

Every battle AI (the faithful rule-based ``ClassicAI`` today, a future
deep-learning ``DeepAI``) implements this interface. Callers (headless runner,
GUI, tests) talk only to this contract via :func:`ai.create_ai`, never to a
concrete class — so swapping the AI is a one-string change.

The three methods mirror fheroes2's per-unit activation order
(retreat -> hero spell -> unit action). A learning agent may answer some of
them trivially (e.g. ``check_retreat`` returning ``None``) and concentrate its
policy in ``decide``; the contract stays forward-compatible.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from engine.actions import Action, CastAction, RetreatAction
from engine.battle_state import BattleState
from engine.unit import Unit


class AIPlayer(ABC):
    """Contract for a pluggable battle AI."""

    @abstractmethod
    def check_retreat(self, battle: BattleState, unit: Unit
                      ) -> Optional[Tuple[Optional[Tuple[CastAction, str]], RetreatAction]]:
        """Before a unit acts, decide whether its hero flees.

        Returns ``(farewell_cast_or_None, RetreatAction)`` when retreating,
        else ``None``.
        """
        raise NotImplementedError

    @abstractmethod
    def maybe_cast_spell(self, battle: BattleState, unit: Unit
                         ) -> Optional[Tuple[CastAction, str]]:
        """Let the unit's hero cast one spell this round.

        Returns ``(CastAction, description)`` or ``None``.
        """
        raise NotImplementedError

    @abstractmethod
    def decide(self, battle: BattleState, unit: Unit) -> Tuple[Action, str]:
        """Choose the unit's own action.

        Returns ``(action, human-readable description)``.
        """
        raise NotImplementedError

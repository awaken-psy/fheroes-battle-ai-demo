"""Battle AI package — pluggable AIs behind a common contract.

Use :func:`create_ai` to build an AI by name; all AIs implement
:class:`AIPlayer`. The faithful rule-based AI lives in ``ai.classic``; future
learning agents go in ``ai.deep``.
"""

from .base import AIPlayer
from .factory import create_ai, register_ai, available_ais

__all__ = ["AIPlayer", "create_ai", "register_ai", "available_ais"]

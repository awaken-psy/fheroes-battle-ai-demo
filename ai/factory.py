"""AI factory — create a battle AI by name, without importing concrete classes.

Callers do ``create_ai("classic")`` (or, in future, ``create_ai("deep")``) and
get back an :class:`~ai.base.AIPlayer`. New AIs register themselves here, so
switching the active AI is a one-string change at the call site.
"""

from typing import Callable, Dict

from ai.base import AIPlayer

# Registry of name -> zero/kwargs constructor returning an AIPlayer.
_REGISTRY: Dict[str, Callable[..., AIPlayer]] = {}


def register_ai(name: str, factory: Callable[..., AIPlayer]) -> None:
    """Register a constructor under ``name`` (overwrites an existing entry)."""
    _REGISTRY[name] = factory


def available_ais() -> list:
    """Names that :func:`create_ai` accepts, sorted."""
    return sorted(_REGISTRY)


def create_ai(kind: str = "classic", **kwargs) -> AIPlayer:
    """Build the AI registered under ``kind``, passing ``kwargs`` through."""
    try:
        factory = _REGISTRY[kind]
    except KeyError:
        raise ValueError(
            f"unknown AI {kind!r}; available: {available_ais()}") from None
    return factory(**kwargs)


# ── built-in registrations ───────────────────────────────────────────
def _make_classic(**kwargs) -> AIPlayer:
    from ai.classic import ClassicAI
    return ClassicAI(**kwargs)


register_ai("classic", _make_classic)

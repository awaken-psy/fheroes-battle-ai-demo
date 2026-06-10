"""Tests for R7 — DeepAI player and make_agent_fn.

Covers:
  1. DeepAI creation and AIPlayer interface compliance
  2. DeepAI.decide() returns valid (Action, description)
  3. Stochastic vs greedy mode
  4. make_agent_fn returns correct action indices
  5. Factory registration: create_ai("deep")
  6. DeepAI vs ClassicAI compatibility (arena)
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.base import AIPlayer
from ai.deep.player import DeepAI, make_agent_fn
from ai.deep.model import BattleNet
from ai.factory import create_ai, available_ais
from ai.env import BattleEnv
from engine.hex_grid import HexGrid
from engine.unit import Unit
from engine.battle_state import BattleState


# ── Fixtures ──────────────────────────────────────────────────────


def _make_battle():
    """Create a simple 1v1 battle for testing."""
    grid = HexGrid()
    units = [
        Unit.from_type("Swordsman", 0, 5, 3),
        Unit.from_type("Swordsman", 1, 5, 6),
    ]
    return BattleState(grid, units, first_team=0, attacker_team=0)


def _simple_env_config():
    return {"units": [("Swordsman", 0, 5, 3), ("Swordsman", 1, 5, 6)]}


# ── DeepAI class tests ───────────────────────────────────────────


class TestDeepAICreation:
    def test_default_creation(self):
        ai = DeepAI()
        assert isinstance(ai.model, BattleNet)
        assert ai.device == "cpu"
        assert ai.stochastic is False

    def test_is_ai_player(self):
        ai = DeepAI()
        assert isinstance(ai, AIPlayer)


class TestDeepAIInterface:
    def test_check_retreat_returns_none(self):
        ai = DeepAI()
        battle = _make_battle()
        unit = battle.alive(0)[0]
        assert ai.check_retreat(battle, unit) is None

    def test_maybe_cast_spell_returns_none(self):
        ai = DeepAI()
        battle = _make_battle()
        unit = battle.alive(0)[0]
        assert ai.maybe_cast_spell(battle, unit) is None

    def test_decide_returns_action_and_description(self):
        ai = DeepAI()
        battle = _make_battle()
        battle.start_round()
        unit = battle.turn_order()[0]
        action, desc = ai.decide(battle, unit)
        assert action is not None
        assert isinstance(desc, str)
        assert "DeepAI" in desc

    def test_decide_greedy_deterministic(self):
        ai = DeepAI()
        battle = _make_battle()
        battle.start_round()
        unit = battle.turn_order()[0]

        results = [ai.decide(battle, unit)[1] for _ in range(5)]
        assert len(set(results)) == 1  # always same action


class TestDeepAIStochastic:
    def test_stochastic_produces_variety(self):
        """Stochastic mode should sometimes produce different actions."""
        ai = DeepAI(stochastic=True)
        battle = _make_battle()
        battle.start_round()
        unit = battle.turn_order()[0]

        results = [ai.decide(battle, unit)[1] for _ in range(20)]
        # With 13566 actions and stochastic, should get >1 unique action
        assert len(set(results)) > 1


# ── make_agent_fn tests ──────────────────────────────────────────


class TestMakeAgentFn:
    def test_returns_callable(self):
        model = BattleNet()
        fn = make_agent_fn(model)
        assert callable(fn)

    def test_returns_int(self):
        model = BattleNet()
        fn = make_agent_fn(model)
        env = BattleEnv(_simple_env_config())
        obs, info = env.reset(seed=42)
        action = fn(obs, info)
        assert isinstance(action, int)
        assert 0 <= action < 13566

    def test_greedy_deterministic(self):
        model = BattleNet()
        fn = make_agent_fn(model)
        env = BattleEnv(_simple_env_config())
        obs, info = env.reset(seed=42)
        actions = [fn(obs, info) for _ in range(5)]
        assert len(set(actions)) == 1

    def test_returns_legal_action(self):
        model = BattleNet()
        fn = make_agent_fn(model)
        env = BattleEnv(_simple_env_config())
        obs, info = env.reset(seed=42)
        action = fn(obs, info)
        legal = np.nonzero(obs["mask"])[0]
        assert action in legal


# ── Factory registration tests ───────────────────────────────────


class TestDeepFactory:
    def test_deep_in_available(self):
        assert "deep" in available_ais()

    def test_create_ai_deep_returns_deep_ai(self):
        ai = create_ai("deep")
        assert isinstance(ai, DeepAI)
        assert isinstance(ai, AIPlayer)

    def test_create_ai_deep_with_kwargs(self):
        ai = create_ai("deep", stochastic=True)
        assert ai.stochastic is True


# ── Arena compatibility ──────────────────────────────────────────


class TestDeepAIArena:
    def test_vs_classic_one_game(self):
        """DeepAI can play a full game vs ClassicAI via eval_vs_classic."""
        from ai.self_play import eval_vs_classic
        model = BattleNet()
        fn = make_agent_fn(model)
        result = eval_vs_classic(
            _simple_env_config(), fn, learning_team=0, games=1, seed=0)
        assert "win_rate" in result
        assert result["games"] == 1

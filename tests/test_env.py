"""Tests for R4 — BattleEnv and self-play utilities."""

import os
import sys
import random

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.env import BattleEnv
from ai.observation import NUM_GRID_CHANNELS, GRID_ROWS, GRID_COLS
from ai.action_space import ACTION_DIM, WAIT_IDX, RETREAT_IDX
from ai.self_play import (run_episode, random_legal_action,
                           eval_vs_random)

# ── Shared fixtures ─────────────────────────────────────────────

BALANCED_CONFIG = {
    "units": [
        ("Swordsman", 0, 1, 2),
        ("Archer",    0, 0, 4),
        ("Ogre Lord", 0, 2, 6),
        ("Swordsman", 1, 9, 2),
        ("Archer",    1, 10, 4),
        ("Ogre Lord", 1, 8, 6),
    ],
}

HERO_CONFIG = {
    "units": [
        ("Swordsman", 0, 1, 4),
        ("Swordsman", 1, 9, 4),
    ],
    "heroes": {
        0: {"power": 5, "spell_points": 30},
        1: {"power": 3, "spell_points": 20},
    },
}

SIEGE_CONFIG = {
    "units": [
        ("Swordsman", 0, 1, 3),
        ("Archer",    0, 0, 5),
        ("Orc Chief", 1, 9, 3),
        ("Orc",       1, 10, 5),
    ],
    "siege": True,
}


def _run_full_episode(config, seed=42):
    """Helper: run a full episode with random legal actions."""
    env = BattleEnv(config)
    obs, info = env.reset(seed=seed)
    steps = 0
    total_reward = 0.0

    while True:
        legal = np.nonzero(obs["mask"])[0]
        action = int(legal[steps % len(legal)])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated or steps > 5000:
            break

    return env, obs, info, steps, total_reward, terminated


# ═══════════════════════════════════════════════════════════════
#  Env construction & spaces
# ═══════════════════════════════════════════════════════════════

class TestEnvConstruction:
    def test_creation(self):
        env = BattleEnv(BALANCED_CONFIG)
        assert env.action_space.n == ACTION_DIM
        assert env.observation_space["grid"].shape == (NUM_GRID_CHANNELS, GRID_ROWS, GRID_COLS)
        assert env.observation_space["global"].shape == (20,)
        assert env.observation_space["mask"].shape == (ACTION_DIM,)

    def test_observation_shapes(self):
        env = BattleEnv(BALANCED_CONFIG)
        obs, info = env.reset(seed=0)
        assert obs["grid"].shape == (NUM_GRID_CHANNELS, GRID_ROWS, GRID_COLS)
        assert obs["grid"].dtype == np.float32
        assert obs["global"].shape == (20,)
        assert obs["global"].dtype == np.float32
        assert obs["mask"].shape == (ACTION_DIM,)
        assert obs["mask"].dtype == np.float32

    def test_legal_mask_has_wait(self):
        env = BattleEnv(BALANCED_CONFIG)
        obs, info = env.reset(seed=0)
        assert obs["mask"][WAIT_IDX] == 1.0

    def test_legal_mask_nonempty(self):
        env = BattleEnv(BALANCED_CONFIG)
        obs, info = env.reset(seed=0)
        assert np.sum(obs["mask"]) > 0

    def test_observation_in_bounds(self):
        env = BattleEnv(BALANCED_CONFIG)
        obs, info = env.reset(seed=0)
        assert np.all(obs["grid"] >= 0.0)
        assert np.all(obs["grid"] <= 1.0)
        assert np.all(obs["global"] >= -1.0)
        assert np.all(obs["global"] <= 1.0)


# ═══════════════════════════════════════════════════════════════
#  Episode lifecycle
# ═══════════════════════════════════════════════════════════════

class TestEpisodeLifecycle:
    def test_step_returns_five_tuple(self):
        env = BattleEnv(BALANCED_CONFIG)
        obs, info = env.reset(seed=42)
        action = int(np.nonzero(obs["mask"])[0][0])
        result = env.step(action)
        assert len(result) == 5
        obs2, reward, terminated, truncated, info2 = result
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert truncated is False

    def test_episode_terminates(self):
        env, obs, info, steps, _, terminated = _run_full_episode(BALANCED_CONFIG)
        assert terminated is True
        assert steps > 0

    def test_terminal_info_has_winner(self):
        env, obs, info, steps, _, terminated = _run_full_episode(BALANCED_CONFIG)
        assert terminated
        assert "winner" in info
        assert info["winner"] in (0, 1)

    def test_terminal_info_has_end_reason(self):
        env, obs, info, steps, _, terminated = _run_full_episode(BALANCED_CONFIG)
        assert "end_reason" in info
        assert info["end_reason"] in ("elim", "retreat", "stalemate", "cap")

    def test_winner_has_alive_units(self):
        env, obs, info, _, _, terminated = _run_full_episode(BALANCED_CONFIG)
        if info["end_reason"] == "elim":
            loser = 1 - info["winner"]
            assert len(env._battle.alive(loser)) == 0
            assert len(env._battle.alive(info["winner"])) > 0

    def test_step_after_end_raises(self):
        env = BattleEnv(BALANCED_CONFIG)
        obs, info = env.reset(seed=42)
        while True:
            legal = np.nonzero(obs["mask"])[0]
            obs, _, terminated, _, info = env.step(int(legal[0]))
            if terminated:
                break
        with pytest.raises(RuntimeError, match="reset"):
            env.step(WAIT_IDX)

    def test_multiple_resets(self):
        env = BattleEnv(BALANCED_CONFIG)
        for seed in range(5):
            obs, info = env.reset(seed=seed)
            assert obs["grid"].shape == (NUM_GRID_CHANNELS, GRID_ROWS, GRID_COLS)
            assert info["round_num"] == 1

    def test_reset_options(self):
        env = BattleEnv(BALANCED_CONFIG)
        obs, info = env.reset(seed=0, options={
            "reward_phase": 3,
            "dense_weight": 0.5,
        })
        assert env._reward_phase == 3
        assert env._dense_weight == 0.5


# ═══════════════════════════════════════════════════════════════
#  Reward computation
# ═══════════════════════════════════════════════════════════════

class TestReward:
    def _collect_rewards(self, config, phase=1, seed=42):
        """Run an episode and collect all rewards."""
        env = BattleEnv(config)
        obs, info = env.reset(seed=seed, options={"reward_phase": phase})
        rewards = []
        while True:
            legal = np.nonzero(obs["mask"])[0]
            action = int(legal[len(rewards) % len(legal)])
            obs, reward, terminated, _, info = env.step(action)
            rewards.append(reward)
            if terminated:
                break
        return rewards, info

    def test_terminal_reward_magnitude(self):
        """Last reward should include ±1 terminal bonus."""
        rewards, info = self._collect_rewards(BALANCED_CONFIG, phase=1)
        # Final reward includes terminal ±1 plus possible dense component
        assert abs(rewards[-1]) >= 1.0

    def test_phase3_sparse_only_intermediate(self):
        """Phase 3: non-terminal rewards should be zero."""
        rewards, info = self._collect_rewards(BALANCED_CONFIG, phase=3)
        # All but last should be 0 (sparse only)
        for r in rewards[:-1]:
            assert r == 0.0

    def test_phase3_terminal_is_plusminus_one(self):
        """Phase 3: terminal reward should be exactly ±1."""
        rewards, info = self._collect_rewards(BALANCED_CONFIG, phase=3)
        assert rewards[-1] in (1.0, -1.0)

    def test_phase1_dense_rewards_nonzero(self):
        """Phase 1: some intermediate rewards should be nonzero."""
        rewards, info = self._collect_rewards(BALANCED_CONFIG, phase=1)
        dense = rewards[:-1]
        # At least some steps should have nonzero dense reward
        assert any(r != 0.0 for r in dense)

    def test_phase2_dense_weight_scales(self):
        """Phase 2: dense component should be scaled by dense_weight."""
        env = BattleEnv(BALANCED_CONFIG)
        obs, info = env.reset(seed=42, options={
            "reward_phase": 2, "dense_weight": 0.0})
        rewards = []
        while True:
            legal = np.nonzero(obs["mask"])[0]
            action = int(legal[len(rewards) % len(legal)])
            obs, r, terminated, _, info = env.step(action)
            rewards.append(r)
            if terminated:
                break
        # With dense_weight=0, only terminal reward remains
        for r in rewards[:-1]:
            assert r == 0.0


# ═══════════════════════════════════════════════════════════════
#  Retreat action
# ═══════════════════════════════════════════════════════════════

class TestRetreat:
    def test_retreat_ends_episode(self):
        env = BattleEnv(HERO_CONFIG)
        obs, info = env.reset(seed=42)
        # Skip cast phase if present
        if info.get("is_cast_phase"):
            obs, _, _, _, info = env.step(WAIT_IDX)
        assert obs["mask"][RETREAT_IDX] == 1.0
        obs, reward, terminated, _, info = env.step(RETREAT_IDX)
        assert terminated is True
        assert info["end_reason"] == "retreat"

    def test_retreat_winner(self):
        env = BattleEnv(HERO_CONFIG)
        obs, info = env.reset(seed=42)
        team = info["current_team"]
        # Skip cast phase if present
        if info.get("is_cast_phase"):
            obs, _, _, _, info = env.step(WAIT_IDX)
        obs, reward, terminated, _, info = env.step(RETREAT_IDX)
        assert terminated
        # Retreating team loses
        assert info["winner"] == 1 - team

    def test_no_hero_no_retreat(self):
        config = {
            "units": [("Swordsman", 0, 1, 4), ("Swordsman", 1, 9, 4)],
        }
        env = BattleEnv(config)
        obs, info = env.reset(seed=0)
        # No heroes → Retreat should be illegal
        assert obs["mask"][RETREAT_IDX] == 0.0


# ═══════════════════════════════════════════════════════════════
#  Config variants
# ═══════════════════════════════════════════════════════════════

class TestConfigVariants:
    def test_siege_episode(self):
        env, obs, info, steps, _, terminated = _run_full_episode(
            SIEGE_CONFIG, seed=42)
        assert terminated
        assert steps > 0

    def test_hero_episode(self):
        env, obs, info, steps, _, terminated = _run_full_episode(
            HERO_CONFIG, seed=42)
        assert terminated
        # With heroes, spell actions should be legal at some point
        assert steps > 0

    def test_siege_castle_exists(self):
        env = BattleEnv(SIEGE_CONFIG)
        obs, info = env.reset(seed=0)
        assert env._battle.castle is not None

    def test_morale_config(self):
        config = {
            "units": [("Swordsman", 0, 1, 4), ("Swordsman", 1, 9, 4)],
            "morale": {0: 3, 1: -3},
        }
        env, _, info, steps, _, terminated = _run_full_episode(config, seed=42)
        assert terminated


# ═══════════════════════════════════════════════════════════════
#  Gymnasium compliance
# ═══════════════════════════════════════════════════════════════

class TestGymnasiumCompliance:
    def test_env_make_directly(self):
        env = BattleEnv(BALANCED_CONFIG)
        obs, info = env.reset(seed=0)
        assert "grid" in obs
        assert "global" in obs
        assert "mask" in obs

    def test_space_contains_obs(self):
        env = BattleEnv(BALANCED_CONFIG)
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs)

    def test_action_in_space(self):
        env = BattleEnv(BALANCED_CONFIG)
        obs, info = env.reset(seed=0)
        legal = np.nonzero(obs["mask"])[0]
        for idx in legal[:5]:
            assert env.action_space.contains(idx)


# ═══════════════════════════════════════════════════════════════
#  Self-play utilities
# ═══════════════════════════════════════════════════════════════

class TestSelfPlay:
    def test_run_episode(self):
        result = run_episode(
            BALANCED_CONFIG,
            agent_fn=random_legal_action,
            seed=42,
        )
        assert result["winner"] in (0, 1)
        assert result["steps"] > 0
        assert result["rounds"] > 0

    def test_eval_vs_random(self):
        result = eval_vs_random(
            BALANCED_CONFIG,
            agent_fn=random_legal_action,
            learning_team=0,
            games=10,
            seed=0,
        )
        assert result["games"] == 10
        assert 0.0 <= result["win_rate"] <= 1.0
        assert result["wins"] <= result["games"]

    def test_run_episode_reward_phases(self):
        for phase in (1, 2, 3):
            result = run_episode(
                BALANCED_CONFIG,
                agent_fn=random_legal_action,
                reward_phase=phase,
                dense_weight=0.5,
                seed=42,
            )
            assert result["winner"] in (0, 1)
            assert result["steps"] > 0

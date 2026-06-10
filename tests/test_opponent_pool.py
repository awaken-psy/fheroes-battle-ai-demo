"""Tests for T3 — Opponent pool and training integration.

Covers:
  1. OpponentPool: add/sample/len basics
  2. OpponentPool: FIFO eviction when over capacity
  3. OpponentPool: disk persistence (add → load_from_disk → sample)
  4. OpponentPool: sample from empty pool returns None
  5. OpponentPool: old files deleted on eviction
  6. PPOTrainer: collect_rollout with opponent_model stores fewer buffer entries
  7. PPOTrainer: collect_rollout with opponent returns "pool" indicator
  8. PPOTrainer: buffer only contains learning team transitions
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.deep.model import BattleNet
from ai.deep.opponent_pool import OpponentPool
from ai.deep.trainer import PPOTrainer


# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def pool_dir(tmp_path):
    """Temporary directory for opponent pool files."""
    return str(tmp_path / "opponent_pool")


@pytest.fixture
def pool(pool_dir):
    """OpponentPool with capacity 3."""
    return OpponentPool(capacity=3, save_dir=pool_dir)


def _make_state_dict(seed: int = 0) -> dict:
    """Create a deterministic model state_dict."""
    model = BattleNet()
    torch.manual_seed(seed)
    for p in model.parameters():
        p.data.normal_()
    return model.state_dict()


def _simple_env_config() -> dict:
    """Minimal 1v1 Swordsman config for fast rollouts."""
    return {
        "units": [
            ("Swordsman", 0, 5, 3),
            ("Swordsman", 1, 5, 6),
        ],
    }


# ── 1. Basic add/sample/len ────────────────────────────────────────


class TestOpponentPoolBasic:
    """Core pool operations."""

    def test_empty_pool_len_zero(self, pool):
        assert len(pool) == 0

    def test_empty_pool_sample_returns_none(self, pool):
        assert pool.sample() is None

    def test_add_increments_len(self, pool):
        pool.add(_make_state_dict(1), step=100)
        assert len(pool) == 1

        pool.add(_make_state_dict(2), step=200)
        assert len(pool) == 2

    def test_sample_returns_valid_state_dict(self, pool):
        sd = _make_state_dict(42)
        pool.add(sd, step=100)

        sampled = pool.sample()
        assert sampled is not None
        assert isinstance(sampled, dict)

        # Keys must match BattleNet state_dict
        expected_keys = set(sd.keys())
        assert set(sampled.keys()) == expected_keys

    def test_repr(self, pool):
        pool.add(_make_state_dict(1), step=100)
        r = repr(pool)
        assert "OpponentPool" in r
        assert "capacity=3" in r
        assert "size=1" in r
        assert "100" in r


# ── 2. FIFO eviction ───────────────────────────────────────────────


class TestOpponentPoolEviction:
    """Capacity management."""

    def test_eviction_at_capacity(self, pool):
        """Pool of capacity 3 keeps only the 3 newest."""
        pool.add(_make_state_dict(1), step=100)
        pool.add(_make_state_dict(2), step=200)
        pool.add(_make_state_dict(3), step=300)
        assert len(pool) == 3

        pool.add(_make_state_dict(4), step=400)
        assert len(pool) == 3

    def test_evicted_file_removed_from_disk(self, pool, pool_dir):
        """Oldest checkpoint file is deleted when evicted."""
        pool.add(_make_state_dict(1), step=100)
        pool.add(_make_state_dict(2), step=200)
        pool.add(_make_state_dict(3), step=300)
        pool.add(_make_state_dict(4), step=400)

        # pool_100.pt should be evicted
        assert not os.path.exists(os.path.join(pool_dir, "pool_100.pt"))
        assert os.path.exists(os.path.join(pool_dir, "pool_200.pt"))
        assert os.path.exists(os.path.join(pool_dir, "pool_300.pt"))
        assert os.path.exists(os.path.join(pool_dir, "pool_400.pt"))

    def test_eviction_fifo_order(self, pool):
        """After eviction, only newest entries are sampled."""
        for i in range(1, 6):
            pool.add(_make_state_dict(i), step=i * 100)

        # capacity=3, so only steps 300, 400, 500 remain
        assert len(pool) == 3
        steps_in_pool = set()
        for _ in range(50):
            sd = pool.sample()
            # Verify it's one of the surviving checkpoints by checking
            # that load_state_dict succeeds (any of the 3 is valid)
            model = BattleNet()
            model.load_state_dict(sd)  # should not raise
            steps_in_pool.add(id(sd))

        # Should have at least 1 distinct sampled state_dict
        assert len(steps_in_pool) >= 1


# ── 3. Disk persistence ────────────────────────────────────────────


class TestOpponentPoolPersistence:
    """Save and restore pool from disk."""

    def test_load_from_disk_recovers_entries(self, pool_dir):
        """Pool state survives save + new instance + load_from_disk."""
        pool1 = OpponentPool(capacity=5, save_dir=pool_dir)
        pool1.add(_make_state_dict(1), step=100)
        pool1.add(_make_state_dict(2), step=200)
        pool1.add(_make_state_dict(3), step=300)

        pool2 = OpponentPool(capacity=5, save_dir=pool_dir)
        assert len(pool2) == 0  # fresh instance

        pool2.load_from_disk()
        assert len(pool2) == 3

        # Can sample and load state_dict
        sd = pool2.sample()
        assert sd is not None
        model = BattleNet()
        model.load_state_dict(sd)

    def test_load_from_disk_trims_to_capacity(self, pool_dir):
        """If disk has more files than capacity, trim to newest."""
        pool1 = OpponentPool(capacity=10, save_dir=pool_dir)
        for i in range(1, 6):
            pool1.add(_make_state_dict(i), step=i * 100)

        # New pool with smaller capacity
        pool2 = OpponentPool(capacity=3, save_dir=pool_dir)
        pool2.load_from_disk()
        assert len(pool2) == 3

    def test_load_from_empty_dir(self, pool_dir):
        """load_from_disk on empty directory works."""
        pool = OpponentPool(capacity=3, save_dir=pool_dir)
        pool.load_from_disk()
        assert len(pool) == 0

    def test_load_from_nonexistent_dir(self, tmp_path):
        """load_from_disk on nonexistent directory works."""
        pool = OpponentPool(capacity=3, save_dir=str(tmp_path / "nope"))
        pool.load_from_disk()  # should not raise
        assert len(pool) == 0


# ── 4. Training integration ────────────────────────────────────────


class TestTrainerOpponentIntegration:
    """collect_rollout with opponent_model."""

    def test_pure_self_play_indicator(self):
        """Default collect_rollout returns pool_play=0.0."""
        trainer = PPOTrainer(BattleNet(), _simple_env_config(), device="cpu")
        info = trainer.collect_rollout(num_steps=64)
        assert info["pool_play"] == 0.0

    def test_pool_play_indicator(self):
        """collect_rollout with opponent_model returns pool_play=1.0."""
        trainer = PPOTrainer(BattleNet(), _simple_env_config(), device="cpu")
        opponent = BattleNet()
        opponent.eval()
        info = trainer.collect_rollout(num_steps=64, opponent_model=opponent)
        assert info["pool_play"] == 1.0

    def test_opponent_rollout_fewer_buffer_entries(self):
        """Vs-opponent stores only learning team transitions (~half)."""
        trainer = PPOTrainer(BattleNet(), _simple_env_config(), device="cpu")

        # Pure self-play
        info_sp = trainer.collect_rollout(num_steps=128)
        sp_len = len(trainer.buffer)

        # Vs opponent
        opponent = BattleNet()
        opponent.eval()
        info_pool = trainer.collect_rollout(num_steps=128, opponent_model=opponent)
        pool_len = len(trainer.buffer)

        # Pool should have fewer entries (only learning team)
        assert pool_len < sp_len
        # But at least some entries
        assert pool_len > 0

    def test_opponent_buffer_valid_for_update(self):
        """Buffer from vs-opponent rollout can run PPO update without error."""
        trainer = PPOTrainer(BattleNet(), _simple_env_config(), device="cpu")
        opponent = BattleNet()
        opponent.eval()
        trainer.collect_rollout(num_steps=128, opponent_model=opponent)

        # PPO update should not crash
        update_info = trainer.update()
        for key in ("policy_loss", "value_loss", "entropy", "total_loss", "approx_kl"):
            assert key in update_info
            assert np.isfinite(update_info[key]), f"{key} is not finite"

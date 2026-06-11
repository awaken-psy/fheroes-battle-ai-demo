"""Tests for T9b — ReplayBuffer and its integration with PPOTrainer."""

import torch
import pytest

from ai.deep.replay_buffer import ReplayBuffer
from ai.deep.trainer import PPOTrainer


# ── Helper ──────────────────────────────────────────────────────


def _make_rollout(steps: int = 64, obs_dim: int = 10):
    """Create a fake rollout dict for testing."""
    return {
        "grid": torch.randn(steps, 9, 11, obs_dim),
        "global": torch.randn(steps, 20),
        "mask": torch.ones(steps, 13566),
        "actions": torch.randint(0, 13566, (steps,)),
        "rewards": torch.randn(steps),
        "log_probs": torch.randn(steps),
        "dones": torch.zeros(steps),
    }


def _make_trainer(replay_buffer=None):
    """Create a PPOTrainer with a minimal env config."""
    from ai.deep.model import BattleNet
    from ai.deep.pipeline import default_battle_config
    model = BattleNet()
    return PPOTrainer(
        model, default_battle_config(),
        replay_buffer=replay_buffer,
    )


# ── ReplayBuffer unit tests ─────────────────────────────────────


class TestReplayBuffer:
    """Unit tests for the ReplayBuffer ring buffer."""

    def test_add_and_len(self):
        buf = ReplayBuffer(capacity=5)
        assert len(buf) == 0
        buf.add(_make_rollout())
        assert len(buf) == 1
        buf.add(_make_rollout())
        assert len(buf) == 2

    def test_fifo_eviction(self):
        buf = ReplayBuffer(capacity=3)
        r1 = _make_rollout()
        r2 = _make_rollout()
        r3 = _make_rollout()
        r4 = _make_rollout()
        buf.add(r1)
        buf.add(r2)
        buf.add(r3)
        assert len(buf) == 3
        # Adding 4th should evict r1
        buf.add(r4)
        assert len(buf) == 3

    def test_sample_returns_valid_data(self):
        buf = ReplayBuffer(capacity=5)
        rollout = _make_rollout(steps=32)
        buf.add(rollout)
        sampled = buf.sample()
        assert isinstance(sampled, dict)
        assert "grid" in sampled
        assert "actions" in sampled
        assert "rewards" in sampled
        assert sampled["grid"].shape == (32, 9, 11, 10)

    def test_sample_from_empty_raises(self):
        buf = ReplayBuffer(capacity=5)
        with pytest.raises(IndexError, match="empty"):
            buf.sample()

    def test_stored_on_cpu(self):
        buf = ReplayBuffer(capacity=5)
        rollout = {
            "grid": torch.randn(10, 9, 11, 10).cuda() if torch.cuda.is_available()
            else torch.randn(10, 9, 11, 10),
            "actions": torch.randint(0, 100, (10,)),
            "rewards": torch.randn(10),
            "log_probs": torch.randn(10),
            "dones": torch.zeros(10),
            "global": torch.randn(10, 20),
            "mask": torch.ones(10, 13566),
        }
        buf.add(rollout)
        sampled = buf.sample()
        assert sampled["grid"].device == torch.device("cpu")
        assert sampled["actions"].device == torch.device("cpu")

    def test_repr(self):
        buf = ReplayBuffer(capacity=10)
        assert "capacity=10" in repr(buf)
        assert "stored=0" in repr(buf)

    def test_sample_is_random(self):
        """Verify that sample returns different rollouts over many calls."""
        buf = ReplayBuffer(capacity=10)
        for i in range(5):
            r = _make_rollout(steps=8)
            r["rewards"] = torch.full((8,), float(i))
            buf.add(r)
        sampled_rewards = set()
        for _ in range(100):
            s = buf.sample()
            sampled_rewards.add(round(s["rewards"][0].item()))
        # Should have seen at least 3 different rollouts in 100 samples
        assert len(sampled_rewards) >= 3

    def test_capacity_one(self):
        buf = ReplayBuffer(capacity=1)
        buf.add(_make_rollout(steps=16))
        buf.add(_make_rollout(steps=32))
        assert len(buf) == 1
        assert buf.sample()["grid"].shape[0] == 32


# ── PPOTrainer integration tests ────────────────────────────────


class TestReplayTrainerIntegration:
    """Integration tests for replay buffer with PPOTrainer."""

    def test_trainer_accepts_replay_buffer(self):
        buf = ReplayBuffer(capacity=5)
        trainer = _make_trainer(replay_buffer=buf)
        assert trainer.replay_buffer is buf

    def test_trainer_without_replay_still_works(self):
        trainer = _make_trainer(replay_buffer=None)
        info = trainer.train_step(num_steps=8)
        assert "policy_loss" in info
        assert "steps" in info

    def test_replay_buffer_fills_during_training(self):
        buf = ReplayBuffer(capacity=5)
        trainer = _make_trainer(replay_buffer=buf)
        assert len(buf) == 0
        trainer.train_step(num_steps=8)
        assert len(buf) == 1
        trainer.train_step(num_steps=8)
        assert len(buf) == 2

    def test_replay_buffer_fifo_during_training(self):
        buf = ReplayBuffer(capacity=2)
        trainer = _make_trainer(replay_buffer=buf)
        trainer.train_step(num_steps=8)
        trainer.train_step(num_steps=8)
        assert len(buf) == 2
        trainer.train_step(num_steps=8)
        assert len(buf) == 2  # FIFO eviction

    def test_replay_mixed_update_runs(self):
        """End-to-end: train a few steps with replay, verify no crash."""
        buf = ReplayBuffer(capacity=10)
        trainer = _make_trainer(replay_buffer=buf)
        # Collect 3 rollouts to fill buffer
        for _ in range(3):
            trainer.train_step(num_steps=16)
        # 4th step should mix replay data
        info = trainer.train_step(num_steps=16)
        assert "policy_loss" in info
        assert "entropy" in info
        assert isinstance(info["policy_loss"], float)

    def test_replay_data_not_on_gpu(self):
        """Verify replay buffer data stays on CPU after training."""
        buf = ReplayBuffer(capacity=5)
        trainer = _make_trainer(replay_buffer=buf)
        trainer.train_step(num_steps=8)
        trainer.train_step(num_steps=8)
        sampled = buf.sample()
        for v in sampled.values():
            assert v.device == torch.device("cpu")

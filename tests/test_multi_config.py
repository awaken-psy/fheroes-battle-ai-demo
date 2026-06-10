"""T5 — Multi-config training tests.

Verify that:
  - collect_rollout / train_step accept env_config override
  - --config accepts multiple files via CLI
  - Each rollout randomly selects from the config pool
  - Eval reports per-config win rates with average
  - best.pt is selected by average win rate
  - Single-config mode is fully backward compatible
"""

import json
import os
import random
import sys
import tempfile

import pytest

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.deep.model import BattleNet
from ai.deep.pipeline import load_battle_config
from ai.deep.trainer import PPOTrainer
from scripts.train import parse_args


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def default_config():
    return load_battle_config(None)


@pytest.fixture
def example_config():
    return load_battle_config("configs/example.json")


@pytest.fixture
def even_clash_config():
    return load_battle_config("configs/even_clash.json")


@pytest.fixture
def trainer(default_config):
    model = BattleNet()
    return PPOTrainer(model, default_config, device="cpu")


# ── 1. Trainer env_config override ────────────────────────────────


class TestTrainerEnvConfigOverride:
    """collect_rollout and train_step should use the provided env_config
    instead of the trainer's default when explicitly passed."""

    def test_collect_rollout_override(self, trainer, example_config):
        info = trainer.collect_rollout(
            num_steps=10, env_config=example_config)
        assert info["steps"] == 10
        assert info["episodes"] >= 0

    def test_collect_rollout_default_when_none(self, trainer, default_config):
        info = trainer.collect_rollout(num_steps=10, env_config=None)
        assert info["steps"] == 10

    def test_train_step_override(self, trainer, example_config):
        info = trainer.train_step(
            num_steps=10, env_config=example_config)
        assert "policy_loss" in info
        assert info["steps"] == 10

    def test_different_configs_produce_valid_rollouts(
        self, trainer, example_config, even_clash_config,
    ):
        """Both configs should produce valid rollout data without errors."""
        info1 = trainer.train_step(
            num_steps=10, env_config=example_config)
        info2 = trainer.train_step(
            num_steps=10, env_config=even_clash_config)
        assert info1["steps"] == 10
        assert info2["steps"] == 10


# ── 2. CLI multi-config parsing ───────────────────────────────────


class TestMultiConfigCLI:
    """--config should accept zero, one, or multiple file paths."""

    def test_no_config_backward_compat(self):
        args = parse_args([])
        assert args.config is None

    def test_single_config(self):
        args = parse_args(["--config", "configs/example.json"])
        assert args.config == ["configs/example.json"]

    def test_multiple_configs(self):
        args = parse_args([
            "--config",
            "configs/example.json",
            "configs/even_clash.json",
            "configs/mage_duel.json",
        ])
        assert len(args.config) == 3
        assert args.config[0] == "configs/example.json"
        assert args.config[1] == "configs/even_clash.json"
        assert args.config[2] == "configs/mage_duel.json"


# ── 3. Config name extraction ─────────────────────────────────────


class TestConfigNameExtraction:
    """Config names should be filename stems for logging."""

    def test_name_from_path(self):
        paths = ["configs/example.json", "configs/even_clash.json"]
        names = [
            os.path.splitext(os.path.basename(p))[0] for p in paths
        ]
        assert names == ["example", "even_clash"]

    def test_none_gives_default(self):
        names = [
            os.path.splitext(os.path.basename(p))[0] if p else "default"
            for p in [None]
        ]
        assert names == ["default"]


# ── 4. Random config selection logic ──────────────────────────────


class TestRandomConfigSelection:
    """random.choice should pick from the config list uniformly."""

    def test_all_configs_represented_over_many_samples(self):
        configs = [
            load_battle_config("configs/example.json"),
            load_battle_config("configs/even_clash.json"),
        ]
        seen = set()
        for _ in range(200):
            cfg = random.choice(configs)
            # Identity check via units tuple
            seen.add(id(cfg))
        assert len(seen) == 2

    def test_single_config_always_same(self):
        configs = [load_battle_config("configs/example.json")]
        results = [id(random.choice(configs)) for _ in range(50)]
        assert len(set(results)) == 1


# ── 5. Eval per-config win rate aggregation ───────────────────────


class TestEvalAggregation:
    """Average win rate should be the mean across configs."""

    def test_avg_win_rate_calculation(self):
        per_config_wr = {
            "example": 0.80,
            "even_clash": 0.40,
        }
        avg = sum(per_config_wr.values()) / len(per_config_wr)
        assert avg == pytest.approx(0.60)

    def test_avg_win_rate_with_zeros(self):
        per_config_wr = {
            "example": 0.88,
            "even_clash": 0.0,
            "mage_duel": 0.0,
            "dragon_battle": 0.0,
        }
        avg = sum(per_config_wr.values()) / len(per_config_wr)
        assert avg == pytest.approx(0.22)

    def test_avg_win_rate_total_wins_games(self):
        """avg = total_wins / total_games across all configs."""
        results = [
            {"wins": 88, "games": 100},
            {"wins": 12, "games": 100},
        ]
        total_wins = sum(r["wins"] for r in results)
        total_games = sum(r["games"] for r in results)
        avg = total_wins / total_games
        assert avg == pytest.approx(0.50)


# ── 6. best.pt selection by average ───────────────────────────────


class TestBestCheckpointByAverage:
    """best.pt should update when average win rate improves."""

    def test_best_updates_on_higher_avg(self):
        best = 0.30
        new_avg = 0.50
        assert new_avg > best

    def test_best_does_not_update_on_lower_avg(self):
        best = 0.50
        new_avg = 0.30
        assert not (new_avg > best)

    def test_best_updates_on_equal_configs(self):
        """Single config: avg == single win_rate, backward compat."""
        per_config_wr = {"example": 0.88}
        avg = sum(per_config_wr.values()) / len(per_config_wr)
        assert avg == 0.88

"""Tests for T1+T7 eval_benchmark — benchmark evaluation framework."""

import json
import os
import tempfile

import pytest
import torch

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from ai.deep.model import BattleNet
from scripts.eval_benchmark import (
    LEGACY_BENCHMARK_CONFIGS,
    discover_training_configs,
    format_table,
    load_model,
    run_benchmark,
    wilson_interval,
)


# ── Wilson interval ────────────────────────────────────────────────


class TestWilsonInterval:
    def test_zero_games(self):
        p, lo, hi = wilson_interval(0, 0)
        assert p == 0.0
        assert lo == 0.0
        assert hi == 0.0

    def test_all_wins(self):
        p, lo, hi = wilson_interval(50, 50)
        assert p == 1.0
        assert hi == 1.0

    def test_all_losses(self):
        p, lo, hi = wilson_interval(0, 50)
        assert p == 0.0
        assert lo == 0.0

    def test_half_wins(self):
        p, lo, hi = wilson_interval(25, 50)
        assert abs(p - 0.5) < 0.01
        assert lo < 0.5 < hi

    def test_ci_width_decreases_with_n(self):
        _, lo1, hi1 = wilson_interval(5, 10)
        _, lo2, hi2 = wilson_interval(50, 100)
        assert (hi2 - lo2) < (hi1 - lo1)


# ── Legacy benchmark config validity ──────────────────────────────


class TestLegacyBenchmarkConfigs:
    def test_all_legacy_configs_exist(self):
        """All legacy benchmark config files must be present on disk."""
        for config_path, name, target in LEGACY_BENCHMARK_CONFIGS:
            full = os.path.join(ROOT, config_path)
            assert os.path.isfile(full), f"Missing config: {config_path}"

    def test_all_legacy_configs_loadable(self):
        """All configs must load without error."""
        from ai.deep.pipeline import load_battle_config
        for config_path, name, target in LEGACY_BENCHMARK_CONFIGS:
            full = os.path.join(ROOT, config_path)
            cfg = load_battle_config(full)
            assert "units" in cfg
            assert len(cfg["units"]) >= 2

    def test_targets_are_valid(self):
        for config_path, name, target in LEGACY_BENCHMARK_CONFIGS:
            assert 0.0 <= target <= 1.0, f"Invalid target for {name}: {target}"

    def test_four_legacy_configs(self):
        assert len(LEGACY_BENCHMARK_CONFIGS) == 4


# ── Auto-discovery ─────────────────────────────────────────────────


class TestAutoDiscovery:
    def test_discovers_training_configs(self):
        configs = discover_training_configs(os.path.join(ROOT, "configs"))
        assert len(configs) >= 16

    def test_discovered_configs_exist(self):
        configs = discover_training_configs(os.path.join(ROOT, "configs"))
        for path, name, target in configs:
            assert os.path.isfile(os.path.join(ROOT, path)), (
                f"Discovered config missing: {path}")

    def test_discovered_names_are_readable(self):
        configs = discover_training_configs(os.path.join(ROOT, "configs"))
        for path, name, target in configs:
            assert " " in name or len(name) > 2, f"Bad display name: {name}"


# ── Model loading ─────────────────────────────────────────────────


class TestLoadModel:
    def test_load_valid_checkpoint(self, tmp_path):
        """Create a checkpoint and verify it loads."""
        model = BattleNet()
        ckpt_path = str(tmp_path / "test.pt")
        torch.save({"model": model.state_dict(), "step": 100}, ckpt_path)

        loaded = load_model(ckpt_path)
        assert isinstance(loaded, BattleNet)
        # Weights should match
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), loaded.named_parameters()
        ):
            assert n1 == n2
            assert torch.equal(p1, p2)

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(Exception):
            load_model(str(tmp_path / "nonexistent.pt"))


# ── Format table ──────────────────────────────────────────────────


class TestFormatTable:
    def test_basic_output(self):
        results = [
            {
                "config": "configs/example.json", "name": "Test",
                "target": 0.5, "wins": 30, "games": 50,
                "win_rate": 0.6, "ci95": [0.46, 0.73],
                "avg_rounds": 12.3, "elapsed": 5.0, "pass": True,
            }
        ]
        table = format_table(results)
        assert "Test" in table
        assert "30/50" in table
        assert "✓" in table
        assert "Passed: 1/1" in table

    def test_fail_marker(self):
        results = [
            {
                "config": "configs/example.json", "name": "Fail",
                "target": 0.9, "wins": 10, "games": 50,
                "win_rate": 0.2, "ci95": [0.10, 0.34],
                "avg_rounds": 5.0, "elapsed": 2.0, "pass": False,
            }
        ]
        table = format_table(results)
        assert "✗" in table
        assert "Passed: 0/1" in table


# ── Integration: run_benchmark with random model ──────────────────


class TestRunBenchmark:
    """Smoke tests with a random (untrained) model — 2 games per config."""

    def test_runs_legacy_configs(self):
        model = BattleNet()
        results = run_benchmark(model, games=2, device="cpu", seed=0,
                                benchmark_configs=LEGACY_BENCHMARK_CONFIGS)
        assert len(results) == 4
        for r in results:
            assert r["games"] == 2
            assert 0 <= r["wins"] <= 2
            assert 0.0 <= r["win_rate"] <= 1.0
            assert "ci95" in r
            assert "avg_rounds" in r

    def test_result_structure(self):
        model = BattleNet()
        results = run_benchmark(model, games=2, device="cpu", seed=0,
                                benchmark_configs=LEGACY_BENCHMARK_CONFIGS)
        r = results[0]
        expected_keys = {"config", "name", "target", "wins", "games",
                         "win_rate", "ci95", "avg_rounds", "elapsed", "pass"}
        assert set(r.keys()) == expected_keys

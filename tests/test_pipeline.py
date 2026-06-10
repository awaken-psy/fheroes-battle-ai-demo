"""Tests for R7 — Training pipeline utilities.

Covers:
  1. Config loading: default, JSON file, tuple pass-through, heroes
  2. Curriculum scheduling: phase transitions and dense_weight decay
  3. Checkpoint save/load round-trip
  4. End-to-end short training run via subprocess
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.deep.model import BattleNet
from ai.deep.pipeline import (
    default_battle_config,
    get_curriculum_phase,
    load_battle_config,
    load_checkpoint,
    save_checkpoint,
)
from ai.deep.trainer import PPOTrainer


# ── Config loading ──────────────────────────────────────────────


class TestDefaultConfig:
    def test_returns_dict_with_units(self):
        cfg = default_battle_config()
        assert isinstance(cfg, dict)
        assert "units" in cfg

    def test_has_two_units(self):
        cfg = default_battle_config()
        assert len(cfg["units"]) == 2


class TestLoadConfig:
    def test_none_returns_default(self):
        cfg = load_battle_config(None)
        assert cfg == default_battle_config()

    def test_loads_json_dict_units(self):
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "units": [
                    {"type": "Archer", "team": 0, "col": 1, "row": 2,
                     "count": 5},
                    {"type": "Orc", "team": 1, "col": 8, "row": 4},
                ],
            }, f)
            f.flush()
            cfg = load_battle_config(f.name)
        os.unlink(f.name)

        assert cfg["units"][0] == ("Archer", 0, 1, 2, 5)
        assert cfg["units"][1] == ("Orc", 1, 8, 4)

    def test_preserves_list_units(self):
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "units": [["Swordsman", 0, 5, 3]],
            }, f)
            f.flush()
            cfg = load_battle_config(f.name)
        os.unlink(f.name)
        assert cfg["units"][0] == ["Swordsman", 0, 5, 3]

    def test_preserves_heroes(self):
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "units": [{"type": "Pikeman", "team": 0, "col": 1, "row": 2}],
                "heroes": {"0": {"name": "Knight", "power": 3}},
            }, f)
            f.flush()
            cfg = load_battle_config(f.name)
        os.unlink(f.name)
        assert "heroes" in cfg
        assert cfg["heroes"]["0"]["name"] == "Knight"


# ── Curriculum scheduling ───────────────────────────────────────


class TestCurriculumPhase:
    def test_phase1_at_zero(self):
        phase, w = get_curriculum_phase(0, 100, 300)
        assert phase == 1
        assert w == 1.0

    def test_phase1_just_before_boundary(self):
        phase, w = get_curriculum_phase(99, 100, 300)
        assert phase == 1
        assert w == 1.0

    def test_phase2_at_boundary(self):
        phase, w = get_curriculum_phase(150, 100, 300)
        assert phase == 2
        assert 0.0 < w < 1.0

    def test_phase2_decays(self):
        _, w1 = get_curriculum_phase(100, 100, 300)
        _, w2 = get_curriculum_phase(200, 100, 300)
        assert w2 < w1

    def test_phase3_at_boundary(self):
        phase, w = get_curriculum_phase(300, 100, 300)
        assert phase == 3
        assert w == 0.0

    def test_phase3_well_beyond(self):
        phase, w = get_curriculum_phase(999_999, 100, 300)
        assert phase == 3
        assert w == 0.0


# ── Checkpoint management ───────────────────────────────────────


class TestCheckpoint:
    def _make_trainer(self):
        model = BattleNet()
        cfg = {"units": [("Swordsman", 0, 5, 3), ("Swordsman", 1, 5, 6)]}
        return PPOTrainer(model, cfg, lr=1e-3, minibatch_size=8,
                          update_epochs=1, device="cpu")

    def test_save_load_roundtrip(self):
        trainer = self._make_trainer()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.pt")
            save_checkpoint(trainer, step=123, path=path)
            assert os.path.exists(path)

            loaded_step = load_checkpoint(trainer, path)
            assert loaded_step == 123

    def test_save_creates_subdirectories(self):
        trainer = self._make_trainer()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "sub", "dir", "ckpt.pt")
            save_checkpoint(trainer, step=0, path=path)
            assert os.path.exists(path)

    def test_weights_preserved_across_roundtrip(self):
        trainer = self._make_trainer()
        w_before = {k: v.clone()
                    for k, v in trainer.model.state_dict().items()}

        # Train a bit
        trainer.train_step(num_steps=10, seed=0)
        w_trained = {k: v.clone()
                     for k, v in trainer.model.state_dict().items()}

        # Weights must differ after training
        assert not torch.equal(w_before["stem_conv.weight"],
                               w_trained["stem_conv.weight"])

        # Save → fresh model → load
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.pt")
            save_checkpoint(trainer, step=10, path=path)

            trainer2 = self._make_trainer()
            load_checkpoint(trainer2, path)

            for k in w_trained:
                assert torch.equal(
                    w_trained[k], trainer2.model.state_dict()[k])


# ── End-to-end training script ──────────────────────────────────


class TestTrainScript:
    def test_short_run(self, tmp_path):
        """train.py runs end-to-end for a few steps and produces JSON."""
        project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(
            [sys.executable, "scripts/train.py",
             "--total-steps", "100",
             "--rollout-steps", "50",
             "--eval-interval", "50",
             "--eval-games", "1",
             "--checkpoint-dir", str(tmp_path / "ckpts")],
            capture_output=True, text=True, timeout=180,
            cwd=project_root,
        )
        assert result.returncode == 0, (
            f"train.py failed:\nstdout: {result.stdout}\n"
            f"stderr: {result.stderr}")

        lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
        assert len(lines) >= 1
        first = json.loads(lines[0])
        assert "step" in first

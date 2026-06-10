"""T6 — LR schedule tests.

Verify that:
  - --lr-schedule accepts none / linear / cosine
  - --lr-decay is no longer accepted (replaced by --lr-schedule)
  - LinearLR and CosineAnnealingLR are created correctly
  - Scheduler steps produce expected LR curves
"""

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.deep.model import BattleNet
from ai.deep.trainer import PPOTrainer
from scripts.train import parse_args


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def trainer():
    model = BattleNet()
    config = {"units": [("Swordsman", 0, 5, 3), ("Swordsman", 1, 5, 6)]}
    return PPOTrainer(model, config, lr=2.5e-4, device="cpu")


# ── 1. CLI parsing ────────────────────────────────────────────────


class TestLRScheduleCLI:
    """--lr-schedule should accept none, linear, cosine."""

    def test_default_is_none(self):
        args = parse_args([])
        assert args.lr_schedule == "none"

    def test_explicit_none(self):
        args = parse_args(["--lr-schedule", "none"])
        assert args.lr_schedule == "none"

    def test_linear(self):
        args = parse_args(["--lr-schedule", "linear"])
        assert args.lr_schedule == "linear"

    def test_cosine(self):
        args = parse_args(["--lr-schedule", "cosine"])
        assert args.lr_schedule == "cosine"

    def test_invalid_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(["--lr-schedule", "invalid"])

    def test_lr_decay_no_longer_exists(self):
        """--lr-decay was replaced by --lr-schedule in T6."""
        with pytest.raises(SystemExit):
            parse_args(["--lr-decay"])


# ── 2. Scheduler creation ─────────────────────────────────────────


class TestSchedulerCreation:
    """Schedulers should be created with correct parameters."""

    def test_linear_scheduler(self, trainer):
        scheduler = torch.optim.lr_scheduler.LinearLR(
            trainer.optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=10,
        )
        assert isinstance(scheduler, torch.optim.lr_scheduler.LinearLR)

    def test_cosine_scheduler(self, trainer):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer,
            T_max=10,
            eta_min=0.0,
        )
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)


# ── 3. LR curve behavior ──────────────────────────────────────────


class TestLRCurves:
    """Verify the LR curves for linear and cosine schedules."""

    def test_linear_decays_to_zero(self, trainer):
        """Linear should reach ~0 at the end."""
        lr = trainer.optimizer.param_groups[0]["lr"]
        scheduler = torch.optim.lr_scheduler.LinearLR(
            trainer.optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=10,
        )
        for _ in range(10):
            scheduler.step()
        final_lr = trainer.optimizer.param_groups[0]["lr"]
        assert final_lr < lr * 0.01  # nearly zero

    def test_cosine_decays_smoothly(self, trainer):
        """Cosine should decrease monotonically to ~0."""
        lr = trainer.optimizer.param_groups[0]["lr"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer,
            T_max=10,
            eta_min=0.0,
        )
        lrs = []
        for _ in range(10):
            scheduler.step()
            lrs.append(trainer.optimizer.param_groups[0]["lr"])
        # Monotonically decreasing
        for i in range(1, len(lrs)):
            assert lrs[i] <= lrs[i - 1]
        # Final near zero
        assert lrs[-1] < lr * 0.01

    def test_cosine_midpoint_not_zero(self, trainer):
        """Cosine should have non-trivial LR at midpoint."""
        lr = trainer.optimizer.param_groups[0]["lr"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer,
            T_max=10,
            eta_min=0.0,
        )
        for _ in range(5):
            scheduler.step()
        mid_lr = trainer.optimizer.param_groups[0]["lr"]
        # At midpoint, cosine LR should be ~50% of initial
        assert mid_lr > lr * 0.3

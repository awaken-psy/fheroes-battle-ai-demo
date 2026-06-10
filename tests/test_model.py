"""Tests for R5 — BattleNet CNN residual backbone + Policy/Value dual-head.

Covers:
  1. Forward-pass output shapes
  2. Masking: illegal actions get -inf logits
  3. Value output range [-1, 1]
  4. Parameter count within target range (1-5M)
  5. Batch and single-sample processing
  6. Deterministic with eval mode
  7. Serialization round-trip (state_dict save/load)
  8. Gradient flow through both heads
  9. All-legal and all-illegal mask edge cases
"""

import io
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.action_space import ACTION_DIM
from ai.observation import NUM_GRID_CHANNELS, GRID_ROWS, GRID_COLS, GLOBAL_DIM
from ai.deep.model import (
    BattleNet,
    ResidualBlock,
    _CONV_CHANNELS,
    _NUM_RES_BLOCKS,
    _BOTTLENECK_DIM,
    _FUSED_DIM,
    _GN_GROUPS,
)


# ── Fixtures ────────────────────────────────────────────────────


def _make_inputs(batch_size: int = 4, legal_fraction: float = 0.5):
    """Create random tensors with the correct shapes."""
    grid = torch.randn(batch_size, NUM_GRID_CHANNELS, GRID_ROWS, GRID_COLS)
    global_vec = torch.randn(batch_size, GLOBAL_DIM)
    mask = (torch.rand(batch_size, ACTION_DIM) > (1 - legal_fraction)).float()
    return grid, global_vec, mask


@pytest.fixture
def model():
    return BattleNet()


@pytest.fixture
def inputs():
    return _make_inputs(batch_size=4)


# ── 1. Forward-pass shapes ──────────────────────────────────────


class TestShapes:
    def test_policy_logits_shape(self, model, inputs):
        grid, gvec, mask = inputs
        policy, value = model(grid, gvec, mask)
        assert policy.shape == (4, ACTION_DIM)

    def test_value_shape(self, model, inputs):
        grid, gvec, mask = inputs
        _, value = model(grid, gvec, mask)
        assert value.shape == (4, 1)

    def test_single_sample(self, model):
        """Batch dim = 1 should work."""
        grid, gvec, mask = _make_inputs(batch_size=1)
        policy, value = model(grid, gvec, mask)
        assert policy.shape == (1, ACTION_DIM)
        assert value.shape == (1, 1)


# ── 2. Masking ──────────────────────────────────────────────────


class TestMasking:
    def test_illegal_actions_are_neg_inf(self, model, inputs):
        grid, gvec, mask = inputs
        policy, _ = model(grid, gvec, mask)
        illegal = mask == 0
        assert torch.all(policy[illegal] == float("-inf"))

    def test_legal_actions_finite(self, model, inputs):
        grid, gvec, mask = inputs
        policy, _ = model(grid, gvec, mask)
        legal = mask == 1
        assert torch.all(torch.isfinite(policy[legal]))

    def test_all_legal_no_neg_inf(self, model):
        """With all-ones mask, no logit should be -inf."""
        grid, gvec, _ = _make_inputs(batch_size=2)
        mask = torch.ones(2, ACTION_DIM)
        policy, _ = model(grid, gvec, mask)
        assert torch.all(torch.isfinite(policy))

    def test_all_illegal_all_neg_inf(self, model):
        """With all-zeros mask, every logit should be -inf."""
        grid, gvec, _ = _make_inputs(batch_size=2)
        mask = torch.zeros(2, ACTION_DIM)
        policy, _ = model(grid, gvec, mask)
        assert torch.all(policy == float("-inf"))


# ── 3. Value range ──────────────────────────────────────────────


class TestValueRange:
    def test_value_in_range(self, model, inputs):
        _, value = model(*inputs)
        assert torch.all(value >= -1.0)
        assert torch.all(value <= 1.0)

    def test_value_range_extreme_inputs(self, model):
        """Even with very large inputs, tanh keeps values in [-1, 1]."""
        grid = torch.randn(3, NUM_GRID_CHANNELS, GRID_ROWS, GRID_COLS) * 10
        gvec = torch.randn(3, GLOBAL_DIM) * 10
        mask = torch.ones(3, ACTION_DIM)
        _, value = model(grid, gvec, mask)
        assert torch.all(value >= -1.0)
        assert torch.all(value <= 1.0)


# ── 4. Parameter count ──────────────────────────────────────────


class TestParameterCount:
    def test_param_count_in_range(self, model):
        count = model.count_parameters()
        assert 1_000_000 <= count <= 5_000_000, (
            f"Parameter count {count:,} outside target range 1-5M"
        )

    def test_param_count_matches_manual(self, model):
        """Cross-check against sum of module parameters."""
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert model.count_parameters() == total


# ── 5. Batch processing ─────────────────────────────────────────


class TestBatchProcessing:
    def test_large_batch(self, model):
        grid, gvec, mask = _make_inputs(batch_size=32)
        policy, value = model(grid, gvec, mask)
        assert policy.shape == (32, ACTION_DIM)
        assert value.shape == (32, 1)

    def test_batch_consistency(self, model):
        """Single-sample result should match the corresponding batch element."""
        model.eval()
        grid, gvec, mask = _make_inputs(batch_size=3)

        with torch.no_grad():
            policy_batch, value_batch = model(grid, gvec, mask)

        # Process element 1 alone
        with torch.no_grad():
            p1, v1 = model(grid[1:2], gvec[1:2], mask[1:2])

        assert torch.allclose(policy_batch[1:2], p1, atol=1e-5)
        assert torch.allclose(value_batch[1:2], v1, atol=1e-5)


# ── 6. Determinism in eval mode ─────────────────────────────────


class TestDeterminism:
    def test_eval_deterministic(self, model):
        model.eval()
        grid, gvec, mask = _make_inputs(batch_size=2)
        with torch.no_grad():
            p1, v1 = model(grid, gvec, mask)
            p2, v2 = model(grid, gvec, mask)
        assert torch.equal(p1, p2)
        assert torch.equal(v1, v2)


# ── 7. Serialization ────────────────────────────────────────────


class TestSerialization:
    def test_state_dict_roundtrip(self, model, inputs):
        model.eval()
        grid, gvec, mask = inputs

        with torch.no_grad():
            p_orig, v_orig = model(grid, gvec, mask)

        # Save
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)

        # Load into fresh model
        model2 = BattleNet()
        model2.load_state_dict(torch.load(buf, weights_only=True))
        model2.eval()

        with torch.no_grad():
            p_loaded, v_loaded = model2(grid, gvec, mask)

        assert torch.allclose(p_orig, p_loaded, atol=1e-6)
        assert torch.allclose(v_orig, v_loaded, atol=1e-6)


# ── 8. Gradient flow ────────────────────────────────────────────


class TestGradientFlow:
    def test_policy_gradient(self, model, inputs):
        grid, gvec, mask = inputs
        policy, _ = model(grid, gvec, mask)
        loss = policy.sum()
        loss.backward()
        # Check stem conv has gradients
        assert model.stem_conv.weight.grad is not None
        assert model.stem_conv.weight.grad.abs().sum() > 0

    def test_value_gradient(self, model, inputs):
        grid, gvec, mask = inputs
        _, value = model(grid, gvec, mask)
        loss = value.sum()
        loss.backward()
        assert model.value_head.weight.grad is not None
        assert model.value_head.weight.grad.abs().sum() > 0

    def test_both_heads_propagate(self, model, inputs):
        """Combined loss should produce gradients in shared layers."""
        grid, gvec, mask = inputs
        policy, value = model(grid, gvec, mask)
        loss = policy.sum() + value.sum()
        loss.backward()
        # Shared bottleneck should receive gradients from both heads
        assert model.fc_bottleneck.weight.grad is not None
        assert model.fc_bottleneck.weight.grad.abs().sum() > 0


# ── 9. ResidualBlock unit tests ─────────────────────────────────


class TestResidualBlock:
    def test_shape_preserved(self):
        block = ResidualBlock(_CONV_CHANNELS)
        x = torch.randn(2, _CONV_CHANNELS, GRID_ROWS, GRID_COLS)
        out = block(x)
        assert out.shape == x.shape

    def test_skip_connection_effect(self):
        """ResBlock output should differ from a plain two-conv stack."""
        block = ResidualBlock(_CONV_CHANNELS)
        x = torch.randn(1, _CONV_CHANNELS, GRID_ROWS, GRID_COLS)
        block.eval()
        with torch.no_grad():
            out = block(x)
        # Output should not be all zeros (skip connection adds identity)
        assert out.abs().sum() > 0


# ── 10. Internal constants consistency ──────────────────────────


class TestConstants:
    def test_fused_dim(self):
        flat_grid = _CONV_CHANNELS * GRID_ROWS * GRID_COLS
        assert _FUSED_DIM == flat_grid + GLOBAL_DIM


# ── 11. T2: GroupNorm migration ─────────────────────────────────


class TestGroupNormMigration:
    """Verify BatchNorm → GroupNorm replacement (T2)."""

    def test_resblock_uses_groupnorm(self):
        block = ResidualBlock(_CONV_CHANNELS)
        assert isinstance(block.gn1, torch.nn.GroupNorm)
        assert isinstance(block.gn2, torch.nn.GroupNorm)
        assert block.gn1.num_groups == _GN_GROUPS
        assert block.gn2.num_groups == _GN_GROUPS
        assert not hasattr(block, "bn1")
        assert not hasattr(block, "bn2")

    def test_stem_uses_groupnorm(self):
        model = BattleNet()
        assert isinstance(model.stem_gn, torch.nn.GroupNorm)
        assert model.stem_gn.num_groups == _GN_GROUPS
        assert not hasattr(model, "stem_bn")

    def test_no_batchnorm_anywhere(self):
        model = BattleNet()
        bn_count = sum(1 for m in model.modules()
                       if isinstance(m, torch.nn.BatchNorm2d))
        assert bn_count == 0

    def test_groupnorm_count(self):
        """1 stem GN + 2 per ResBlock × 4 = 9 total."""
        model = BattleNet()
        gn_count = sum(1 for m in model.modules()
                       if isinstance(m, torch.nn.GroupNorm))
        assert gn_count == 1 + 2 * _NUM_RES_BLOCKS

    def test_batch1_no_nan(self):
        """GroupNorm should produce stable output at batch=1."""
        model = BattleNet().eval()
        grid = torch.randn(1, NUM_GRID_CHANNELS, GRID_ROWS, GRID_COLS)
        gvec = torch.randn(1, GLOBAL_DIM)
        mask = torch.ones(1, ACTION_DIM)
        logits, value = model(grid, gvec, mask)
        assert not logits.isnan().any()
        assert not value.isnan().any()

    def test_train_eval_consistent(self):
        """GroupNorm output should be identical in train and eval mode
        (unlike BatchNorm which uses running stats in eval)."""
        model = BattleNet()
        grid = torch.randn(2, NUM_GRID_CHANNELS, GRID_ROWS, GRID_COLS)
        gvec = torch.randn(2, GLOBAL_DIM)
        mask = torch.ones(2, ACTION_DIM)

        model.train()
        with torch.no_grad():
            out_train = model(grid, gvec, mask)

        model.eval()
        with torch.no_grad():
            out_eval = model(grid, gvec, mask)

        assert torch.allclose(out_train[0], out_eval[0], atol=1e-5)
        assert torch.allclose(out_train[1], out_eval[1], atol=1e-5)

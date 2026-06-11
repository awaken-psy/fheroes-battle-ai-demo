"""Tests for T9c — SoftMoELayer and BattleNet MoE integration.

Covers:
  1. SoftMoELayer output shapes
  2. Router weight properties (softmax: sums to 1, non-negative)
  3. Gradient flow through each expert
  4. Active expert control (set_active_expert)
  5. BattleNet + MoE forward pass (output shape unchanged)
  6. Backward compatibility (num_experts=0 identical to original)
  7. Parameter count verification
  8. State dict round-trip (save / load reproduces outputs)
  9. Partial weight loading (load_backbone_weights)
  10. Freeze / unfreeze helpers
"""

import io
import sys
import os

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.action_space import ACTION_DIM
from ai.deep.model import (
    BattleNet,
    SoftMoELayer,
    _BOTTLENECK_DIM,
    _DEFAULT_MOE_HIDDEN_DIM,
)
from ai.deep.pipeline import load_backbone_weights


# -- Helpers ----------------------------------------------------------------

def _make_inputs(batch_size: int = 4, legal_fraction: float = 0.5):
    """Create random (grid, global_vec, mask) tensors."""
    grid = torch.randn(batch_size, 35, 9, 11)
    global_vec = torch.randn(batch_size, 20)
    mask = (torch.rand(batch_size, ACTION_DIM) < legal_fraction).float()
    return grid, global_vec, mask


def _make_moe_layer(
    input_dim: int = _BOTTLENECK_DIM,
    hidden_dim: int = 64,
    num_experts: int = 4,
) -> SoftMoELayer:
    return SoftMoELayer(input_dim, hidden_dim, num_experts)


# -- 1. SoftMoELayer shapes ------------------------------------------------


class TestSoftMoEShapes:
    def test_output_shape(self):
        moe = _make_moe_layer(input_dim=384, hidden_dim=64, num_experts=4)
        x = torch.randn(8, 384)
        out, weights = moe(x)
        assert out.shape == (8, 384)

    def test_weights_shape(self):
        moe = _make_moe_layer(num_experts=4)
        x = torch.randn(8, 384)
        _, weights = moe(x)
        assert weights.shape == (8, 4)

    def test_batch_size_1(self):
        moe = _make_moe_layer()
        x = torch.randn(1, 384)
        out, weights = moe(x)
        assert out.shape == (1, 384)
        assert weights.shape == (1, 4)


# -- 2. Router weights -----------------------------------------------------


class TestRouterWeights:
    def test_weights_sum_to_one(self):
        moe = _make_moe_layer(num_experts=4)
        x = torch.randn(16, 384)
        _, weights = moe(x)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(16), atol=1e-5)

    def test_weights_non_negative(self):
        moe = _make_moe_layer()
        x = torch.randn(8, 384)
        _, weights = moe(x)
        assert (weights >= 0).all()

    def test_weights_bounded(self):
        moe = _make_moe_layer()
        x = torch.randn(8, 384)
        _, weights = moe(x)
        assert (weights <= 1.0).all()


# -- 3. Gradient flow ------------------------------------------------------


class TestGradientFlow:
    def test_expert_receives_gradient(self):
        moe = _make_moe_layer(num_experts=4)
        x = torch.randn(4, 384)
        out, _ = moe(x)
        out.sum().backward()
        for i, expert in enumerate(moe.experts):
            assert expert[0].weight.grad is not None, (
                f"Expert {i} received no gradient"
            )

    def test_router_receives_gradient(self):
        moe = _make_moe_layer()
        x = torch.randn(4, 384)
        out, _ = moe(x)
        out.sum().backward()
        assert moe.router.weight.grad is not None

    def test_merge_receives_gradient(self):
        moe = _make_moe_layer()
        x = torch.randn(4, 384)
        out, _ = moe(x)
        out.sum().backward()
        assert moe.merge.weight.grad is not None


# -- 4. Active expert control ----------------------------------------------


class TestActiveExpert:
    def test_set_active_expert_freezes_others(self):
        moe = _make_moe_layer(num_experts=4)
        moe.set_active_expert(2)

        for i, expert in enumerate(moe.experts):
            for param in expert.parameters():
                if i == 2:
                    assert param.requires_grad, f"Expert {i} should be trainable"
                else:
                    assert not param.requires_grad, f"Expert {i} should be frozen"

    def test_router_frozen(self):
        moe = _make_moe_layer(num_experts=4)
        moe.set_active_expert(0)
        assert not moe.router.weight.requires_grad
        assert not moe.router.bias.requires_grad

    def test_merge_frozen(self):
        moe = _make_moe_layer(num_experts=4)
        moe.set_active_expert(0)
        assert not moe.merge.weight.requires_grad

    def test_inactive_expert_no_gradient(self):
        moe = _make_moe_layer(num_experts=4)
        moe.set_active_expert(0)
        x = torch.randn(4, 384)
        out, _ = moe(x)
        out.sum().backward()

        # Expert 0 gets gradient, expert 1 does not
        assert moe.experts[0][0].weight.grad is not None
        assert moe.experts[1][0].weight.grad is None

    def test_freeze_all_experts(self):
        moe = _make_moe_layer(num_experts=4)
        moe.freeze_all_experts()
        for param in moe.parameters():
            assert not param.requires_grad

    def test_unfreeze_all(self):
        moe = _make_moe_layer(num_experts=4)
        moe.freeze_all_experts()
        moe.unfreeze_all()
        for param in moe.parameters():
            assert param.requires_grad

    def test_freeze_experts_and_merge(self):
        moe = _make_moe_layer(num_experts=4)
        moe.freeze_experts_and_merge()
        # Router stays trainable
        assert moe.router.weight.requires_grad
        # Experts and merge are frozen
        for expert in moe.experts:
            for param in expert.parameters():
                assert not param.requires_grad
        assert not moe.merge.weight.requires_grad


# -- 5. BattleNet + MoE integration ----------------------------------------


class TestBattleNetMoE:
    def test_forward_shapes(self):
        model = BattleNet(num_experts=4)
        grid, gvec, mask = _make_inputs(batch_size=4)
        policy, value = model(grid, gvec, mask)
        assert policy.shape == (4, ACTION_DIM)
        assert value.shape == (4, 1)

    def test_extract_bottleneck(self):
        model = BattleNet(num_experts=4)
        grid, gvec, _ = _make_inputs(batch_size=4)
        feat = model.extract_bottleneck(grid, gvec)
        assert feat.shape == (4, _BOTTLENECK_DIM)

    def test_value_range(self):
        model = BattleNet(num_experts=4)
        grid, gvec, mask = _make_inputs()
        _, value = model(grid, gvec, mask)
        assert torch.all(value >= -1.0)
        assert torch.all(value <= 1.0)

    def test_masking_inf_for_illegal(self):
        model = BattleNet(num_experts=4)
        grid, gvec, _ = _make_inputs(batch_size=2)
        mask = torch.zeros(2, ACTION_DIM)
        mask[:, :5] = 1.0  # only first 5 actions legal
        policy, _ = model(grid, gvec, mask)
        # Illegal actions should be -inf
        assert torch.all(policy[:, 5:] == float("-inf"))
        # Legal actions should be finite
        assert torch.all(torch.isfinite(policy[:, :5]))


# -- 6. Backward compatibility ---------------------------------------------


class TestBackwardCompat:
    def test_no_moe_by_default(self):
        """Default BattleNet() has no MoE, identical to T8."""
        model = BattleNet()
        assert model.moe is None
        assert model.num_experts == 0

    def test_output_identical_to_original(self):
        """Without MoE, forward pass produces same shapes as T8."""
        model = BattleNet()
        grid, gvec, mask = _make_inputs()
        policy, value = model(grid, gvec, mask)
        assert policy.shape == (4, ACTION_DIM)
        assert value.shape == (4, 1)

    def test_no_moe_keys_in_state_dict(self):
        """State dict has no 'moe' keys when num_experts=0."""
        model = BattleNet()
        keys = set(model.state_dict().keys())
        assert not any(k.startswith("moe") for k in keys)


# -- 7. Parameter count ----------------------------------------------------


class TestParameterCount:
    def test_moe_adds_parameters(self):
        base = BattleNet(num_experts=0)
        moe = BattleNet(num_experts=4)
        base_count = sum(p.numel() for p in base.parameters())
        moe_count = sum(p.numel() for p in moe.parameters())
        added = moe_count - base_count
        # Router: (384*4+4) = 1540
        # 4 experts: (384*128+128)*4 = 197120
        # Merge: (128*384+384) = 49536
        expected = 1540 + 197120 + 49536  # = 248196
        assert added == expected, f"Expected {expected} added params, got {added}"

    def test_moe_hidden_dim_affects_count(self):
        m1 = BattleNet(num_experts=4, moe_hidden_dim=64)
        m2 = BattleNet(num_experts=4, moe_hidden_dim=128)
        c1 = sum(p.numel() for p in m1.parameters())
        c2 = sum(p.numel() for p in m2.parameters())
        assert c2 > c1


# -- 8. State dict round-trip ----------------------------------------------


class TestStateDict:
    def test_roundtrip(self):
        model = BattleNet(num_experts=4)
        grid, gvec, mask = _make_inputs()
        model.eval()
        with torch.no_grad():
            p_orig, v_orig = model(grid, gvec, mask)

        # Save and reload
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)

        model2 = BattleNet(num_experts=4)
        model2.load_state_dict(torch.load(buf, weights_only=True))
        model2.eval()
        with torch.no_grad():
            p_loaded, v_loaded = model2(grid, gvec, mask)

        assert torch.allclose(p_orig, p_loaded, atol=1e-6)
        assert torch.allclose(v_orig, v_loaded, atol=1e-6)


# -- 9. Partial weight loading ---------------------------------------------


class TestPartialLoading:
    def test_load_non_moe_into_moe(self):
        """Loading a T8 checkpoint into a MoE model only loads backbone keys."""
        # Save a non-MoE checkpoint
        base_model = BattleNet(num_experts=0)
        buf = io.BytesIO()
        torch.save({"model": base_model.state_dict(), "step": 100}, buf)
        buf.seek(0)

        # Load into MoE model
        moe_model = BattleNet(num_experts=4)
        tmp_path = "/tmp/_test_t8_ckpt.pt"
        with open(tmp_path, "wb") as f:
            f.write(buf.getvalue())

        matched, skipped = load_backbone_weights(moe_model, tmp_path, "cpu")
        # All backbone keys should match
        assert matched > 0
        os.unlink(tmp_path)

    def test_preserves_moe_random_init(self):
        """MoE layers should NOT be overwritten by backbone loading."""
        moe_model = BattleNet(num_experts=4)
        # Save MoE router weights before loading
        router_w_before = moe_model.moe.router.weight.data.clone()

        # Save a non-MoE checkpoint
        base_model = BattleNet(num_experts=0)
        tmp_path = "/tmp/_test_t8_ckpt2.pt"
        torch.save({"model": base_model.state_dict()}, tmp_path)

        load_backbone_weights(moe_model, tmp_path, "cpu")
        # Router weights should be unchanged (not in old checkpoint)
        assert torch.equal(moe_model.moe.router.weight.data, router_w_before)
        os.unlink(tmp_path)


# -- 10. Freeze / unfreeze -------------------------------------------------


class TestFreezeUnfreeze:
    def test_freeze_backbone(self):
        model = BattleNet(num_experts=4)
        model.freeze_backbone()

        # Backbone params should be frozen
        assert not model.stem_conv.weight.requires_grad
        assert not model.fc_bottleneck.weight.requires_grad
        assert not model.unit_embed.weight.requires_grad

        # Heads and MoE should still be trainable
        assert model.policy_head.weight.requires_grad
        assert model.moe.router.weight.requires_grad
        assert model.moe.experts[0][0].weight.requires_grad

    def test_freeze_experts_and_merge(self):
        model = BattleNet(num_experts=4)
        model.freeze_experts_and_merge()

        # Router trainable
        assert model.moe.router.weight.requires_grad
        # Experts frozen
        assert not model.moe.experts[0][0].weight.requires_grad
        # Merge frozen
        assert not model.moe.merge.weight.requires_grad
        # Heads still trainable (not part of MoE freeze)
        assert model.policy_head.weight.requires_grad

    def test_unfreeze_all(self):
        model = BattleNet(num_experts=4)
        model.freeze_backbone()
        model.freeze_experts_and_merge()
        model.unfreeze_all()

        for param in model.parameters():
            assert param.requires_grad

    def test_set_active_expert_on_battlenet(self):
        model = BattleNet(num_experts=4)
        model.freeze_backbone()
        model.set_active_expert(2)

        # Backbone frozen
        assert not model.stem_conv.weight.requires_grad
        # Expert 2 trainable, others frozen
        assert model.moe.experts[2][0].weight.requires_grad
        assert not model.moe.experts[0][0].weight.requires_grad
        assert not model.moe.experts[1][0].weight.requires_grad
        assert not model.moe.experts[3][0].weight.requires_grad
        # Router and merge frozen (Stage 2 fix), heads trainable
        assert not model.moe.router.weight.requires_grad
        assert not model.moe.merge.weight.requires_grad
        assert model.policy_head.weight.requires_grad

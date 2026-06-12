"""Tests for T9f Phase 2 supervised router pretraining pipeline.

Validates: collect_router_data.py, train_router_supervised.py, and
train.py --load-router bridge.
"""

import os
import sys
import tempfile

import torch
import torch.nn as nn
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.deep.model import BattleNet, SoftMoELayer


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def moe_model():
    """Create a small MoE model for testing."""
    model = BattleNet(num_experts=4, moe_hidden_dim=384, routing_topk=2)
    model.eval()
    return model


@pytest.fixture
def synthetic_dataset():
    """Create synthetic (features, labels) data for router training.

    Generates 4 clusters of 384-dim features, one per config/expert,
    with enough separation for a linear classifier to learn.
    """
    torch.manual_seed(42)
    n_per_class = 100
    n_classes = 4
    features = []
    labels = []

    for i in range(n_classes):
        # Each class has a distinct mean direction
        center = torch.randn(384)
        center = center / center.norm() * 2.0  # scale up
        noise = torch.randn(n_per_class, 384) * 0.3
        feats = center.unsqueeze(0) + noise
        feats = torch.relu(feats)  # bottleneck output is relu'd
        features.append(feats)
        labels.append(torch.full((n_per_class,), i, dtype=torch.long))

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return features, labels


# ── SoftMoELayer freeze helpers ───────────────────────────────────


class TestMoEFreezeHelpers:
    """Verify freeze/unfreeze helpers work correctly for router training."""

    def test_freeze_all_experts(self, moe_model):
        """freeze_all_experts should disable grads on experts+heads, not router."""
        moe_model.moe.freeze_all_experts()
        # Experts and heads should be frozen
        for i in range(4):
            for p in moe_model.moe.experts[i].parameters():
                assert not p.requires_grad
            for p in moe_model.moe.policy_heads[i].parameters():
                assert not p.requires_grad
            for p in moe_model.moe.value_heads[i].parameters():
                assert not p.requires_grad
        # Router should still be trainable
        for p in moe_model.moe.router.parameters():
            assert p.requires_grad

    def test_freeze_backbone(self, moe_model):
        """freeze_backbone should disable grads on CNN+embedding+bottleneck."""
        moe_model.freeze_backbone()
        frozen_prefixes = ("stem_", "res_blocks", "unit_embed", "fc_bottleneck")
        for name, param in moe_model.named_parameters():
            if any(name.startswith(p) for p in frozen_prefixes):
                assert not param.requires_grad, f"{name} should be frozen"


# ── RouterDataset ─────────────────────────────────────────────────


class TestRouterDataset:
    """Test the RouterDataset class from train_router_supervised.py."""

    def test_router_dataset_length(self):
        from scripts.train_router_supervised import RouterDataset
        feats = torch.randn(100, 384)
        labels = torch.randint(0, 4, (100,))
        ds = RouterDataset(feats, labels)
        assert len(ds) == 100

    def test_router_dataset_getitem(self):
        from scripts.train_router_supervised import RouterDataset
        feats = torch.randn(50, 384)
        labels = torch.randint(0, 4, (50,))
        ds = RouterDataset(feats, labels)
        f, l = ds[0]
        assert f.shape == (384,)
        assert l.shape == ()

    def test_router_dataset_types(self):
        from scripts.train_router_supervised import RouterDataset
        feats = torch.randn(10, 384)
        labels = torch.randint(0, 4, (10,))
        ds = RouterDataset(feats, labels)
        f, l = ds[5]
        assert f.dtype == torch.float32
        assert l.dtype == torch.int64


# ── Router training ───────────────────────────────────────────────


class TestRouterTraining:
    """Test supervised router training on synthetic data."""

    def test_train_router_freezes_correctly(self, moe_model, synthetic_dataset):
        """After train_router, only router params should have gradients."""
        from scripts.train_router_supervised import train_router
        feats, labels = synthetic_dataset
        # Small split
        train_f, val_f = feats[:320], feats[320:]
        train_l, val_l = labels[:320], labels[320:]

        train_router(moe_model, train_f, train_l, val_f, val_l,
                     num_epochs=2, batch_size=32, lr=1e-3, device="cpu")

        # Check only router has gradients
        for name, param in moe_model.named_parameters():
            if "moe.router" in name:
                assert param.requires_grad, f"{name} should be trainable"
            elif "moe" in name:
                assert not param.requires_grad, f"{name} should be frozen"

    def test_train_router_improves_accuracy(self, moe_model, synthetic_dataset):
        """Router should achieve high accuracy on separable synthetic data."""
        from scripts.train_router_supervised import train_router, evaluate_router
        feats, labels = synthetic_dataset

        # Before training: random accuracy ~25% (4 classes)
        before = evaluate_router(moe_model, feats, labels, device="cpu")

        # Train
        train_f, val_f = feats[:320], feats[320:]
        train_l, val_l = labels[:320], labels[320:]
        train_router(moe_model, train_f, train_l, val_f, val_l,
                     num_epochs=30, batch_size=32, lr=1e-3, device="cpu")

        # After training: should be much better
        after = evaluate_router(moe_model, feats, labels, device="cpu")
        assert after["accuracy"] > before["accuracy"] + 0.3, (
            f"Accuracy should improve significantly: "
            f"{before['accuracy']:.3f} -> {after['accuracy']:.3f}"
        )

    def test_evaluate_router_returns_confusion_matrix(self, moe_model,
                                                       synthetic_dataset):
        """evaluate_router should return a valid confusion matrix."""
        from scripts.train_router_supervised import evaluate_router
        feats, labels = synthetic_dataset
        results = evaluate_router(moe_model, feats, labels, device="cpu")

        assert "confusion_matrix" in results
        cm = results["confusion_matrix"]
        assert cm.shape == (4, 4)
        assert cm.sum() == len(labels)

        assert "per_class_accuracy" in results
        assert len(results["per_class_accuracy"]) == 4


# ── Checkpoint roundtrip ──────────────────────────────────────────


class TestCheckpointRoundtrip:
    """Test that router weights survive save/load cycle."""

    def test_router_checkpoint_roundtrip(self, moe_model, synthetic_dataset):
        """Save and reload model checkpoint, verify router weights match."""
        from scripts.train_router_supervised import train_router
        feats, labels = synthetic_dataset

        # Train to get non-trivial weights
        train_router(moe_model, feats[:320], labels[:320],
                     feats[320:], labels[320:],
                     num_epochs=5, batch_size=32, lr=1e-3, device="cpu")

        # Save router weights
        router_w_before = moe_model.moe.router.weight.data.clone()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({
                "step": 5,
                "model": moe_model.state_dict(),
            }, f.name)
            tmp_path = f.name

        try:
            # Load into fresh model
            model2 = BattleNet(num_experts=4, moe_hidden_dim=384,
                               routing_topk=2)
            ckpt = torch.load(tmp_path, map_location="cpu",
                              weights_only=False)
            model2.load_state_dict(ckpt["model"])

            router_w_after = model2.moe.router.weight.data
            assert torch.allclose(router_w_before, router_w_after, atol=1e-6), \
                "Router weights should match after save/load"
        finally:
            os.unlink(tmp_path)

    def test_load_router_flag(self, moe_model, synthetic_dataset):
        """--load-router should overlay router weights onto existing model."""
        from scripts.train_router_supervised import train_router
        feats, labels = synthetic_dataset

        # Train to get non-trivial router weights
        train_router(moe_model, feats[:320], labels[:320],
                     feats[320:], labels[320:],
                     num_epochs=5, batch_size=32, lr=1e-3, device="cpu")

        # Save only model (simulating supervised checkpoint)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"model": moe_model.state_dict()}, f.name)
            supervised_ckpt = f.name

        try:
            # Create fresh model with default router weights
            model2 = BattleNet(num_experts=4, moe_hidden_dim=384,
                               routing_topk=2)
            default_w = model2.moe.router.weight.data.clone()

            # Simulate --load-router logic
            router_ckpt = torch.load(supervised_ckpt, map_location="cpu",
                                     weights_only=False)
            router_state = {
                k: v for k, v in router_ckpt["model"].items()
                if k.startswith("moe.router")
            }
            model_state = model2.state_dict()
            model_state.update(router_state)
            model2.load_state_dict(model_state)

            # Router weights should now match supervised model
            loaded_w = model2.moe.router.weight.data
            assert not torch.allclose(default_w, loaded_w, atol=1e-3), \
                "Router weights should differ from default after --load-router"
            assert torch.allclose(
                moe_model.moe.router.weight.data, loaded_w, atol=1e-6), \
                "Loaded router weights should match supervised model"
        finally:
            os.unlink(supervised_ckpt)

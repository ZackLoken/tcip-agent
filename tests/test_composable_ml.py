"""Tests for composable ML primitives — Phases 0-4."""

from __future__ import annotations

import pytest
torch = pytest.importorskip("torch")


# ====================================================================
# Phase 0: Component Registry
# ====================================================================

class TestComponentRegistry:
    def test_register_and_get(self):
        from tcip_mcp.pipelines.registry import ComponentRegistry
        reg = ComponentRegistry("test")
        @reg.register("foo", category="a", metadata={"x": 1})
        def build_foo(**kw):
            return {"built": True, **kw}
        assert "foo" in reg
        assert reg.build("foo") == {"built": True}
        assert reg.build("foo", y=2) == {"built": True, "y": 2}

    def test_duplicate_rejection(self):
        from tcip_mcp.pipelines.registry import ComponentRegistry
        reg = ComponentRegistry("test_dup")
        reg.register_factory("a", lambda: 1)
        with pytest.raises(ValueError, match="Duplicate"):
            reg.register_factory("a", lambda: 2)

    def test_list_and_filter(self):
        from tcip_mcp.pipelines.registry import ComponentRegistry
        reg = ComponentRegistry("test_list")
        reg.register_factory("x", lambda: 1, category="cat_a", metadata={"size": 10})
        reg.register_factory("y", lambda: 2, category="cat_b", metadata={"size": 20})
        reg.register_factory("z", lambda: 3, category="cat_a", metadata={"size": 30})

        all_items = reg.list()
        assert len(all_items) == 3
        cat_a = reg.list(category="cat_a")
        assert len(cat_a) == 2
        big = reg.list(filter_fn=lambda d: d.get("size", 0) > 15)
        assert len(big) == 2

    def test_describe(self):
        from tcip_mcp.pipelines.registry import ComponentRegistry
        reg = ComponentRegistry("test_desc")
        reg.register_factory("q", lambda: 1, category="c", metadata={"desc": "hi"})
        info = reg.describe("q")
        assert info["name"] == "q"
        assert info["category"] == "c"
        assert info["desc"] == "hi"

    def test_unknown_key_error(self):
        from tcip_mcp.pipelines.registry import ComponentRegistry
        reg = ComponentRegistry("test_err")
        with pytest.raises(KeyError, match="Unknown"):
            reg.get("nonexistent")

    def test_names(self):
        from tcip_mcp.pipelines.registry import ComponentRegistry
        reg = ComponentRegistry("test_names")
        reg.register_factory("b", lambda: 1)
        reg.register_factory("a", lambda: 2)
        assert reg.names() == ["a", "b"]


# ====================================================================
# Phase 1: Backbones + Necks — using global registries
# ====================================================================

class TestGlobalRegistries:
    def test_backbones_registered(self):
        from tcip_mcp.pipelines.registry import BACKBONES
        # Trigger registration
        import tcip_mcp.pipelines.components.backbones  # noqa: F401
        assert len(BACKBONES) > 0
        assert "resnet50" in BACKBONES

    def test_necks_registered(self):
        from tcip_mcp.pipelines.registry import NECKS
        import tcip_mcp.pipelines.components.necks  # noqa: F401
        assert "fpn" in NECKS
        assert "gap" in NECKS
        assert "identity" in NECKS
        assert "pan" in NECKS

    def test_heads_registered(self):
        from tcip_mcp.pipelines.registry import HEADS
        import tcip_mcp.pipelines.components.heads  # noqa: F401
        assert "classification" in HEADS
        assert "ordinal" in HEADS
        assert "regression" in HEADS

    def test_losses_registered(self):
        from tcip_mcp.pipelines.registry import LOSSES
        import tcip_mcp.pipelines.components.losses  # noqa: F401
        assert "cross_entropy" in LOSSES
        assert "focal" in LOSSES
        assert "corn" in LOSSES
        assert "coral" in LOSSES
        assert "dice" in LOSSES
        assert "giou" in LOSSES


# ====================================================================
# Phase 1: Neck forward tests
# ====================================================================

class TestNecks:
    def _dummy_features(self, channels=(64, 128, 256, 512)):
        return {
            f"s{i}": torch.randn(1, c, 32 // (2 ** i), 32 // (2 ** i))
            for i, c in enumerate(channels)
        }

    def test_fpn_forward(self):
        from tcip_mcp.pipelines.components.necks import FPN
        fpn = FPN([64, 128, 256, 512], out_channels=64)
        out = fpn(self._dummy_features())
        assert len(out) == 4
        for v in out.values():
            assert v.shape[1] == 64

    def test_pan_forward(self):
        from tcip_mcp.pipelines.components.necks import PAN
        pan = PAN([64, 128, 256, 512], out_channels=64)
        out = pan(self._dummy_features())
        assert len(out) == 4

    def test_gap_forward(self):
        from tcip_mcp.pipelines.components.necks import GlobalAvgPoolNeck
        gap = GlobalAvgPoolNeck([64, 128, 256, 512])
        out = gap(self._dummy_features())
        assert out.shape == (1, 512)

    def test_identity_forward(self):
        from tcip_mcp.pipelines.components.necks import IdentityNeck
        neck = IdentityNeck([64, 128, 256, 512])
        feats = self._dummy_features()
        out = neck(feats)
        assert set(out.keys()) == set(feats.keys())


# ====================================================================
# Phase 2: Head tests
# ====================================================================

class TestHeads:
    def test_classification_head(self):
        from tcip_mcp.pipelines.components.heads import ClassificationHead
        head = ClassificationHead(in_channels=512, num_classes=5)
        features = torch.randn(2, 512)
        out = head(features)
        assert out["logits"].shape == (2, 5)

        targets = {"labels": torch.tensor([0, 3])}
        loss = head.compute_loss(out, targets)
        assert "cls_loss" in loss
        assert loss["cls_loss"].requires_grad

        decoded = head.decode(out)
        assert decoded["labels"].shape == (2,)
        assert decoded["confidences"].shape == (2,)

    def test_ordinal_head(self):
        from tcip_mcp.pipelines.components.heads import OrdinalHead
        head = OrdinalHead(in_channels=256, num_ranks=9)
        features = torch.randn(4, 256)
        out = head(features)
        assert out["logits"].shape == (4, 8)  # K-1 classifiers

        targets = {"ranks": torch.tensor([0, 3, 5, 8])}
        loss = head.compute_loss(out, targets)
        assert "ordinal_loss" in loss
        assert loss["ordinal_loss"].requires_grad

        decoded = head.decode(out)
        assert decoded["ranks"].shape == (4,)

    def test_regression_head(self):
        from tcip_mcp.pipelines.components.heads import RegressionHead
        head = RegressionHead(in_channels=256)
        features = torch.randn(3, 256)
        out = head(features)
        assert out["values"].shape == (3,)

        targets = {"values": torch.tensor([1.0, 2.5, 0.3])}
        loss = head.compute_loss(out, targets)
        assert "reg_loss" in loss

    def test_semantic_seg_head(self):
        from tcip_mcp.pipelines.components.heads import SemanticSegHead
        head = SemanticSegHead(in_channels=64, num_classes=3)
        features = {"p0": torch.randn(1, 64, 32, 32)}
        out = head(features)
        assert out["logits"].shape == (1, 3, 32, 32)

        targets = {"masks": torch.randint(0, 3, (1, 32, 32))}
        loss = head.compute_loss(out, targets)
        assert "ce_loss" in loss
        assert "dice_loss" in loss


# ====================================================================
# Phase 3: Loss tests
# ====================================================================

class TestLosses:
    def test_focal_loss(self):
        from tcip_mcp.pipelines.components.losses import FocalLoss
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        preds = torch.randn(8, 5, requires_grad=True)
        targets = torch.randint(0, 5, (8,))
        loss = loss_fn(preds, targets)
        assert loss.shape == ()
        loss.backward()

    def test_corn_loss(self):
        from tcip_mcp.pipelines.components.losses import CORNLoss
        loss_fn = CORNLoss(num_ranks=9)
        preds = torch.randn(10, 8, requires_grad=True)  # K-1=8
        targets = torch.randint(0, 9, (10,))
        loss = loss_fn(preds, targets)
        assert loss.shape == ()
        loss.backward()

    def test_coral_loss(self):
        from tcip_mcp.pipelines.components.losses import CORALLoss
        loss_fn = CORALLoss(num_ranks=5)
        preds = torch.randn(6, 4, requires_grad=True)
        targets = torch.randint(0, 5, (6,))
        loss = loss_fn(preds, targets)
        loss.backward()

    def test_dice_loss(self):
        from tcip_mcp.pipelines.components.losses import DiceLoss
        loss_fn = DiceLoss()
        preds = torch.randn(2, 1, 16, 16, requires_grad=True)
        targets = torch.randint(0, 2, (2, 1, 16, 16))
        loss = loss_fn(preds, targets)
        loss.backward()

    def test_build_combined(self):
        from tcip_mcp.pipelines.components.losses import build_loss
        combined = build_loss("smooth_l1+huber")
        assert hasattr(combined, "losses")
        assert len(combined.losses) == 2

    def test_giou_loss(self):
        from tcip_mcp.pipelines.components.losses import GIoULoss
        loss_fn = GIoULoss()
        pred = torch.tensor([[10., 10., 50., 50.]], requires_grad=True)
        gt = torch.tensor([[15., 15., 55., 55.]])
        loss = loss_fn(pred, gt)
        loss.backward()


# ====================================================================
# Phase 4: Composer tests
# ====================================================================

class TestComposer:
    def test_validate_model_spec_valid(self):
        import tcip_mcp.pipelines.components.backbones  # noqa: F401
        import tcip_mcp.pipelines.components.necks  # noqa: F401
        import tcip_mcp.pipelines.components.heads  # noqa: F401
        from tcip_mcp.pipelines.composer import validate_model_spec
        spec = {
            "backbone": {"name": "resnet50"},
            "neck": {"name": "gap"},
            "heads": [{"name": "classification", "num_classes": 5}],
        }
        assert validate_model_spec(spec) == []

    def test_validate_model_spec_invalid(self):
        from tcip_mcp.pipelines.composer import validate_model_spec
        spec = {"backbone": {"name": "nonexistent"}, "heads": []}
        issues = validate_model_spec(spec)
        assert len(issues) >= 2  # bad backbone + empty heads

    def test_recommend_classification(self):
        from tcip_mcp.pipelines.composer import recommend_model_spec
        spec = recommend_model_spec("classification", dataset_size=300, num_classes=5)
        assert spec["backbone"]["name"] == "efficientnet_b0"
        assert spec["neck"]["name"] == "gap"
        assert spec["heads"][0]["name"] == "classification"

    def test_recommend_detection(self):
        from tcip_mcp.pipelines.composer import recommend_model_spec
        spec = recommend_model_spec("detection", dataset_size=1000, num_classes=3)
        assert spec["neck"]["name"] == "fpn"
        assert spec["heads"][0]["name"] == "anchor_detection"

    def test_recommend_ordinal(self):
        from tcip_mcp.pipelines.composer import recommend_model_spec
        spec = recommend_model_spec("ordinal", dataset_size=500, num_ranks=9)
        assert spec["heads"][0]["name"] == "ordinal"
        assert spec["heads"][0]["num_ranks"] == 9

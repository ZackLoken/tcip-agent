"""Tests for composable ML primitives — Phases 0-4."""

from __future__ import annotations

import pytest
torch = pytest.importorskip("torch")


# ====================================================================
# Phase 1: Neck forward tests
# ====================================================================

class TestBackboneOrdering:
    """The wrapper renames stages so pyramid order survives into the necks.

    The necks rebuild order with ``sorted(features.keys())``. A module that emits its own names
    would be consumed alphabetically — and with uniform stage widths there is no shape error to
    catch it, so the finest map silently lands where a detector assigns the smallest anchors.
    """

    def _staged(self):
        import torch.nn as nn

        class Staged(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Conv2d(3, 64, 3, stride=2, padding=1)
                self.b = nn.Conv2d(64, 64, 3, stride=2, padding=1)
                self.c = nn.Conv2d(64, 64, 3, stride=2, padding=1)

            def forward(self, x):
                lo = self.a(x)
                mid = self.b(lo)
                return {"low": lo, "mid": mid, "high": self.c(mid)}

        return Staged()

    def test_self_named_stages_are_renamed_finest_first(self):
        from tcip_mcp.pipelines.components.backbones import BackboneWrapper

        out = BackboneWrapper(self._staged(), [64, 64, 64])(torch.zeros(1, 3, 64, 64))
        assert list(out.keys()) == ["s0", "s1", "s2"]
        # s0 is the finest stage, not whatever sorted() would have put first ("high", 8x8).
        assert out["s0"].shape[-1] == 32
        assert out["s2"].shape[-1] == 8

    def test_pyramid_order_survives_into_the_neck(self):
        from tcip_mcp.pipelines.components.backbones import BackboneWrapper
        from tcip_mcp.pipelines.components.necks import FPN

        feats = BackboneWrapper(self._staged(), [64, 64, 64])(torch.zeros(1, 3, 64, 64))
        p = FPN([64, 64, 64], out_channels=64)(feats)
        assert [p[k].shape[-1] for k in sorted(p)] == [32, 16, 8]  # finest -> coarsest

    def test_feature_keys_names_the_stages(self):
        from tcip_mcp.pipelines.components.backbones import BackboneWrapper

        w = BackboneWrapper(self._staged(), [64, 64, 64], feature_keys=["p3", "p4", "p5"])
        assert list(w(torch.zeros(1, 3, 64, 64)).keys()) == ["p3", "p4", "p5"]

    def test_feature_keys_length_mismatch_raises(self):
        from tcip_mcp.pipelines.components.backbones import BackboneWrapper

        w = BackboneWrapper(self._staged(), [64, 64, 64], feature_keys=["only_one"])
        with pytest.raises(ValueError, match="1 names but the backbone emitted 3"):
            w(torch.zeros(1, 3, 64, 64))


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

    def test_combined_loss_receives_class_weighting(self):
        """The weighting context must reach a combined loss's weightable term.

        It used to be dropped: the `+` branch recursed without forwarding class_distribution,
        so imbalance handling silently vanished for every combined loss.
        """
        from tcip_mcp.pipelines.components.losses import build_loss
        combined = build_loss("cross_entropy+dice", class_distribution={0: 100, 1: 10, 2: 5},
                              num_classes=3)
        weights = dict(combined.losses[0].named_buffers()).get("ce.weight")
        assert weights is not None, "class weighting did not reach the cross_entropy term"
        assert weights.numel() == 3
        assert weights[0] < weights[2]  # the majority class is down-weighted

    def test_combined_loss_refuses_unusable_class_weighting(self):
        """A class_distribution no term can consume must raise, not be silently discarded."""
        from tcip_mcp.pipelines.components.losses import build_loss
        with pytest.raises(ValueError, match="weightable"):
            build_loss("bce+dice", class_distribution={0: 100, 1: 10}, num_classes=2)

    def test_giou_loss(self):
        from tcip_mcp.pipelines.components.losses import GIoULoss
        loss_fn = GIoULoss()
        pred = torch.tensor([[10., 10., 50., 50.]], requires_grad=True)
        gt = torch.tensor([[15., 15., 55., 55.]])
        loss = loss_fn(pred, gt)
        loss.backward()

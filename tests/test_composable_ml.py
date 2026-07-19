"""Tests for composable ML primitives — Phases 0-4."""

from __future__ import annotations

import pytest
torch = pytest.importorskip("torch")


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

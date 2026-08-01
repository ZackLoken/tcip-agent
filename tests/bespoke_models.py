"""A hand-written bespoke detector + custom training loop — the S7 proof fixture.

This is *agent-authored* model code the platform runs through its audited envelope. It exercises
two things the CV-scientist boundary must support:

  (a) **modified architecture internals** — a from-scratch backbone + tiny FPN that use ``GroupNorm``
      (detector batches are tiny, so BatchNorm statistics are unreliable), fed into a torchvision
      Faster R-CNN whose ``AnchorGenerator`` is built from *this dataset's* GT box shapes
      (aspect ratios via ``pipelines.derivations.gt_aspect_ratios``, sizes from the GT size
      distribution) rather than torchvision's fixed defaults; and

  (b) a **custom ``train(ctx)`` loop** (``train_bespoke``) — not ``ctx.default_train()`` — that drives
      training through the envelope's ``ctx`` sinks so it stays audited, immutable, and provenanced.

Importable-builder only (the envelope re-imports it, never ``exec``). Not a ``test_*`` module — it is
imported by ``test_bespoke_model_e2e.py`` via its dotted name.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tcip_mcp.pipelines.derivations import gt_aspect_ratios


# ---------------------------------------------------------------------------
# (a) architecture with modified internals — GroupNorm backbone/FPN + GT anchors
# ---------------------------------------------------------------------------

class _GNBlock(nn.Module):
    """Conv -> GroupNorm -> ReLU. GroupNorm because detector batches are tiny (BN stats are noisy)."""

    def __init__(self, cin: int, cout: int, stride: int, groups: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.norm = nn.GroupNorm(groups, cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class _GNBackboneFPN(nn.Module):
    """From-scratch backbone + minimal top-down FPN, fully GroupNorm (batch-independent).

    Returns a single fused feature map; torchvision's GeneralizedRCNN wraps a bare tensor as the
    one-level feature dict ``{"0": ...}`` the detector's anchors / RoI pool are configured for.
    """

    def __init__(self, in_chans: int = 3, out_channels: int = 64, gn_groups: int = 8) -> None:
        super().__init__()
        self.stem = _GNBlock(in_chans, out_channels // 2, stride=2, groups=gn_groups)   # /2
        self.c2 = _GNBlock(out_channels // 2, out_channels, stride=2, groups=gn_groups)  # /4
        self.c3 = _GNBlock(out_channels, out_channels, stride=2, groups=gn_groups)       # /8
        self.lat2 = nn.Conv2d(out_channels, out_channels, 1)
        self.lat3 = nn.Conv2d(out_channels, out_channels, 1)
        self.fpn_norm = nn.GroupNorm(gn_groups, out_channels)   # GroupNorm in the FPN too, same reason
        self.smooth = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_norm = nn.GroupNorm(gn_groups, out_channels)
        self.act = nn.ReLU(inplace=True)
        self.out_channels = out_channels  # torchvision FasterRCNN reads this

    def forward(self, x):
        c2 = self.c2(self.stem(x))
        c3 = self.c3(c2)
        p2 = self.lat2(c2) + F.interpolate(self.lat3(c3), size=c2.shape[-2:], mode="nearest")
        p2 = self.act(self.fpn_norm(p2))
        return self.act(self.smooth_norm(self.smooth(p2)))


class BespokeGNDetector(nn.Module):
    """Wraps the torchvision Faster R-CNN as ``.detector`` (so the platform's eval / operating-point
    utilities that look for ``.detector`` work), with the same forward contract as the composed
    ``DetectionModel``: loss dict in train mode, ``list[dict]`` predictions in eval mode."""

    def __init__(self, detector: nn.Module) -> None:
        super().__init__()
        self.detector = detector

    def forward(self, images, targets=None):
        if isinstance(images, torch.Tensor):
            images = [images[i] for i in range(images.shape[0])]
        if self.training and targets is not None:
            return self.detector(images, targets)
        return self.detector(images)


def gt_anchor_sizes(gt_boxes_wh) -> tuple[int, ...]:
    """Anchor scales from the GT object-size distribution (p10/p50/p90 of sqrt(area)).

    Derived from the data in hand, not torchvision's fixed (32,64,128,256,512) — the anchors cover
    the sizes objects actually take in this dataset.
    """
    import numpy as np

    scales = [float(np.sqrt(w * h)) for (w, h) in gt_boxes_wh if w > 0 and h > 0]
    if not scales:
        return (32,)
    qs = np.quantile(scales, [0.1, 0.5, 0.9])
    return tuple(sorted({int(round(float(s))) for s in qs}))


def build_bespoke_detector(*, gt_boxes_wh, num_classes: int = 1, in_chans: int = 3,
                           out_channels: int = 64, gn_groups: int = 8,
                           min_size: int = 64, max_size: int = 128) -> BespokeGNDetector:
    """Build the bespoke detector with GT-derived anchors and a GroupNorm backbone/FPN.

    ``gt_boxes_wh`` is a list of ``(w, h)`` in pixels — the dataset's GT box shapes that drive the
    anchor derivation. Deterministic in its inputs, so re-importing the builder at inference rebuilds
    the identical architecture and the trained ``state_dict`` loads cleanly.
    """
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.rpn import AnchorGenerator
    from torchvision.ops import MultiScaleRoIAlign

    derived = gt_aspect_ratios(gt_boxes_wh)         # anchor aspect ratios from the GT box shapes
    ratios = tuple(derived) if derived is not None else (0.5, 1.0, 2.0)  # underivable -> stamped
    sizes = gt_anchor_sizes(gt_boxes_wh)            # anchor scales from the GT size distribution
    backbone = _GNBackboneFPN(in_chans, out_channels, gn_groups)
    anchor_generator = AnchorGenerator(sizes=(sizes,), aspect_ratios=(ratios,))  # single fused level
    roi_pool = MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)
    detector = FasterRCNN(
        backbone, num_classes=num_classes + 1,  # +1 background
        rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pool,
        min_size=min_size, max_size=max_size,
    )
    return BespokeGNDetector(detector)


# ---------------------------------------------------------------------------
# (b) a custom train(ctx) loop — hand-written, drives training through ctx sinks
# ---------------------------------------------------------------------------

def train_bespoke(ctx) -> None:
    """A from-scratch training loop (not ``ctx.default_train``).

    Composes the envelope's craft utilities (``build_optimizer`` / ``build_scheduler`` / ``evaluate``)
    and routes every metric + checkpoint through the audited/immutable ctx sinks, so integrity is the
    envelope's guarantee, not this loop's responsibility.
    """
    ctx.set_seed()
    device = ctx.device
    model = ctx.build_model().to(device)

    optimizer = ctx.build_optimizer("adamw", model, backbone_lr=1e-3, head_lr=1e-3, weight_decay=0.0)
    epochs = int(ctx.config.get("epochs", 2))
    scheduler = ctx.build_scheduler(optimizer, {"scheduler": {"type": "cosine"}}, epochs)

    best = float("inf")
    for epoch in range(1, epochs + 1):
        if ctx.should_cancel():
            break
        model.train()
        running, n = 0.0, 0
        for images, targets in ctx.train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
                       for t in targets]
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
            n += 1
        if scheduler is not None:
            scheduler.step()

        metrics = {"train_loss": running / max(n, 1)}
        if ctx.val_loader is not None:
            val = ctx.evaluate(model)
            metrics["val_loss"] = val.get("loss")
            metrics["val_map50"] = val.get("map50")
        ctx.log_metrics(epoch, metrics)   # envelope sink -> experiment store + metrics.jsonl + TB

        sel = metrics.get("val_loss")
        sel = metrics["train_loss"] if sel is None else sel
        if sel < best:
            best = sel
            ctx.run.best_metric = best
            ctx.save_checkpoint(   # envelope sink -> stamped (kind/model_source/config) + atomic
                {"model_state_dict": model.state_dict(), "metrics": {**metrics, "epoch": epoch}},
                "model_best")

    ctx.run.current_epoch = epochs
    # Guarantee a final checkpoint exists even if val never improved.
    ctx.save_checkpoint(
        {"model_state_dict": model.state_dict(), "metrics": {"epoch": epochs}}, "model_final")


# ---------------------------------------------------------------------------
# Sibling builders — agent-authored modules that wire the kept nn.Module blocks
# for the non-detector tasks. Same forward contract the trainer/eval expect:
# a loss dict in train mode (``head{i}_{k}``), decoded predictions in eval mode.
# ---------------------------------------------------------------------------

from tcip_mcp.pipelines.components.backbones import BackboneWrapper
from tcip_mcp.pipelines.components.detectors import BackboneNeckAdapter, build_detector
from tcip_mcp.pipelines.components.heads import (
    ClassificationHead,
    OrdinalHead,
    RegressionHead,
    SemanticSegHead,
)
from tcip_mcp.pipelines.components.necks import FPN, GlobalAvgPoolNeck


def _resnet18(in_chans: int = 3):
    import timm

    m = timm.create_model("resnet18", pretrained=False, features_only=True,
                          out_indices=(1, 2, 3, 4), in_chans=in_chans)
    return BackboneWrapper(m, m.feature_info.channels())


class BespokeComposed(nn.Module):
    """Backbone + neck + task head — the sibling of the removed ``ComposedModel``."""

    def __init__(self, backbone: nn.Module, neck: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.heads = nn.ModuleList([head])

    def forward(self, images, targets=None):
        feats = self.neck(self.backbone(images))
        out: dict[str, torch.Tensor] = {}
        if self.training and targets is not None:
            for i, head in enumerate(self.heads):
                o = head(feats, targets)
                for k, v in head.compute_loss(o, targets).items():
                    out[f"head{i}_{k}"] = v
        else:
            for i, head in enumerate(self.heads):
                for k, v in head.decode(head(feats)).items():
                    out[f"head{i}_{k}"] = v
        return out

    def freeze_backbone(self, to_stage: int) -> None:
        if hasattr(self.backbone, "freeze_to"):
            self.backbone.freeze_to(to_stage)

    def get_param_groups(self, backbone_lr: float = 1e-4, head_lr: float = 1e-3) -> list[dict]:
        head_params = [p for h in self.heads for p in h.parameters()]
        return [
            {"params": [p for p in self.backbone.parameters() if p.requires_grad], "lr": backbone_lr},
            {"params": [p for p in self.neck.parameters() if p.requires_grad], "lr": head_lr},
            {"params": [p for p in head_params if p.requires_grad], "lr": head_lr},
        ]


class BespokeDetection(nn.Module):
    """Real backbone + FPN fed into a torchvision detector via ``BackboneNeckAdapter`` —
    the detection / instance-seg sibling of the removed ``DetectionModel``."""

    def __init__(self, num_classes: int, *, in_chans: int = 3, detector: str = "faster_rcnn",
                 min_size: int = 800, max_size: int = 1333, **det_kwargs) -> None:
        super().__init__()
        self.backbone = _resnet18(in_chans)
        self.neck = FPN(self.backbone.out_channels, out_channels=256)
        adapter = BackboneNeckAdapter(self.backbone, self.neck)
        with torch.no_grad():
            names = list(adapter(torch.zeros(1, in_chans, 64, 64)).keys())
        self.detector = build_detector(
            detector, adapter, num_classes,
            featmap_names=names, num_levels=len(names),
            min_size=min_size, max_size=max_size, **det_kwargs,
        )

    def forward(self, images, targets=None):
        if isinstance(images, torch.Tensor):
            images = [images[i] for i in range(images.shape[0])]
        if self.training and targets is not None:
            return self.detector(images, targets)
        return self.detector(images)

    def freeze_backbone(self, to_stage: int) -> None:
        if hasattr(self.backbone, "freeze_to"):
            self.backbone.freeze_to(to_stage)


def build_bespoke_classifier(*, num_classes: int, in_chans: int = 3, dropout: float = 0.0):
    bb = _resnet18(in_chans)
    neck = GlobalAvgPoolNeck(bb.out_channels)
    return BespokeComposed(bb, neck, ClassificationHead(neck.out_channels, num_classes, dropout=dropout))


def build_bespoke_ordinal(*, num_ranks: int, in_chans: int = 3):
    bb = _resnet18(in_chans)
    neck = GlobalAvgPoolNeck(bb.out_channels)
    return BespokeComposed(bb, neck, OrdinalHead(neck.out_channels, num_ranks))


def build_bespoke_regressor(*, in_chans: int = 3):
    bb = _resnet18(in_chans)
    neck = GlobalAvgPoolNeck(bb.out_channels)
    return BespokeComposed(bb, neck, RegressionHead(neck.out_channels))


def build_bespoke_semantic_seg(*, num_classes: int, in_chans: int = 3):
    bb = _resnet18(in_chans)
    neck = FPN(bb.out_channels, out_channels=256)
    return BespokeComposed(bb, neck, SemanticSegHead(256, num_classes))


def build_bespoke_instance_seg(*, num_classes: int = 1, in_chans: int = 3,
                               min_size: int = 800, max_size: int = 1333, **det_kwargs):
    return BespokeDetection(num_classes, in_chans=in_chans, detector="mask_rcnn",
                            min_size=min_size, max_size=max_size, **det_kwargs)


def build_bespoke_detection(*, num_classes: int = 1, in_chans: int = 3, detector: str = "faster_rcnn",
                            min_size: int = 800, max_size: int = 1333, **det_kwargs):
    return BespokeDetection(num_classes, in_chans=in_chans, detector=detector,
                            min_size=min_size, max_size=max_size, **det_kwargs)

"""Bespoke builders and a custom training loop the tests name by dotted reference.

  (a) a from-scratch ``GroupNorm`` backbone and tiny FPN feeding a torchvision Faster R-CNN whose
      ``AnchorGenerator`` is built from the dataset's own GT box shapes
      (``pipelines.derivations.gt_aspect_ratios`` and the GT size distribution); and

  (b) a custom ``train(ctx)`` loop (``train_bespoke``) that trains through the ``ctx`` sinks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional

from tcip_mcp.pipelines.derivations import gt_aspect_ratios
from tcip_mcp.pipelines.model_build import METRICS_KEY, STATE_DICT_KEY


# ---------------------------------------------------------------------------
# (a) architecture with modified internals: GroupNorm backbone/FPN + GT anchors
# ---------------------------------------------------------------------------

class _GNBlock(nn.Module):
    """Conv -> GroupNorm -> ReLU."""

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
        # GroupNorm in the FPN too, same reason
        self.fpn_norm = nn.GroupNorm(gn_groups, out_channels)
        self.smooth = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_norm = nn.GroupNorm(gn_groups, out_channels)
        self.act = nn.ReLU(inplace=True)
        self.out_channels = out_channels  # torchvision FasterRCNN reads this

    def forward(self, x):
        c2 = self.c2(self.stem(x))
        c3 = self.c3(c2)
        p2 = self.lat2(c2) + functional.interpolate(
            self.lat3(c3), size=c2.shape[-2:], mode="nearest")
        p2 = self.act(self.fpn_norm(p2))
        return self.act(self.smooth_norm(self.smooth(p2)))


class _HeldDetector(nn.Module):
    """A torchvision detector held as ``.detector``: a loss dict in train mode, ``list[dict]``
    predictions in eval mode, a batched tensor split into its images first."""

    detector: nn.Module

    def forward(self, images, targets=None):
        if isinstance(images, torch.Tensor):
            images = [images[i] for i in range(images.shape[0])]
        if self.training and targets is not None:
            return self.detector(images, targets)
        return self.detector(images)


class _FreezesBackbone:
    """``freeze_backbone`` over a ``.backbone`` that may declare ``freeze_to``."""

    backbone: nn.Module

    def freeze_backbone(self, to_stage: int) -> None:
        if hasattr(self.backbone, "freeze_to"):
            self.backbone.freeze_to(to_stage)


class BespokeGNDetector(_HeldDetector):
    """The torchvision Faster R-CNN held as ``.detector``."""

    def __init__(self, detector: nn.Module) -> None:
        super().__init__()
        self.detector = detector


def gt_anchor_sizes(gt_boxes_wh) -> tuple[int, ...]:
    """Anchor scales from the GT object-size distribution (p10/p50/p90 of sqrt(area)), ``(32,)``
    for no positive box."""
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

    ``gt_boxes_wh`` is a list of ``(w, h)`` in pixels: the dataset's GT box shapes that drive the
    anchor derivation. Deterministic in its inputs, so re-importing the builder at inference
    rebuilds the identical architecture and the trained ``state_dict`` loads cleanly.
    """
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.rpn import AnchorGenerator
    from torchvision.ops import MultiScaleRoIAlign

    derived = gt_aspect_ratios(gt_boxes_wh)         # anchor aspect ratios from the GT box shapes
    ratios = tuple(derived) if derived is not None else (0.5, 1.0, 2.0)  # underivable -> stamped
    sizes = gt_anchor_sizes(gt_boxes_wh)            # anchor scales from the GT size distribution
    backbone = _GNBackboneFPN(in_chans, out_channels, gn_groups)
    # single fused level
    anchor_generator = AnchorGenerator(sizes=(sizes,), aspect_ratios=(ratios,))
    roi_pool = MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)
    detector = FasterRCNN(
        backbone, num_classes=num_classes + 1,  # +1 background
        rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pool,
        min_size=min_size, max_size=max_size,
    )
    return BespokeGNDetector(detector)


# ---------------------------------------------------------------------------
# (b) a custom train(ctx) loop: hand-written, drives training through ctx sinks
# ---------------------------------------------------------------------------

def train_bespoke(ctx) -> None:
    """A from-scratch training loop composing the envelope's craft utilities (``build_optimizer`` /
    ``build_scheduler`` / ``evaluate``) that records every metric and checkpoint through the ctx
    sinks, its best epoch's weights saved once as ``model_best`` when the loop ends.
    """
    ctx.set_seed()
    device = ctx.device
    model = ctx.build_model().to(device)

    optimizer = ctx.build_optimizer(
        "adamw", model, backbone_lr=1e-3, head_lr=1e-3, weight_decay=0.0
    )
    epochs = int(ctx.config.get("epochs", 2))
    scheduler = ctx.build_scheduler(optimizer, {"type": "cosine"}, epochs)

    best = float("inf")
    best_state = None
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
            best_state = {STATE_DICT_KEY: {k: v.detach().cpu().clone()
                                           for k, v in model.state_dict().items()},
                          METRICS_KEY: {**metrics, "epoch": epoch}}

    ctx.run.current_epoch = epochs
    if best_state is not None:
        ctx.save_checkpoint(best_state, "model_best")
    ctx.save_checkpoint(
        {STATE_DICT_KEY: model.state_dict(), METRICS_KEY: {"epoch": epochs}}, "model_final")


def save_built_weights(ctx) -> None:
    """A training body that takes no step: logs each of the config's ``fixture_rows`` (an
    ``epoch`` plus its metrics) through the run's own metrics log, then saves the model its config
    builds, as built under the run's own seed, as the run's final weights, carrying the config's
    ``fixture_metrics`` as the checkpoint's metrics when it states any."""
    for row in ctx.config.get("fixture_rows") or []:
        ctx.log_metrics(row["epoch"], {k: v for k, v in row.items() if k != "epoch"})
    ctx.set_seed()
    state = {STATE_DICT_KEY: ctx.build_model().state_dict()}
    if ctx.config.get("fixture_metrics"):
        state[METRICS_KEY] = dict(ctx.config["fixture_metrics"])
    ctx.save_checkpoint(state, "model_final")


# ---------------------------------------------------------------------------
# Sibling builders: agent-authored modules that wire the kept nn.Module blocks
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


class BespokeComposed(_FreezesBackbone, nn.Module):
    """Backbone + neck + task head: the sibling of the removed ``ComposedModel``."""

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

    def get_param_groups(self, backbone_lr: float = 1e-4, head_lr: float = 1e-3) -> list[dict]:
        head_params = [p for h in self.heads for p in h.parameters()]
        return [
            {"params": [p for p in self.backbone.parameters() if p.requires_grad],
             "lr": backbone_lr},
            {"params": [p for p in self.neck.parameters() if p.requires_grad], "lr": head_lr},
            {"params": [p for p in head_params if p.requires_grad], "lr": head_lr},
        ]


class BespokeDetection(_FreezesBackbone, _HeldDetector):
    """Real backbone + FPN fed into a torchvision detector via ``BackboneNeckAdapter``:
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


def build_bespoke_classifier(*, num_classes: int, in_chans: int = 3, dropout: float = 0.0):
    bb = _resnet18(in_chans)
    neck = GlobalAvgPoolNeck(bb.out_channels)
    return BespokeComposed(
        bb, neck, ClassificationHead(neck.out_channels, num_classes, dropout=dropout)
    )


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


def build_bespoke_detection(*, num_classes: int = 1, in_chans: int = 3,
                            detector: str = "faster_rcnn",
                            min_size: int = 800, max_size: int = 1333, **det_kwargs):
    return BespokeDetection(num_classes, in_chans=in_chans, detector=detector,
                            min_size=min_size, max_size=max_size, **det_kwargs)


class _OneConvDetector(nn.Module):
    """A detector whose one parameter is a 1x1 convolution, carrying ``score_thresh`` on itself;
    :meth:`zero_loss` is its training forward's answer."""

    def __init__(self, in_chans: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_chans, 1, 1)
        self.score_thresh = 0.0

    def zero_loss(self, images) -> dict:
        return {"loss": sum(self.conv(im.unsqueeze(0)).sum() for im in images) * 0.0}


class FixedMaskSegmenter(_OneConvDetector):
    """An instance segmenter whose eval forward returns, per image, one box over its central half
    scored 0.9 and a mask of ones filling that box, so every pass carries a real mask."""

    def forward(self, images, targets=None):
        from tests._producer_fixtures import painted_array

        if self.training:
            return self.zero_loss(images)
        results = []
        for im in images:
            h, w = int(im.shape[-2]), int(im.shape[-1])
            y0, y1, x0, x1 = h // 4, 3 * h // 4, w // 4, 3 * w // 4
            mask = torch.as_tensor(painted_array(w, h, [((x0, y0, x1, y1), 1.0)],
                                                 background=0.0, mode="F"))[None, None]
            results.append({"boxes": torch.tensor([[x0, y0, x1, y1]], dtype=torch.float32),
                            "scores": torch.tensor([0.9]), "labels": torch.tensor([1]),
                            "masks": mask})
        return results


def build_fixed_mask_instance_seg(*, num_classes: int = 1, in_chans: int = 3):
    return FixedMaskSegmenter(in_chans=in_chans)


# A non-torchvision detector, no .detector to route through.
class BareScoreThreshDetector(_OneConvDetector):
    """A hand-rolled detector with no ``.detector``: exposes ``score_thresh`` on itself, and
    honors it in its own eval-mode forward, one fixed box per image kept only when it clears the
    threshold. The proof that the operating-point holder resolves to the module itself when it is
    the only thing exposing a knob.
    """

    def forward(self, images):
        if self.training:
            return self.zero_loss(images)
        results = []
        for im in images:
            h, w = int(im.shape[-2]), int(im.shape[-1])
            boxes = torch.tensor([[w * 0.25, h * 0.25, w * 0.75, h * 0.75]], dtype=torch.float32)
            scores = torch.tensor([0.9])
            labels = torch.tensor([1])
            keep = scores >= self.score_thresh
            results.append({"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]})
        return results


# The same shape with no operating-point knob under any recognized name anywhere: the case the
# row demonstrates, where the platform genuinely has nothing it can set.
class BareNoKnobDetector(nn.Module):
    def __init__(self, in_chans: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_chans, 1, 1)

    def forward(self, images):
        if self.training:
            return {"loss": sum(self.conv(im.unsqueeze(0)).sum() for im in images) * 0.0}
        results = []
        for im in images:
            h, w = int(im.shape[-2]), int(im.shape[-1])
            boxes = torch.tensor([[w * 0.25, h * 0.25, w * 0.75, h * 0.75]], dtype=torch.float32)
            results.append(
                {"boxes": boxes, "scores": torch.tensor([0.9]), "labels": torch.tensor([1])})
        return results


def build_bare_score_thresh_detector(*, in_chans: int = 3, num_classes: int = 1):
    return BareScoreThreshDetector(in_chans=in_chans)


def build_bare_no_knob_detector(*, in_chans: int = 3, num_classes: int = 1):
    return BareNoKnobDetector(in_chans=in_chans)


class DivergingDetection(nn.Module):
    """Wraps ``BespokeDetection``, adding one parameter whose loss term drives the training loss
    to ``nan`` a few optimizer steps in: the fixture ``overfit_check``'s non-finite-loss rendering
    guards against, since a real training pass can diverge exactly this way."""

    def __init__(self, num_classes: int = 1, **kwargs) -> None:
        super().__init__()
        self.inner = BespokeDetection(num_classes, **kwargs)
        self.w = nn.Parameter(torch.tensor(0.02))

    @property
    def detector(self):
        return self.inner.detector

    def forward(self, images, targets=None):
        out = self.inner(images, targets)
        if self.training and targets is not None:
            out = dict(out)
            out["diverge"] = torch.log(self.w)  # drives w negative under Adam, losses -> nan
        return out

    def freeze_backbone(self, to_stage: int) -> None:
        self.inner.freeze_backbone(to_stage)


def build_diverging_detection(*, num_classes: int = 1, in_chans: int = 3, **det_kwargs):
    return DivergingDetection(num_classes, in_chans=in_chans, **det_kwargs)


class BrightRegionDetector(nn.Module):
    """A detector whose box geometry is analytic and whose confidence is learned.

    Finds the bright region of a frame and reports its extent as one box, beside one low-scored
    decoy, so a conf sweep has two score levels to choose between and a count-unbiased operating
    point exists between them. The learned ``logit`` scales the confidence the true box carries,
    fit against the labels, so a training pass is a real optimization over real data while the
    geometry stays fixed and a chain measured over this model's counts is measuring the chain.
    """

    def __init__(self, in_chans: int = 3, bright: float = 0.5, decoy_score: float = 0.03) -> None:
        super().__init__()
        self.logit = nn.Parameter(torch.tensor([2.5]))
        self.bright = float(bright)
        # Just above the floor a calibration stages a reference at, so the swept curve sees the
        # low-confidence tail it assumes rather than reading the reference as truncated.
        self.decoy_score = float(decoy_score)
        self.score_thresh = 0.0
        self.nms_thresh = 0.5
        self.detections_per_img = 100

    def _extent(self, image):
        """The bounding box of the frame's bright pixels, or ``None`` when it holds none."""
        mask = image.mean(dim=0) > self.bright
        ys, xs = torch.nonzero(mask, as_tuple=True)
        if xs.numel() == 0:
            return None
        return torch.tensor(
            [float(xs.min()), float(ys.min()), float(xs.max()) + 1.0, float(ys.max()) + 1.0],
            dtype=torch.float32, device=image.device,
        )

    def forward(self, images, targets=None):
        if isinstance(images, torch.Tensor):
            images = [images[i] for i in range(images.shape[0])]
        confidence = torch.sigmoid(self.logit)
        if self.training and targets is not None:
            # Fit the reported confidence toward one per labeled object present, so the
            # parameter answers to the data rather than drifting free.
            present = torch.tensor(
                [1.0 if len(t.get("boxes", [])) else 0.0 for t in targets],
                dtype=torch.float32,
            )
            return {"confidence": ((confidence - present.mean()) ** 2)}
        results = []
        for image in images:
            device = image.device
            extent = self._extent(image)
            if extent is None:
                results.append({"boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                                "scores": torch.zeros(0, device=device),
                                "labels": torch.zeros(0, dtype=torch.int64, device=device)})
                continue
            decoy = extent + torch.ones(4, dtype=torch.float32, device=device)
            boxes = torch.stack([extent, decoy])
            reported = confidence.squeeze().detach().to(device)
            scores = torch.stack(
                [reported, torch.tensor(self.decoy_score, dtype=torch.float32, device=device)])
            labels = torch.ones(2, dtype=torch.int64, device=device)
            keep = scores >= self.score_thresh
            results.append({"boxes": boxes[keep][: self.detections_per_img],
                            "scores": scores[keep][: self.detections_per_img],
                            "labels": labels[keep][: self.detections_per_img]})
        return results


def build_bright_region_detector(*, in_chans: int = 3, num_classes: int = 1, bright: float = 0.5,
                                 decoy_score: float = 0.03) -> BrightRegionDetector:
    """``model_source`` builder for :class:`BrightRegionDetector`, deterministic in its inputs."""
    return BrightRegionDetector(in_chans=in_chans, bright=bright, decoy_score=decoy_score)


class BrightBlobDetector(nn.Module):
    """One detection per connected bright blob of a frame, labeled the one subject and scored by
    a learned confidence (0.5 untrained) plus the fraction of the frame the blob covers, capped
    at 1, with the blob's own soft mask when ``with_masks``. A blob is bright where the band mean
    exceeds 0.5; with ``attributes`` (the count of attributes it is built at) it is bright where
    one band exceeds 0.5, and carries that band's index as its first attribute's id and 0 for
    every other. A training step's loss is the squared distance of the confidence from each
    frame's label presence (one when it holds a labeled object, zero when not). Every inference
    forward records each input's channel count in ``seen_channels`` and its per-band peak value
    in ``seen_band_peaks``."""

    def __init__(self, in_chans: int = 3, with_masks: bool = False, attributes: int = 0) -> None:
        super().__init__()
        self.logit = nn.Parameter(torch.tensor([0.0]))
        self.with_masks = with_masks
        self.attributes = attributes
        self.score_thresh = 0.0
        self.nms_thresh = 0.45  # never applied here; a sentinel a pass must leave unchanged
        self.seen_channels: list[int] = []
        self.seen_band_peaks: list[list[float]] = []

    def forward(self, images, targets=None):
        import cv2
        import numpy as np

        confidence = torch.sigmoid(self.logit)
        if self.training and targets is not None:
            present = torch.tensor([1.0 if len(t.get("boxes", [])) else 0.0 for t in targets])
            return {"loss": ((confidence - present.to(confidence.device)) ** 2).mean()}
        reported = float(confidence.detach())
        results = []
        for image in images:
            self.seen_channels.append(int(image.shape[0]))
            self.seen_band_peaks.append(image.amax(dim=(1, 2)).tolist())
            h, w = int(image.shape[-2]), int(image.shape[-1])
            planes = ([(c, image[c] > 0.5) for c in range(image.shape[0])]
                      if self.attributes else [(0, image.mean(dim=0) > 0.5)])
            boxes, scores, masks, values = [], [], [], []
            for value, plane in planes:
                bright = plane.cpu().numpy().astype(np.uint8)
                n, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=4)
                for k in range(1, n):
                    x, y, bw, bh, area = (int(v) for v in stats[k])
                    boxes.append([float(x), float(y), float(x + bw), float(y + bh)])
                    scores.append(min(1.0, reported + area / float(h * w)))
                    masks.append(torch.from_numpy((labels == k).astype(np.float32))[None])
                    values.append(value)
            out = {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                   "scores": torch.tensor(scores, dtype=torch.float32),
                   "labels": torch.ones(len(boxes), dtype=torch.int64)}
            if self.attributes:
                out["attributes"] = torch.tensor(
                    [[v] + [0] * (self.attributes - 1) for v in values],
                    dtype=torch.int64).reshape(-1, self.attributes)
            if self.with_masks:
                out["masks"] = (torch.stack(masks) if masks
                                else torch.zeros((0, 1, h, w), dtype=torch.float32))
            results.append(out)
        return results


def build_bright_blob_detector(*, in_chans: int = 3, num_classes: int = 1, with_masks: bool = False,
                               attributes: tuple = ()) -> BrightBlobDetector:
    """``model_source`` builder for :class:`BrightBlobDetector`, at the attributes the platform
    hands it."""
    return BrightBlobDetector(in_chans=in_chans, with_masks=with_masks,
                              attributes=len(attributes))


class WholeBlobDetector(BrightBlobDetector):
    """A :class:`BrightBlobDetector` whose first attribute's id says whether the frame holds the
    blob whole: 1 for a blob clear of every border, 0 for one a border cuts."""

    def forward(self, images, targets=None):
        results = super().forward(images, targets)
        if self.training:
            return results
        for image, out in zip(images, results):
            h, w = image.shape[-2:]
            b = out["boxes"]
            out["attributes"][:, 0] = ((b[:, 0] > 0) & (b[:, 1] > 0) & (b[:, 2] < w)
                                       & (b[:, 3] < h)).long()
        return results


def build_whole_blob_detector(*, in_chans: int = 3, num_classes: int = 1,
                              attributes: tuple = ()) -> WholeBlobDetector:
    """``model_source`` builder for :class:`WholeBlobDetector`, at the attributes the platform
    hands it."""
    return WholeBlobDetector(in_chans=in_chans, attributes=len(attributes))

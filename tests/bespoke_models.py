"""Bespoke model builders and training sources importable by dotted reference.

Among them are (a) a from-scratch ``GroupNorm`` backbone and tiny FPN feeding a torchvision
RetinaNet whose ``AnchorGenerator`` is built at the anchor geometry it is handed, and (b) a
custom ``train(ctx)`` loop (``train_bespoke``) that trains through the ``ctx`` sinks.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.nn import functional

from tcip_mcp.pipelines.model_build import METRICS_KEY, STATE_DICT_KEY

if TYPE_CHECKING:
    from torchvision.models.detection.anchor_utils import AnchorGenerator

    from tcip_mcp.pipelines.training.envelope import TrainContext
    from tcip_mcp.subject_registry import Attribute

DetectorOutput = dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]
"""A detection model's forward: its losses in train mode, one prediction per image in eval."""

# ---------------------------------------------------------------------------
# (a) architecture with modified internals: GroupNorm backbone/FPN + GT anchors
# ---------------------------------------------------------------------------


class _GNBlock(nn.Module):
    """A block of a convolution, a GroupNorm and a ReLU, in that order."""

    def __init__(self, cin: int, cout: int, stride: int, groups: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            cin, cout, 3, stride=stride, padding=1, bias=False
        )
        self.norm = nn.GroupNorm(groups, cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class _GNBackboneFPN(nn.Module):
    """From-scratch backbone + minimal top-down FPN, fully GroupNorm (batch-independent).

    Its forward returns one fused feature map, keyed ``"0"``.
    """

    def __init__(
        self, in_chans: int = 3, out_channels: int = 64, gn_groups: int = 8
    ) -> None:
        super().__init__()
        self.stem = _GNBlock(
            in_chans, out_channels // 2, stride=2, groups=gn_groups
        )  # /2
        self.c2 = _GNBlock(
            out_channels // 2, out_channels, stride=2, groups=gn_groups
        )  # /4
        self.c3 = _GNBlock(
            out_channels, out_channels, stride=2, groups=gn_groups
        )  # /8
        self.lat2 = nn.Conv2d(out_channels, out_channels, 1)
        self.lat3 = nn.Conv2d(out_channels, out_channels, 1)
        # GroupNorm in the FPN too, same reason
        self.fpn_norm = nn.GroupNorm(gn_groups, out_channels)
        self.smooth = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_norm = nn.GroupNorm(gn_groups, out_channels)
        self.act = nn.ReLU(inplace=True)
        self.out_channels = out_channels  # a torchvision detector reads this

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        c2 = self.c2(self.stem(x))
        c3 = self.c3(c2)
        p2 = self.lat2(c2) + functional.interpolate(
            self.lat3(c3), size=c2.shape[-2:], mode="nearest"
        )
        p2 = self.act(self.fpn_norm(p2))
        return OrderedDict({"0": self.act(self.smooth_norm(self.smooth(p2)))})


class _FreezesBackbone:
    """A mixin giving ``freeze_backbone`` through its ``.backbone``'s own ``freeze_to``."""

    backbone: nn.Module

    def freeze_backbone(self, to_stage: int) -> None:
        self.backbone.freeze_to(to_stage)


class _KeepsItsOperatingPoint:
    """A mixin keeping a detector's eval output to its own two operating-point knobs."""

    score_thresh: float
    detections_per_img: int

    def governed(self, out: dict) -> dict:
        """Return ``out``'s rows scoring at least ``score_thresh``, at most ``detections_per_img``.

        The rows kept are the highest-scoring ones, chosen by one index selection over every key.
        """
        order = torch.argsort(out["scores"], descending=True)
        keep = order[out["scores"][order] >= self.score_thresh][
            : self.detections_per_img
        ]
        return {key: value[keep] for key, value in out.items()}


def build_bespoke_detector(
    *,
    min_size: int,
    max_size: int,
    anchor_base_size: int,
    aspect_ratios: tuple[float, ...],
    image_mean: list[float],
    image_std: list[float],
    num_classes: int = 1,
    in_chans: int = 3,
    out_channels: int = 64,
    gn_groups: int = 8,
    attributes: tuple[Attribute, ...] = (),
) -> BespokeDetection:
    """Build a :class:`BespokeDetection` over a RetinaNet on a GroupNorm backbone/FPN.

    The RetinaNet is built at ``min_size``, ``max_size``, ``image_mean`` and ``image_std``, its
    one fused level's anchors at ``anchor_base_size`` and its next two doublings over
    ``aspect_ratios``; the module carries one head per record of ``attributes``.
    """
    from torchvision.models.detection import RetinaNet
    from torchvision.models.detection.anchor_utils import AnchorGenerator

    backbone = _GNBackboneFPN(in_chans, out_channels, gn_groups)
    library = RetinaNet(
        backbone,
        num_classes + 1,
        anchor_generator=AnchorGenerator(
            sizes=(tuple(anchor_base_size * 2**i for i in range(3)),),
            aspect_ratios=(aspect_ratios,),
        ),
        min_size=min_size,
        max_size=max_size,
        image_mean=image_mean,
        image_std=image_std,
    )
    return BespokeDetection(
        library, backbone, featmap_names=["0"], attributes=attributes
    )


# ---------------------------------------------------------------------------
# (b) a custom train(ctx) loop: hand-written, drives training through ctx sinks
# ---------------------------------------------------------------------------


def train_bespoke(ctx: TrainContext) -> None:
    """Train from scratch, composing the envelope's craft utilities.

    The loop composes ``build_optimizer`` / ``build_scheduler`` / ``evaluate`` and records every
    metric and checkpoint through the ctx sinks; when the loop ends its best epoch's weights are
    saved as ``model_best`` and its last weights as ``model_final``.
    """
    ctx.set_seed()
    device = ctx.device
    model = ctx.build_model().to(device)

    optimizer = ctx.build_optimizer(model)
    scheduler = ctx.build_scheduler(optimizer)
    epochs = ctx.spec.scheduler.horizon_epochs

    best = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        if ctx.should_cancel():
            break
        model.train()
        running, n = 0.0, 0
        for images, targets in ctx.train_loader:
            images = [img.to(device) for img in images]
            targets = [
                {
                    k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in t.items()
                }
                for t in targets
            ]
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
        ctx.log_metrics(
            epoch, metrics
        )  # envelope sink -> experiment store + metrics.jsonl + TB

        sel = metrics.get("val_loss")
        sel = metrics["train_loss"] if sel is None else sel
        if sel < best:
            best = sel
            ctx.run.best_metric = best
            best_state = {
                STATE_DICT_KEY: {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                },
                METRICS_KEY: {**metrics, "epoch": epoch},
            }

    ctx.run.current_epoch = epochs
    if best_state is not None:
        ctx.save_checkpoint(best_state, "model_best")
    ctx.save_checkpoint(
        {STATE_DICT_KEY: model.state_dict(), METRICS_KEY: {"epoch": epochs}},
        "model_final",
    )


def save_built_weights(ctx: TrainContext) -> None:
    """Save the model the config builds as the run's final weights, taking no training step.

    Each of the config's ``fixture_rows`` (an ``epoch`` plus its metrics) is logged through the
    run's own metrics log first; the model is saved as built under the run's own seed, carrying
    the config's ``fixture_metrics`` as the checkpoint's metrics when it states any.
    """
    own = ctx.spec.model_extra
    for row in own.get("fixture_rows") or []:
        ctx.log_metrics(
            row["epoch"], {k: v for k, v in row.items() if k != "epoch"}
        )
    ctx.set_seed()
    state = {STATE_DICT_KEY: ctx.build_model().state_dict()}
    if own.get("fixture_metrics"):
        state[METRICS_KEY] = own["fixture_metrics"]
    ctx.save_checkpoint(state, "model_final")


UNENCODABLE_METRICS: dict[str, dict] = {
    "integer_key": {"val_per_class_iou": {0: 0.5}},
    "non_finite": {"val_loss": float("nan")},
    "bytes_key": {b"val_loss": 0.5},
}
"""Checkpoint metrics no run config can state, each one JSON cannot hold, by the case it is."""


def save_built_weights_with_unencodable_metrics(ctx: TrainContext) -> None:
    """Save the built model as the run's final weights with metrics JSON cannot hold.

    The metrics are the :data:`UNENCODABLE_METRICS` case the config's ``fixture_unencodable``
    names.
    """
    ctx.set_seed()
    ctx.save_checkpoint(
        {
            STATE_DICT_KEY: ctx.build_model().state_dict(),
            METRICS_KEY: UNENCODABLE_METRICS[
                ctx.spec.model_extra["fixture_unencodable"]
            ],
        },
        "model_final",
    )


SECOND_CHECKPOINT_TAG = "second"
"""The tag :func:`save_built_weights_and_a_second_checkpoint` saves its second checkpoint under."""


def save_built_weights_and_a_second_checkpoint(ctx: TrainContext) -> None:
    """Run :func:`save_built_weights`, then save the built model again as a second checkpoint.

    The second save is under :data:`SECOND_CHECKPOINT_TAG`, beside the config's
    ``fixture_state`` when it states any: a checkpoint the run wrote beside the deliverable its
    completion registers.
    """
    save_built_weights(ctx)
    ctx.save_checkpoint(
        {
            STATE_DICT_KEY: ctx.build_model().state_dict(),
            **(ctx.spec.model_extra.get("fixture_state") or {}),
        },
        SECOND_CHECKPOINT_TAG,
    )


# ---------------------------------------------------------------------------
# Sibling builders: agent-authored modules that wire the kept nn.Module blocks
# for the non-detector tasks. Same forward contract the trainer/eval expect:
# a loss dict in train mode (``head{i}_{k}``), decoded predictions in eval mode.
# ---------------------------------------------------------------------------

from tcip_mcp.pipelines.components.backbones import BackboneWrapper
from tcip_mcp.pipelines.components.heads import (
    ClassificationHead,
    OrdinalHead,
    RegressionHead,
    SemanticSegHead,
)
from tcip_mcp.pipelines.components.necks import FPN, GlobalAvgPoolNeck


def _resnet18(in_chans: int = 3) -> BackboneWrapper:
    import timm

    m = timm.create_model(
        "resnet18",
        pretrained=False,
        features_only=True,
        out_indices=(1, 2, 3, 4),
        in_chans=in_chans,
    )
    return BackboneWrapper(m, m.feature_info.channels())


class BespokeComposed(_FreezesBackbone, nn.Module):
    """Backbone + neck + task head over normalized images.

    The images are normalized by the per-band ``image_mean`` and ``image_std`` it is built with.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        head: nn.Module,
        image_mean: list[float],
        image_std: list[float],
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.heads = nn.ModuleList([head])
        self.image_mean, self.image_std = image_mean, image_std

    def forward(
        self, images: torch.Tensor, targets: dict | None = None
    ) -> dict[str, torch.Tensor]:
        from torchvision.transforms.functional import normalize

        images = normalize(images, self.image_mean, self.image_std)
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

    def get_param_groups(
        self, backbone_lr: float = 1e-4, head_lr: float = 1e-3
    ) -> list[dict]:
        head_params = [p for h in self.heads for p in h.parameters()]
        return [
            {
                "params": [
                    p for p in self.backbone.parameters() if p.requires_grad
                ],
                "lr": backbone_lr,
            },
            {
                "params": [
                    p for p in self.neck.parameters() if p.requires_grad
                ],
                "lr": head_lr,
            },
            {
                "params": [p for p in head_params if p.requires_grad],
                "lr": head_lr,
            },
        ]


class _Pyramid(nn.Module):
    """A ResNet-18 and a 256-channel FPN over it, read by a torchvision detector as its backbone.

    Its forward returns the FPN's levels keyed by ``featmap_names``, ``"0"`` the finest;
    ``out_channels`` is the FPN's width.
    """

    def __init__(self, in_chans: int) -> None:
        super().__init__()
        self.backbone = _resnet18(in_chans)
        self.neck = FPN(self.backbone.out_channels, out_channels=256)
        self.out_channels = self.neck.out_channels
        self.featmap_names = [
            str(i) for i in range(len(self.backbone.out_channels))
        ]

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        levels = self.neck(self.backbone(x)).values()
        return OrderedDict(zip(self.featmap_names, levels, strict=True))

    def anchors(
        self, anchor_base_size: int, aspect_ratios: tuple[float, ...]
    ) -> AnchorGenerator:
        """Return one anchor size per level, ``anchor_base_size`` at the finest, doubling each.

        Every level's anchors take ``aspect_ratios``.
        """
        from torchvision.models.detection.anchor_utils import AnchorGenerator

        levels = len(self.featmap_names)
        return AnchorGenerator(
            sizes=tuple((anchor_base_size * 2**i,) for i in range(levels)),
            aspect_ratios=(tuple(aspect_ratios),) * levels,
        )


class BespokeDetection(_FreezesBackbone, _KeepsItsOperatingPoint, nn.Module):
    """A thin module over a torchvision detector, owning its operating point and attribute heads.

    ``library`` is the torchvision detector it calls and ``backbone`` the module inside it
    trained at the backbone rate. ``score_thresh`` and ``detections_per_img`` start at the
    library's own; the library's floor is then held at zero and its cap set to the module's
    before each eval forward, so no floor or cap of the library's is stricter than the module's
    (its own candidate limits and NMS still apply), and each frame's output passes through
    :meth:`governed`. Each of ``attributes``, the scope's records,
    gets a ``ClassificationHead`` (categorical) or an ``OrdinalHead`` (ordinal) reading the
    channel means ``MultiScaleRoIAlign`` pools at a box from the library backbone's levels
    ``featmap_names``, run over the frame as given. In train mode each head's loss over the rows
    its target column assesses (not ``UNASSESSED``) joins the library's losses as
    ``attribute_loss_<name>``, zero when no row does; in eval every output carries
    ``attributes``, one int64 row per box, its columns in declared order.
    """

    def __init__(
        self,
        library: nn.Module,
        backbone: nn.Module,
        *,
        featmap_names: list[str],
        attributes: tuple[Attribute, ...] = (),
    ) -> None:
        from torchvision.ops import MultiScaleRoIAlign

        from tcip_annotation.state import ATTR_TYPES

        super().__init__()
        self.library = library
        self.backbone = backbone
        held = self.library_knobs
        self.score_thresh = float(held.score_thresh)
        self.detections_per_img = int(held.detections_per_img)
        held.score_thresh = 0.0
        self.attributes = tuple(attributes)
        # torchvision's own box-head pool: a 7-cell edge, two samples per cell.
        self.pool = MultiScaleRoIAlign(
            featmap_names, output_size=7, sampling_ratio=2
        )
        head_of = dict(
            zip(ATTR_TYPES, (ClassificationHead, OrdinalHead), strict=True)
        )
        self.attribute_heads = nn.ModuleList(
            head_of[a.type](library.backbone.out_channels, len(a.values))
            for a in self.attributes
        )

    @property
    def library_knobs(self) -> nn.Module:
        """The module holding the library's own two operating-point knobs."""
        return self.library

    def forward(
        self,
        images: torch.Tensor | list[torch.Tensor],
        targets: list[dict] | None = None,
    ) -> DetectorOutput:
        if isinstance(images, torch.Tensor):
            images = list(images.unbind(0))
        if self.training and targets is not None:
            return {
                **self.library(images, targets),
                **self._attribute_losses(images, targets),
            }
        self.library_knobs.detections_per_img = self.detections_per_img
        results = self.library(images)
        if self.attributes:
            boxes = [r["boxes"] for r in results]
            for result, rows in zip(
                results, self._attribute_ids(images, boxes)
            ):
                result["attributes"] = rows
        return [self.governed(r) for r in results]

    def _pooled(
        self, images: list[torch.Tensor], boxes: list[torch.Tensor]
    ) -> torch.Tensor:
        """Return the channel means pooled at each image's ``boxes``, every image's in turn."""
        transform, levels = self.library.transform, self.library.backbone
        return torch.cat(
            [
                self.pool(
                    levels(transform.normalize(image)[None]),
                    [at],
                    [tuple(image.shape[-2:])],
                ).mean(dim=(2, 3))
                for image, at in zip(images, boxes, strict=True)
            ]
        )

    def _attribute_ids(
        self, images: list[torch.Tensor], boxes: list[torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        """Return each image's attribute ids at its ``boxes``, boxes by attributes, int64."""
        pooled = self._pooled(images, boxes)
        ids = torch.stack(
            [
                head.decode(head(pooled))[head.target_key]
                for head in self.attribute_heads
            ],
            dim=1,
        ).long()
        return ids.split([len(at) for at in boxes])

    def _attribute_losses(
        self, images: list[torch.Tensor], targets: list[dict]
    ) -> dict[str, torch.Tensor]:
        """Return each head's loss over the target rows its attribute assesses, by name."""
        from tcip_annotation.json_io import UNASSESSED

        if not self.attributes:
            return {}
        pooled = self._pooled(images, [t["boxes"] for t in targets])
        truth = torch.cat([t["attributes"] for t in targets]).reshape(
            -1, len(self.attributes)
        )
        losses = {}
        for i, (attribute, head) in enumerate(
            zip(self.attributes, self.attribute_heads)
        ):
            logits = head(pooled)["logits"]
            assessed = truth[:, i] != UNASSESSED
            losses[f"attribute_loss_{attribute.name}"] = (
                sum(
                    head.compute_loss(
                        {"logits": logits[assessed]},
                        {head.target_key: truth[assessed, i]},
                    ).values()
                )
                if assessed.any()
                else logits.sum() * 0.0
            )
        return losses


class BespokeInstanceSeg(BespokeDetection):
    """A :class:`BespokeDetection` over a two-stage library, its knobs on its ``roi_heads``."""

    @property
    def library_knobs(self) -> nn.Module:
        """The library's ``roi_heads``, which hold its two operating-point knobs."""
        return self.library.roi_heads


def _over_retinanet(
    cls: type[BespokeDetection],
    *,
    anchor_base_size: int,
    aspect_ratios: tuple[float, ...],
    num_classes: int,
    in_chans: int,
    attributes: tuple[Attribute, ...],
    **library_kwargs: object,
) -> BespokeDetection:
    """Build ``cls`` over a RetinaNet on a :class:`_Pyramid` of ``in_chans`` bands.

    The RetinaNet's anchors are :meth:`_Pyramid.anchors`' at ``anchor_base_size`` and
    ``aspect_ratios``; ``library_kwargs`` (its resize bounds, ``image_mean``, ``image_std`` and
    any other RetinaNet keyword) pass to its constructor.
    """
    from torchvision.models.detection import RetinaNet

    pyramid = _Pyramid(in_chans)
    library = RetinaNet(
        pyramid,
        num_classes + 1,
        anchor_generator=pyramid.anchors(anchor_base_size, aspect_ratios),
        **library_kwargs,
    )
    return cls(
        library,
        pyramid.backbone,
        featmap_names=pyramid.featmap_names,
        attributes=attributes,
    )


def build_bespoke_classifier(
    *,
    num_classes: int,
    image_mean: list[float],
    image_std: list[float],
    in_chans: int = 3,
    dropout: float = 0.0,
) -> BespokeComposed:
    bb = _resnet18(in_chans)
    neck = GlobalAvgPoolNeck(bb.out_channels)
    return BespokeComposed(
        bb,
        neck,
        ClassificationHead(neck.out_channels, num_classes, dropout=dropout),
        image_mean,
        image_std,
    )


def build_bespoke_ordinal(
    *,
    num_ranks: int,
    image_mean: list[float],
    image_std: list[float],
    in_chans: int = 3,
) -> BespokeComposed:
    bb = _resnet18(in_chans)
    neck = GlobalAvgPoolNeck(bb.out_channels)
    return BespokeComposed(
        bb,
        neck,
        OrdinalHead(neck.out_channels, num_ranks),
        image_mean,
        image_std,
    )


def build_bespoke_regressor(
    *, image_mean: list[float], image_std: list[float], in_chans: int = 3
) -> BespokeComposed:
    bb = _resnet18(in_chans)
    neck = GlobalAvgPoolNeck(bb.out_channels)
    return BespokeComposed(
        bb, neck, RegressionHead(neck.out_channels), image_mean, image_std
    )


def build_bespoke_semantic_seg(
    *,
    num_classes: int,
    image_mean: list[float],
    image_std: list[float],
    in_chans: int = 3,
) -> BespokeComposed:
    bb = _resnet18(in_chans)
    neck = FPN(bb.out_channels, out_channels=256)
    return BespokeComposed(
        bb, neck, SemanticSegHead(256, num_classes), image_mean, image_std
    )


def build_bespoke_instance_seg(
    *,
    anchor_base_size: int,
    aspect_ratios: tuple[float, ...],
    num_classes: int = 1,
    in_chans: int = 3,
    attributes: tuple[Attribute, ...] = (),
    **library_kwargs: object,
) -> BespokeInstanceSeg:
    """Build a :class:`BespokeInstanceSeg` over a Mask R-CNN on a :class:`_Pyramid`.

    Its RPN anchors are :meth:`_Pyramid.anchors`' at ``anchor_base_size`` and
    ``aspect_ratios``, its box and mask pools torchvision's own over the pyramid's four levels;
    ``library_kwargs`` (its resize bounds, ``image_mean``, ``image_std`` and any other Mask
    R-CNN keyword) pass to its constructor.
    """
    from torchvision.models.detection import MaskRCNN

    pyramid = _Pyramid(in_chans)
    library = MaskRCNN(
        pyramid,
        num_classes + 1,
        rpn_anchor_generator=pyramid.anchors(anchor_base_size, aspect_ratios),
        **library_kwargs,
    )
    return BespokeInstanceSeg(
        library,
        pyramid.backbone,
        featmap_names=pyramid.featmap_names,
        attributes=attributes,
    )


def build_bespoke_detection(
    *,
    num_classes: int = 1,
    in_chans: int = 3,
    attributes: tuple[Attribute, ...] = (),
    **library_kwargs: object,
) -> BespokeDetection:
    """Build a :class:`BespokeDetection` through :func:`_over_retinanet`."""
    return _over_retinanet(
        BespokeDetection,
        num_classes=num_classes,
        in_chans=in_chans,
        attributes=attributes,
        **library_kwargs,
    )


class _OneConvDetector(_KeepsItsOperatingPoint, nn.Module):
    """A detector whose one parameter is a 1x1 convolution.

    It carries ``score_thresh`` and ``detections_per_img`` on itself; :meth:`zero_loss` is its
    training forward's answer and :meth:`governed` its eval forward's output kept to both knobs.
    """

    def __init__(self, in_chans: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_chans, 1, 1)
        self.score_thresh = 0.0
        self.detections_per_img = 100

    def zero_loss(self, images: list[torch.Tensor]) -> dict:
        return {
            "loss": sum(self.conv(im.unsqueeze(0)).sum() for im in images)
            * 0.0
        }


class FixedMaskSegmenter(_OneConvDetector):
    """An instance segmenter whose eval forward offers one fixed detection per image.

    The detection is scored 0.9 at ``box`` (``[x0, y0, x1, y1]``), its mask the frame-sized
    :func:`~tests._producer_fixtures.painted_array` of ``regions`` at ``value`` over 0, and it
    passes through :meth:`governed`.
    """

    def __init__(
        self, *, in_chans: int, regions: list, value: float, box: list
    ) -> None:
        super().__init__(in_chans=in_chans)
        self.regions, self.value, self.box = regions, value, box

    def forward(
        self, images: list[torch.Tensor], targets: list[dict] | None = None
    ) -> DetectorOutput:
        from tests._producer_fixtures import painted_array

        if self.training:
            return self.zero_loss(images)
        results = []
        for im in images:
            h, w = int(im.shape[-2]), int(im.shape[-1])
            x0, y0, x1, y1 = self.box
            mask = torch.as_tensor(
                painted_array(
                    w,
                    h,
                    [(tuple(r), self.value) for r in self.regions],
                    background=0.0,
                    mode="F",
                )
            )[None, None]
            results.append(
                self.governed(
                    {
                        "boxes": torch.tensor(
                            [[x0, y0, x1, y1]], dtype=torch.float32
                        ),
                        "scores": torch.tensor([0.9]),
                        "labels": torch.tensor([1]),
                        "masks": mask,
                    }
                )
            )
        return results


def build_fixed_mask_instance_seg(
    *,
    num_classes: int = 1,
    in_chans: int = 3,
    regions: list,
    value: float,
    box: list,
    **_handed: object,
) -> FixedMaskSegmenter:
    """Build a :class:`FixedMaskSegmenter` at ``regions``, ``value`` and ``box``.

    A ``model_source`` builder; its fixed boxes read none of the detector geometry or statistics
    the platform hands it.
    """
    return FixedMaskSegmenter(
        in_chans=in_chans, regions=regions, value=value, box=box
    )


# A non-torchvision detector, no .detector to route through.
class BareScoreThreshDetector(_OneConvDetector):
    """A hand-rolled detector with no ``.detector``.

    In eval its forward offers, per image, one box over the frame's middle half scored 0.9,
    passed through :meth:`governed`; in train it answers :meth:`zero_loss`.
    """

    def forward(self, images: list[torch.Tensor]) -> DetectorOutput:
        if self.training:
            return self.zero_loss(images)
        results = []
        for im in images:
            h, w = int(im.shape[-2]), int(im.shape[-1])
            boxes = torch.tensor(
                [[w * 0.25, h * 0.25, w * 0.75, h * 0.75]], dtype=torch.float32
            )
            results.append(
                self.governed(
                    {
                        "boxes": boxes,
                        "scores": torch.tensor([0.9]),
                        "labels": torch.tensor([1]),
                    }
                )
            )
        return results


# The same shape with no operating-point knob under any recognized name anywhere: the case the
# row demonstrates, where the platform genuinely has nothing it can set.
class BareNoKnobDetector(nn.Module):
    def __init__(self, in_chans: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_chans, 1, 1)

    def forward(self, images: list[torch.Tensor]) -> DetectorOutput:
        if self.training:
            return {
                "loss": sum(self.conv(im.unsqueeze(0)).sum() for im in images)
                * 0.0
            }
        results = []
        for im in images:
            h, w = int(im.shape[-2]), int(im.shape[-1])
            boxes = torch.tensor(
                [[w * 0.25, h * 0.25, w * 0.75, h * 0.75]], dtype=torch.float32
            )
            results.append(
                {
                    "boxes": boxes,
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([1]),
                }
            )
        return results


def build_bare_score_thresh_detector(
    *, in_chans: int = 3, num_classes: int = 1, **_handed: object
) -> BareScoreThreshDetector:
    """Build a :class:`BareScoreThreshDetector` as a ``model_source`` builder.

    It reads none of the detector geometry or statistics the platform hands it.
    """
    return BareScoreThreshDetector(in_chans=in_chans)


def build_bare_no_knob_detector(
    *, in_chans: int = 3, num_classes: int = 1, **_handed: object
) -> BareNoKnobDetector:
    """Build a :class:`BareNoKnobDetector` as a ``model_source`` builder.

    It reads none of the detector geometry or statistics the platform hands it.
    """
    return BareNoKnobDetector(in_chans=in_chans)


class DivergingDetection(BespokeDetection):
    """A :class:`BespokeDetection` whose training losses add a ``diverge`` term, ``log(w)``.

    ``w`` is one extra parameter, starting at 0.02.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.w = nn.Parameter(torch.tensor(0.02))

    def forward(
        self,
        images: torch.Tensor | list[torch.Tensor],
        targets: list[dict] | None = None,
    ) -> DetectorOutput:
        out = super().forward(images, targets)
        if isinstance(out, dict):
            # drives w negative under Adam, losses -> nan
            out["diverge"] = torch.log(self.w)
        return out


def build_diverging_detection(
    *,
    num_classes: int = 1,
    in_chans: int = 3,
    attributes: tuple[Attribute, ...] = (),
    **library_kwargs: object,
) -> BespokeDetection:
    """Build a :class:`DivergingDetection` through :func:`_over_retinanet`."""
    return _over_retinanet(
        DivergingDetection,
        num_classes=num_classes,
        in_chans=in_chans,
        attributes=attributes,
        **library_kwargs,
    )


class BrightRegionDetector(_KeepsItsOperatingPoint, nn.Module):
    """A detector whose box geometry is analytic and whose confidence is learned.

    It reports the extent of a frame's pixels whose band mean exceeds ``bright`` as one box scored
    by the sigmoid of the learned ``logit``, beside a decoy box, that extent with every
    coordinate plus one, scored ``decoy_score``, kept by :meth:`governed`; a frame with no such
    pixel gets no box.
    In training its loss is the squared distance of that confidence from the batch's mean label
    presence.
    """

    def __init__(
        self, in_chans: int = 3, bright: float = 0.5, decoy_score: float = 0.03
    ) -> None:
        super().__init__()
        self.logit = nn.Parameter(torch.tensor([2.5]))
        self.bright = float(bright)
        # Just above the floor a calibration stages a reference at, so the swept curve sees the
        # low-confidence tail it assumes rather than reading the reference as truncated.
        self.decoy_score = float(decoy_score)
        self.score_thresh = 0.0
        self.nms_thresh = 0.5
        self.detections_per_img = 100

    def _extent(self, image: torch.Tensor) -> torch.Tensor | None:
        """Return the bounding box of the frame's bright pixels, or ``None`` when it holds none."""
        mask = image.mean(dim=0) > self.bright
        ys, xs = torch.nonzero(mask, as_tuple=True)
        if xs.numel() == 0:
            return None
        return torch.tensor(
            [
                float(xs.min()),
                float(ys.min()),
                float(xs.max()) + 1.0,
                float(ys.max()) + 1.0,
            ],
            dtype=torch.float32,
            device=image.device,
        )

    def forward(
        self,
        images: torch.Tensor | list[torch.Tensor],
        targets: list[dict] | None = None,
    ) -> DetectorOutput:
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
                results.append(
                    {
                        "boxes": torch.zeros(
                            (0, 4), dtype=torch.float32, device=device
                        ),
                        "scores": torch.zeros(0, device=device),
                        "labels": torch.zeros(
                            0, dtype=torch.int64, device=device
                        ),
                    }
                )
                continue
            decoy = extent + torch.ones(4, dtype=torch.float32, device=device)
            boxes = torch.stack([extent, decoy])
            reported = confidence.squeeze().detach().to(device)
            scores = torch.stack(
                [
                    reported,
                    torch.tensor(
                        self.decoy_score, dtype=torch.float32, device=device
                    ),
                ]
            )
            labels = torch.ones(2, dtype=torch.int64, device=device)
            results.append(
                self.governed(
                    {"boxes": boxes, "scores": scores, "labels": labels}
                )
            )
        return results


def build_bright_region_detector(
    *,
    in_chans: int = 3,
    num_classes: int = 1,
    bright: float = 0.5,
    decoy_score: float = 0.03,
    **_handed: object,
) -> BrightRegionDetector:
    """Build a :class:`BrightRegionDetector` as a ``model_source`` builder.

    It passes ``in_chans``, ``bright`` and ``decoy_score`` and reads none of the detector
    geometry or statistics the platform hands it.
    """
    return BrightRegionDetector(
        in_chans=in_chans, bright=bright, decoy_score=decoy_score
    )


class BrightBlobDetector(_KeepsItsOperatingPoint, nn.Module):
    """A detector offering one detection per 4-connected bright blob of a frame.

    Each detection is labeled 1 and scored by the sigmoid of a learned logit (0 when built) plus
    the fraction of the frame the blob covers, capped at 1, with the blob's 0/1 mask when
    ``with_masks``. A blob is bright where the band mean exceeds 0.5; with ``attributes``
    (the count of attributes it is built at) it is bright where one band exceeds 0.5, and
    carries that band's index as its first attribute's id and 0 for every other. A training
    step's loss is the mean squared distance of the confidence from each frame's label presence
    (one when its target holds a box, zero when not). An inference forward keeps each frame's
    blobs scoring at least ``score_thresh``, the ``detections_per_img`` highest, and records each
    input's channel count in ``seen_channels`` and its per-band peak value in
    ``seen_band_peaks``.
    """

    def __init__(
        self, in_chans: int = 3, with_masks: bool = False, attributes: int = 0
    ) -> None:
        super().__init__()
        self.logit = nn.Parameter(torch.tensor([0.0]))
        self.with_masks = with_masks
        self.attributes = attributes
        self.score_thresh = 0.0
        self.detections_per_img = 100
        self.nms_thresh = (
            0.45  # never applied here; a sentinel a pass must leave unchanged
        )
        self.seen_channels: list[int] = []
        self.seen_band_peaks: list[list[float]] = []

    def forward(
        self, images: list[torch.Tensor], targets: list[dict] | None = None
    ) -> DetectorOutput:
        import cv2
        import numpy as np

        confidence = torch.sigmoid(self.logit)
        if self.training and targets is not None:
            present = torch.tensor(
                [1.0 if len(t.get("boxes", [])) else 0.0 for t in targets]
            )
            return {
                "loss": (
                    (confidence - present.to(confidence.device)) ** 2
                ).mean()
            }
        reported = float(confidence.detach())
        results = []
        for image in images:
            self.seen_channels.append(int(image.shape[0]))
            self.seen_band_peaks.append(image.amax(dim=(1, 2)).tolist())
            h, w = int(image.shape[-2]), int(image.shape[-1])
            planes = (
                [(c, image[c] > 0.5) for c in range(image.shape[0])]
                if self.attributes
                else [(0, image.mean(dim=0) > 0.5)]
            )
            boxes, scores, masks, values = [], [], [], []
            for value, plane in planes:
                bright = plane.cpu().numpy().astype(np.uint8)
                n, labels, stats, _ = cv2.connectedComponentsWithStats(
                    bright, connectivity=4
                )
                for k in range(1, n):
                    x, y, bw, bh, area = (int(v) for v in stats[k])
                    boxes.append(
                        [float(x), float(y), float(x + bw), float(y + bh)]
                    )
                    scores.append(min(1.0, reported + area / float(h * w)))
                    masks.append(
                        torch.from_numpy((labels == k).astype(np.float32))[
                            None
                        ]
                    )
                    values.append(value)
            out = {
                "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(
                    -1, 4
                ),
                "scores": torch.tensor(scores, dtype=torch.float32),
                "labels": torch.ones(len(boxes), dtype=torch.int64),
            }
            if self.attributes:
                out["attributes"] = torch.tensor(
                    [[v] + [0] * (self.attributes - 1) for v in values],
                    dtype=torch.int64,
                ).reshape(-1, self.attributes)
            if self.with_masks:
                out["masks"] = (
                    torch.stack(masks)
                    if masks
                    else torch.zeros((0, 1, h, w), dtype=torch.float32)
                )
            results.append(self.governed(out))
        return results


def build_bright_blob_detector(
    *,
    in_chans: int = 3,
    num_classes: int = 1,
    with_masks: bool = False,
    attributes: tuple = (),
    **_handed: object,
) -> BrightBlobDetector:
    """Build a :class:`BrightBlobDetector` at the attributes the platform hands it.

    A ``model_source`` builder; its blobs read none of the detector geometry or statistics.
    """
    return BrightBlobDetector(
        in_chans=in_chans, with_masks=with_masks, attributes=len(attributes)
    )


class WholeBlobDetector(BrightBlobDetector):
    """A :class:`BrightBlobDetector` whose first attribute marks a box clear of the frame border.

    In eval its first attribute's id is 1 for a box with ``x1 > 0``, ``y1 > 0``, ``x2`` below the
    width and ``y2`` below the height, and 0 otherwise.
    """

    def forward(
        self, images: list[torch.Tensor], targets: list[dict] | None = None
    ) -> DetectorOutput:
        results = super().forward(images, targets)
        if self.training:
            return results
        for image, out in zip(images, results):
            h, w = image.shape[-2:]
            b = out["boxes"]
            out["attributes"][:, 0] = (
                (b[:, 0] > 0) & (b[:, 1] > 0) & (b[:, 2] < w) & (b[:, 3] < h)
            ).long()
        return results


def build_whole_blob_detector(
    *,
    in_chans: int = 3,
    num_classes: int = 1,
    attributes: tuple = (),
    **_handed: object,
) -> WholeBlobDetector:
    """Build a :class:`WholeBlobDetector` at the attributes the platform hands it.

    A ``model_source`` builder reading none of the detector geometry or statistics.
    """
    return WholeBlobDetector(in_chans=in_chans, attributes=len(attributes))

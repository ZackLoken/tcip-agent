"""Model composer — wires backbone + neck + head(s) into a ComposedModel.

This is the core of the agentic ML system: the agent designs a ModelSpec
dict, and the composer builds a fully functional nn.Module from it.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn

from tcip_mcp.pipelines.registry import BACKBONES, NECKS, HEADS
from tcip_mcp.pipelines.components.heads import BaseHead
from tcip_mcp.pipelines.components.backbones import HAS_TIMM
# Side-effect imports: registering necks + detectors makes the composer self-contained
# (BACKBONES/HEADS already register via the imports above), so validate/compose never
# see an empty registry regardless of import order.
import tcip_mcp.pipelines.components.necks  # noqa: F401
import tcip_mcp.pipelines.components.detectors  # noqa: F401

_DETECTION_HEADS = {"anchor_detection", "anchor_free_detection"}

# An FPN over a standard 4-stage CNN backbone yields 4 pyramid levels; add_p2 adds one.
_FPN_BASE_LEVELS = 4
# Backbone output_format values that are single-scale (incompatible with a pyramid neck).
_SINGLE_SCALE_FORMATS = {"single", "flat", "flat_vector", "pooled", "vector"}


class ComposedModel(nn.Module):
    """A model assembled from composable backbone + neck + head(s).

    Supports multi-head models (shared backbone, multiple task heads).
    Training mode: returns combined loss dict from all heads.
    Eval mode: returns combined predictions dict from all heads.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        heads: list[BaseHead],
        spec: dict | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.heads = nn.ModuleList(heads)
        self.spec = spec or {}

    def forward(
        self,
        images: torch.Tensor,
        targets: list[dict] | dict | None = None,
    ) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        neck_out = self.neck(features)

        all_outputs: dict[str, torch.Tensor] = {}

        if self.training and targets is not None:
            # Return loss dict
            for i, head in enumerate(self.heads):
                head_targets = targets[i] if isinstance(targets, list) else targets
                outputs = head(neck_out, head_targets)
                losses = head.compute_loss(outputs, head_targets)
                for k, v in losses.items():
                    all_outputs[f"head{i}_{k}"] = v
        else:
            # Return predictions
            for i, head in enumerate(self.heads):
                outputs = head(neck_out)
                decoded = head.decode(outputs)
                for k, v in decoded.items():
                    all_outputs[f"head{i}_{k}"] = v

        return all_outputs

    def freeze_backbone(self, to_stage: int) -> None:
        """Freeze backbone up to stage index (0=none, num_stages=all)."""
        if hasattr(self.backbone, "freeze_to"):
            self.backbone.freeze_to(to_stage)
        elif to_stage > 0:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.requires_grad = True

    def get_param_groups(
        self,
        backbone_lr: float = 1e-4,
        head_lr: float = 1e-3,
    ) -> list[dict]:
        """Return parameter groups for differential learning rates."""
        backbone_params = list(self.backbone.parameters())
        neck_params = list(self.neck.parameters())
        head_params = []
        for h in self.heads:
            head_params.extend(h.parameters())
        return [
            {"params": [p for p in backbone_params if p.requires_grad], "lr": backbone_lr},
            {"params": [p for p in neck_params if p.requires_grad], "lr": head_lr},
            {"params": [p for p in head_params if p.requires_grad], "lr": head_lr},
        ]

    def total_loss(self, loss_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Sum all loss terms into a single scalar for backward()."""
        total = None
        for v in loss_dict.values():
            if torch.is_tensor(v) and v.requires_grad:
                total = v if total is None else total + v
        return total if total is not None else torch.tensor(0.0)


# ====================================================================
# Detection model — wraps backbone+neck into torchvision Faster R-CNN
# ====================================================================

class _BackboneNeckAdapter(nn.Module):
    """Wraps a composed backbone+neck so torchvision's FasterRCNN can use it."""

    def __init__(self, backbone: nn.Module, neck: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.out_channels = (
            neck.out_channels if isinstance(neck.out_channels, int)
            else neck.out_channels[-1]
        )

    def forward(self, x: torch.Tensor) -> OrderedDict:
        features = self.backbone(x)
        neck_out = self.neck(features)
        if isinstance(neck_out, dict):
            return OrderedDict(sorted(neck_out.items()))
        return OrderedDict({"0": neck_out})


class DetectionModel(nn.Module):
    """Detection model using composed backbone+neck fed into torchvision Faster R-CNN.

    Exposes the same interface as ComposedModel (freeze_backbone, get_param_groups, etc.)
    so the trainer can use it interchangeably.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        num_classes: int,
        spec: dict | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.spec = spec or {}

        adapter = _BackboneNeckAdapter(backbone, neck)

        # Determine feature map names from the neck output.
        dummy = torch.zeros(1, 3, 64, 64)
        with torch.no_grad():
            sample_out = adapter(dummy)
        featmap_names = list(sample_out.keys())
        num_levels = len(featmap_names)

        # Registry-driven detector construction (see components/detectors.py): add a new
        # detector (Mask R-CNN, DETR, external framework) by registering a builder, not
        # by editing this class.
        from tcip_mcp.pipelines.components.detectors import build_detector

        detector = kwargs.get("detector", "faster_rcnn")
        det_kwargs = {k: v for k, v in kwargs.items() if k != "detector"}
        self.detector = build_detector(
            detector, adapter, num_classes,
            featmap_names=featmap_names, num_levels=num_levels, **det_kwargs,
        )

    def forward(
        self,
        images: list[torch.Tensor] | torch.Tensor,
        targets: list[dict] | None = None,
    ) -> dict[str, torch.Tensor] | list[dict]:
        if isinstance(images, torch.Tensor):
            images = [images[i] for i in range(images.shape[0])]
        if self.training and targets is not None:
            return self.detector(images, targets)
        return self.detector(images)

    def freeze_backbone(self, to_stage: int) -> None:
        if hasattr(self.backbone, "freeze_to"):
            self.backbone.freeze_to(to_stage)
        elif to_stage > 0:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def get_param_groups(
        self,
        backbone_lr: float = 1e-4,
        head_lr: float = 1e-3,
    ) -> list[dict]:
        bb_params = set(id(p) for p in self.backbone.parameters())
        neck_params = set(id(p) for p in self.neck.parameters())
        backbone_group = [p for p in self.backbone.parameters() if p.requires_grad]
        head_group = [
            p for p in self.detector.parameters()
            if p.requires_grad and id(p) not in bb_params and id(p) not in neck_params
        ]
        neck_group = [p for p in self.neck.parameters() if p.requires_grad]
        return [
            {"params": backbone_group, "lr": backbone_lr},
            {"params": neck_group + head_group, "lr": head_lr},
        ]


def compose_model(spec: dict) -> ComposedModel | DetectionModel:
    """Build a ComposedModel (or DetectionModel) from a spec dict.

    Spec format::

        {
            "backbone": {"name": "resnet50", "pretrained": true},
            "neck": {"name": "fpn", "out_channels": 256},
            "heads": [
                {"name": "classification", "num_classes": 5}
            ]
        }

    For detection heads (``anchor_detection``), a ``DetectionModel``
    is returned instead.  It wraps the composed backbone+neck into
    torchvision's Faster R-CNN so the full pipeline flows through the
    shared backbone.
    """
    issues = validate_model_spec(spec)
    if issues:
        raise ValueError(f"Invalid model spec: {issues}")

    # Build backbone
    bb_spec = spec["backbone"]
    bb_name = bb_spec["name"]
    bb_kwargs = {k: v for k, v in bb_spec.items() if k != "name"}
    backbone = BACKBONES.build(bb_name, **bb_kwargs)

    # Build neck
    neck_spec = spec.get("neck", {"name": "gap"})
    neck_name = neck_spec["name"]
    neck_kwargs = {k: v for k, v in neck_spec.items() if k != "name"}
    neck = NECKS.build(neck_name, in_channels_list=backbone.out_channels, **neck_kwargs)

    # Check if this is a detection spec
    head_specs = spec["heads"]
    if len(head_specs) == 1 and head_specs[0]["name"] in _DETECTION_HEADS:
        h = head_specs[0]
        det_kwargs = {k: v for k, v in h.items() if k not in ("name", "in_channels")}
        det_kwargs.setdefault("detector", "fcos" if h["name"] == "anchor_free_detection" else "faster_rcnn")
        return DetectionModel(backbone, neck, spec=spec, **det_kwargs)

    # Non-detection: build heads normally
    heads: list[BaseHead] = []
    for h_spec in head_specs:
        h_name = h_spec["name"]
        h_kwargs = {k: v for k, v in h_spec.items() if k != "name"}
        # Inject in_channels from neck
        if "in_channels" not in h_kwargs:
            h_kwargs["in_channels"] = (
                neck.out_channels if isinstance(neck.out_channels, int)
                else neck.out_channels[-1]
            )
        heads.append(HEADS.build(h_name, **h_kwargs))

    return ComposedModel(backbone, neck, heads, spec=spec)


def validate_model_spec(spec: dict) -> list[str]:
    """Validate a model spec dict. Returns list of issues (empty = valid)."""
    issues: list[str] = []

    if not isinstance(spec, dict):
        return ["spec must be a dict"]

    # Backbone
    bb = spec.get("backbone")
    if not bb or "name" not in bb:
        issues.append("Missing backbone.name")
    elif bb["name"] not in BACKBONES:
        issues.append(f"Unknown backbone: {bb['name']}. Available: {BACKBONES.names()}")

    # Neck
    neck = spec.get("neck", {"name": "gap"})
    if "name" not in neck:
        issues.append("Missing neck.name")
    elif neck["name"] not in NECKS:
        issues.append(f"Unknown neck: {neck['name']}. Available: {NECKS.names()}")

    # Heads
    heads = spec.get("heads")
    if not heads or not isinstance(heads, list):
        issues.append("Missing or empty heads list")
    else:
        for i, h in enumerate(heads):
            if "name" not in h:
                issues.append(f"heads[{i}] missing 'name'")
            elif h["name"] not in HEADS:
                issues.append(f"Unknown head: {h['name']}. Available: {HEADS.names()}")

    issues.extend(_channel_compat_issues(spec))
    return issues


def _channel_compat_issues(spec: dict) -> list[str]:
    """Flag neck/head format + channel mismatches (W7).

    Fully defensive: a no-op unless both ``neck`` and ``head`` are dicts whose
    names resolve in the registries (so existence reporting and string-typed specs
    are unaffected). Catches e.g. a flat-vector ``gap`` neck feeding a multi-scale
    detection/segmentation head, or an explicit ``in_channels`` that disagrees with
    an FPN/PAN neck's ``out_channels``.
    """
    issues: list[str] = []
    if not isinstance(spec, dict):
        return issues
    neck = spec.get("neck", {"name": "gap"})
    heads = spec.get("heads", [])
    if not isinstance(neck, dict) or neck.get("name") not in NECKS or not isinstance(heads, list):
        return issues

    neck_name = neck["name"]
    neck_fmt = NECKS.describe(neck_name).get("output_format")
    neck_out = neck.get("out_channels", 256)
    for i, h in enumerate(heads):
        if not isinstance(h, dict) or h.get("name") not in HEADS:
            continue
        head_fmt = HEADS.describe(h["name"]).get("input_format")
        if neck_fmt and head_fmt and neck_fmt != head_fmt:
            issues.append(
                f"heads[{i}] '{h['name']}' expects {head_fmt} input but neck "
                f"'{neck_name}' outputs {neck_fmt}")
        if (neck_name in ("fpn", "pan") and h["name"] not in _DETECTION_HEADS
                and "in_channels" in h and h["in_channels"] != neck_out):
            issues.append(
                f"heads[{i}] '{h['name']}' in_channels={h['in_channels']} != "
                f"neck '{neck_name}' out_channels={neck_out}")

    # Backbone-stage compatibility: pyramid necks (fpn/pan) need a multi-scale backbone.
    backbone = spec.get("backbone")
    if (isinstance(backbone, dict) and backbone.get("name") in BACKBONES
            and neck_name in ("fpn", "pan")):
        bb_fmt = BACKBONES.describe(backbone["name"]).get("output_format")
        if bb_fmt in _SINGLE_SCALE_FORMATS:
            issues.append(
                f"neck '{neck_name}' needs a multi-scale backbone but backbone "
                f"'{backbone['name']}' outputs single-scale '{bb_fmt}'")

    # Pyramid-level compatibility: add_p2 adds an extra finer (P2) level; a head that
    # caps the level count below the produced count would break at forward().
    if isinstance(neck, dict) and neck.get("add_p2"):
        produced = _FPN_BASE_LEVELS + 1
        for i, h in enumerate(heads):
            if isinstance(h, dict) and h.get("name") in HEADS:
                cap = HEADS.describe(h["name"]).get("max_pyramid_levels")
                if isinstance(cap, int) and cap < produced:
                    issues.append(
                        f"heads[{i}] '{h['name']}' supports at most {cap} pyramid levels "
                        f"but neck '{neck_name}' with add_p2 produces {produced}")
    return issues


def recommend_model_spec(
    task: str,
    dataset_size: int,
    sensor: str = "rgb",
    num_classes: int = 1,
    num_ranks: int | None = None,
    object_size: str = "medium",
) -> dict:
    """Generate a recommended ModelSpec from task parameters.

    The agent can use this as a starting point and customize further.
    ``object_size`` (tiny/small/medium/large) tunes the detection recommendation:
    tiny/small objects get the anchor-free FCOS detector with smaller anchors (and
    an extra ``add_p2`` pyramid level for tiny); medium/large keep Faster R-CNN.
    """
    if task == "instance_seg":
        # Deferred (Phase 0.3 task-honesty): instance_seg would route to a box-only
        # detector and silently discard mask labels. Refuse to recommend it until the
        # Mask R-CNN mask head lands rather than hand back a model that drops masks.
        raise ValueError(
            "instance_seg is not a supported recommended task yet: it currently trains "
            "box-only and discards mask ground truth. Use 'detection', or wait for the "
            "Mask R-CNN mask head (deferred)."
        )
    # Backbone selection by dataset size. Without timm, fall back to the
    # torchvision-only backbones (resnet18/50/101 and efficientnet_* are all
    # timm-only — recommending them with no timm yields an unbuildable spec).
    if dataset_size < 500:
        bb = "efficientnet_b0" if HAS_TIMM else "tv_resnet50"
    elif dataset_size < 2000:
        bb = "resnet50" if HAS_TIMM else "tv_resnet50"
    else:
        bb = "resnet101" if HAS_TIMM else "tv_resnet101"

    # Neck and head by task (instance_seg is rejected above pending a mask head)
    if task == "detection":
        if object_size in ("tiny", "small"):
            neck = {"name": "fpn", "out_channels": 256}
            if object_size == "tiny":
                neck["add_p2"] = True
            head = {
                "name": "anchor_free_detection", "num_classes": num_classes,
                "detector": "fcos", "anchor_base_size": 8 if object_size == "tiny" else 16,
            }
        else:
            neck = {"name": "fpn", "out_channels": 256}
            head = {"name": "anchor_detection", "num_classes": num_classes}
    elif task == "semantic_seg":
        neck = {"name": "fpn", "out_channels": 256}
        head = {"name": "semantic_seg", "num_classes": num_classes}
    elif task == "classification":
        neck = {"name": "gap"}
        head = {"name": "classification", "num_classes": num_classes}
    elif task == "ordinal":
        neck = {"name": "gap"}
        head = {"name": "ordinal", "num_ranks": num_ranks or 9}
    elif task == "regression":
        neck = {"name": "gap"}
        head = {"name": "regression"}
    else:
        neck = {"name": "gap"}
        head = {"name": "classification", "num_classes": num_classes}

    return {
        "backbone": {"name": bb, "pretrained": True},
        "neck": neck,
        "heads": [head],
    }

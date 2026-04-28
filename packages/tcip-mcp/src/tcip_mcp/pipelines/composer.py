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

_DETECTION_HEADS = {"anchor_detection"}


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

        from torchvision.models.detection import FasterRCNN
        from torchvision.models.detection.rpn import AnchorGenerator
        from torchvision.ops import MultiScaleRoIAlign

        adapter = _BackboneNeckAdapter(backbone, neck)

        # Determine feature map names from neck output
        dummy = torch.zeros(1, 3, 64, 64)
        with torch.no_grad():
            sample_out = adapter(dummy)
        featmap_names = list(sample_out.keys())
        num_levels = len(featmap_names)

        anchor_sizes = kwargs.get(
            "anchor_sizes",
            tuple((s,) for s in [32, 64, 128, 256][:num_levels]),
        )
        aspect_ratios = kwargs.get(
            "aspect_ratios",
            ((0.5, 1.0, 2.0),) * num_levels,
        )
        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes, aspect_ratios=aspect_ratios,
        )
        roi_pool = MultiScaleRoIAlign(
            featmap_names=featmap_names, output_size=7, sampling_ratio=2,
        )

        self.detector = FasterRCNN(
            adapter,
            num_classes=num_classes + 1,  # +1 for background
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pool,
            min_size=kwargs.get("min_size", 800),
            max_size=kwargs.get("max_size", 1333),
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

    return issues


def recommend_model_spec(
    task: str,
    dataset_size: int,
    sensor: str = "rgb",
    num_classes: int = 1,
    num_ranks: int | None = None,
) -> dict:
    """Generate a recommended ModelSpec from task parameters.

    The agent can use this as a starting point and customize further.
    """
    # Backbone selection by dataset size
    if dataset_size < 500:
        bb = "efficientnet_b0" if HAS_TIMM else "resnet18"
    elif dataset_size < 2000:
        bb = "resnet50"
    else:
        bb = "resnet101"

    # Neck and head by task
    if task in ("detection", "instance_seg"):
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

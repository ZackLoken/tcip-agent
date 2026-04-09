"""Progressive unfreezing stage definitions.

Each stage is a dict with:
    freeze_pattern: which layers to freeze ('all_backbone', 'top_backbone', 'none')
    lr: learning rate
    epochs: number of epochs
"""

from __future__ import annotations

import torch.nn as nn


def get_default_stages() -> list[dict]:
    """Return the default 3-stage progressive unfreezing schedule."""
    return [
        {"freeze_pattern": "all_backbone", "lr": 1e-3, "epochs": 5},
        {"freeze_pattern": "top_backbone", "lr": 1e-4, "epochs": 10},
        {"freeze_pattern": "none", "lr": 1e-5, "epochs": 5},
    ]


def apply_stage(model: nn.Module, stage_config: dict) -> None:
    """Apply freeze/unfreeze pattern for a training stage.

    This works with torchvision detection models which have:
        model.backbone — the feature extractor
        model.rpn — region proposal network (FasterRCNN)
        model.roi_heads — classification/regression heads (FasterRCNN)

    For single-stage detectors (RetinaNet, FCOS):
        model.backbone — feature extractor
        model.head — detection head
    """
    pattern = stage_config.get("freeze_pattern", "none")

    if pattern == "all_backbone":
        # Freeze entire backbone, train only head
        _freeze_backbone(model, freeze_all=True)

    elif pattern == "top_backbone":
        # Unfreeze top layers of backbone
        _freeze_backbone(model, freeze_all=False)

    elif pattern == "none":
        # Unfreeze everything
        for param in model.parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"Unknown freeze pattern: {pattern}")


def _freeze_backbone(model: nn.Module, freeze_all: bool = True) -> None:
    """Freeze or partially freeze the backbone."""
    # First unfreeze everything
    for param in model.parameters():
        param.requires_grad = True

    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return

    if freeze_all:
        for param in backbone.parameters():
            param.requires_grad = False
    else:
        # Freeze only early layers — for FPN backbones, freeze body layers 0-2
        body = getattr(backbone, "body", backbone)
        named = list(body.named_children())
        # Freeze the first 2/3 of layers
        freeze_count = max(1, len(named) * 2 // 3)
        for i, (name, module) in enumerate(named):
            if i < freeze_count:
                for param in module.parameters():
                    param.requires_grad = False

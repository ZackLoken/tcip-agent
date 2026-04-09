"""Backbone factories — pure torchvision models only."""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as tvmodels


_BACKBONE_REGISTRY: dict[str, dict] = {
    "resnet50": {"factory": "torchvision", "out_channels": 2048},
    "resnet101": {"factory": "torchvision", "out_channels": 2048},
    "mobilenet_v2": {"factory": "torchvision", "out_channels": 1280},
}


def list_backbones() -> list[str]:
    """Return names of available backbones."""
    return sorted(_BACKBONE_REGISTRY.keys())


def build_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    """Build a backbone feature extractor.

    Returns:
        (backbone_module, out_channels)
    """
    if name not in _BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone: {name}. Available: {list_backbones()}")

    weights = "DEFAULT" if pretrained else None

    if name == "resnet50":
        base = tvmodels.resnet50(weights=weights)
        layers = list(base.children())[:-2]  # Remove avgpool and fc
        backbone = nn.Sequential(*layers)
        return backbone, 2048
    elif name == "resnet101":
        base = tvmodels.resnet101(weights=weights)
        layers = list(base.children())[:-2]
        backbone = nn.Sequential(*layers)
        return backbone, 2048
    elif name == "mobilenet_v2":
        base = tvmodels.mobilenet_v2(weights=weights)
        backbone = base.features
        return backbone, 1280
    else:
        raise ValueError(f"No factory for backbone: {name}")

"""Composable backbone feature extractors.

Wraps timm models into a standardized interface with multi-scale feature
output, known out_channels, and stage-based freezing for progressive
unfreezing.  Falls back to torchvision when timm is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from tcip_mcp.pipelines.registry import BACKBONES

logger = logging.getLogger(__name__)

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    logger.info("timm not installed — only torchvision backbones available")


class BackboneWrapper(nn.Module):
    """Uniform interface around a feature extractor.

    Attributes:
        out_channels: List of channel counts per feature level
            (e.g. [256, 512, 1024, 2048] for ResNet-50).
        num_stages: Number of feature levels.
    """

    def __init__(
        self,
        model: nn.Module,
        out_channels: list[int],
        *,
        feature_keys: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.out_channels = out_channels
        self.num_stages = len(out_channels)
        self._feature_keys = feature_keys

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract multi-scale features.

        Returns:
            Dict mapping ``"s0"`` … ``"sN"`` to feature tensors, where
            ``s0`` is the earliest (highest-resolution) stage.
        """
        if hasattr(self.model, "forward_features"):
            # timm features_only model
            features = self.model(x)
            if isinstance(features, (list, tuple)):
                return {f"s{i}": f for i, f in enumerate(features)}
            return {"s0": features}
        # Fallback: single feature map from backbone
        out = self.model(x)
        if isinstance(out, dict):
            return out
        if isinstance(out, (list, tuple)):
            return {f"s{i}": f for i, f in enumerate(out)}
        return {"s0": out}

    def freeze_to(self, stage: int) -> None:
        """Freeze all parameters up to (exclusive) *stage*.

        ``stage=0`` freezes nothing; ``stage=self.num_stages`` freezes
        the entire backbone.
        """
        if stage <= 0:
            for p in self.model.parameters():
                p.requires_grad = True
            return
        if stage >= self.num_stages:
            for p in self.model.parameters():
                p.requires_grad = False
            return

        # For timm models: freeze by feature_info index
        if HAS_TIMM and hasattr(self.model, "feature_info"):
            _freeze_timm_stages(self.model, stage)
        else:
            _freeze_sequential_fraction(self.model, stage, self.num_stages)


# ---------------------------------------------------------------------------
# timm-based backbone builder
# ---------------------------------------------------------------------------

def _build_timm_backbone(
    name: str,
    pretrained: bool = True,
    **kwargs: Any,
) -> BackboneWrapper:
    """Build a backbone from timm with multi-scale features."""
    if not HAS_TIMM:
        raise ImportError(f"timm is required for backbone '{name}'")
    model = timm.create_model(
        name,
        pretrained=pretrained,
        features_only=True,
        out_indices=(1, 2, 3, 4),
        **kwargs,
    )
    channels = model.feature_info.channels()
    return BackboneWrapper(model, channels)


# ---------------------------------------------------------------------------
# Torchvision fallback builders
# ---------------------------------------------------------------------------

def _build_tv_resnet(variant: str, pretrained: bool = True, **_: Any) -> BackboneWrapper:
    import torchvision.models as tvm
    factory = getattr(tvm, variant)
    weights = "DEFAULT" if pretrained else None
    base = factory(weights=weights)
    # Extract stage modules (layer1 … layer4)
    stages = nn.ModuleList([
        nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool, base.layer1),
        base.layer2,
        base.layer3,
        base.layer4,
    ])
    channels = [
        stages[0][-1].conv1.in_channels if hasattr(stages[0][-1], "conv1") else 256,
        512 if "50" in variant or "101" in variant else 128,
        1024 if "50" in variant or "101" in variant else 256,
        2048 if "50" in variant or "101" in variant else 512,
    ]
    # Use a simple sequential extractor
    model = _MultiStageExtractor(stages)
    return BackboneWrapper(model, channels)


class _MultiStageExtractor(nn.Module):
    """Runs sequential stages, returns multi-scale dict."""
    def __init__(self, stages: nn.ModuleList) -> None:
        super().__init__()
        self.stages = stages

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {}
        for i, stage in enumerate(self.stages):
            x = stage(x)
            out[f"s{i}"] = x
        return out

    def named_children_list(self):
        return list(self.stages.named_children())


# ---------------------------------------------------------------------------
# Stage freezing helpers
# ---------------------------------------------------------------------------

def _freeze_timm_stages(model: nn.Module, freeze_to: int) -> None:
    """Freeze timm backbone up to a feature stage index."""
    for p in model.parameters():
        p.requires_grad = True
    # timm models expose feature_info with reduction info per stage
    fi = model.feature_info
    freeze_modules = set()
    for i in range(min(freeze_to, len(fi))):
        mod_name = fi[i]["module"]
        freeze_modules.add(mod_name)
    for name, param in model.named_parameters():
        for fm in freeze_modules:
            if name.startswith(fm):
                param.requires_grad = False
                break


def _freeze_sequential_fraction(
    model: nn.Module, freeze_to: int, total: int,
) -> None:
    """Freeze first *freeze_to/total* of named children."""
    for p in model.parameters():
        p.requires_grad = True
    children = list(model.named_children())
    if not children:
        children = list(model.named_modules())
    n_freeze = max(1, len(children) * freeze_to // total)
    for i, (_, module) in enumerate(children):
        if i < n_freeze:
            for p in module.parameters():
                p.requires_grad = False


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_TIMM_BACKBONES: dict[str, dict] = {
    "resnet18": {"category": "cnn", "params_M": 11.7, "dataset_size": "any", "tasks": "all"},
    "resnet34": {"category": "cnn", "params_M": 21.8, "dataset_size": "any", "tasks": "all"},
    "resnet50": {"category": "cnn", "params_M": 25.6, "dataset_size": ">500", "tasks": "all"},
    "resnet101": {"category": "cnn", "params_M": 44.5, "dataset_size": ">2000", "tasks": "all"},
    "efficientnet_b0": {"category": "cnn", "params_M": 5.3, "dataset_size": "<500", "tasks": "all"},
    "efficientnet_b1": {"category": "cnn", "params_M": 7.8, "dataset_size": "<1000", "tasks": "all"},
    "efficientnet_b2": {"category": "cnn", "params_M": 9.2, "dataset_size": "<2000", "tasks": "all"},
    "efficientnet_b3": {"category": "cnn", "params_M": 12.0, "dataset_size": "<5000", "tasks": "all"},
    "efficientnet_b4": {"category": "cnn", "params_M": 19.3, "dataset_size": ">2000", "tasks": "all"},
    "mobilenetv2_100": {"category": "cnn", "params_M": 3.5, "dataset_size": "<500", "tasks": "all"},
    "mobilenetv3_small_100": {"category": "cnn", "params_M": 2.5, "dataset_size": "<500", "tasks": "all"},
    "mobilenetv3_large_100": {"category": "cnn", "params_M": 5.5, "dataset_size": "<1000", "tasks": "all"},
    "convnext_tiny": {"category": "cnn", "params_M": 28.6, "dataset_size": ">1000", "tasks": "all"},
    "convnext_small": {"category": "cnn", "params_M": 50.2, "dataset_size": ">2000", "tasks": "all"},
    "vit_small_patch16_224": {"category": "vit", "params_M": 22.1, "dataset_size": ">2000", "tasks": "classification,regression,ordinal"},
    "vit_base_patch16_224": {"category": "vit", "params_M": 86.6, "dataset_size": ">5000", "tasks": "classification,regression,ordinal"},
}

for _name, _info in _TIMM_BACKBONES.items():
    meta = {
        "description": f"{_name} backbone via timm",
        "params_M": _info["params_M"],
        "dataset_size_hint": _info["dataset_size"],
        "valid_tasks": _info["tasks"].split(","),
        "requires": "timm",
    }
    BACKBONES.register_factory(
        _name,
        lambda n=_name, **kw: _build_timm_backbone(n, **kw),
        category=_info["category"],
        metadata=meta,
    )

# Torchvision fallbacks (no timm needed)
for _tv_name in ("resnet50", "resnet101"):
    _fb_name = f"tv_{_tv_name}"
    BACKBONES.register_factory(
        _fb_name,
        lambda v=_tv_name, **kw: _build_tv_resnet(v, **kw),
        category="cnn_torchvision",
        metadata={
            "description": f"{_tv_name} via torchvision (no timm required)",
            "valid_tasks": ["all"],
            "requires": "torchvision",
        },
    )

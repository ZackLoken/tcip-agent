"""``BackboneWrapper``: the interface a backbone must expose to the necks and detectors here.

Wrap a module that already emits a list, tuple, or dict of feature maps: timm's
``features_only=True``, torchvision's ``create_feature_extractor`` / ``IntermediateLayerGetter``,
or a staged module you wrote. The wrapper renames those to ``{"s0": ..., "sN": ...}``, carries a
declared ``out_channels``, and adds per-stage ``freeze_to`` for progressive unfreezing.

Emit finest first: stages are renamed in the order the module returns them (a dict's insertion
order), and the necks rebuild pyramid order from those names, so ``s0`` must be the
highest-resolution stage. The renaming is what makes that ordering survive: ``FPN``/``PAN`` sort the
keys they are handed, so a module's own names (``feat1``/``feat4``, ``low``/``mid``/``high``) would
otherwise be consumed in alphabetical order, silently, and without a shape error whenever the
stage widths are uniform. Pass ``feature_keys=[...]`` to name the stages yourself instead.

It does not turn a classifier into a feature extractor: wrapping a plain
``torchvision.models.resnet50`` yields ``{"s0": (1, 1000)}`` (the logit vector, one entry)
because that is what the module returns. Extract features first, then wrap.

There is deliberately no backbone-building helper. Constructing the backbone is your decision and
its arguments are yours to choose: a helper would have to pin ``out_indices``, which decides
whether you get a pyramid at all (a plain ViT returns four maps at one reduction; ``convnext_*``
and ``swin_*`` expose only indices 0-3 and raise on a request for 4). Choose them against the
model you are building, then wrap it::

    import timm
    from tcip_mcp.pipelines.components.backbones import BackboneWrapper

    m = timm.create_model("resnet50", pretrained=True, features_only=True,
                          out_indices=(1, 2, 3, 4), in_chans=5)
    backbone = BackboneWrapper(m, m.feature_info.channels())
"""

from __future__ import annotations

import logging
from typing import Any, cast

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


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
        out = self.model(x)
        if isinstance(out, dict):
            # Renamed, not passed through: the necks reconstruct pyramid order from these keys
            # (``sorted(features.keys())``), so a module emitting its own names (torchvision's
            # ``feat1``/``feat4``, or a staged module's ``low``/``mid``/``high``) would be
            # consumed out of order, silently, whenever the stage widths are uniform.
            feats = list(out.values())
        elif isinstance(out, (list, tuple)):
            feats = list(out)
        else:
            feats = [out]
        keys = self._feature_keys or [f"s{i}" for i in range(len(feats))]
        if len(keys) != len(feats):
            raise ValueError(
                f"feature_keys has {len(keys)} names but the backbone emitted {len(feats)} "
                f"feature maps"
            )
        return dict(zip(keys, feats))

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

        # A model carrying timm-style feature_info can be frozen by stage index;
        # anything else falls back to freezing a fraction of its named children.
        if hasattr(self.model, "feature_info"):
            _freeze_timm_stages(self.model, stage)
        else:
            _freeze_sequential_fraction(self.model, stage, self.num_stages)


# ---------------------------------------------------------------------------
# Stage freezing helpers
# ---------------------------------------------------------------------------

def _freeze_timm_stages(model: nn.Module, freeze_to: int) -> None:
    """Freeze timm backbone up to a feature stage index."""
    for p in model.parameters():
        p.requires_grad = True
    # timm models expose feature_info with reduction info per stage; its shape is a timm runtime
    # detail nn.Module's stub can't see.
    fi = cast(Any, model.feature_info)
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
    # A wrapper whose sole child is a ModuleList keeps its real stages one level
    # down, so descend to freeze per-stage rather than all-or-nothing. This is
    # the common shape when an agent wraps its own staged backbone.
    if len(children) == 1 and isinstance(children[0][1], nn.ModuleList):
        children = list(children[0][1].named_children())
    if not children:
        children = list(model.named_modules())
    n_freeze = len(children) * freeze_to // total
    for i, (_, module) in enumerate(children):
        if i < n_freeze:
            for p in module.parameters():
                p.requires_grad = False



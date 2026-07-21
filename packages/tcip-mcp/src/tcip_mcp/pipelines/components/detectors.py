"""2D object-detector builders — plain torchvision detector factories.

Bespoke model code imports these directly: build a ``BackboneNeckAdapter`` over an
agent-composed backbone+neck, then call ``build_detector`` (or a ``_build_*`` builder
directly) to get an ``nn.Module`` honoring the torchvision-detection forward contract:
``model(images, targets)`` returns a loss dict in train mode and ``list[dict]``
predictions in eval mode.
"""

from __future__ import annotations

import inspect
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn


class BackboneNeckAdapter(nn.Module):
    """Wrap a backbone+neck so a torchvision detector can consume it as its backbone."""

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


def _default_anchor_sizes(num_levels: int, base: int = 32) -> tuple[tuple[int, ...], ...]:
    """One anchor size per pyramid level, doubling each level.

    ``base=32, num_levels=4`` -> ``((32,),(64,),(128,),(256,))`` (the historical
    default). Generated for ``num_levels`` so ``add_p2`` (5+ levels) doesn't crash.
    """
    return tuple((base * 2 ** i,) for i in range(num_levels))


def _build_faster_rcnn(
    adapter: Any, num_classes: int, *, featmap_names: list[str], num_levels: int,
    anchor_base_size: int = 32, min_size: int = 800, max_size: int = 1333,
    aspect_ratios: tuple[float, ...] = (0.5, 1.0, 2.0), **_: Any,
) -> Any:
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.rpn import AnchorGenerator
    from torchvision.ops import MultiScaleRoIAlign

    sizes = _default_anchor_sizes(num_levels, anchor_base_size)
    # aspect_ratios is a builder kwarg (was hardcoded): set/derive it per trait — elongated catkins
    # (~1:3-1:6) need a tall ratio the default (0.5,1,2) can't match.
    ar = tuple(float(r) for r in aspect_ratios)
    anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=(ar,) * num_levels)
    roi_pool = MultiScaleRoIAlign(featmap_names=featmap_names, output_size=7, sampling_ratio=2)
    return FasterRCNN(
        adapter, num_classes=num_classes + 1,  # +1 for background
        rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pool,
        min_size=min_size, max_size=max_size,
    )


def _build_fcos(
    adapter: Any, num_classes: int, *, featmap_names: list[str], num_levels: int,
    anchor_base_size: int = 32, min_size: int = 800, max_size: int = 1333, **_: Any,
) -> Any:
    from torchvision.models.detection import FCOS
    from torchvision.models.detection.rpn import AnchorGenerator

    sizes = _default_anchor_sizes(num_levels, anchor_base_size)
    # FCOS is anchor-free: exactly one point/anchor per location (ratio 1.0).
    anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=((1.0,),) * num_levels)
    return FCOS(
        adapter, num_classes=num_classes + 1,
        anchor_generator=anchor_generator, min_size=min_size, max_size=max_size,
    )


def _build_retinanet(
    adapter: Any, num_classes: int, *, featmap_names: list[str], num_levels: int,
    anchor_base_size: int = 32, min_size: int = 800, max_size: int = 1333,
    aspect_ratios: tuple[float, ...] = (0.5, 1.0, 2.0), **_: Any,
) -> Any:
    from torchvision.models.detection import RetinaNet
    from torchvision.models.detection.rpn import AnchorGenerator

    sizes = _default_anchor_sizes(num_levels, anchor_base_size)
    # RetinaNet: 3 octave scales x len(ratios) anchors/location.
    octave_sizes = tuple(tuple(int(s[0] * 2 ** (k / 3)) for k in range(3)) for s in sizes)
    ar = tuple(float(r) for r in aspect_ratios)
    anchor_generator = AnchorGenerator(sizes=octave_sizes, aspect_ratios=(ar,) * num_levels)
    return RetinaNet(
        adapter, num_classes=num_classes + 1,
        anchor_generator=anchor_generator, min_size=min_size, max_size=max_size,
    )


def _build_mask_rcnn(
    adapter: Any, num_classes: int, *, featmap_names: list[str], num_levels: int,
    anchor_base_size: int = 32, min_size: int = 800, max_size: int = 1333,
    aspect_ratios: tuple[float, ...] = (0.5, 1.0, 2.0), **_: Any,
) -> Any:
    from torchvision.models.detection import MaskRCNN
    from torchvision.models.detection.rpn import AnchorGenerator
    from torchvision.ops import MultiScaleRoIAlign

    sizes = _default_anchor_sizes(num_levels, anchor_base_size)
    ar = tuple(float(r) for r in aspect_ratios)
    anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=(ar,) * num_levels)
    box_roi_pool = MultiScaleRoIAlign(featmap_names=featmap_names, output_size=7, sampling_ratio=2)
    mask_roi_pool = MultiScaleRoIAlign(featmap_names=featmap_names, output_size=14, sampling_ratio=2)
    return MaskRCNN(
        adapter, num_classes=num_classes + 1,  # +1 for background
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=box_roi_pool, mask_roi_pool=mask_roi_pool,
        min_size=min_size, max_size=max_size,
    )


_DETECTOR_BUILDERS = {
    "faster_rcnn": _build_faster_rcnn,
    "fcos": _build_fcos,
    "retinanet": _build_retinanet,
    "mask_rcnn": _build_mask_rcnn,
}


def build_detector(name: str, adapter: Any, num_classes: int, **kwargs: Any) -> Any:
    """Instantiate a detector builder by name.

    Raises ``KeyError`` for an unknown name and ``TypeError`` for an unrecognized kwarg. The
    ``_build_*`` functions end in ``**_: Any``, so without this check a mistyped or unsupported
    key is swallowed and the parameter it was meant to set stays at its pinned default — the
    derived value silently does not apply, with no error and no record.
    """
    try:
        fn = _DETECTOR_BUILDERS[name]
    except KeyError:
        raise KeyError(
            f"Unknown detector '{name}'. Available: {sorted(_DETECTOR_BUILDERS)}"
        ) from None
    # Named parameters only — the trailing VAR_KEYWORD is what we are guarding against, so
    # including it would make the accepted set universal and the check a no-op.
    params = inspect.signature(fn).parameters
    accepted = {n for n, p in params.items()
                if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)} - {"adapter", "num_classes"}
    unknown = sorted(set(kwargs) - accepted)
    if unknown:
        raise TypeError(
            f"build_detector('{name}', ...) got unexpected keyword argument(s) {unknown}. "
            f"Accepted: {sorted(accepted)}"
        )
    return fn(adapter, num_classes, **kwargs)

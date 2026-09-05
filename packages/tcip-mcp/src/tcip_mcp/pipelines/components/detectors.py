"""2D object-detector builders: plain torchvision detector factories.

Bespoke model code imports these directly: build a ``BackboneNeckAdapter`` over an
agent-composed backbone+neck, then call ``build_detector`` (or a ``_build_*`` builder
directly) to get an ``nn.Module`` honoring the torchvision-detection forward contract:
``model(images, targets)`` returns a loss dict in train mode and ``list[dict]``
predictions in eval mode.
"""

from __future__ import annotations

import inspect
from collections import OrderedDict
from typing import Any, cast

import torch
import torch.nn as nn


class BackboneNeckAdapter(nn.Module):
    """Wrap a backbone+neck so a torchvision detector can consume it as its backbone."""

    def __init__(self, backbone: nn.Module, neck: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        # neck.out_channels is a neck-specific attribute nn.Module's stub can't see; its concrete
        # shape (int or sequence) is decided at runtime by which neck was composed.
        neck_out_channels = cast(Any, neck).out_channels
        self.out_channels = (
            neck_out_channels if isinstance(neck_out_channels, int)
            else neck_out_channels[-1]
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


def _probe_in_chans(adapter: Any) -> int | None:
    """A *hint* at the input band count, from the first ``Conv2d`` in registration order.

    Only ever consulted when the caller passes no ``in_chans``, and never allowed to contradict
    one. ``modules()`` yields in attribute-assignment order, which says nothing about the forward
    graph: an adapter that registers its neck first reports the neck's width, and one that
    band-projects N bands through a 1x1 conv into a pretrained 3-channel backbone reports whichever
    conv was assigned first. Treating this as authoritative both blocks correct builds and lets
    wrong ones through, so it stays a fallback. ``None`` when there is no conv at all.
    """
    for module in adapter.modules():
        if isinstance(module, nn.Conv2d):
            return int(module.in_channels)
    return None


def _normalization(adapter: Any, in_chans: int | None, image_mean, image_std,
                   detector: str) -> dict:
    """Validated ``image_mean``/``image_std`` kwargs for a torchvision detector.

    torchvision's ``GeneralizedRCNNTransform`` defaults to 3-element ImageNet statistics and
    broadcasts them against a ``[C, H, W]`` input, so any ``C != 3`` raises inside the transform
    with an error naming no channel concept. Nothing is synthesized here: per-band statistics are a
    property of the dataset, so an N-channel build without them is refused rather than normalized
    against numbers the platform picked. Derive them with
    ``pipelines.derivations.band_normalization_stats`` and pass them.
    """
    # The caller is authoritative: only they know the band count, and the probe is a registration-
    # order guess that is wrong for band-projection and neck-first adapters alike.
    if in_chans is None:
        probed = _probe_in_chans(adapter)
        in_chans = probed if probed is not None else 3
    if image_mean is None and image_std is None:
        if in_chans == 3:
            return {}  # torchvision's own ImageNet default applies
        raise ValueError(
            f"build_detector('{detector}', ..., in_chans={in_chans}) needs per-band image_mean and "
            f"image_std of length {in_chans}: torchvision normalizes with 3-element ImageNet "
            f"statistics by default, which do not describe a {in_chans}-band image: at 1 channel "
            f"they silently broadcast it to 3, and at any other count they raise inside the "
            f"transform. Derive them from your training split with "
            f"pipelines.derivations.band_normalization_stats(...) and pass both."
        )
    if image_mean is None or image_std is None:
        raise ValueError(
            f"build_detector('{detector}', ...) needs image_mean and image_std together; got only "
            f"{'image_mean' if image_mean is not None else 'image_std'}."
        )
    mean, std = [float(v) for v in image_mean], [float(v) for v in image_std]
    if len(mean) != in_chans or len(std) != in_chans:
        raise ValueError(
            f"build_detector('{detector}', ..., in_chans={in_chans}) got image_mean of length "
            f"{len(mean)} and image_std of length {len(std)}; both must be {in_chans}."
        )
    return {"image_mean": mean, "image_std": std}


def _build_faster_rcnn(
    adapter: Any, num_classes: int, *, featmap_names: list[str], num_levels: int,
    anchor_base_size: int = 32, min_size: int = 800, max_size: int = 1333,
    aspect_ratios: tuple[float, ...] = (0.5, 1.0, 2.0), in_chans: int | None = None,
    image_mean: Any = None, image_std: Any = None, **kwargs: Any,
) -> Any:
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.rpn import AnchorGenerator
    from torchvision.ops import MultiScaleRoIAlign

    sizes = _default_anchor_sizes(num_levels, anchor_base_size)
    # aspect_ratios is a builder kwarg (was hardcoded): set/derive it per trait, since an elongated
    # object class (~1:3-1:6) needs a tall ratio the default (0.5,1,2) can't match.
    ar = tuple(float(r) for r in aspect_ratios)
    anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=(ar,) * num_levels)
    roi_pool = MultiScaleRoIAlign(featmap_names=featmap_names, output_size=7, sampling_ratio=2)
    return FasterRCNN(
        adapter, num_classes=num_classes + 1,  # +1 for background
        rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pool,
        min_size=min_size, max_size=max_size,
        **_normalization(adapter, in_chans, image_mean, image_std, "faster_rcnn"), **kwargs,
    )


def _build_fcos(
    adapter: Any, num_classes: int, *, featmap_names: list[str], num_levels: int,
    anchor_base_size: int = 32, min_size: int = 800, max_size: int = 1333,
    in_chans: int | None = None, image_mean: Any = None, image_std: Any = None,
    **kwargs: Any,
) -> Any:
    from torchvision.models.detection import FCOS
    from torchvision.models.detection.rpn import AnchorGenerator

    sizes = _default_anchor_sizes(num_levels, anchor_base_size)
    # FCOS is anchor-free: exactly one point/anchor per location (ratio 1.0).
    anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=((1.0,),) * num_levels)
    return FCOS(
        adapter, num_classes=num_classes + 1,
        anchor_generator=anchor_generator, min_size=min_size, max_size=max_size,
        **_normalization(adapter, in_chans, image_mean, image_std, "fcos"), **kwargs,
    )


def _build_retinanet(
    adapter: Any, num_classes: int, *, featmap_names: list[str], num_levels: int,
    anchor_base_size: int = 32, min_size: int = 800, max_size: int = 1333,
    aspect_ratios: tuple[float, ...] = (0.5, 1.0, 2.0), in_chans: int | None = None,
    image_mean: Any = None, image_std: Any = None, **kwargs: Any,
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
        **_normalization(adapter, in_chans, image_mean, image_std, "retinanet"), **kwargs,
    )


def _build_mask_rcnn(
    adapter: Any, num_classes: int, *, featmap_names: list[str], num_levels: int,
    anchor_base_size: int = 32, min_size: int = 800, max_size: int = 1333,
    aspect_ratios: tuple[float, ...] = (0.5, 1.0, 2.0), in_chans: int | None = None,
    image_mean: Any = None, image_std: Any = None, **kwargs: Any,
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
        **_normalization(adapter, in_chans, image_mean, image_std, "mask_rcnn"), **kwargs,
    )


_DETECTOR_BUILDERS = {
    "faster_rcnn": _build_faster_rcnn,
    "fcos": _build_fcos,
    "retinanet": _build_retinanet,
    "mask_rcnn": _build_mask_rcnn,
}

# The torchvision class each builder constructs. Its own constructor parameters (box_score_thresh,
# rpn_nms_thresh, detections_per_img, ...) are part of the surface an agent may tune, so they are
# accepted and forwarded rather than rejected as unknown, while a typo still raises.
_DETECTOR_CLASSES = {
    "faster_rcnn": ("torchvision.models.detection", "FasterRCNN"),
    "fcos": ("torchvision.models.detection", "FCOS"),
    "retinanet": ("torchvision.models.detection", "RetinaNet"),
    "mask_rcnn": ("torchvision.models.detection", "MaskRCNN"),
}


# Structural arguments the builders construct and pass themselves. Accepting them would let a
# caller past the guard only to hit "got multiple values for keyword argument" from torchvision;
# they are shaped through anchor_base_size / aspect_ratios / featmap_names / num_levels instead.
_BUILDER_SUPPLIED = frozenset({
    "rpn_anchor_generator", "anchor_generator", "box_roi_pool", "mask_roi_pool",
})


def _accepted_kwargs(name: str) -> set[str]:
    """Every keyword ``build_detector(name, ...)` accepts.

    The builder's own named parameters plus the torchvision detector class's, since the builder
    forwards its ``**kwargs`` to that constructor.
    """
    import importlib

    def _named(obj) -> set[str]:
        return {n for n, p in inspect.signature(obj).parameters.items()
                if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}

    module, cls_name = _DETECTOR_CLASSES[name]
    cls = getattr(importlib.import_module(module), cls_name)
    return (_named(_DETECTOR_BUILDERS[name]) | _named(cls)) - {
        "adapter", "backbone", "num_classes",
    } - _BUILDER_SUPPLIED


def build_detector(name: str, adapter: Any, num_classes: int, **kwargs: Any) -> Any:
    """Instantiate a detector builder by name.

    Raises ``KeyError`` for an unknown name and ``TypeError`` for an unrecognized kwarg, so a
    mistyped or inapplicable key cannot leave the parameter it was meant to set at a pinned default
    with no error and no record. Accepted keys are the builder's own plus the torchvision detector
    class's, which the builder forwards: the library's real surface, not a shorter list of it.

    An ``in_chans != 3`` build additionally requires ``image_mean``/``image_std`` of that length;
    see ``_normalization``.
    """
    try:
        fn = _DETECTOR_BUILDERS[name]
    except KeyError:
        raise KeyError(
            f"Unknown detector '{name}'. Available: {sorted(_DETECTOR_BUILDERS)}"
        ) from None
    accepted = _accepted_kwargs(name)
    unknown = sorted(set(kwargs) - accepted)
    if unknown:
        # Name the detectors that do take each rejected kwarg. A key valid elsewhere is an
        # architecture difference (fcos is anchor-free, so it has no aspect_ratios) rather than a
        # typo, and the two need different responses from the caller.
        elsewhere = {
            k: [o for o in _DETECTOR_BUILDERS if k in _accepted_kwargs(o)]
            for k in unknown
        }
        detail = "; ".join(
            f"{k!r} is accepted by {others}" if (others := elsewhere[k]) else f"{k!r} by none"
            for k in unknown
        )
        raise TypeError(
            f"build_detector('{name}', ...) got unexpected keyword argument(s) {unknown}. "
            f"Accepted by '{name}': {sorted(accepted)}. {detail}."
        )
    return fn(adapter, num_classes, **kwargs)

"""Detector factory — registry-driven 2D object detectors.

A detection spec (a single head named ``anchor_detection`` / ``anchor_free_detection``)
is composed into a ``DetectionModel`` (see ``composer.py``) whose *actual* detector is
looked up here by name. New detectors — Mask R-CNN, DETR, YOLO, or an external framework
— are added by **registering a builder**, not by editing ``DetectionModel``.

Each builder receives the composed backbone+neck ``adapter`` and the resolved
``featmap_names`` / ``num_levels``, and returns an ``nn.Module`` honoring the
torchvision-detection forward contract: ``model(images, targets)`` returns a loss dict in
train mode and ``list[dict]`` predictions in eval mode.

External packages register via :func:`register_external_detector` or a ``tcip.components``
entry point (see ``registry.load_plugins``).
"""

from __future__ import annotations

from typing import Any

from tcip_mcp.pipelines.registry import DETECTORS

_BASE_META = {
    "valid_tasks": ["detection"],
    "input_format": "multi_scale_dict",
    "output_format": "boxes",
    "required_deps": ["torchvision"],
}


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
    # aspect_ratios is a spec kwarg (was hardcoded): set/derive it per trait — elongated catkins
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


DETECTORS.register_factory("faster_rcnn", _build_faster_rcnn, category="anchor_based", metadata={
    **_BASE_META, "description": "torchvision Faster R-CNN over a composed backbone+neck",
    "anchor_free": False,
})
DETECTORS.register_factory("fcos", _build_fcos, category="anchor_free", metadata={
    **_BASE_META, "description": "torchvision FCOS (anchor-free) over a composed backbone+neck",
    "anchor_free": True,
})
DETECTORS.register_factory("retinanet", _build_retinanet, category="anchor_based", metadata={
    **_BASE_META, "description": "torchvision RetinaNet (single-stage) over a composed backbone+neck",
    "anchor_free": False,
})


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


DETECTORS.register_factory("mask_rcnn", _build_mask_rcnn, category="instance_seg", metadata={
    **_BASE_META, "valid_tasks": ["instance_seg"], "output_format": "boxes+masks",
    "description": "torchvision Mask R-CNN (instance segmentation) over a composed backbone+neck",
    "anchor_free": False,
})


def build_detector(name: str, adapter: Any, num_classes: int, **kwargs: Any) -> Any:
    """Look up a registered detector builder by name and instantiate it.

    Raises ``KeyError`` (listing available detectors) for an unknown name.
    """
    return DETECTORS.build(name, adapter=adapter, num_classes=num_classes, **kwargs)


def register_external_detector(
    name: str, builder: Any, *, metadata: dict | None = None
) -> None:
    """Register a detector builder defined outside this package (plugins/experiments).

    ``builder`` must accept ``(adapter, num_classes, *, featmap_names, num_levels, **kwargs)``
    and return an nn.Module with the torchvision-detection forward contract.
    """
    DETECTORS.register_external(name, builder, category="external", metadata=metadata)

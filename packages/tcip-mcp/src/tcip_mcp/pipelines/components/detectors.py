"""2D object-detector builders: plain torchvision detector factories.

``build_detector`` over a ``BackboneNeckAdapter`` answers an ``nn.Module`` honoring the
torchvision-detection forward contract: ``model(images, targets)`` returns a loss dict in train
mode and ``list[dict]`` predictions in eval mode.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn

from tcip_mcp.pipelines.operating_point import DETECTIONS_PER_IMG, SCORE_THRESH

if TYPE_CHECKING:
    from collections.abc import Callable

    from torchvision.models.detection import (
        FCOS,
        FasterRCNN,
        MaskRCNN,
        RetinaNet,
    )
    from torchvision.models.detection.image_list import ImageList

    from tcip_mcp.subject_registry import Attribute

ROI_OUTPUT_SIZE = 7
"""The edge, in feature cells, every box's pooled features are resampled to: torchvision's own
box-head pool edge."""


class BackboneNeckAdapter(nn.Module):
    """Wrap a backbone+neck so a torchvision detector can consume it as its backbone."""

    def __init__(self, backbone: nn.Module, neck: nn.Module) -> None:
        """Hold ``backbone`` and ``neck``, taking the neck's last channel count as its own."""
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        # neck.out_channels is a neck-specific attribute nn.Module's stub can't see; its concrete
        # shape (int or sequence) is decided at runtime by which neck was composed.
        neck_out_channels = cast(Any, neck).out_channels
        self.out_channels = (
            neck_out_channels
            if isinstance(neck_out_channels, int)
            else neck_out_channels[-1]
        )

    def forward(self, x: torch.Tensor) -> OrderedDict:
        """Return the neck's features of ``x`` in sorted key order, a lone map under ``"0"``."""
        features = self.backbone(x)
        neck_out = self.neck(features)
        if isinstance(neck_out, dict):
            return OrderedDict(sorted(neck_out.items()))
        return OrderedDict({"0": neck_out})


def _anchor_sizes(num_levels: int, base: int) -> tuple[tuple[int, ...], ...]:
    """Return one anchor size per pyramid level, ``base`` at the finest, doubling each level.

    ``base=8, num_levels=4`` -> ``((8,),(16,),(32,),(64,))``.
    """
    return tuple((base * 2**i,) for i in range(num_levels))


def _probe_in_chans(adapter: nn.Module) -> int | None:
    """Return the ``in_channels`` of ``adapter``'s first ``Conv2d`` in registration order.

    ``None`` when it holds no conv.
    """
    for module in adapter.modules():
        if isinstance(module, nn.Conv2d):
            return int(module.in_channels)
    return None


def _normalization(
    adapter: nn.Module,
    in_chans: int | None,
    image_mean: list[float],
    image_std: list[float],
    detector: str,
) -> dict:
    """Validate ``image_mean``/``image_std`` kwargs for a torchvision detector at ``in_chans``.

    ``in_chans`` is probed from ``adapter`` when ``None``; one value per band. Refuses
    (``ValueError``) statistics of another length than the band count, naming the stated route.
    """
    mean, std = [float(v) for v in image_mean], [float(v) for v in image_std]
    # The caller is authoritative: only they know the band count, and the probe is a registration-
    # order guess that is wrong for band-projection and neck-first adapters alike.
    if in_chans is None:
        in_chans = _probe_in_chans(adapter)
    bands = in_chans if in_chans is not None else len(mean)
    if len(mean) != bands or len(std) != bands:
        raise ValueError(
            f"build_detector('{detector}', ...) got image_mean of length {len(mean)} and "
            f"image_std of length {len(std)}; both must be {bands}, the band count this detector "
            "reads. A model reading more bands than its pretrained weights carry statistics for "
            "states every band's mean and std with its reason (schemas.StatedBeyondWeights)."
        )
    return {"image_mean": mean, "image_std": std}


def _rcnn_anchors_and_box_pool(
    featmap_names: list[str],
    num_levels: int,
    anchor_base_size: int,
    aspect_ratios: tuple[float, ...],
) -> tuple[Any, Any]:
    """Build the anchor generator and box RoI pool a two-stage R-CNN reads its levels with.

    Every level takes ``aspect_ratios``.
    """
    from torchvision.models.detection.rpn import AnchorGenerator
    from torchvision.ops import MultiScaleRoIAlign

    ratios = tuple(float(r) for r in aspect_ratios)
    return (
        AnchorGenerator(
            sizes=_anchor_sizes(num_levels, anchor_base_size),
            aspect_ratios=(ratios,) * num_levels,
        ),
        MultiScaleRoIAlign(
            featmap_names=featmap_names,
            output_size=ROI_OUTPUT_SIZE,
            sampling_ratio=2,
        ),
    )


def _build_faster_rcnn(
    adapter: nn.Module,
    num_classes: int,
    *,
    featmap_names: list[str],
    num_levels: int,
    anchor_base_size: int,
    min_size: int,
    max_size: int,
    aspect_ratios: tuple[float, ...],
    in_chans: int | None = None,
    image_mean: list[float],
    image_std: list[float],
    **kwargs: object,
) -> FasterRCNN:
    from torchvision.models.detection import FasterRCNN

    anchor_generator, roi_pool = _rcnn_anchors_and_box_pool(
        featmap_names, num_levels, anchor_base_size, aspect_ratios
    )
    return FasterRCNN(
        adapter,
        num_classes=num_classes + 1,  # +1 for background
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pool,
        min_size=min_size,
        max_size=max_size,
        **_normalization(
            adapter, in_chans, image_mean, image_std, "faster_rcnn"
        ),
        **kwargs,
    )


def _build_fcos(
    adapter: nn.Module,
    num_classes: int,
    *,
    featmap_names: list[str],
    num_levels: int,
    anchor_base_size: int,
    min_size: int,
    max_size: int,
    in_chans: int | None = None,
    image_mean: list[float],
    image_std: list[float],
    **kwargs: object,
) -> FCOS:
    from torchvision.models.detection import FCOS
    from torchvision.models.detection.rpn import AnchorGenerator

    sizes = _anchor_sizes(num_levels, anchor_base_size)
    # FCOS is anchor-free: exactly one point/anchor per location (ratio 1.0).
    anchor_generator = AnchorGenerator(
        sizes=sizes, aspect_ratios=((1.0,),) * num_levels
    )
    return FCOS(
        adapter,
        num_classes=num_classes + 1,
        anchor_generator=anchor_generator,
        min_size=min_size,
        max_size=max_size,
        **_normalization(adapter, in_chans, image_mean, image_std, "fcos"),
        **kwargs,
    )


def _build_retinanet(
    adapter: nn.Module,
    num_classes: int,
    *,
    featmap_names: list[str],
    num_levels: int,
    anchor_base_size: int,
    min_size: int,
    max_size: int,
    aspect_ratios: tuple[float, ...],
    in_chans: int | None = None,
    image_mean: list[float],
    image_std: list[float],
    **kwargs: object,
) -> RetinaNet:
    from torchvision.models.detection import RetinaNet
    from torchvision.models.detection.rpn import AnchorGenerator

    sizes = _anchor_sizes(num_levels, anchor_base_size)
    # RetinaNet: 3 octave scales x len(ratios) anchors/location.
    octave_sizes = tuple(
        tuple(int(s[0] * 2 ** (k / 3)) for k in range(3)) for s in sizes
    )
    ar = tuple(float(r) for r in aspect_ratios)
    anchor_generator = AnchorGenerator(
        sizes=octave_sizes, aspect_ratios=(ar,) * num_levels
    )
    return RetinaNet(
        adapter,
        num_classes=num_classes + 1,
        anchor_generator=anchor_generator,
        min_size=min_size,
        max_size=max_size,
        **_normalization(
            adapter, in_chans, image_mean, image_std, "retinanet"
        ),
        **kwargs,
    )


def _build_mask_rcnn(
    adapter: nn.Module,
    num_classes: int,
    *,
    featmap_names: list[str],
    num_levels: int,
    anchor_base_size: int,
    min_size: int,
    max_size: int,
    aspect_ratios: tuple[float, ...],
    in_chans: int | None = None,
    image_mean: list[float],
    image_std: list[float],
    **kwargs: object,
) -> MaskRCNN:
    from torchvision.models.detection import MaskRCNN
    from torchvision.ops import MultiScaleRoIAlign

    anchor_generator, box_roi_pool = _rcnn_anchors_and_box_pool(
        featmap_names, num_levels, anchor_base_size, aspect_ratios
    )
    mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=featmap_names, output_size=14, sampling_ratio=2
    )
    return MaskRCNN(
        adapter,
        num_classes=num_classes + 1,  # +1 for background
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=box_roi_pool,
        mask_roi_pool=mask_roi_pool,
        min_size=min_size,
        max_size=max_size,
        **_normalization(
            adapter, in_chans, image_mean, image_std, "mask_rcnn"
        ),
        **kwargs,
    )


_DETECTORS: dict[str, tuple[Callable[..., nn.Module], str]] = {
    "faster_rcnn": (_build_faster_rcnn, "FasterRCNN"),
    "fcos": (_build_fcos, "FCOS"),
    "retinanet": (_build_retinanet, "RetinaNet"),
    "mask_rcnn": (_build_mask_rcnn, "MaskRCNN"),
}
"""Each detector name's builder and the ``torchvision.models.detection`` class it constructs, whose
own constructor parameters the builder forwards."""


# Structural arguments the builders construct and pass themselves; a caller shapes them through
# anchor_base_size / aspect_ratios / featmap_names / num_levels instead.
_BUILDER_SUPPLIED = frozenset(
    {
        "rpn_anchor_generator",
        "anchor_generator",
        "box_roi_pool",
        "mask_roi_pool",
    }
)


def _accepted_kwargs(name: str) -> set[str]:
    """Return every keyword the detector ``name`` accepts.

    That is its builder's own named parameters plus its torchvision class's, less what the builder
    supplies itself.
    """
    import torchvision.models.detection as detection

    from tcip_mcp.pipelines.model_build import keyword_parameters

    builder, cls_name = _DETECTORS[name]
    return (
        (
            keyword_parameters(builder)[0]
            | keyword_parameters(getattr(detection, cls_name))[0]
        )
        - {"adapter", "backbone", "num_classes"}
        - _BUILDER_SUPPLIED
    )


def _held(name: str) -> property:
    """Return a property reading and writing ``name`` on the detector an attribute model holds."""
    return property(
        lambda self: getattr(self.detector, name),
        lambda self, value: setattr(self.detector, name, value),
    )


DetectorOutput = dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]
"""A torchvision detector's forward: its losses in train mode, one prediction per image in eval."""


class AttributeDetector(nn.Module):
    """A torchvision detector with one per-instance head per attribute of its subject.

    Each head reads the channel means of the features :class:`~torchvision.ops.MultiScaleRoIAlign`
    pools at a box. ``attributes`` are the scope's attribute records, each carrying ``name``,
    ``type`` and ``values``: a categorical one gets a :class:`~tcip_mcp.pipelines.components.heads.
    ClassificationHead` trained by cross-entropy, an ordinal one an :class:`~tcip_mcp.pipelines.
    components.heads.OrdinalHead` trained by its CORN loss. Forward hooks on the detector's
    ``backbone`` and ``transform`` hold the feature maps, resized images and resized targets of
    the one forward. In train mode each head pools at the resized target boxes and its loss,
    masked to the rows its target column assesses (``attributes`` other than
    :data:`~tcip_annotation.json_io.UNASSESSED`), joins the detector's losses; in eval mode each
    head pools at the detector's output boxes in the resized frame and every output dict carries
    ``attributes``, boxes by attributes in declared order, int64. ``backbone``, ``roi_heads``,
    ``score_thresh`` and ``detections_per_img`` read and write the held detector's.
    """

    backbone = _held("backbone")
    roi_heads = _held("roi_heads")
    score_thresh = _held(SCORE_THRESH)
    detections_per_img = _held(DETECTIONS_PER_IMG)
    _features: dict[str, torch.Tensor]
    """The held backbone's feature maps of the one forward, by name."""
    _transformed: tuple[ImageList, list[dict[str, torch.Tensor]] | None]
    """The held transform's resized images and resized targets of the one forward."""

    def __init__(
        self,
        detector: nn.Module,
        attributes: tuple[Attribute, ...],
        *,
        featmap_names: list[str],
        in_channels: int,
    ) -> None:
        """Hold ``detector`` and build one head per record of ``attributes``.

        Each head reads ``in_channels`` pooled channel means off ``featmap_names``; the hooks that
        hold one forward's features and resized inputs are registered on ``detector``.
        """
        from torchvision.ops import MultiScaleRoIAlign

        from tcip_annotation.state import ATTR_TYPES
        from tcip_mcp.pipelines.components.heads import (
            ClassificationHead,
            OrdinalHead,
        )

        super().__init__()
        self.detector = detector
        self.attributes = tuple(attributes)
        self.pool = MultiScaleRoIAlign(
            featmap_names=featmap_names,
            output_size=ROI_OUTPUT_SIZE,
            sampling_ratio=2,
        )
        head_of = dict(
            zip(ATTR_TYPES, (ClassificationHead, OrdinalHead), strict=True)
        )
        self.attribute_heads = nn.ModuleList(
            head_of[a.type](in_channels, len(a.values))
            for a in self.attributes
        )
        cast(Any, detector).backbone.register_forward_hook(self._hold_features)
        cast(Any, detector).transform.register_forward_hook(
            self._hold_transformed
        )

    def _hold_features(
        self,
        _module: nn.Module,
        _inputs: object,
        output: dict[str, torch.Tensor],
    ) -> None:
        self._features = output

    def _hold_transformed(
        self,
        _module: nn.Module,
        _inputs: object,
        output: tuple[ImageList, list[dict[str, torch.Tensor]] | None],
    ) -> None:
        self._transformed = output

    def _pooled(self, boxes: list[torch.Tensor]) -> torch.Tensor:
        return self.pool(
            self._features, boxes, self._transformed[0].image_sizes
        ).mean(dim=(2, 3))

    def forward(
        self, images: list[torch.Tensor], targets: list[dict] | None = None
    ) -> DetectorOutput:
        """Run the detector and its attribute heads over ``images``.

        In train mode, return the detector's losses joined by each head's loss over its assessed
        target rows; in eval mode, return the detector's outputs, each carrying ``attributes``.
        """
        from tcip_annotation.json_io import UNASSESSED

        out = self.detector(images, targets)
        if self.training:
            resized = self._transformed[1]
            assert resized is not None, "a detector trains on resized targets"
            pooled = self._pooled([t["boxes"] for t in resized])
            truth = torch.cat([t["attributes"] for t in resized]).reshape(
                -1, len(self.attributes)
            )
            losses = {}
            for i, (attribute, head) in enumerate(
                zip(self.attributes, cast(Any, self.attribute_heads))
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
            return {**out, **losses}
        sizes = self._transformed[0].image_sizes
        boxes = []
        for image, (h, w), result in zip(images, sizes, out):
            scale = result["boxes"].new_tensor(
                [w / image.shape[-1], h / image.shape[-2]] * 2
            )
            boxes.append(result["boxes"] * scale)
        pooled = self._pooled(boxes)
        columns = [
            head.decode(head(pooled))[head.target_key]
            for head in cast(Any, self.attribute_heads)
        ]
        ids = torch.stack(columns, dim=1).long()
        for result, rows in zip(
            out, ids.split([len(r["boxes"]) for r in out])
        ):
            result["attributes"] = rows
        return out


def build_detector(
    name: str,
    adapter: BackboneNeckAdapter,
    num_classes: int,
    *,
    featmap_names: list[str],
    attributes: tuple[Attribute, ...] = (),
    **kwargs: object,
) -> nn.Module:
    """Instantiate a detector builder by name, held by an :class:`AttributeDetector` if need be.

    The :class:`AttributeDetector` carries one head per record of ``attributes`` when it names
    any. Raises ``ValueError`` for an unknown name and ``TypeError`` for an unrecognized kwarg,
    naming the detectors that do take it. Accepted keys are the builder's own plus the torchvision
    detector class's, which the builder forwards, but an operating-point knob (a detection cap,
    ``box_detections_per_img`` or ``detections_per_img``, or a score threshold,
    ``box_score_thresh`` or ``score_thresh``), which refuses (``ValueError``). Every build
    requires ``anchor_base_size``, ``min_size``, ``max_size`` and, but for ``fcos``,
    ``aspect_ratios`` (``schemas.DetectorGeometry``), and ``image_mean`` and ``image_std`` of the
    band count's length.
    """
    from tcip_mcp.pipelines.model_build import resolve_named

    fn, _cls_name = resolve_named(name, _DETECTORS, kind="detector")
    stated_point = sorted(
        {
            "box_detections_per_img",
            DETECTIONS_PER_IMG,
            "box_score_thresh",
            SCORE_THRESH,
        }
        & set(kwargs)
    )
    if stated_point:
        raise ValueError(
            f"build_detector('{name}', ...) was given {stated_point}: a detector's operating "
            "point is set at every forward, its cap each frame's object density times its "
            "pixels and its score threshold the pass's conf, or none at validation. Drop "
            f"{stated_point}."
        )
    accepted = _accepted_kwargs(name)
    unknown = sorted(set(kwargs) - accepted)
    if unknown:
        elsewhere = {
            k: [o for o in _DETECTORS if k in _accepted_kwargs(o)]
            for k in unknown
        }
        detail = "; ".join(
            f"{k!r} is accepted by {others}"
            if (others := elsewhere[k])
            else f"{k!r} by none"
            for k in unknown
        )
        raise TypeError(
            f"build_detector('{name}', ...) got unexpected keyword argument(s) {unknown}. "
            f"Accepted by '{name}': {sorted(accepted)}. {detail}."
        )
    detector = fn(adapter, num_classes, featmap_names=featmap_names, **kwargs)
    if not attributes:
        return detector
    return AttributeDetector(
        detector,
        attributes,
        featmap_names=featmap_names,
        in_channels=adapter.out_channels,
    )

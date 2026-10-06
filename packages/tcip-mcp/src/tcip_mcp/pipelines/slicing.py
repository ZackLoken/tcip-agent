"""Tiled inference over the ``sahi`` library: the tile geometry a pass runs at, the slice lattice,
a platform checkpoint wrapped as a SAHI detection model, and the one cross-tile merge every tiled
path runs."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from sahi.models.base import DetectionModel
from sahi.postprocess.backends import set_postprocess_backend
from sahi.postprocess.combine import PostprocessPredictions
from sahi.predict import POSTPROCESS_NAME_TO_CLASS
from sahi.prediction import ObjectPrediction
from sahi.slicing import get_slice_bboxes

from tcip_annotation.mask_contours import mask_to_polygon_rings

from tcip_mcp.pipelines.execution import DEFAULT_OVERLAP

if TYPE_CHECKING:
    from tcip_mcp.pipelines.execution import Execution
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor

logger = logging.getLogger(__name__)


def _native_ratio_tile_size(train_native_size: Any) -> int | None:
    """The tile edge a checkpoint's own uniform untiled training frame justifies: the frame's edge
    when every training frame shared one square size (``train_native_size``, stamped ``[width,
    height]``), else ``None``."""
    if not isinstance(train_native_size, (list, tuple)) or len(train_native_size) != 2:
        return None
    try:
        width, height = int(train_native_size[0]), int(train_native_size[1])
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    if width != height:
        logger.info(
            "no tile edge derivable from this checkpoint's untiled training frame (%dx%d): tile "
            "geometry is a single square edge, and no square edge reproduces a rectangular frame's "
            "scale on both axes. Pass tile_size explicitly, or run untiled.", width, height)
        return None
    return width


class TileEdgeContradictionError(ValueError):
    """A caller-stated tile edge that differs from the checkpoint's own recorded tile geometry."""


@dataclass(frozen=True)
class TileGeometry:
    """A pass's tile edge and overlap, each beside the source it came from, and the resize each
    tile runs through.

    ``tile_size_source`` is ``"explicit"``, ``"derived"`` (the checkpoint's persisted training
    geometry), ``"native_ratio"`` (the square frame the checkpoint trained untiled at) or
    ``"unavailable"`` (``tile_size`` ``None``); ``tile_size_derived_from`` says why a stated edge on
    a tiled pass is trusted and is ``None`` otherwise. ``overlap_source`` is ``"explicit"``,
    ``"derived"`` or ``"default"``.
    """

    tile_size: int | None
    tile_size_source: str
    tile_size_derived_from: str | None
    overlap: float
    overlap_source: str
    tile_resize: tuple[int, int] | None


def resolve_tile_geometry(
    predictor: GenericPredictor, *, tiled: bool, tile_size: int | None, overlap: float | None,
) -> TileGeometry:
    """The tile geometry a pass of ``predictor`` runs at, each value by precedence: stated, then
    the checkpoint's persisted training geometry (``train_tile_size``/``train_overlap``), then for
    the edge the square frame it trained untiled at (``train_native_size``), else the edge
    ``None`` and the overlap ``DEFAULT_OVERLAP``.

    A stated edge on a tiled pass that differs from the checkpoint's recorded edge (persisted, else
    native) raises :class:`TileEdgeContradictionError` naming both. A tiled pass whose edge came
    from the native frame runs each tile through the resize the run's recorded augmentation chain
    applied (:func:`~tcip_mcp.pipelines.data.augmentations.recorded_resize`); every other pass
    runs none.
    """
    persisted = predictor.train_tile_size
    native = _native_ratio_tile_size(predictor.train_native_size)
    recorded = int(persisted) if persisted is not None else native
    edge: int | None = None
    derived_from = None
    if tile_size is not None:
        edge, source = int(tile_size), "explicit"
        if tiled and recorded is None:
            derived_from = "stated on a checkpoint that records no tile geometry"
        elif tiled:
            kind = ("persisted training tile geometry" if persisted is not None
                    else "recorded untiled training frame")
            if recorded != edge:
                raise TileEdgeContradictionError(
                    f"stated tile_size {edge} contradicts this checkpoint's own {kind} of "
                    f"{recorded}. Pass tile_size {recorded} to match the checkpoint, or leave "
                    "tile_size unset to derive it from the checkpoint.")
            derived_from = (
                "equal to the checkpoint's persisted training tile geometry"
                if persisted is not None else
                "equal to the edge the checkpoint's recorded untiled training frame yields, run "
                "without that frame's own recorded resize")
    elif persisted is not None:
        edge, source = int(persisted), "derived"
    elif native is not None:
        edge, source = native, "native_ratio"
    else:
        source = "unavailable"

    if overlap is not None:
        resolved_overlap, overlap_source = float(overlap), "explicit"
    elif predictor.train_overlap is not None:
        resolved_overlap, overlap_source = float(predictor.train_overlap), "derived"
    else:
        resolved_overlap, overlap_source = DEFAULT_OVERLAP, "default"

    tile_resize = None
    if tiled and source == "native_ratio":
        from tcip_mcp.pipelines.data.augmentations import recorded_resize

        tile_resize = recorded_resize(predictor.train_augmentation)
    return TileGeometry(edge, source, derived_from, resolved_overlap, overlap_source, tile_resize)


def packed_category(label: int, ids: Sequence[int], sizes: Sequence[int]) -> int:
    """One detection's label and attribute ids as the one integer a SAHI category carries: mixed
    radix, the first attribute's id least significant, each attribute's radix its value count
    ``sizes``, the label most significant; the label alone for no attributes."""
    packed = int(label)
    for value, size in zip(reversed(ids), reversed(sizes), strict=True):
        packed = packed * size + int(value)
    return packed


def unpacked_category(packed: int, sizes: Sequence[int]) -> tuple[int, list[int]]:
    """``(label, attribute ids)`` from a :func:`packed_category` integer."""
    ids = []
    for size in sizes:
        packed, value = divmod(packed, size)
        ids.append(value)
    return packed, ids


def slice_lattice(
    height: int, width: int, tile_size: int, overlap: float,
) -> list[tuple[int, int, int, int]]:
    """The half-open ``(x0, y0, x1, y1)`` slices SAHI's ``get_slice_bboxes`` lays over a ``height``
    x ``width`` frame at ``tile_size`` and ``overlap``, every slice parameter stated and automatic
    slice sizing off. A slice past the far edge is pulled back to end at it, so no slice is padded
    and none exceeds the frame; a frame shorter than ``tile_size`` on an axis gets slices that
    short on it."""
    return [(int(x0), int(y0), int(x1), int(y1)) for x0, y0, x1, y1 in get_slice_bboxes(
        image_height=height, image_width=width, slice_height=tile_size, slice_width=tile_size,
        auto_slice_resolution=False, overlap_height_ratio=overlap, overlap_width_ratio=overlap)]


def is_full_slice(box: tuple[int, int, int, int], tile_size: int) -> bool:
    """Whether a lattice slice spans the whole ``tile_size`` on both axes, which it does unless
    the frame is shorter than the tile on that axis."""
    x0, y0, x1, y1 = box
    return x1 - x0 == tile_size and y1 - y0 == tile_size


def prediction_rows(predictions: list[ObjectPrediction], sizes: Sequence[int]) -> list[dict]:
    """Full-frame predictions as plain rows (xyxy ``bbox``, ``category_id`` the label, the
    ``attributes`` ids :func:`unpacked_category` reads off the category under the value counts
    ``sizes``, ``score``, ``segmentation`` polygons or ``None``)."""
    rows = []
    for p in predictions:
        label, ids = unpacked_category(p.category.id, sizes)
        rows.append({"bbox": [float(v) for v in p.bbox.to_xyxy()], "category_id": label,
                     "attributes": ids, "score": float(p.score.value),
                     "segmentation": p.mask.segmentation if p.mask else None})
    return rows


def predictions_from_rows(rows: list[dict], full_shape: list[int],
                          sizes: Sequence[int]) -> list[ObjectPrediction]:
    """The predictions :func:`prediction_rows` recorded, over a ``[height, width]`` frame."""
    return [ObjectPrediction(bbox=r["bbox"], score=r["score"],
                             category_id=packed_category(r["category_id"], r["attributes"], sizes),
                             segmentation=r["segmentation"],
                             full_shape=full_shape if r["segmentation"] else None) for r in rows]


def cross_tile_merge(execution: Execution) -> PostprocessPredictions:
    """The SAHI postprocess a tiled execution record names, at its threshold over its own match
    metric, class-agnostic over the one subject so two slices' calls of one object merge whatever
    their attribute values (a merged box keeps the category of the member SAHI keeps, the
    higher-scoring), on SAHI's numpy backend, so the merge leaves the process environment as it
    found it."""
    # SAHI's torchvision backend picks its device by writing CUDA_VISIBLE_DEVICES for the process.
    set_postprocess_backend("numpy")
    assert execution.merge_type is not None, "a tiled record names its merge"
    return POSTPROCESS_NAME_TO_CLASS[execution.merge_type](
        match_threshold=execution.cross_tile_nms, match_metric=execution.match_metric,
        class_agnostic=True)


class TcipDetectionModel(DetectionModel):
    """A platform predictor as a SAHI detection model.

    Each slice, an ``[H, W, C]`` array at whatever band count the source carries, reaches the model
    through the predictor's own tensor conversion; ``tile_resize`` stretches a slice PIL represents
    faithfully to that ``(width, height)`` and maps its boxes and masks back per axis, and a batch
    holding a slice PIL does not represent logs that the resize was skipped for it. Detections
    scoring at least ``conf`` (all of them for ``None``) become ``ObjectPrediction``s clipped to
    the slice, each carrying its label and its ``attributes`` ids under the predictor's
    ``attribute_sizes`` as one :func:`packed_category`, each mask cut at ``mask_binarize``'s value
    into the rings
    :func:`~tcip_annotation.mask_contours.mask_to_polygon_rings` extracts when ``collect_masks``.
    ``mask_binarize`` is the threshold's provenance, kept whole on the model it cut masks for.
    """

    def __init__(
        self, predictor: GenericPredictor, *, conf: float | None,
        tile_resize: tuple[int, int] | None,
        band_interpretations: tuple[str, ...] | None, collect_masks: bool, mask_binarize: dict,
    ) -> None:
        self._predictor = predictor
        self._conf = conf
        self._tile_resize = tile_resize
        self._band_interpretations = band_interpretations
        self.collect_masks = collect_masks
        self.mask_binarize = mask_binarize
        super().__init__(model=predictor.model, mask_threshold=mask_binarize["value"])

    def set_device(self, device: str | None = None) -> None:
        """The predictor's own device."""
        self.device = self._predictor.device

    def set_model(self, model: Any, **kwargs: Any) -> None:
        self.model = model

    def perform_inference(self, image: np.ndarray) -> None:
        self.perform_batch_inference([image])

    def perform_batch_inference(self, images: list[np.ndarray]) -> None:
        """One forward over every slice in ``images``."""
        from PIL import Image

        from tcip_mcp.pipelines.image_utils import pil_to_tensor, to_pil_if_faithful

        tensors, meta, unresized = [], [], False
        for arr in images:
            h, w = arr.shape[:2]
            pil = to_pil_if_faithful(arr, band_interpretations=self._band_interpretations)
            model_input: Any
            if self._tile_resize is not None and isinstance(pil, Image.Image):
                from tcip_mcp.pipelines.data.augmentations import Resize

                target = (int(self._tile_resize[0]), int(self._tile_resize[1]))
                model_input, _ = Resize(size=target)(pil, {})
                scale = (target[0] / w, target[1] / h)
            else:
                model_input, scale = arr, (1.0, 1.0)
                unresized = unresized or self._tile_resize is not None
            tensors.append(pil_to_tensor(model_input).to(self.device))
            meta.append((scale, h, w))
        if unresized:
            logger.warning(
                "tiled inference: the checkpoint's recorded train-time resize %s was not applied, "
                "these tiles are in no PIL mode and the training loader's own transform chain "
                "skipped such samples too.", self._tile_resize)
        self._original_predictions = list(zip(self.model(tensors), meta))

    def _create_object_prediction_list_from_original_predictions(
        self, shift_amount_list: list[list[int | float]] | None = None,
        full_shape_list: list[list[int | float]] | None = None,
    ) -> None:
        import torch

        assert shift_amount_list is not None and full_shape_list is not None
        per_image: list[list[ObjectPrediction]] = []
        for (out, ((sx, sy), h, w)), shift, full in zip(
                self._original_predictions, shift_amount_list, full_shape_list):
            keep = (out["scores"] >= self._conf if self._conf is not None
                    else torch.ones_like(out["scores"], dtype=torch.bool))
            boxes = out["boxes"][keep].cpu().numpy().astype(np.float64)
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] / sx, 0, w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] / sy, 0, h)
            scores = out["scores"][keep].cpu().tolist()
            sizes = self._predictor.attribute_sizes
            categories = [packed_category(label, ids, sizes) for label, ids in zip(
                out["labels"][keep].cpu().tolist(),
                out["attributes"][keep].cpu().tolist() if sizes else [[]] * len(scores),
                strict=True)]
            segmentations: list = [None] * len(scores)
            if self.collect_masks:
                masks = out["masks"][keep]
                if masks.dim() == 4 and masks.shape[1] == 1:
                    masks = masks[:, 0]
                if (sx, sy) != (1.0, 1.0) and len(masks):
                    masks = torch.nn.functional.interpolate(
                        masks.unsqueeze(1).float(), size=(h, w), mode="bilinear",
                        align_corners=False).squeeze(1)
                segmentations = [
                    [[c for point in ring for c in point]
                     for ring in mask_to_polygon_rings(m, threshold=self.mask_threshold)] or None
                    for m in masks.cpu().numpy()]
            per_image.append([
                ObjectPrediction(bbox=box.tolist(), category_id=category, score=float(score),
                                 segmentation=seg, shift_amount=shift, full_shape=full)
                for box, score, category, seg in zip(boxes, scores, categories, segmentations)])
        self._object_prediction_list_per_image = per_image

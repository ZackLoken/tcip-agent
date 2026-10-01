"""Tiled inference over the ``sahi`` library: the slice lattice, a platform checkpoint wrapped as a
SAHI detection model, and the one cross-tile merge every tiled path runs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from sahi.models.base import DetectionModel
from sahi.postprocess.backends import set_postprocess_backend
from sahi.postprocess.combine import PostprocessPredictions
from sahi.predict import POSTPROCESS_NAME_TO_CLASS
from sahi.prediction import ObjectPrediction
from sahi.slicing import get_slice_bboxes

from tcip_annotation.mask_contours import mask_to_polygon_rings

if TYPE_CHECKING:
    from tcip_mcp.pipelines.execution import Execution

CLASS_AGNOSTIC = False
"""Whether the cross-tile merge compares detections across classes; it never does."""


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


def prediction_rows(predictions: list[ObjectPrediction]) -> list[dict]:
    """Full-frame predictions as plain rows (xyxy ``bbox``, ``category_id``, ``score``,
    ``segmentation`` polygons or ``None``): the one projection a paused pass records and a detection
    record is built from."""
    return [{"bbox": [float(v) for v in p.bbox.to_xyxy()], "category_id": p.category.id,
             "score": float(p.score.value),
             "segmentation": p.mask.segmentation if p.mask else None} for p in predictions]


def predictions_from_rows(rows: list[dict], full_shape: list[int]) -> list[ObjectPrediction]:
    """The predictions :func:`prediction_rows` recorded, over a ``[height, width]`` frame."""
    return [ObjectPrediction(bbox=r["bbox"], category_id=int(r["category_id"]), score=r["score"],
                             segmentation=r["segmentation"],
                             full_shape=full_shape if r["segmentation"] else None) for r in rows]


def cross_tile_merge(execution: Execution) -> PostprocessPredictions:
    """The SAHI postprocess a tiled execution record names, at its threshold over its own match
    metric and class-aware, on SAHI's numpy backend, so the merge leaves the process environment
    as it found it."""
    # SAHI's torchvision backend picks its device by writing CUDA_VISIBLE_DEVICES for the process.
    set_postprocess_backend("numpy")
    assert execution.merge_type is not None, "a tiled record names its merge"
    return POSTPROCESS_NAME_TO_CLASS[execution.merge_type](
        match_threshold=execution.cross_tile_nms, match_metric=execution.match_metric,
        class_agnostic=CLASS_AGNOSTIC)


class TcipDetectionModel(DetectionModel):
    """A platform predictor as a SAHI detection model.

    Each slice, an ``[H, W, C]`` array at whatever band count the source carries, reaches the model
    through the predictor's own tensor conversion; ``tile_resize`` stretches a slice PIL represents
    faithfully to that ``(width, height)`` and maps its boxes and masks back per axis, and a batch
    holding a slice PIL does not represent logs that the resize was skipped for it. Detections
    scoring at least ``conf`` (all of them for ``None``) become ``ObjectPrediction``s clipped to
    the slice, each mask cut at ``mask_binarize``'s value into the rings
    :func:`~tcip_annotation.mask_contours.mask_to_polygon_rings` extracts when ``collect_masks``.
    ``mask_binarize`` is the threshold's provenance, kept whole on the model it cut masks for.
    """

    def __init__(
        self, predictor: Any, *, conf: float | None, tile_resize: tuple[int, int] | None,
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
            logging.getLogger(__name__).warning(
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
            labels = out["labels"][keep].cpu().tolist()
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
                ObjectPrediction(bbox=box.tolist(), category_id=int(label), score=float(score),
                                 segmentation=seg, shift_amount=shift, full_shape=full)
                for box, score, label, seg in zip(boxes, scores, labels, segmentations)])
        self._object_prediction_list_per_image = per_image

"""Encoding a pass's predictions as per-image documents, and delivering a bucket's per-image
counts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from tcip_annotation.state import BBox, Polygon
    from tcip_mcp.pipelines.data.selection import ClassScope

logger = logging.getLogger(__name__)


def _clip(value: float, upper: float | None) -> float:
    """Clamp ``value`` into ``[0, upper]``; ``upper=None`` (no known image extent) leaves it as-is."""
    if upper is None:
        return value
    return max(0.0, min(float(value), float(upper)))


def stored_geometries(image_result: dict) -> dict[int, BBox | Polygon]:
    """Each detection of one raw predictor result that a write stores, by index, as the geometry
    stored: its mask's rings when it carries one (:func:`_mask_geometry_for_export`), else its box,
    kept only when that geometry has extent on the stored grid
    (:func:`~tcip_annotation.json_io.geometry_extent_ok`).
    """
    from tcip_annotation.json_io import geometry_extent_ok
    from tcip_annotation.state import BBox

    masks = image_result.get("masks")
    size = (image_result["width"], image_result["height"])
    kept: dict[int, BBox | Polygon] = {}
    for i, box in enumerate(image_result["boxes"]):
        geometry: BBox | Polygon = BBox(*box)
        if masks is not None and i < len(masks):
            geometry = _mask_geometry_for_export(masks[i], tuple(box), f"detection {i}",
                                                 image_size=size)
        if geometry_extent_ok(geometry):
            kept[i] = geometry
    return kept


def encode_predictions(
    result: dict, created_by: str, *, scope: "ClassScope",
) -> tuple[bytes, int]:
    """A detection result encoded as its name-based per-image prediction document
    (:func:`~tcip_annotation.json_io.encode_annotations`), and the number of detections dropped.

    ``result`` carries its source ``image``, pixel-xyxy ``boxes``, 1-indexed ``labels``
    (background=0), ``scores``, image ``width``/``height``, and, when ``scope`` declares
    attributes, ``attributes``, one id row per detection; a result missing one of the first six,
    or whose boxes, scores and labels differ in length, refuses (``ValueError``) naming it, and so does an
    image of a reserved stem and a ``scope`` naming no subject or attributes
    (:meth:`~tcip_mcp.pipelines.data.selection.ClassScope.admitted_for`). Every prediction's
    ``subject`` is ``scope``'s, and its ``attributes`` each attribute's value its row names
    (:func:`~tcip_annotation.json_io.attribute_values`), none for a scope declaring none. An
    image with no detection encodes ``{"annotations": []}``. ``created_by`` stamps every
    prediction.

    ``masks`` (``instance_seg``), one ``{"segmentation"}`` of flat full-image polygons per
    detection, become one ``Polygon`` ring per polygon; a mask that binarized to nothing stores the
    detection's ``BBox`` (logged). A detection whose stored geometry has no extent is dropped, and
    the same entries are dropped from ``result``'s ``boxes``/``scores``/``labels``/
    ``attributes``/``masks``/``count`` in place.
    """
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation
    from tcip_mcp.audit import now_iso
    from tcip_mcp.pipelines.data.selection import DOCUMENT
    from tcip_mcp.pipelines.inference.generic_predictor import DETECTION_ROWS

    attributes = cast(tuple, scope.admitted_for(DOCUMENT, "this bucket's scope").attributes)
    missing = [k for k in ("image", "width", "height", "boxes", "scores", "labels")
               if k not in result]
    if missing:
        raise ValueError(f"the result for {result.get('image')!r} carries no {missing}: a "
                         "prediction document holds a detection head's boxes, scores and "
                         "labels for one source image, so nothing here is publishable.")
    p = Path(result["image"])
    boxes, scores, labels = result["boxes"], result["scores"], result["labels"]
    if not len(boxes) == len(scores) == len(labels):
        raise ValueError(f"{p.name}: the result carries {len(boxes)} boxes, {len(scores)} scores "
                         f"and {len(labels)} labels, which name no one set of detections.")
    rows = result["attributes"] if attributes else [()] * len(boxes)
    w, h = result["width"], result["height"]
    created_at = now_iso()
    stored = stored_geometries(result)
    preds: list[Annotation] = []
    kept_indices: list[int] = []
    dropped = 0
    for i, (score, ids) in enumerate(zip(scores, rows, strict=True)):
        geometry = stored.get(i)
        if geometry is None:
            dropped += 1
            continue
        preds.append(Annotation(subject=cast(str, scope.subject), geometry=geometry,
                                score=float(score),
                                attributes=json_io.attribute_values(ids, attributes),
                                created_by=created_by, created_at=created_at))
        kept_indices.append(i)
    _path, data = json_io.encode_annotations(p, preds, int(w), int(h), keep_empty=True)
    assert data is not None, "keep_empty encodes every document"
    if dropped:
        kept = set(kept_indices)
        for key in DETECTION_ROWS:
            if result.get(key) is not None:
                result[key] = [v for i, v in enumerate(result[key]) if i in kept]
        result["count"] = len(kept_indices)
    return data, dropped


def encode_head_output(result: dict, *, task: str) -> tuple[bytes, int]:
    """A classification, ordinal, regression or semantic segmentation result encoded as its
    document: the source image's stem, its ``width`` and ``height``, the head's ``task`` and every
    output the head returned under ``outputs``, as returned; no detection is ever dropped. A
    result carrying no output, and an image of a reserved stem, refuse (``ValueError``)."""
    from tcip_annotation.json_io import is_reserved_stem
    from tcip_store import encode_record

    stem = Path(result["image"]).stem
    outputs = {k: v for k, v in result.items() if k not in ("image", "width", "height")}
    if not outputs:
        raise ValueError(f"the {task} result for {result['image']!r} carries no output, so "
                         "nothing here is publishable.")
    if is_reserved_stem(stem):
        raise ValueError(f"{stem} would name its document after a prediction bucket's own "
                         "record, so it can never have a per-image document.")
    return encode_record({"image": stem, "width": int(result["width"]),
                          "height": int(result["height"]), "task": task,
                          "outputs": outputs}), 0


def _mask_geometry_for_export(
    mask: dict, bbox_xyxy: tuple[float, float, float, float], subject: str, *,
    image_size: tuple[int, int] | None = None,
) -> BBox | Polygon:
    """One detection's mask -> a real (possibly multi-ring) Polygon, or BBox if empty.

    ``mask`` is a ``{"segmentation"}`` dict of flat ``[x0, y0, x1, y1, ...]`` polygons in
    full-image pixels (the record
    :meth:`tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor._detection_record`
    builds), each polygon one ring with every point clipped to ``image_size`` (``(width,
    height)``).
    """
    from tcip_annotation.state import BBox, Polygon

    max_x, max_y = image_size if image_size else (None, None)
    rings = [[(_clip(flat[i], max_x), _clip(flat[i + 1], max_y))
              for i in range(0, len(flat) - 1, 2)] for flat in mask["segmentation"]]
    if rings:
        return Polygon(rings=rings)
    logger.warning("%s: mask binarized to nothing, exporting BBox (no contour to store).", subject)
    x1, y1, x2, y2 = bbox_xyxy
    return BBox(x1, y1, x2, y2)


def deliver_per_image_counts_csv(
    project: Path, bucket_path: Path, output_path: str, *, trait: str,
    acknowledgment_id: str | None, door: str, actor: str | None,
) -> dict:
    """Deliver the per-image counts of the published bucket at ``bucket_path`` as the CSV at
    ``output_path``, by ``actor``, under ``trait``'s latest confirmed revision stating a
    ``per_image_count`` operationalization.

    The population is the bucket's own documents, as its record names them, and a raster bucket
    refuses. It clears the one gate (:func:`~tcip_mcp.delivery.gate`), the recorded
    acknowledgment ``acknowledgment_id`` shipping it unvalidated when the gate does not validate
    it. Each row is one document in stem
    order: ``image`` (its source file name), ``detection_count``, ``avg_confidence`` and the
    delivery's own cells, written with the delivery's one event under ``door``
    (:func:`~tcip_mcp.delivery.deliver_csv`); returns what was delivered.
    """
    from tcip_mcp.buckets import detection_rows, read_bucket
    from tcip_mcp.delivery import DELIVERY_COLUMNS, Result, deliver_csv, gate
    from tcip_mcp.operationalization import confirmed_revision
    from tcip_mcp.traits import PER_IMAGE_COUNT

    revision = confirmed_revision(PER_IMAGE_COUNT, project=project, trait=trait)
    bucket = read_bucket(bucket_path)
    if bucket.raster_path is not None:
        raise ValueError(f"{bucket_path} holds one whole-raster prediction: a mosaic total is not "
                         "a per-image count. Deliver per-plant counts from it through "
                         "deliver_orthomosaic_plant_counts.")
    rows = detection_rows(bucket)
    result = Result(
        ["image", "detection_count", "avg_confidence", *DELIVERY_COLUMNS],
        [{"image": r["image"], "detection_count": r["count"],
          "avg_confidence": round(sum(r["scores"]) / len(r["scores"]), 4) if r["scores"] else ""}
         for r in rows],
        population=sorted(bucket.documents))
    clearance = gate(project, [bucket], delivery_kind=PER_IMAGE_COUNT, revision=revision,
                     result=result, acknowledgment_id=acknowledgment_id)
    delivered = deliver_csv(project, output_path, result, clearance=clearance, revision=revision,
                            door=door, delivery_kind=PER_IMAGE_COUNT, actor=actor)
    return {**delivered, "image_count": len(rows),
            "total_detections": sum(r["count"] for r in rows), "predictions_dir": str(bucket_path)}

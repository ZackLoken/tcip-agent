"""A pass's predictions encoded as per-image documents, and a bucket's per-image count delivery."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from tcip_annotation.state import BBox, Polygon
    from tcip_mcp.delivery import PerImageCountsDelivered
    from tcip_mcp.pipelines.data.selection import ClassScope
    from tcip_mcp.pipelines.inference.passes import DetectionPass, HeadPass

logger = logging.getLogger(__name__)


def _clip(value: float, upper: float | None) -> float:
    """Clamp ``value`` into ``[0, upper]``; ``upper=None`` leaves it as-is."""
    if upper is None:
        return value
    return max(0.0, min(float(value), float(upper)))


def stored_geometries(result: DetectionPass) -> dict[int, BBox | Polygon]:
    """Return the geometry stored for each kept detection of one pass, by index.

    A detection's geometry is its mask's rings when it carries one
    (:func:`_mask_geometry_for_export`), else its box, kept only when that geometry has extent on
    the stored grid (:func:`~tcip_annotation.json_io.geometry_extent_ok`).
    """
    from tcip_annotation.json_io import geometry_extent_ok
    from tcip_annotation.state import BBox

    size = (result.width, result.height)
    kept: dict[int, BBox | Polygon] = {}
    for i, d in enumerate(result.detections):
        geometry: BBox | Polygon = (
            BBox(*d.box)
            if d.mask is None
            else _mask_geometry_for_export(
                d.mask, d.box, f"detection {i}", image_size=size
            )
        )
        if geometry_extent_ok(geometry):
            kept[i] = geometry
    return kept


def encode_predictions(
    result: DetectionPass,
    created_by: str,
    *,
    scope: ClassScope,
) -> tuple[dict, int]:
    """Encode a detector pass as its name-based per-image prediction document.

    Returns the document (:func:`~tcip_annotation.json_io.document_payload`), carrying the frame's
    ``cap``, and the number of detections dropped.

    A ``scope`` naming no subject or attributes refuses (``ValueError``)
    (:meth:`~tcip_mcp.pipelines.data.selection.ClassScope.document_attributes`), and so does a
    detection whose attribute ids are not one per attribute the scope declares. Every
    prediction's ``subject`` is ``scope``'s, and its ``attributes`` each attribute's value its ids
    name (:func:`~tcip_annotation.json_io.attribute_values`), none for a scope declaring none. An
    image with no detection encodes ``{"annotations": []}``. ``created_by`` stamps every
    prediction.

    Each mask becomes one ``Polygon`` ring per polygon; a mask that binarized to nothing stores the
    detection's ``BBox`` (logged). A detection whose stored geometry has no extent is dropped.
    """
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation
    from tcip_mcp.audit import now

    read = scope.document_attributes("this bucket's scope")
    attributes = read.attributes
    created_at = now().isoformat()
    stored = stored_geometries(result)
    preds: list[Annotation] = []
    for i, d in enumerate(result.detections):
        geometry = stored.get(i)
        if geometry is None:
            continue
        preds.append(
            Annotation(
                subject=read.subject,
                geometry=geometry,
                score=d.score,
                attributes=json_io.attribute_values(
                    d.attributes if attributes else [], attributes
                ),
                created_by=created_by,
                created_at=created_at,
            )
        )
    data = json_io.document_payload(
        preds, result.width, result.height, cap=result.cap
    )
    return data, len(result.detections) - len(preds)


def encode_head_output(result: HeadPass, *, task: str) -> tuple[dict, int]:
    """Encode a classification, ordinal, regression or semantic segmentation pass.

    Returns the document and a dropped count of zero: the document holds the source image's
    ``width`` and ``height``, the head's ``task`` and every output the head returned under
    ``outputs``, as returned. A pass carrying no output refuses (``ValueError``).
    """
    if not result.outputs:
        raise ValueError(
            f"the {task} result for {result.image!r} carries no output, so "
            "nothing here is publishable."
        )
    return {
        "width": result.width,
        "height": result.height,
        "task": task,
        "outputs": result.outputs,
    }, 0


def _mask_geometry_for_export(
    mask: list[list[float]],
    bbox_xyxy: tuple[float, float, float, float],
    subject: str,
    *,
    image_size: tuple[int, int] | None = None,
) -> BBox | Polygon:
    """Return the detection's (possibly multi-ring) polygon, or its box when the mask is empty.

    Each of ``mask``'s polygons is one ring with every point clipped to ``image_size``
    (``(width, height)``).
    """
    from tcip_annotation.state import BBox, Polygon

    max_x, max_y = image_size if image_size else (None, None)
    rings = [
        [
            (_clip(flat[i], max_x), _clip(flat[i + 1], max_y))
            for i in range(0, len(flat) - 1, 2)
        ]
        for flat in mask
    ]
    if rings:
        return Polygon(rings=rings)
    logger.warning(
        "%s: mask binarized to nothing, exporting BBox (no contour to store).",
        subject,
    )
    x1, y1, x2, y2 = bbox_xyxy
    return BBox(x1, y1, x2, y2)


def deliver_per_image_counts_csv(
    project: Path,
    dataset_root: str | Path,
    name: str,
    output_path: str,
    *,
    trait: str,
    acknowledgment_id: str | None,
    door: str,
    actor: str | None,
) -> PerImageCountsDelivered:
    """Deliver the per-image counts of a published bucket as the CSV at ``output_path``.

    The bucket is ``name`` published under ``dataset_root``; the delivery is by ``actor``, under
    ``trait``'s latest confirmed revision stating a ``per_image_count`` operationalization. The
    population is the bucket's own documents, as its record names them, and a raster bucket
    refuses. It clears :func:`~tcip_mcp.delivery.gate`, the recorded acknowledgment
    ``acknowledgment_id`` shipping it unvalidated when the gate does not validate it. Each row is
    one document in stem order: ``image`` (its source file name), ``detection_count``,
    ``avg_confidence`` and the delivery's own cells, written with the delivery's one event under
    ``door`` (:func:`~tcip_mcp.delivery.deliver_csv`); returns what was delivered.
    """
    from tcip_mcp.buckets import detection_rows, read_bucket
    from tcip_mcp.delivery import (
        DELIVERY_COLUMNS,
        PerImageCountsDelivered,
        Result,
        deliver_csv,
        gate,
    )
    from tcip_mcp.operationalization import confirmed_revision
    from tcip_mcp.traits import PER_IMAGE_COUNT

    revision = confirmed_revision(
        PER_IMAGE_COUNT, project=project, trait=trait
    )
    bucket = read_bucket(dataset_root, name)
    if bucket.raster_path is not None:
        raise ValueError(
            f"bucket {name!r} holds one whole-raster prediction: a mosaic total is "
            "not a per-image count. Deliver per-plant counts from it through "
            "deliver_orthomosaic_plant_counts."
        )
    rows = detection_rows(bucket)
    result = Result(
        ["image", "detection_count", "avg_confidence", *DELIVERY_COLUMNS],
        [
            {
                "image": r["image"],
                "detection_count": r["count"],
                "avg_confidence": round(sum(r["scores"]) / len(r["scores"]), 4)
                if r["scores"]
                else "",
            }
            for r in rows
        ],
        population=sorted(bucket.documents),
    )
    clearance = gate(
        project,
        [bucket],
        delivery_kind=PER_IMAGE_COUNT,
        revision=revision,
        result=result,
        acknowledgment_id=acknowledgment_id,
    )
    delivered = deliver_csv(
        project,
        output_path,
        result,
        clearance=clearance,
        revision=revision,
        door=door,
        delivery_kind=PER_IMAGE_COUNT,
        actor=actor,
    )
    return PerImageCountsDelivered(
        **dict(delivered),
        image_count=len(rows),
        total_detections=sum(r["count"] for r in rows),
    )

"""CSV export for per-plant phenotyping results."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from tcip_annotation.state import BBox, Polygon
    from tcip_mcp.pipelines.data.selection import ClassScope
    from tcip_mcp.pipelines.resolution import Acknowledgment
    from tcip_mcp.traits import TraitRevision

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
    for i, box in enumerate(image_result.get("boxes", [])):
        geometry: BBox | Polygon = BBox(*box)
        if masks is not None and i < len(masks):
            geometry = _mask_geometry_for_export(masks[i], tuple(box), f"detection {i}",
                                                 image_size=size)
        if geometry_extent_ok(geometry):
            kept[i] = geometry
    return kept


def positive_detections(image_result: dict) -> tuple[int, list[float]]:
    """One image's raw predictor result narrowed to real detections (:func:`stored_geometries`):
    the kept count and the kept confidence scores, as :func:`write_predictions_json` persists them.

    Falls back to ``image_result["count"]`` when no ``boxes`` are present at all.
    """
    scores = image_result.get("scores", [])
    if not image_result.get("boxes"):
        return image_result.get("count", 0), scores
    kept = stored_geometries(image_result)
    return len(kept), [s for i, s in enumerate(scores) if i in kept]


def _run_class_id(label: int) -> int:
    """The 0-indexed run class id a 1-indexed torchvision label (background 0) stands for."""
    return max(int(label) - 1, 0)


def write_predictions_json(
    json_path: str | Path, result: dict, created_by: str | None = None, *, scope: "ClassScope",
) -> int:
    """Write a ``GenericPredictor`` detection result as a name-based per-image prediction file.

    ``result`` carries pixel-xyxy ``boxes``, 1-indexed ``labels`` (background=0), ``scores``, and
    image ``width``/``height``. Each detection's numeric label is decoded via ``scope``'s map, the
    run's own admitted class space, as its checkpoint records it.

    Under a classified scope every decoded name lands in ``attributes[attribute]`` and ``subject``
    carries the object class itself; otherwise ``subject`` carries the decoded name and
    ``attributes`` stays empty. The first label the map cannot decode refuses (``ValueError``,
    naming the id and the map's own ids). ``keep_empty=True`` so a processed image with zero detections
    still yields an ``{"annotations": []}`` file. ``created_by`` stamps the producing model on every
    prediction.

    When ``result`` carries ``masks`` (``instance_seg``), each is a ``{"segmentation"}`` dict of
    flat polygons in full-image pixels, binarized by the predictor at the threshold the result's
    own ``mask_binarize`` records, and becomes a ``Polygon`` with one ring per polygon. A mask
    that binarized to nothing falls back to the detection's ``BBox`` (a warning is logged). The
    threshold is recorded once in the run's ``operating_point.json``, not on each annotation.

    A detection whose box or mask-derived box has no extent is dropped. Returns the number dropped.
    Mutates ``result`` in place to drop the same entries from its
    ``boxes``/``scores``/``labels``/``masks``/``count``.
    """
    from datetime import datetime, timezone

    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation
    from tcip_mcp.subject_registry import decode_class_ids

    p = Path(json_path)
    if json_io.is_sidecar_name(p.name):
        raise ValueError(
            f"{p.name} names one of a prediction bucket's own provenance stamps; an image whose "
            "stem is reserved this way can never be written as a bucket's per-image prediction "
            "document, since the stamp write would then destroy or refuse over it."
        )
    attribute = scope.attribute
    w, h = result["width"], result["height"]
    created_at = datetime.now(timezone.utc).isoformat() if created_by else None
    id_to_name = decode_class_ids(scope.id_map or {})
    masks = result.get("masks")
    boxes = result.get("boxes", [])
    scores = result.get("scores", [])
    labels = result.get("labels", [])
    stored = stored_geometries(result)
    preds: list[Annotation] = []
    kept_indices: list[int] = []
    dropped = 0
    for i, (score, label) in enumerate(zip(scores, labels)):
        cid = _run_class_id(label)
        if cid not in id_to_name:
            raise ValueError(
                f"{p.name}: detection {i} decoded to id {cid}, not a key of this run's recorded "
                f"id_map ({sorted(id_to_name)}); a name no admitted class map declares cannot be "
                "written."
            )
        name = id_to_name[cid]
        geometry = stored.get(i)
        if geometry is None:
            dropped += 1
            continue
        # A classified scope names its subject (ClassScope's own admission).
        pred_subject = cast(str, scope.subject) if attribute is not None else name
        pred_attributes = {attribute: name} if attribute is not None else {}
        preds.append(Annotation(subject=pred_subject, geometry=geometry, score=float(score),
                                attributes=pred_attributes,
                                created_by=created_by, created_at=created_at))
        kept_indices.append(i)
    json_io.write_annotations(str(json_path), preds, int(w), int(h), keep_empty=True)
    if dropped:
        kept = set(kept_indices)
        result["boxes"] = [b for i, b in enumerate(boxes) if i in kept]
        result["scores"] = [s for i, s in enumerate(scores) if i in kept]
        result["labels"] = [l for i, l in enumerate(labels) if i in kept]
        result["count"] = len(kept_indices)
        if masks is not None:
            result["masks"] = [m for i, m in enumerate(masks) if i in kept]
    return dropped


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


_PROVENANCE_COLUMNS = ["producer_model_sha256", "producing_experiment_id", "operating_point_conf",
                       "produced_at", "operating_point_validated", "unvalidated_dimensions",
                       "validation_record", "acknowledged_by", "acknowledgment_reason"]

_MEASUREMENT_DOCUMENT = "operating_point"
"""The sidecar document every per-image count rests on: the count operating point."""


def export_detection_csv(
    image_results: list[dict],
    output_path: str,
    provenance: dict | None = None,
    *,
    revision: TraitRevision,
    operating_point_validated: str | None = None,
    pred_dirs: list[str] | None = None,
    acknowledgment: Acknowledgment | None = None,
    project: Path,
) -> tuple[str, dict, dict]:
    """Export per-image detection counts to CSV.

    A delivery door: it refuses a bare write (an unvalidated count with no acknowledgment) via
    ``check_delivery_gate`` and stamps the reconciled validity into every row. ``pred_dirs`` (the
    prediction buckets the counts came from) has the count operating point's validity read from
    each ``operating_point.json`` sidecar and floored against ``operating_point_validated``. A
    bucket produced by a tiled run gates on its ``tile_size`` too; untiled buckets never do. The
    physical-scale dimension is never operative here. Without ``pred_dirs`` the operating_point
    dimension floors to unvalidated. ``acknowledgment`` is the breeder's own act of shipping this
    delivery unvalidated, or ``None``.

    The ``provenance`` cells are built by ``delivered_tail`` from the verification the gate ran: a
    producer this delivery cannot corroborate reads unknown, ``produced_at`` is the write's own
    timestamp, and ``validation_record`` names the record the claim was earned against.
    ``acknowledged_by``/``acknowledgment_reason`` carry the gate's own effective acknowledgment
    (blank together on a fully validated delivery, since the gate discards an acknowledgment that
    cleared nothing).

    Every row also carries ``measurement_document``, always ``"operating_point"``.

    Before composing the gate's flags, every bucket in ``pred_dirs`` is bound to ``revision``
    (``operationalization.bind``): its stamp must record ``revision``'s trait or none, and the
    object classes it counted (its scope's subject, else its recorded ``id_map`` keys; nothing for
    a bucket with no stamp) must include the ``per_image_count``
    operationalization's measured subject. The delivery event names ``revision``.

    Args:
        image_results: List of dicts with 'image', 'count', 'boxes', etc.
        output_path: Path for the output CSV file.
        provenance: Optional producing-model / operating-point stamp added as trailing columns.
        revision: The confirmed trait revision stating the ``per_image_count`` operationalization
            this delivery ships under (``operationalization.confirmed_revision``).
        operating_point_validated: The count operating point's reconciled validity reference.
            Floored against each bucket's on-disk sidecar when ``pred_dirs`` is given; floored to
            unvalidated otherwise.
        pred_dirs: Prediction buckets to reconcile the count operating point's (and, if tiled, the
            tile-geometry) validity from.
        acknowledgment: The breeder's own act of shipping this delivery unvalidated, or ``None``.
        project: The project this delivery belongs to, whose runs verify its buckets and
            whose log records its event.

    Returns:
        ``(path, tail, summary)``: the path to the written CSV, the ``_PROVENANCE_COLUMNS`` tail
        ``delivered_tail`` composed and wrote into every row, and the gate's own evaluation summary
        (``stamp``, ``unvalidated``, ``tile_size_operative``, ``tile_size_validated``,
        ``binding_notes``).

    Raises:
        DeliveryRefused: the gate refused (an unvalidated dimension with no acknowledgment that
            clears it); carries the ``DeliveryGateResult`` and both reconcilers' binding notes.
        OperationalizationRefused (``tcip_mcp.operationalization``): a bucket ``revision`` does
            not bind.
        AuditEntryNotWritten (``tcip_mcp.audit``): the delivery-event audit line could not be
            appended, raised by ``record_delivery_binding_event`` after the CSV and the
            ``delivery_events`` record were written.
    """
    from tcip_mcp.operationalization import bind
    from tcip_mcp.pipelines.data.selection import ClassScope
    from tcip_mcp.traits import PER_IMAGE_COUNT
    from tcip_annotation.json_io import safe_score
    from tcip_mcp.pipelines.resolution import (
        VALIDATED_FALSE,
        DeliveryRefused,
        binding_notes_text,
        check_delivery_gate,
        delivered_tail,
        read_operating_point_sidecar,
        record_delivery_binding_event,
        reconcile_operating_point_validity,
        reconcile_tile_size_validity,
    )

    buckets: dict[str, tuple[str | None, set[str]]] = {}
    for d in (pred_dirs or []):
        stamp = read_operating_point_sidecar(Path(d))
        scope = ClassScope.of(stamp) if stamp is not None else ClassScope()
        buckets[d] = (stamp["trait"] if stamp is not None else None,
                      set(scope.id_map) if scope.id_map and not scope.classified
                      else {scope.subject} if scope.subject else set())
    bind(revision, PER_IMAGE_COUNT, buckets=buckets)
    trait = revision.entry.name

    # With no pred_dirs nothing on disk backs the count's validity, so the dimension floors to
    # unvalidated rather than trusting the caller's bare string (mirrors export_aggregated_csv).
    flags: dict[str, str | None] = {"operating_point": VALIDATED_FALSE}
    operating_point_recon: dict | None = None
    tile_recon: dict | None = None
    if pred_dirs:
        # Reconciled from the buckets' own sidecars, floored against the caller assertion, never
        # trusted from the string alone (mirrors export_aggregated_csv's count-trait gating).
        operating_point_recon = reconcile_operating_point_validity(
            pred_dirs, project=project, trait=trait, asserted=operating_point_validated)
        flags["operating_point"] = operating_point_recon["validated"]
        tile_recon = reconcile_tile_size_validity(pred_dirs, project=project)
        if tile_recon["operative"]:
            flags["tile_size"] = tile_recon["validated"]

    notes = binding_notes_text({
        **(operating_point_recon or {}).get("binding_notes", {}),
        **(tile_recon or {}).get("binding_notes", {}),
    })
    gate = check_delivery_gate(flags, acknowledgment=acknowledgment)
    if not gate.ok:
        raise DeliveryRefused(gate, notes)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    stamp = delivered_tail(provenance, (operating_point_recon or {}).get("bindings", {}), gate,
                           columns=_PROVENANCE_COLUMNS, project=project)
    fieldnames = (["image", "detection_count", "avg_confidence", "measurement_document"]
                 + _PROVENANCE_COLUMNS)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in image_results:
            detection_count, scores = positive_detections(r)
            # Quantized at the persisted precision before averaging, never a re-spelled round(x, 4).
            safe_scores = [safe_score(s) for s in scores]
            avg_conf = sum(safe_scores) / len(safe_scores) if safe_scores else 0.0
            writer.writerow({
                "image": Path(r.get("image", "")).name,
                "detection_count": detection_count,
                "avg_confidence": round(avg_conf, 4),
                "measurement_document": _MEASUREMENT_DOCUMENT,
                **stamp,
            })

    record_delivery_binding_event(
        "export_detection_csv", output_path, pred_dirs,
        document_reconciliations=(
            {_MEASUREMENT_DOCUMENT: operating_point_recon} if operating_point_recon is not None
            else {}
        ),
        dimension_reconciliations={"tile_size": tile_recon} if tile_recon is not None else {},
        acknowledgment=gate.effective_acknowledgment(), revision=revision,
        delivery_kind=PER_IMAGE_COUNT, project=project)
    summary = {
        "stamp": gate.stamp,
        "unvalidated": gate.unvalidated,
        "tile_size_operative": (tile_recon or {}).get("operative", False),
        "tile_size_validated": (tile_recon or {}).get("validated"),
        "binding_notes": notes,
    }
    return output_path, stamp, summary

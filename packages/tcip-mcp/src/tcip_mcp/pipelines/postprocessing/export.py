"""CSV export for per-plant phenotyping results."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tcip_annotation.state import BBox, Polygon
    from tcip_mcp.pipelines.resolution import Acknowledgment

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


def unmapped_label_ids(results: list[dict], id_map: dict[str, int] | None) -> list[int]:
    """Every 0-indexed label id across ``results`` that ``id_map`` cannot decode, sorted; empty
    when every one decodes, or when ``id_map`` is ``None`` (the detector-run case, where the raw
    index is itself the name). Only meaningful for a classified run (``attribute`` set).
    """
    if id_map is None:
        return []
    known = set(id_map.values())
    unmapped: set[int] = set()
    for r in results:
        for label in r.get("labels", []):
            cid = _run_class_id(label)
            if cid not in known:
                unmapped.add(cid)
    return sorted(unmapped)


def write_predictions_json(
    json_path: str | Path, result: dict, created_by: str | None = None, *,
    subject: str | None, attribute: str | None, id_map: dict[str, int] | None = None,
) -> int:
    """Write a ``GenericPredictor`` detection result as a name-based per-image prediction file.

    ``result`` carries pixel-xyxy ``boxes``, 1-indexed ``labels`` (background=0), ``scores``, and
    image ``width``/``height``. Each detection's numeric label is decoded via ``id_map`` (the run's
    recorded ``operating_point.json`` name->id map).

    ``subject`` and ``attribute`` are the run's own scope
    (:func:`~tcip_mcp.tools.inference_tools.run_scope`). With ``attribute`` set, every decoded name
    lands in ``attributes[attribute]`` and ``subject`` carries the object class itself. With
    ``attribute=None``, ``subject`` carries the decoded name and ``attributes`` stays empty. An
    ``attribute`` with no ``subject`` refuses (``ValueError``) before the first document is
    written. Absent a recorded map, a detector run uses the raw 0-indexed id as the name; a
    classified run refuses (``ValueError``, naming the id and the map's own ids) the first label it
    cannot decode. ``keep_empty=True`` so a processed image with zero detections still yields an
    ``{"annotations": []}`` file. ``created_by`` stamps the producing model on every prediction.

    When ``result`` carries ``masks`` (``instance_seg``), each mask is binarized via
    :func:`tcip_mcp.pipelines.measurement.mask_geometry.resolve_binarize_threshold` and converted
    to a ``Polygon`` with one ring per connected component. A mask that binarizes to nothing falls
    back to the detection's ``BBox`` (a warning is logged). Each entry is either a dense ``[H, W]``
    array already in full-image coordinates (the untiled predictors' shape) or a ``{"mask_patch",
    "offset_x", "offset_y"}`` dict (the tiled predictors' shape); the offset, when present, is
    added to every ring point, clipped to the image's own ``width``/``height``.

    The threshold is an unvalidated run constant recorded in the run's ``operating_point.json``
    through :func:`mask_binarize_provenance`, not on each annotation.

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
    if attribute is not None and subject is None:
        raise ValueError(
            f"{p.name}: attribute {attribute!r} was given with no subject; a value with no "
            "object class names nothing a reader could hold this record to."
        )
    w, h = result["width"], result["height"]
    created_at = datetime.now(timezone.utc).isoformat() if created_by else None
    id_to_name = decode_class_ids(id_map) if id_map else {}
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
        if attribute is not None:
            if cid not in id_to_name:
                raise ValueError(
                    f"{p.name}: detection {i} decoded to id {cid}, not a key of this run's "
                    f"recorded id_map ({sorted(id_to_name)}); a value no vocabulary declares "
                    f"cannot be written under attribute {attribute!r}."
                )
            name = id_to_name[cid]
        else:
            name = id_to_name.get(cid, str(cid))  # decode via the recorded map, never a fresh derivation
        geometry = stored.get(i)
        if geometry is None:
            dropped += 1
            continue
        if attribute is not None:
            assert subject is not None  # refused above when attribute is set with no subject
            pred_subject = subject
        else:
            pred_subject = name
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


def mask_binarize_provenance() -> dict:
    """The run-constant unvalidated binarize threshold ``_mask_geometry_for_export`` actually used,
    as a stamp for the caller's own ``operating_point.json``, never a per-annotation attribute (see
    :func:`write_predictions_json`'s docstring). Call once per run, when ``masks`` were present."""
    from tcip_mcp.pipelines.measurement.mask_geometry import resolve_binarize_threshold

    return resolve_binarize_threshold().to_provenance()


def _mask_geometry_for_export(
    mask, bbox_xyxy: tuple[float, float, float, float], subject: str, *,
    image_size: tuple[int, int] | None = None,
) -> BBox | Polygon:
    """One detection's soft mask -> a real (possibly multi-ring) Polygon, or BBox if empty.

    ``mask`` is either a dense array already in full-image coordinates (the untiled predictors'
    shape) or a ``{"mask_patch", "offset_x", "offset_y"}`` dict (the tiled predictors' shape, see
    :meth:`tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor._tiled_infer_core`): the
    contour is extracted from the patch in its own local coordinates, then every ring point is
    shifted by the patch's offset and clipped to ``image_size`` (``(width, height)``), so a patch
    drawn from a zero-padded boundary tile can never place a stored point outside the image.
    """
    from tcip_annotation.state import BBox, Polygon
    from tcip_mcp.pipelines.measurement.mask_geometry import (
        mask_to_polygon_points, resolve_binarize_threshold,
    )

    offset_x = offset_y = 0
    patch = mask
    if isinstance(mask, dict):
        patch = mask["mask_patch"]
        offset_x, offset_y = int(mask["offset_x"]), int(mask["offset_y"])

    threshold = resolve_binarize_threshold().unvalidated_value(acknowledge_unvalidated=True)
    rings = mask_to_polygon_points(patch, threshold=threshold)
    if offset_x or offset_y:
        max_x = image_size[0] if image_size else None
        max_y = image_size[1] if image_size else None
        rings = [
            [(_clip(x + offset_x, max_x), _clip(y + offset_y, max_y)) for x, y in ring]
            for ring in rings
        ]
    if rings:
        return Polygon(rings=rings)
    logger.warning(
        "%s: mask binarized to nothing at threshold=%.3f, exporting BBox (no contour to store).",
        subject, threshold,
    )
    x1, y1, x2, y2 = bbox_xyxy
    return BBox(x1, y1, x2, y2)


_PROVENANCE_COLUMNS = ["producer_model_sha256", "producing_experiment_id", "operating_point_conf",
                       "produced_at", "operating_point_validated", "unvalidated_dimensions",
                       "validation_record", "acknowledged_by", "acknowledgment_reason"]

_MEASUREMENT_DOCUMENT = "operating_point"
"""What this CSV's counts always rest on: a per-image detection count is always the count
operating point, never a scalar head or a physical scale, so ``measurement_document`` is a
constant here rather than a per-row statement (contrast ``export_aggregated_csv``, whose rows can
carry any of the three per-plant measurement documents)."""


def export_detection_csv(
    image_results: list[dict],
    output_path: str,
    provenance: dict | None = None,
    *,
    trait: str,
    operating_point_validated: str | None = None,
    pred_dirs: list[str] | None = None,
    acknowledgment: Acknowledgment | None = None,
    project_root: str | Path | None = None,
) -> tuple[str, dict, dict, bool]:
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

    Refuses before composing the gate's flags unless ``trait``'s ``per_image_count``
    operationalization is recorded and breeder-confirmed. The record's ``measured_subject`` is
    checked against the object classes every bucket's own scope says its detections are of: a
    classified bucket's own subject, an unscoped bucket's recorded ``id_map`` keys. A delivery
    whose buckets contribute neither, and one called with no ``pred_dirs``, carries the subject
    unchecked.

    Args:
        image_results: List of dicts with 'image', 'count', 'boxes', etc.
        output_path: Path for the output CSV file.
        provenance: Optional producing-model / operating-point stamp added as trailing columns.
        trait: The registered trait whose confirmed per-image-count operationalization this
            delivery rests on. Required.
        operating_point_validated: The count operating point's reconciled validity reference.
            Floored against each bucket's on-disk sidecar when ``pred_dirs`` is given; floored to
            unvalidated otherwise.
        pred_dirs: Prediction buckets to reconcile the count operating point's (and, if tiled, the
            tile-geometry) validity from.
        acknowledgment: The breeder's own act of shipping this delivery unvalidated, or ``None``.
        project_root: The project this delivery's meaning-record reads and delivery event belong
            to. ``None`` resolves against this process's pinned platform root.

    Returns:
        ``(path, tail, summary, event_recorded)``: the path to the written CSV, the
        ``_PROVENANCE_COLUMNS`` tail ``delivered_tail`` composed and wrote into every row, the
        gate's own evaluation summary (``stamp``, ``unvalidated``, ``tile_size_operative``,
        ``tile_size_validated``, ``binding_notes``), and whether the best-effort delivery-event
        write landed (``record_delivery_binding_event``'s own return).

    Raises:
        DeliveryRefused: the gate refused (an unvalidated dimension with no acknowledgment that
            clears it); carries the ``DeliveryGateResult`` and both reconcilers' binding notes.
        OperationalizationRefused (``tcip_mcp.operationalization``): the ``trait``'s
            ``per_image_count`` operationalization is unrecorded, not breeder-confirmed, or was
            withdrawn since the first check; carries the failed check and no counts.
        AuditEntryNotWritten (``tcip_mcp.audit``): the dataset-scoped delivery-event audit line
            could not be appended, raised by ``record_delivery_binding_event`` after the CSV was
            already written to ``output_path``.
    """
    from tcip_mcp.operationalization import (
        PER_IMAGE_COUNT,
        OperationalizationRefused,
        check_operationalization,
        resolve_trait_and_record,
    )
    from tcip_annotation.json_io import safe_score
    from tcip_mcp.pipelines.postprocessing.phenology import bucket_id_map
    from tcip_mcp.pipelines.resolution import (
        VALIDATED_FALSE,
        DeliveryRefused,
        binding_notes_text,
        bucket_scope,
        check_delivery_gate,
        delivered_tail,
        record_delivery_binding_event,
        reconcile_operating_point_validity,
        reconcile_tile_size_validity,
    )

    # A bucket's own counted object classes: a classified stamp's subject, else its map's keys.
    counted_subjects: dict[str, set[str]] = {}
    for d in (pred_dirs or []):
        scope = bucket_scope(Path(d))
        if scope is not None and scope.classified:
            if scope.subject is not None:
                counted_subjects[d] = {scope.subject}
            continue
        recorded_map = bucket_id_map(Path(d))
        if recorded_map:
            counted_subjects[d] = set(recorded_map)
    spec, record, _specs_dir = resolve_trait_and_record(trait, PER_IMAGE_COUNT, project_root=project_root)
    # This door never delivers a crossing kind, so it has no registry to check a positive class against.
    stated = check_operationalization(
        spec, record, PER_IMAGE_COUNT, counted_subjects=counted_subjects or None, registry=None)
    if not stated.ok:
        raise OperationalizationRefused(stated)

    # With no pred_dirs nothing on disk backs the count's validity, so the dimension floors to
    # unvalidated rather than trusting the caller's bare string (mirrors export_aggregated_csv).
    flags: dict[str, str | None] = {"operating_point": VALIDATED_FALSE}
    operating_point_recon: dict | None = None
    tile_recon: dict | None = None
    if pred_dirs:
        # Reconciled from the buckets' own sidecars, floored against the caller assertion, never
        # trusted from the string alone (mirrors export_aggregated_csv's count-trait gating).
        operating_point_recon = reconcile_operating_point_validity(
            pred_dirs, trait=trait, asserted=operating_point_validated)
        flags["operating_point"] = operating_point_recon["validated"]
        tile_recon = reconcile_tile_size_validity(pred_dirs)
        if tile_recon["operative"]:
            flags["tile_size"] = tile_recon["validated"]

    notes = binding_notes_text({
        **(operating_point_recon or {}).get("binding_notes", {}),
        **(tile_recon or {}).get("binding_notes", {}),
    })
    gate = check_delivery_gate(flags, acknowledgment=acknowledgment)
    if not gate.ok:
        raise DeliveryRefused(gate, notes)

    # A confirmation withdrawn or a field moved since the first check refuses here, before anything.
    spec_now, record_now, _ = resolve_trait_and_record(
        trait, PER_IMAGE_COUNT, project_root=project_root)
    still_stated = check_operationalization(
        spec_now, record_now, PER_IMAGE_COUNT, counted_subjects=counted_subjects or None,
        registry=None, basis=stated.basis)
    if not still_stated.ok:
        raise OperationalizationRefused(still_stated)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    stamp = delivered_tail(provenance, (operating_point_recon or {}).get("bindings", {}), gate,
                           columns=_PROVENANCE_COLUMNS)
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

    event_recorded = record_delivery_binding_event(
        "export_detection_csv", output_path, pred_dirs,
        document_reconciliations=(
            {_MEASUREMENT_DOCUMENT: operating_point_recon} if operating_point_recon is not None
            else {}
        ),
        dimension_reconciliations={"tile_size": tile_recon} if tile_recon is not None else {},
        measurement_documents=[_MEASUREMENT_DOCUMENT],
        acknowledgment=gate.effective_acknowledgment(), trait=trait,
        delivery_kind=PER_IMAGE_COUNT, project_root=project_root)
    summary = {
        "stamp": gate.stamp,
        "unvalidated": gate.unvalidated,
        "tile_size_operative": (tile_recon or {}).get("operative", False),
        "tile_size_validated": (tile_recon or {}).get("validated"),
        "binding_notes": notes,
    }
    return output_path, stamp, summary, event_recorded

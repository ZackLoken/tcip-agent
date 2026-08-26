"""CSV export for per-plant phenotyping results."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tcip_annotation.state import BBox, Polygon

logger = logging.getLogger(__name__)


def _clip(value: float, upper: float | None) -> float:
    """Clamp ``value`` into ``[0, upper]``; ``upper=None`` (no known image extent) leaves it as-is."""
    if upper is None:
        return value
    return max(0.0, min(float(value), float(upper)))


def positive_detections(image_result: dict) -> tuple[int, list[float]]:
    """One image's raw predictor result narrowed to real detections: a box with no positive
    extent was never a detection, so it counts toward neither the kept count nor the confidence
    scores. The one predicate :func:`write_predictions_json` itself drops by, so a delivered count
    or a CSV row computed here always agrees with what that write actually persists.

    Falls back to ``image_result["count"]`` when no ``boxes`` are present at all (a caller that
    states only a bare count, never a per-box result to narrow).
    """
    from tcip_annotation.json_io import box_extent_ok
    from tcip_annotation.state import BBox

    boxes = image_result.get("boxes", [])
    scores = image_result.get("scores", [])
    if not boxes:
        return image_result.get("count", 0), scores
    keep = [box_extent_ok(BBox(*b)) for b in boxes]
    return keep.count(True), [s for s, k in zip(scores, keep) if k]


def write_predictions_json(
    json_path: str | Path, result: dict, created_by: str | None = None, *,
    id_map: dict[str, int] | None = None,
) -> int:
    """Write a ``GenericPredictor`` detection result as a name-based per-image prediction file.

    ``result`` carries pixel-xyxy ``boxes``, 1-indexed ``labels`` (background=0), ``scores``, and
    image ``width``/``height``. Each detection's numeric label is decoded to a name (its
    ``subject``) via ``id_map`` (the run's *recorded* ``operating_point.json`` name→id map), so a
    prediction on disk carries the same names its labels do, and decode is never a fresh
    ``assign_class_ids``. Absent a recorded map, the raw 0-indexed id is used as the name (a degraded
    but honest fallback, never a re-derivation). ``keep_empty=True`` so a processed image with zero
    detections still yields an ``{"annotations": []}`` file. ``created_by`` stamps the producing model
    on every prediction so the origin travels into GT when a human accepts it.

    When ``result`` carries ``masks`` (``instance_seg``, see
    :mod:`tcip_mcp.pipelines.inference.generic_predictor`), each mask is binarized via
    :func:`tcip_mcp.pipelines.measurement.mask_geometry.resolve_binarize_threshold` (never a bare
    hardcoded threshold) and converted to a real ``Polygon``: every connected component becomes its
    own ring (an occlusion-split instance, routine in this imagery, is genuinely more than one
    region), so the stored geometry never silently drops part of the object. A mask that binarizes to
    nothing falls back to the detection's ``BBox`` (a warning is logged) since there is no contour to
    store at all. Each entry is either a dense ``[H, W]`` array already in full-image coordinates
    (the untiled predictors' shape) or a ``{"mask_patch", "offset_x", "offset_y"}`` dict (the tiled
    predictors' shape, a tile-local patch plus its full-image-space origin); either way the polygon
    this function stores ends up in the same full-image pixel space, the offset (when present) is
    added to every ring point before it is stored, clipped to the image's own ``width``/``height`` so
    a patch drawn from a zero-padded boundary tile can never place a point outside the image.

    The threshold is an unvalidated default; its own :class:`ResolvedParam` contract requires the
    consuming output to carry ``validated=false`` so the uncertainty travels on. ``Annotation.attributes``
    is the wrong home for that (it is the domain trait namespace, a fixed vocabulary of names, and a
    machine-provenance float stamped there survives unfixed into GT the moment a breeder accepts the
    prediction). The threshold is a run constant, not a per-detection fact, so it belongs in the run's
    own ``operating_point.json`` instead; see :func:`mask_binarize_provenance`, which the two doors
    that write predictions to disk (``export_predictions``, the web inference route) call once and
    fold into that same stamp, mirroring how ``tiled``/``tile_size``/``conf`` already travel there.

    A detector's own box can collapse to zero extent at an image edge (clipping), or a mask can
    binarize to a sliver whose own derived box carries no real extent either; either way there is
    no detection to store, so it is dropped here rather than raising
    :func:`~tcip_annotation.json_io.write_annotations`'s own persistence-boundary refusal and
    failing the whole run over one degenerate detection. Returns the number dropped, for the
    caller's own run summary. Mutates ``result`` in place to drop the same entries from its
    ``boxes``/``scores``/``labels``/``masks``/``count``, so a caller that reads ``result`` again
    after this call (a delivered count, a CSV row) sees exactly what landed rather than a stale
    pre-drop figure.
    """
    from datetime import datetime, timezone

    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox, bbox_of
    from tcip_mcp.class_registry import decode_class_ids

    w = result.get("width") or 0
    h = result.get("height") or 0
    created_at = datetime.now(timezone.utc).isoformat() if created_by else None
    id_to_name = decode_class_ids(id_map) if id_map else {}
    masks = result.get("masks")
    boxes = result.get("boxes", [])
    scores = result.get("scores", [])
    labels = result.get("labels", [])
    preds: list[Annotation] = []
    kept_indices: list[int] = []
    dropped = 0
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = box
        cid = max(int(label) - 1, 0)  # undo the 1-indexed torchvision label -> 0-indexed run id
        name = id_to_name.get(cid, str(cid))  # decode via the recorded map, never a fresh derivation
        geometry: BBox | Polygon = BBox(x1, y1, x2, y2)
        if masks is not None and i < len(masks):
            geometry = _mask_geometry_for_export(masks[i], (x1, y1, x2, y2), name, image_size=(w, h))
        if not json_io.stored_box_extent_ok(bbox_of(geometry)):
            dropped += 1
            continue
        preds.append(Annotation(subject=name, geometry=geometry, score=float(score),
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


_PROVENANCE_COLUMNS = ["producer_model_sha256", "experiment_id", "operating_point_conf",
                       "produced_at", "measurement_validated", "validation_record"]

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
    measurement_validated: str | None = None,
    pred_dirs: list[str] | None = None,
    acknowledge_unvalidated: bool = False,
) -> str:
    """Export per-image detection counts to CSV.

    The count is the phenotype for count traits, so this is a delivery door: it refuses a *bare*
    write (an unvalidated count with no acknowledgement) via the shared ``check_delivery_gate`` and
    stamps the reconciled validity into every row. Pass ``pred_dirs`` (the prediction buckets the
    counts came from) so the count operating point's validity is read from each
    ``operating_point.json`` sidecar and floored against ``measurement_validated`` (never trusted
    from the string alone). A bucket produced by a tiled run gates on its ``tile_size`` too, the same
    operating point's other gating dimension: the tile edge scales the per-image counts this CSV
    reports, so a run with no persisted training geometry, no recoverable native-frame edge, and no
    explicit caller override refuses here. Untiled buckets are never gated on it. This CSV carries
    no dimensional
    value (its rows are ``detection_count``/``avg_confidence``, never an area/length/diameter), so
    the physical-scale dimension (see ``export_aggregated_csv``'s gate on
    ``resolve_scale.json``/``reconcile_scale_validity``) is never operative here: there is nothing in
    this CSV's own shape for a physical scale to have produced, so gating on it would manufacture a
    refusal over a dimension that can't apply to a count. Without ``pred_dirs`` there is no on-disk
    source for the count's validity, so the measurement dimension floors to unvalidated regardless
    of the caller's string, mirroring ``export_aggregated_csv``: a bare caller-asserted reference is
    never trusted on its own, and ``acknowledge_unvalidated=True`` is the only route to delivery on
    that path. Either way, ``acknowledge_unvalidated=True`` writes a clearly-flagged provisional
    CSV stamped ``validated=false``. The ``provenance`` stamp (producing checkpoint sha, experiment
    id, operating-point conf, timestamp) travels alongside; the number is only as trustworthy as the
    operating point + model behind it. Those cells are built by ``delivered_provenance`` from the
    verification the gate already ran, so a producer this delivery cannot corroborate is reported
    unknown rather than repeated from the stamp that asserted it, and ``validation_record`` names the
    record a reader can open to see what the claim was earned against.

    Every row also carries ``measurement_document``, always ``"operating_point"``: a detection count
    never rests on a scalar head or a physical scale, so this is a constant column, unlike
    ``export_aggregated_csv``'s own per-row statement.

    Meaning door: a count nobody defined is not a measurement, so this refuses before it composes
    the gate's flags unless ``trait``'s ``per_image_count`` operationalization is recorded and
    breeder-confirmed. The counts are counts of that record's own ``measured_subject``, which is
    checked against the ``id_map`` of every bucket that recorded one; a delivery whose buckets
    recorded none, and one called with no ``pred_dirs`` at all, carries the subject unchecked
    because nothing in it names what the labels decoded to. The record names no delivered
    phenotype, because this CSV's columns name none.

    Args:
        image_results: List of dicts with 'image', 'count', 'boxes', etc.
        output_path: Path for the output CSV file.
        provenance: Optional producing-model / operating-point stamp added as trailing columns.
        trait: The registered trait whose confirmed per-image-count operationalization this
            delivery rests on. Required: a count CSV under no trait states nothing about what was
            counted.
        measurement_validated: The count operating point's reconciled validity reference. Floored
            against each bucket's on-disk sidecar when ``pred_dirs`` is given; floored to
            unvalidated otherwise, since nothing on disk backs it.
        pred_dirs: Prediction buckets to reconcile the count operating point's (and, if tiled, the
            tile-geometry) validity from.
        acknowledge_unvalidated: Write an unvalidated count as a flagged provisional CSV.

    Returns:
        Path to the written CSV file.
    """
    from tcip_mcp.operationalization import (
        PER_IMAGE_COUNT,
        check_operationalization,
        resolve_trait_and_record,
    )
    from tcip_mcp.pipelines.postprocessing.phenology import bucket_id_map
    from tcip_mcp.pipelines.resolution import (
        VALIDATED_FALSE,
        binding_notes_text,
        check_delivery_gate,
        delivered_provenance,
        record_delivery_binding_event,
        reconcile_operating_point_validity,
        reconcile_tile_size_validity,
    )

    # Only buckets that recorded a map: one that recorded none says nothing about what was counted.
    recorded_maps = {d: bucket_id_map(Path(d)) for d in (pred_dirs or [])}
    id_maps = {d: m for d, m in recorded_maps.items() if m is not None} or None
    spec, record, _specs_dir = resolve_trait_and_record(trait, PER_IMAGE_COUNT)
    # This door never delivers a crossing kind, so it has no registry to check a positive class against.
    stated = check_operationalization(spec, record, PER_IMAGE_COUNT, id_maps=id_maps, registry=None)
    if not stated.ok:
        raise ValueError(stated.message)

    # With no pred_dirs nothing on disk backs the count's validity, so the dimension floors to
    # unvalidated rather than trusting the caller's bare string (mirrors export_aggregated_csv).
    flags: dict[str, str | None] = {"measurement": VALIDATED_FALSE}
    measurement_recon: dict = {"bindings": {}}
    if pred_dirs:
        # Reconciled from the buckets' own sidecars, floored against the caller assertion, never
        # trusted from the string alone (mirrors export_aggregated_csv's count-trait gating).
        measurement_recon = reconcile_operating_point_validity(
            pred_dirs, trait=trait, asserted=measurement_validated)
        flags["measurement"] = measurement_recon["validated"]
        tile_recon = reconcile_tile_size_validity(pred_dirs)
        if tile_recon["operative"]:
            flags["tile_size"] = tile_recon["validated"]

    gate = check_delivery_gate(flags, acknowledge_unvalidated=acknowledge_unvalidated)
    if not gate.ok:
        notes = binding_notes_text(measurement_recon.get("binding_notes", {}))
        raise ValueError(f"{gate.reason} {notes}".rstrip())

    # A confirmation withdrawn or a field moved since the first check refuses here, before anything.
    spec_now, record_now, _ = resolve_trait_and_record(trait, PER_IMAGE_COUNT)
    still_stated = check_operationalization(
        spec_now, record_now, PER_IMAGE_COUNT, id_maps=id_maps, registry=None, basis=stated.basis)
    if not still_stated.ok:
        raise ValueError(still_stated.message)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    stamp = delivered_provenance(provenance, measurement_recon["bindings"],
                                 columns=_PROVENANCE_COLUMNS)
    stamp["measurement_validated"] = gate.column_stamp("measurement")
    fieldnames = (["image", "detection_count", "avg_confidence", "measurement_document"]
                 + _PROVENANCE_COLUMNS)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in image_results:
            detection_count, scores = positive_detections(r)
            avg_conf = sum(scores) / len(scores) if scores else 0.0
            writer.writerow({
                "image": Path(r.get("image", "")).name,
                "detection_count": detection_count,
                "avg_confidence": round(avg_conf, 4),
                "measurement_document": _MEASUREMENT_DOCUMENT,
                **stamp,
            })

    record_delivery_binding_event("export_detection_csv", output_path, pred_dirs,
                                  measurement_recon["bindings"],
                                  measurement_documents=[_MEASUREMENT_DOCUMENT],
                                  scale_document=None,
                                  trait=trait, delivery_kind=PER_IMAGE_COUNT)
    return output_path

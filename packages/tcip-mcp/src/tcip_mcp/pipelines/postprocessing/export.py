"""CSV export for per-plant phenotyping results."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tcip_annotation.state import BBox, Polygon
    from tcip_mcp.pipelines.resolution import Acknowledgement

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


def unmapped_label_ids(results: list[dict], id_map: dict[str, int] | None) -> list[int]:
    """Every 0-indexed label id across ``results`` that ``id_map`` cannot decode, sorted; empty
    when every one decodes, or when ``id_map`` is ``None`` (the detector-run case, where the raw
    index is itself a legitimate, honest name).

    The preflight a door that publishes several documents from one run makes ahead of its first
    write: :func:`write_predictions_json` refuses one such id per document, partway through a
    multi-image bucket, so a caller checks the whole run first and refuses before anything lands
    rather than after part of it has. Only meaningful for a classified run (``attribute`` set); a
    detector run's raw-index fallback never refuses on this, whatever it returns.
    """
    if id_map is None:
        return []
    known = set(id_map.values())
    unmapped: set[int] = set()
    for r in results:
        for label in r.get("labels", []):
            cid = max(int(label) - 1, 0)
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
    *recorded* ``operating_point.json`` name→id map), so a prediction on disk carries the same
    names its labels do, and decode is never a fresh ``assign_class_ids``.

    ``subject`` and ``attribute`` are the run's own scope (:func:`~tcip_mcp.tools.inference_tools.
    run_scope`). With ``attribute`` set, every decoded name lands in ``attributes[attribute]`` and
    ``subject`` carries the object class itself, so a classified prediction carries the shape
    ground truth carries (``state.py``): a value under an attribute of a named object class, never
    the value alone in ``subject``. With ``attribute=None`` the output is byte-identical to a
    detector run's: ``subject`` carries the decoded name, ``attributes`` stays empty. An
    ``attribute`` with no ``subject`` refuses (``ValueError``) before the first document is
    written, since the record it would write carries a value under no object class. Absent a
    recorded map, the raw 0-indexed id is used as the name (a degraded but honest fallback, never
    a re-derivation) for a detector run; a classified run instead refuses (``ValueError``, naming
    the id and the map's own ids) the first label it cannot decode, since a value no vocabulary
    declares landing in ``attributes[attribute]`` would be a fabricated state. ``keep_empty=True``
    so a processed image with zero detections still yields an ``{"annotations": []}`` file.
    ``created_by`` stamps the producing model on every prediction so the origin travels into GT
    when a human accepts it.

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
    own ``operating_point.json`` instead; see :func:`mask_binarize_provenance`, which the two entry
    points that write predictions to disk (``run_inference``, the web inference route) call once and
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
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.class_registry import decode_class_ids

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
        geometry: BBox | Polygon = BBox(x1, y1, x2, y2)
        if masks is not None and i < len(masks):
            geometry = _mask_geometry_for_export(masks[i], (x1, y1, x2, y2), name, image_size=(w, h))
        if not json_io.geometry_extent_ok(geometry):
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
                       "validation_record", "acknowledged_by", "acknowledgement_reason"]

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
    acknowledgement: Acknowledgement | None = None,
    project_root: str | Path | None = None,
) -> tuple[str, dict, dict, bool]:
    """Export per-image detection counts to CSV.

    The count is the phenotype for count traits, so this is a delivery door: it refuses a *bare*
    write (an unvalidated count with no acknowledgement) via the shared ``check_delivery_gate`` and
    stamps the reconciled validity into every row. Pass ``pred_dirs`` (the prediction buckets the
    counts came from) so the count operating point's validity is read from each
    ``operating_point.json`` sidecar and floored against ``operating_point_validated`` (never
    trusted from the string alone). A bucket produced by a tiled run gates on its ``tile_size`` too, the same
    operating point's other gating dimension: the tile edge scales the per-image counts this CSV
    reports, so a run with no persisted training geometry, no recoverable native-frame edge, and no
    explicit caller override refuses here. Untiled buckets are never gated on it. This CSV carries
    no dimensional
    value (its rows are ``detection_count``/``avg_confidence``, never an area/length/diameter), so
    the physical-scale dimension (see ``export_aggregated_csv``'s gate on
    ``resolve_scale.json``/``reconcile_scale_validity``) is never operative here: there is nothing in
    this CSV's own shape for a physical scale to have produced, so gating on it would manufacture a
    refusal over a dimension that can't apply to a count. Without ``pred_dirs`` there is no on-disk
    source for the count's validity, so the operating_point dimension floors to unvalidated
    regardless of the caller's string, mirroring ``export_aggregated_csv``. ``acknowledgement`` is
    the breeder's own act of shipping this delivery unvalidated (the web results route's per-image
    count export is the one surface that builds one), or ``None`` for every MCP-tool call, which
    still refuses a bare unvalidated write here exactly as before; the promotion route (through the
    review validation door, then a re-delivery) still applies with no acknowledgement at hand. The
    ``provenance``
    stamp (producing checkpoint sha, experiment
    id, operating-point conf, timestamp) travels alongside; the number is only as trustworthy as the
    operating point + model behind it. Those cells are built by ``delivered_tail`` from the
    verification the gate already ran, so a producer this delivery cannot corroborate is reported
    unknown rather than repeated from the stamp that asserted it, ``produced_at`` is the write's own
    timestamp rather than one the caller asserts, and ``validation_record`` names the record a
    reader can open to see what the claim was earned against. ``acknowledged_by``/
    ``acknowledgement_reason`` carry the gate's own effective acknowledgement (blank together on a
    fully validated delivery, even one posted with one, since the gate discards an acknowledgement
    that cleared nothing), never the caller's ``acknowledgement`` verbatim, so the CSV tail and the
    recorded delivery event can never disagree about who acknowledged what.

    Every row also carries ``measurement_document``, always ``"operating_point"``: a detection count
    never rests on a scalar head or a physical scale, so this is a constant column, unlike
    ``export_aggregated_csv``'s own per-row statement.

    Meaning door: a count nobody defined is not a measurement, so this refuses before it composes
    the gate's flags unless ``trait``'s ``per_image_count`` operationalization is recorded and
    breeder-confirmed. The counts are counts of that record's own ``measured_subject``, which is
    checked against the object classes every bucket's own scope says its detections are of: a
    classified bucket's own subject, an unscoped bucket's recorded ``id_map`` keys. A delivery
    whose buckets contribute neither, and one called with no ``pred_dirs`` at all, carries the
    subject unchecked because nothing in it names what the labels decoded to. The record names no
    delivered phenotype, because this CSV's columns name none.

    Args:
        image_results: List of dicts with 'image', 'count', 'boxes', etc.
        output_path: Path for the output CSV file.
        provenance: Optional producing-model / operating-point stamp added as trailing columns.
        trait: The registered trait whose confirmed per-image-count operationalization this
            delivery rests on. Required: a count CSV under no trait states nothing about what was
            counted.
        operating_point_validated: The count operating point's reconciled validity reference.
            Floored against each bucket's on-disk sidecar when ``pred_dirs`` is given; floored to
            unvalidated otherwise, since nothing on disk backs it.
        pred_dirs: Prediction buckets to reconcile the count operating point's (and, if tiled, the
            tile-geometry) validity from.
        acknowledgement: The breeder's own act of shipping this delivery unvalidated, or ``None``
            for an ordinary validated export or an MCP-tool call, which never builds one.
        project_root: The project this delivery's meaning-record reads and delivery event belong
            to. ``None`` (every MCP-tool call) resolves against this process's pinned platform
            root, correct since an MCP process serves exactly one project; a web route already
            holding its own guarded, resolved root passes it explicitly.

    Returns:
        ``(path, tail, summary, event_recorded)``: the path to the written CSV, the
        ``_PROVENANCE_COLUMNS`` tail ``delivered_tail`` composed and wrote into every row (so a
        caller that needs one of those cells back, a response echoing the CSV's own
        ``operating_point_validated``, say, reads the value actually written rather than
        re-deriving or re-asserting it a second time), the gate's own evaluation summary
        (``stamp``, ``unvalidated``, ``tile_size_operative``, ``tile_size_validated``,
        ``binding_notes``) so a door composes its response fields from this call's single
        authoritative gate rather than re-reconciling the same buckets itself, and whether the
        best-effort delivery-event write landed (``record_delivery_binding_event``'s own return).

    Raises:
        DeliveryRefused: the gate refused (an unvalidated dimension with no acknowledgement that
            clears it); carries the ``DeliveryGateResult`` and both reconcilers' binding notes.
        OperationalizationRefused (``tcip_mcp.operationalization``): the ``trait``'s
            ``per_image_count`` operationalization is unrecorded, not breeder-confirmed, or was
            withdrawn since the first check; carries the failed check and no counts, so a caller
            must not read a delivered count off this raise.
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
    operating_point_recon: dict = {"bindings": {}}
    tile_recon: dict = {"operative": False, "validated": None, "binding_notes": {}}
    if pred_dirs:
        # Reconciled from the buckets' own sidecars, floored against the caller assertion, never
        # trusted from the string alone (mirrors export_aggregated_csv's count-trait gating).
        operating_point_recon = reconcile_operating_point_validity(
            pred_dirs, trait=trait, asserted=operating_point_validated)
        flags["operating_point"] = operating_point_recon["validated"]
        tile_recon = reconcile_tile_size_validity(pred_dirs)
        if tile_recon["operative"]:
            flags["tile_size"] = tile_recon["validated"]

    gate = check_delivery_gate(flags, acknowledgement=acknowledgement)
    if not gate.ok:
        notes = binding_notes_text(
            {**operating_point_recon.get("binding_notes", {}), **tile_recon.get("binding_notes", {})})
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

    stamp = delivered_tail(provenance, operating_point_recon["bindings"], gate,
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
        "export_detection_csv", output_path, pred_dirs, operating_point_recon["bindings"],
        measurement_documents=[_MEASUREMENT_DOCUMENT], scale_document=None,
        acknowledgement=gate.effective_acknowledgement(), trait=trait,
        delivery_kind=PER_IMAGE_COUNT, project_root=project_root)
    summary = {
        "stamp": gate.stamp,
        "unvalidated": gate.unvalidated,
        "tile_size_operative": tile_recon["operative"],
        "tile_size_validated": tile_recon.get("validated"),
        "binding_notes": binding_notes_text(
            {**operating_point_recon.get("binding_notes", {}), **tile_recon.get("binding_notes", {})}),
    }
    return output_path, stamp, summary, event_recorded

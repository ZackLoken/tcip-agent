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


def write_predictions_json(
    json_path: str | Path, result: dict, created_by: str | None = None, *,
    id_map: dict[str, int] | None = None,
) -> None:
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
    """
    from datetime import datetime, timezone

    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.class_registry import decode_class_ids

    w = result.get("width") or 0
    h = result.get("height") or 0
    created_at = datetime.now(timezone.utc).isoformat() if created_by else None
    id_to_name = decode_class_ids(id_map) if id_map else {}
    masks = result.get("masks")
    preds: list[Annotation] = []
    for i, (box, score, label) in enumerate(zip(
        result.get("boxes", []), result.get("scores", []), result.get("labels", [])
    )):
        x1, y1, x2, y2 = box
        cid = max(int(label) - 1, 0)  # undo the 1-indexed torchvision label -> 0-indexed run id
        name = id_to_name.get(cid, str(cid))  # decode via the recorded map, never a fresh derivation
        geometry: BBox | Polygon = BBox(x1, y1, x2, y2)
        if masks is not None and i < len(masks):
            geometry = _mask_geometry_for_export(masks[i], (x1, y1, x2, y2), name, image_size=(w, h))
        preds.append(Annotation(subject=name, geometry=geometry, score=float(score),
                                created_by=created_by, created_at=created_at))
    json_io.write_annotations(str(json_path), preds, int(w), int(h), keep_empty=True)


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
                       "produced_at", "measurement_validated"]


def export_detection_csv(
    image_results: list[dict],
    output_path: str,
    provenance: dict | None = None,
    *,
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
    reports, so a run with no persisted training geometry and no explicit caller override refuses
    here. Untiled buckets are never gated on it. This CSV carries no dimensional
    value (its rows are ``detection_count``/``avg_confidence``, never an area/length/diameter), so
    the physical-scale dimension (see ``export_aggregated_csv``'s gate on
    ``resolve_scale.json``/``reconcile_scale_validity``) is never operative here: there is nothing in
    this CSV's own shape for a physical scale to have produced, so gating on it would manufacture a
    refusal over a dimension that can't apply to a count. Without ``pred_dirs`` (no buckets
    to reconcile from, e.g. a caller that already resolved the gate against a live run's own bundle),
    ``measurement_validated`` is taken as a bare caller-asserted reference with no on-disk
    reconciliation. Either way, ``acknowledge_unvalidated=True`` writes a clearly-flagged provisional
    CSV stamped ``validated=false``. The ``provenance`` stamp (producing checkpoint sha, experiment
    id, operating-point conf, timestamp) travels alongside; the number is only as trustworthy as the
    operating point + model behind it.

    Args:
        image_results: List of dicts with 'image', 'count', 'boxes', etc.
        output_path: Path for the output CSV file.
        provenance: Optional producing-model / operating-point stamp added as trailing columns.
        measurement_validated: The count operating point's reconciled validity reference. Floored
            against each bucket's on-disk sidecar when ``pred_dirs`` is given; taken as-is otherwise.
        pred_dirs: Prediction buckets to reconcile the count operating point's (and, if tiled, the
            tile-geometry) validity from.
        acknowledge_unvalidated: Write an unvalidated count as a flagged provisional CSV.

    Returns:
        Path to the written CSV file.
    """
    from tcip_mcp.pipelines.resolution import (
        check_delivery_gate,
        reconcile_operating_point_validity,
        reconcile_tile_size_validity,
    )

    flags: dict[str, str | None] = {"measurement": measurement_validated}
    if pred_dirs:
        # Reconciled from the buckets' own sidecars, floored against the caller assertion, never
        # trusted from the string alone (mirrors export_aggregated_csv's count-trait gating).
        flags["measurement"] = reconcile_operating_point_validity(
            pred_dirs, asserted=measurement_validated,
        )["validated"]
        tile_recon = reconcile_tile_size_validity(pred_dirs)
        if tile_recon["operative"]:
            flags["tile_size"] = tile_recon["validated"]

    gate = check_delivery_gate(flags, acknowledge_unvalidated=acknowledge_unvalidated)
    if not gate.ok:
        raise ValueError(gate.reason)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    stamp = {k: (provenance or {}).get(k) for k in _PROVENANCE_COLUMNS}
    stamp["measurement_validated"] = gate.column_stamp("measurement")
    fieldnames = ["image", "detection_count", "avg_confidence"] + _PROVENANCE_COLUMNS

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in image_results:
            scores = r.get("scores", [])
            avg_conf = sum(scores) / len(scores) if scores else 0.0
            writer.writerow({
                "image": Path(r.get("image", "")).name,
                "detection_count": r.get("count", len(r.get("boxes", []))),
                "avg_confidence": round(avg_conf, 4),
                **stamp,
            })

    return output_path

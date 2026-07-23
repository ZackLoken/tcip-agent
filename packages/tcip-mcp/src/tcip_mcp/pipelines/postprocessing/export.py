"""CSV export for per-plant phenotyping results."""

from __future__ import annotations

import csv
from pathlib import Path


def write_predictions_json(
    json_path: str | Path, result: dict, created_by: str | None = None, *,
    id_map: dict[str, int] | None = None,
) -> None:
    """Write a ``GenericPredictor`` detection result as a name-based per-image prediction file.

    ``result`` carries pixel-xyxy ``boxes``, 1-indexed ``labels`` (background=0), ``scores``, and
    image ``width``/``height``. Each detection's numeric label is decoded to a **name** (its
    ``subject``) via ``id_map`` — the run's *recorded* ``operating_point.json`` name→id map — so a
    prediction on disk carries the same names its labels do, and decode is never a fresh
    ``assign_class_ids``. Absent a recorded map, the raw 0-indexed id is used as the name (a degraded
    but honest fallback, never a re-derivation). ``keep_empty=True`` so a processed image with zero
    detections still yields an ``{"annotations": []}`` file. ``created_by`` stamps the producing model
    on every prediction so the origin travels into GT when a human accepts it.
    """
    from datetime import datetime, timezone

    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.class_registry import decode_class_ids

    w = result.get("width") or 0
    h = result.get("height") or 0
    created_at = datetime.now(timezone.utc).isoformat() if created_by else None
    id_to_name = decode_class_ids(id_map) if id_map else {}
    preds: list[Annotation] = []
    for box, score, label in zip(
        result.get("boxes", []), result.get("scores", []), result.get("labels", [])
    ):
        x1, y1, x2, y2 = box
        cid = max(int(label) - 1, 0)  # undo the 1-indexed torchvision label -> 0-indexed run id
        name = id_to_name.get(cid, str(cid))  # decode via the recorded map, never a fresh derivation
        preds.append(Annotation(subject=name, geometry=BBox(x1, y1, x2, y2), score=float(score),
                                created_by=created_by, created_at=created_at))
    json_io.write_annotations(str(json_path), preds, int(w), int(h), keep_empty=True)


_PROVENANCE_COLUMNS = ["producer_model_sha256", "experiment_id", "operating_point_conf",
                       "produced_at", "measurement_validated"]


def export_detection_csv(
    image_results: list[dict],
    output_path: str,
    provenance: dict | None = None,
    *,
    measurement_validated: str | None = None,
    acknowledge_unvalidated: bool = False,
) -> str:
    """Export per-image detection counts to CSV.

    The count is the phenotype for count traits, so this is a delivery door: it refuses a *bare*
    write (an unvalidated count with no acknowledgement) via the shared ``check_delivery_gate`` and
    stamps the reconciled validity into every row. Pass ``measurement_validated`` = the count
    operating point's reconciled state (a shippable reference), or ``acknowledge_unvalidated=True``
    to write a clearly-flagged provisional CSV stamped ``validated=false``. The ``provenance`` stamp
    (producing checkpoint sha, experiment id, operating-point conf, timestamp) travels alongside — the
    number is only as trustworthy as the operating point + model behind it.

    Args:
        image_results: List of dicts with 'image', 'count', 'boxes', etc.
        output_path: Path for the output CSV file.
        provenance: Optional producing-model / operating-point stamp added as trailing columns.
        measurement_validated: The count operating point's reconciled validity reference.
        acknowledge_unvalidated: Write an unvalidated count as a flagged provisional CSV.

    Returns:
        Path to the written CSV file.
    """
    from tcip_mcp.pipelines.resolution import check_delivery_gate

    gate = check_delivery_gate({"measurement": measurement_validated},
                               acknowledge_unvalidated=acknowledge_unvalidated)
    if not gate.ok:
        raise ValueError(gate.reason)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    stamp = {k: (provenance or {}).get(k) for k in _PROVENANCE_COLUMNS}
    stamp["measurement_validated"] = gate.stamp["measurement"]
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

"""CSV export for per-plant phenotyping results."""

from __future__ import annotations

import csv
from pathlib import Path


def result_to_yolo_lines(result: dict) -> list[str]:
    """Convert a ``GenericPredictor`` detection result to YOLO prediction lines.

    ``result`` carries pixel-xyxy ``boxes``, 1-indexed ``labels`` (background=0),
    ``scores``, and image ``width``/``height``. Returns ``"cls conf cx cy w h"`` lines
    (normalized) — the single format that ``parse_detect_predictions`` / the Review tab
    consume. Shared by the MCP ``export_predictions_yolo`` tool and the web inference job
    so there is exactly one detection→YOLO conversion.
    """
    w = result.get("width") or 1
    h = result.get("height") or 1
    lines: list[str] = []
    for box, score, label in zip(
        result.get("boxes", []), result.get("scores", []), result.get("labels", [])
    ):
        x1, y1, x2, y2 = box
        cls = max(int(label) - 1, 0)
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{cls} {float(score):.4f} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def write_predictions_json(json_path: str | Path, result: dict) -> None:
    """Write a ``GenericPredictor`` detection result as a per-image COCO/JSON prediction file.

    JSON counterpart of :func:`result_to_yolo_lines`: each detection becomes a pixel-xyxy
    ``PredBBox`` (``class_id = max(label-1, 0)`` to undo the 1-indexed torchvision label,
    ``confidence`` from the score) and is written via ``json_io.write_detect``. ``keep_empty=True``
    so a processed image with zero detections still yields a ``{"objects": []}`` confirmed-negative
    file — preserving the behavior of the YOLO text writer that emitted an empty ``<stem>.txt``.
    """
    from tcip_annotation import json_io
    from tcip_annotation.state import PredBBox

    w = result.get("width") or 0
    h = result.get("height") or 0
    preds: list[PredBBox] = []
    for box, score, label in zip(
        result.get("boxes", []), result.get("scores", []), result.get("labels", [])
    ):
        x1, y1, x2, y2 = box
        preds.append(PredBBox(x1, y1, x2, y2, max(int(label) - 1, 0), confidence=float(score)))
    json_io.write_detect(str(json_path), preds, int(w), int(h), keep_empty=True)


def export_detection_csv(
    image_results: list[dict],
    output_path: str,
) -> str:
    """Export per-image detection counts to CSV.

    Args:
        image_results: List of dicts with 'image', 'count', 'boxes', etc.
        output_path: Path for the output CSV file.

    Returns:
        Path to the written CSV file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["image", "detection_count", "avg_confidence"]

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
            })

    return output_path

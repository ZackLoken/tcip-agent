"""YOLO label file I/O — parse and write detect / segment label files.

All functions are pure (no GUI dependencies).  YOLO format uses normalised
coordinates (0-1).  Internal representation uses pixel coordinates.
"""

from __future__ import annotations

import os
import tempfile

from tcip_annotation.state import BBox, Polygon, PredBBox, PredPolygon


# ── Parsing ─────────────────────────────────────────────────────────────────


def parse_detect_labels(
    path: str, img_w: int, img_h: int
) -> tuple[list[BBox], set[int]]:
    """Parse YOLO detect label file into pixel-coordinate boxes.

    Format per line: ``class_id cx cy w h`` (normalised 0-1).
    """
    boxes: list[BBox] = []
    class_ids: set[int] = set()
    if not os.path.exists(path):
        return boxes, class_ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cid = int(parts[0])
                cx, cy, w, h = (float(v) for v in parts[1:])
            except ValueError:
                continue
            pw, ph = w * img_w, h * img_h
            pcx, pcy = cx * img_w, cy * img_h
            boxes.append(BBox(pcx - pw / 2, pcy - ph / 2, pcx + pw / 2, pcy + ph / 2, cid))
            class_ids.add(cid)
    return boxes, class_ids


def parse_segment_labels(
    path: str, img_w: int, img_h: int
) -> tuple[list[Polygon], set[int]]:
    """Parse YOLO segment label file into pixel-coordinate polygons.

    Format per line: ``class_id x1 y1 x2 y2 ...`` (normalised 0-1).
    """
    polygons: list[Polygon] = []
    class_ids: set[int] = set()
    if not os.path.exists(path):
        return polygons, class_ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7 or len(parts) % 2 != 1:
                continue
            try:
                cid = int(parts[0])
                vals = [float(v) for v in parts[1:]]
            except ValueError:
                continue
            points = [(vals[i] * img_w, vals[i + 1] * img_h) for i in range(0, len(vals), 2)]
            polygons.append(Polygon(points, cid))
            class_ids.add(cid)
    return polygons, class_ids


def parse_detect_predictions(
    path: str, img_w: int, img_h: int
) -> tuple[list[PredBBox], set[int]]:
    """Parse YOLO detect prediction file (with confidence).

    Format per line: ``class_id confidence cx cy w h`` (normalised 0-1).
    """
    pred_boxes: list[PredBBox] = []
    class_ids: set[int] = set()
    if not os.path.exists(path):
        return pred_boxes, class_ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            try:
                cid = int(parts[0])
                conf = float(parts[1])
                cx, cy, w, h = (float(v) for v in parts[2:])
            except ValueError:
                continue
            pw, ph = w * img_w, h * img_h
            pcx, pcy = cx * img_w, cy * img_h
            pred_boxes.append(
                PredBBox(pcx - pw / 2, pcy - ph / 2, pcx + pw / 2, pcy + ph / 2, cid, conf)
            )
            class_ids.add(cid)
    return pred_boxes, class_ids


def parse_segment_predictions(
    path: str, img_w: int, img_h: int
) -> tuple[list[PredPolygon], set[int]]:
    """Parse YOLO segment prediction file (with confidence).

    Format per line: ``class_id confidence x1 y1 x2 y2 ...`` (normalised).
    """
    pred_polygons: list[PredPolygon] = []
    class_ids: set[int] = set()
    if not os.path.exists(path):
        return pred_polygons, class_ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8 or (len(parts) - 2) % 2 != 0:
                continue
            try:
                cid = int(parts[0])
                conf = float(parts[1])
                vals = [float(v) for v in parts[2:]]
            except ValueError:
                continue
            points = [(vals[i] * img_w, vals[i + 1] * img_h) for i in range(0, len(vals), 2)]
            pred_polygons.append(PredPolygon(points, cid, conf))
            class_ids.add(cid)
    return pred_polygons, class_ids


# ── Writing ─────────────────────────────────────────────────────────────────


def _atomic_write_lines(path: str, lines: list[str]) -> None:
    """Atomically write ``lines`` (already newline-terminated) to ``path``."""
    dir_name = os.path.dirname(path) or "."
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.writelines(lines)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def write_detect_labels(
    path: str, boxes: list[BBox], img_w: int, img_h: int, *, keep_empty: bool = False
) -> None:
    """Write detect boxes to a YOLO label file (atomic write).

    When ``boxes`` is empty: by default the file is removed. Pass ``keep_empty=True``
    to instead write a 0-byte file — used by the interactive annotator so that clearing
    all boxes records a *confirmed negative* (empty label file = valid negative) rather
    than deleting the record (see CLAUDE.md invariant).
    """
    if boxes:
        lines = []
        for b in boxes:
            cx = ((b.x1 + b.x2) / 2) / img_w
            cy = ((b.y1 + b.y2) / 2) / img_h
            w = (b.x2 - b.x1) / img_w
            h = (b.y2 - b.y1) / img_h
            lines.append(f"{b.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        _atomic_write_lines(path, lines)
    elif keep_empty:
        _atomic_write_lines(path, [])
    elif os.path.exists(path):
        os.remove(path)


def write_segment_labels(
    path: str, polygons: list[Polygon], img_w: int, img_h: int, *, keep_empty: bool = False
) -> None:
    """Write segment polygons to a YOLO label file (atomic write).

    ``keep_empty=True`` writes a 0-byte file for an empty polygon list (confirmed
    negative) instead of removing it; see :func:`write_detect_labels`.
    """
    if polygons:
        lines = []
        for poly in polygons:
            coords = " ".join(
                f"{x / img_w:.6f} {y / img_h:.6f}" for x, y in poly.points
            )
            lines.append(f"{poly.class_id} {coords}\n")
        _atomic_write_lines(path, lines)
    elif keep_empty:
        _atomic_write_lines(path, [])
    elif os.path.exists(path):
        os.remove(path)

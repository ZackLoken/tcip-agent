"""Per-image COCO-shaped JSON — the canonical on-disk label format (GT + predictions).

One JSON file per (image, task): ``<stem>.json`` under ``detect/`` or ``segment/``. Each object
carries geometry, an optional ``score`` (predictions), and provenance (``created_by/at``,
``accepted_by/at``) **natively** — so a prediction's origin travels with it into ground truth on
accept, with no sidecar. Field names are COCO-conventional so the per-image files concatenate into
a valid dataset COCO for training/export (see :func:`to_coco_dataset`).

Schema::

    { "image": "<stem>", "width": W, "height": H,
      "objects": [
        { "category_id": int,
          "bbox": [x, y, w, h],            # COCO xywh, pixel  (detect)
          "segmentation": [[x1,y1, ...]],  # pixel polygon     (segment)
          "score": 0.91,                   # predictions only
          "created_by": "sam", "created_at": "...",
          "accepted_by": "user:zack", "accepted_at": "..." } ] }

Negative invariant (unchanged from the YOLO writers this replaces): a present file with
``"objects": []`` is a **confirmed negative**; a **missing** file is unannotated. ``keep_empty=True``
writes the empty-objects file; without it an empty write removes the file.

Readers never raise on malformed input — a bad file / object is skipped and the reader returns what it
could parse. Writers and readers are symmetric (a shape a writer emits is one a reader accepts), and a
degenerate polygon (<3 points) is skipped on write so it can never masquerade as a confirmed negative.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

from tcip_annotation.state import BBox, Polygon, PredBBox, PredPolygon

SCHEMA_VERSION = 1
_PROV_KEYS = ("created_by", "created_at", "accepted_by", "accepted_at")


# ── low-level file I/O (tcip-annotation must not depend on tcip-mcp) ──────────


def _atomic_write_json(path: str, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            # allow_nan=False: a non-finite value here is a bug we sanitize before this call, but
            # fail loudly rather than emit non-standard `NaN` that strict parsers (JS, jq) reject.
            json.dump(payload, f, ensure_ascii=False, indent=1, allow_nan=False)
        os.replace(tmp, str(p))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _load(path: str) -> dict | None:
    """Parsed dict, or None if the file is missing/unreadable/not a dict."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _prov_kwargs(obj: dict) -> dict:
    """Provenance fields present on a JSON object, as BBox/Polygon kwargs."""
    return {k: obj[k] for k in _PROV_KEYS if obj.get(k) is not None}


def _emit_prov(rec: dict, shape) -> None:
    """Copy any set provenance fields from a shape onto a JSON object."""
    for k in _PROV_KEYS:
        v = getattr(shape, k, None)
        if v is not None:
            rec[k] = v


def _safe_score(x) -> float:
    """A JSON-safe confidence: finite, rounded; non-finite (NaN/inf) collapses to 0.0."""
    v = float(x)
    return round(v, 4) if math.isfinite(v) else 0.0


def _score_of(obj: dict) -> float:
    """Read a prediction score tolerantly: a null / absent / non-numeric score is 0.0."""
    s = obj.get("score")
    return float(s) if isinstance(s, (int, float)) and not isinstance(s, bool) else 0.0


def _coerce_bbox(bb) -> list[float] | None:
    """A 4-list of floats, or None (wrong shape / non-numeric) → the object is skipped."""
    if not isinstance(bb, list) or len(bb) != 4:
        return None
    try:
        return [float(v) for v in bb]
    except (TypeError, ValueError):
        return None


def _ring_to_points(seg) -> list[tuple[float, float]] | None:
    """First polygon ring of a COCO ``segmentation`` → pixel points, or None if unusable.

    Needs an even coord count and >=3 points (6 coords); anything else (RLE dict, too few) is skipped.
    """
    if not isinstance(seg, list) or not seg:
        return None
    ring = seg[0]
    if not isinstance(ring, list) or len(ring) < 6 or len(ring) % 2 != 0:
        return None
    try:
        return [(float(ring[i]), float(ring[i + 1])) for i in range(0, len(ring), 2)]
    except (TypeError, ValueError):
        return None


def _objects(data: dict | None) -> list:
    """The ``objects`` list, filtered to dict entries only (never raises on junk)."""
    if not data:
        return []
    objs = data.get("objects")
    if not isinstance(objs, list):
        return []
    return [o for o in objs if isinstance(o, dict)]


# ── readers ──────────────────────────────────────────────────────────────────
# img_w/img_h are accepted for signature-parity with the YOLO parsers; JSON is already pixel-space
# so they are unused. Missing/empty/malformed file → ([], set()) (absent == unannotated, never raises).


def read_detect(path, img_w: int = 0, img_h: int = 0) -> tuple[list[BBox], set[int]]:
    boxes: list[BBox] = []
    class_ids: set[int] = set()
    for o in _objects(_load(str(path))):
        bb = _coerce_bbox(o.get("bbox"))
        if bb is None:
            continue
        try:
            cid = int(o.get("category_id", 0))
        except (TypeError, ValueError):
            continue
        x, y, w, h = bb
        boxes.append(BBox(x, y, x + w, y + h, cid, **_prov_kwargs(o)))
        class_ids.add(cid)
    return boxes, class_ids


def read_detect_pred(path, img_w: int = 0, img_h: int = 0) -> tuple[list[PredBBox], set[int]]:
    boxes: list[PredBBox] = []
    class_ids: set[int] = set()
    for o in _objects(_load(str(path))):
        bb = _coerce_bbox(o.get("bbox"))
        if bb is None:
            continue
        try:
            cid = int(o.get("category_id", 0))
        except (TypeError, ValueError):
            continue
        x, y, w, h = bb
        boxes.append(PredBBox(x, y, x + w, y + h, cid, confidence=_score_of(o), **_prov_kwargs(o)))
        class_ids.add(cid)
    return boxes, class_ids


def read_segment(path, img_w: int = 0, img_h: int = 0) -> tuple[list[Polygon], set[int]]:
    polys: list[Polygon] = []
    class_ids: set[int] = set()
    for o in _objects(_load(str(path))):
        pts = _ring_to_points(o.get("segmentation"))
        if pts is None:
            continue
        try:
            cid = int(o.get("category_id", 0))
        except (TypeError, ValueError):
            continue
        polys.append(Polygon(pts, cid, **_prov_kwargs(o)))
        class_ids.add(cid)
    return polys, class_ids


def read_segment_pred(path, img_w: int = 0, img_h: int = 0) -> tuple[list[PredPolygon], set[int]]:
    polys: list[PredPolygon] = []
    class_ids: set[int] = set()
    for o in _objects(_load(str(path))):
        pts = _ring_to_points(o.get("segmentation"))
        if pts is None:
            continue
        try:
            cid = int(o.get("category_id", 0))
        except (TypeError, ValueError):
            continue
        polys.append(PredPolygon(pts, cid, confidence=_score_of(o), **_prov_kwargs(o)))
        class_ids.add(cid)
    return polys, class_ids


# ── writers ──────────────────────────────────────────────────────────────────
# A GT shape is a BBox/Polygon; a prediction is PredBBox/PredPolygon (writes `score`). Provenance is
# written when set. Empty list: keep_empty → write {"objects": []} (confirmed negative), else remove.


def _finish_write(path, img_w: int, img_h: int, objects: list[dict], keep_empty: bool) -> None:
    p = str(path)
    if not objects and not keep_empty:
        if os.path.exists(p):
            os.remove(p)
        return
    _atomic_write_json(
        p, {"image": Path(p).stem, "width": int(img_w), "height": int(img_h), "objects": objects}
    )


def write_detect(path, boxes, img_w: int, img_h: int, *, keep_empty: bool = False) -> None:
    objects: list[dict] = []
    for b in boxes:
        rec = {
            "category_id": int(b.class_id),
            "bbox": [round(b.x1, 2), round(b.y1, 2), round(b.x2 - b.x1, 2), round(b.y2 - b.y1, 2)],
        }
        if isinstance(b, PredBBox):
            rec["score"] = _safe_score(b.confidence)
        _emit_prov(rec, b)
        objects.append(rec)
    _finish_write(path, img_w, img_h, objects, keep_empty)


def write_segment(path, polygons, img_w: int, img_h: int, *, keep_empty: bool = False) -> None:
    objects: list[dict] = []
    for poly in polygons:
        if len(poly.points) < 3:
            continue  # a degenerate polygon is not a shape; skip so write<->read stays symmetric
        flat = [round(float(c), 2) for xy in poly.points for c in xy]
        rec = {"category_id": int(poly.class_id), "segmentation": [flat]}
        if isinstance(poly, PredPolygon):
            rec["score"] = _safe_score(poly.confidence)
        _emit_prov(rec, poly)
        objects.append(rec)
    _finish_write(path, img_w, img_h, objects, keep_empty)


# ── dataset-COCO assembly (for training / export) ────────────────────────────


def to_coco_dataset(
    entries: list[tuple[str, str]],
    *,
    categories: list[dict] | None = None,
) -> dict:
    """Concatenate per-image JSON label files into one COCO dataset dict.

    ``entries``: ``[(label_json_path, image_file_name), ...]``. A **missing** label file is
    *unannotated* and is **skipped entirely** (so an unlabeled image never becomes a training
    negative); a **present** file — including a confirmed negative (empty objects) — yields an
    ``images`` record. Every object becomes an ``annotations`` record with ``bbox`` (xywh) + ``area``
    + (for polygons) ``segmentation``, plus ``score``/provenance as COCO-extension keys. All geometry
    is float-coerced, so the result parses cleanly with ``format_io.parse_coco_detect`` /
    ``parse_coco_segment``.
    """
    coco: dict = {"images": [], "annotations": [], "categories": categories or []}
    ann_id = 1
    img_id = 0
    for label_path, file_name in entries:
        if not os.path.exists(str(label_path)):
            continue  # unannotated — not part of the training set
        data = _load(str(label_path)) or {}
        img_id += 1
        coco["images"].append({
            "id": img_id, "file_name": file_name,
            "width": int(data.get("width", 0) or 0), "height": int(data.get("height", 0) or 0),
        })
        for o in _objects(data):
            try:
                cid = int(o.get("category_id", 0))
            except (TypeError, ValueError):
                cid = 0
            rec: dict = {"id": ann_id, "image_id": img_id, "category_id": cid, "iscrowd": 0}
            bbox = _coerce_bbox(o.get("bbox"))
            if bbox is not None:
                rec["bbox"] = bbox
                rec["area"] = round(bbox[2] * bbox[3], 2)
            pts = _ring_to_points(o.get("segmentation"))
            if pts is not None:
                rec["segmentation"] = [[round(c, 2) for xy in pts for c in xy]]
                if "bbox" not in rec:  # derive the box from the polygon
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                    rec["bbox"] = [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)]
                    rec["area"] = round((x1 - x0) * (y1 - y0), 2)
            if "bbox" not in rec:  # no usable geometry at all — not an annotation
                continue
            sc = o.get("score")
            if isinstance(sc, (int, float)) and not isinstance(sc, bool):
                rec["score"] = _safe_score(sc)
            for k in _PROV_KEYS:
                if o.get(k) is not None:
                    rec[k] = o[k]
            coco["annotations"].append(rec)
            ann_id += 1
    return coco

"""Per-image JSON — the canonical on-disk label format (ground truth + predictions).

**One JSON file per image**, holding every subject's annotations by name. Each annotation carries its
``subject``, an optional geometry (``bbox`` xywh or ``segmentation`` polygon, or neither for an
image/plant-level label), its attribute values by name, an optional ``score`` (predictions), and
provenance (``created_by/at``, ``accepted_by/at``) — so a prediction's origin travels with it into
ground truth on accept, with no sidecar.

Schema::

    { "image": "<stem>", "width": W, "height": H,
      "annotations": [
        { "subject": "catkin",
          "bbox": [x, y, w, h],                 # COCO xywh, pixel      (optional)
          "segmentation": [[x1,y1, ...]],       # pixel polygon         (optional)
          "attributes": {"elongation": "elongated"},   # attr name -> value name
          "score": 0.91,                        # predictions only
          "created_by": "sam", "created_at": "...",
          "accepted_by": "user:zack", "accepted_at": "..." } ] }

Integer class ids never appear on disk — a name→id assignment is a per-training-run artifact
(:mod:`tcip_mcp.class_registry`). :func:`to_coco_dataset` takes that run's ``id_map`` (a plain dict,
so this package never imports tcip-mcp) and assigns COCO ``category_id`` from it.

Negative invariant: a **missing** file is unannotated, and a present file with ``"annotations": []`` is
*still* unannotated until a human marks that image Complete — recorded as ``"negative"`` in
``.tcip/state/image_status.json``, scoped to the subject. Only that confirmation makes it a training
negative; :func:`to_coco_dataset` skips an unconfirmed empty. An annotation with a subject but no
geometry (an image-level label) is a real annotation — it keeps the image out of the empty-negative
bucket and never collapses to nothing.

Readers never raise on malformed input — a bad file / annotation is skipped and the reader returns what
it could parse. Writers and readers are symmetric, and a degenerate polygon (<3 points) is skipped on
write so it can never masquerade as an empty record.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

from tcip_annotation.state import Annotation, BBox, Polygon, bbox_of

SCHEMA_VERSION = 2
ANNOTATIONS_KEY = "annotations"  # the one top-level list key; format_io.detect_format shares it
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


def _safe_score(x) -> float:
    """A JSON-safe confidence: finite, rounded; non-finite (NaN/inf) collapses to 0.0."""
    v = float(x)
    return round(v, 4) if math.isfinite(v) else 0.0


def _score_of(obj: dict) -> float | None:
    """A prediction score, or ``None`` when absent/non-numeric (ground truth has none)."""
    s = obj.get("score")
    return float(s) if isinstance(s, (int, float)) and not isinstance(s, bool) else None


def _coerce_bbox(bb) -> list[float] | None:
    """A 4-list of floats, or None (wrong shape / non-numeric)."""
    if not isinstance(bb, list) or len(bb) != 4:
        return None
    try:
        return [float(v) for v in bb]
    except (TypeError, ValueError):
        return None


def _ring_to_points(seg) -> list[tuple[float, float]] | None:
    """First polygon ring of a ``segmentation`` → pixel points, or None if unusable.

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


def _attributes_of(obj: dict) -> dict[str, str]:
    """The annotation's attribute values (name → value name); non-string entries dropped."""
    raw = obj.get("attributes")
    if not isinstance(raw, dict):
        return {}
    return {str(k): v for k, v in raw.items() if isinstance(v, str) and v}


def _prov_kwargs(obj: dict) -> dict:
    return {k: obj[k] for k in _PROV_KEYS if obj.get(k) is not None}


def _annotations_of(data: dict | None) -> list[Annotation]:
    """Parse a loaded per-image dict into :class:`Annotation` records (the one shared parse).

    Prefers a polygon geometry over a box when both are present (the polygon is the source of truth and
    its box is derivable). An annotation with a ``subject`` but no geometry is kept (an image-level
    label). An entry with no ``subject`` is skipped — a name-based label is undecodable without it.
    """
    if not data:
        return []
    raw = data.get(ANNOTATIONS_KEY)
    if not isinstance(raw, list):
        return []
    out: list[Annotation] = []
    for o in raw:
        if not isinstance(o, dict):
            continue
        subject = o.get("subject")
        if not isinstance(subject, str) or not subject:
            continue
        geometry: BBox | Polygon | None = None
        pts = _ring_to_points(o.get("segmentation"))
        if pts is not None:
            geometry = Polygon(pts)
        else:
            bb = _coerce_bbox(o.get("bbox"))
            if bb is not None:
                x, y, w, h = bb
                geometry = BBox(x, y, x + w, y + h)
        out.append(Annotation(
            subject=subject, geometry=geometry, attributes=_attributes_of(o),
            score=_score_of(o), **_prov_kwargs(o),
        ))
    return out


# ── reader ─────────────────────────────────────────────────────────────────
# A missing/empty/malformed file → [] (absent == unannotated, never raises). One reader for GT and
# predictions; ``score`` set means a prediction. Callers filter by subject and derive the geometry a
# task needs (``state.bbox_of`` reads a box from a polygon).


def read_annotations(path) -> list[Annotation]:
    return _annotations_of(_load(str(path)))


# ── writer ─────────────────────────────────────────────────────────────────


def _annotation_record(a: Annotation) -> dict | None:
    """One annotation → its JSON object, or None if a degenerate polygon should be skipped."""
    rec: dict = {"subject": a.subject}
    geom = a.geometry
    if isinstance(geom, Polygon):
        if len(geom.points) < 3:
            return None  # a degenerate polygon is not a shape; skip so write<->read stays symmetric
        rec["segmentation"] = [[round(float(c), 2) for xy in geom.points for c in xy]]
    elif isinstance(geom, BBox):
        rec["bbox"] = [round(geom.x1, 2), round(geom.y1, 2),
                       round(geom.x2 - geom.x1, 2), round(geom.y2 - geom.y1, 2)]
    if a.attributes:
        rec["attributes"] = dict(a.attributes)
    if a.score is not None:
        rec["score"] = _safe_score(a.score)
    for k in _PROV_KEYS:
        v = getattr(a, k, None)
        if v is not None:
            rec[k] = v
    return rec


def write_annotations(path, annotations, img_w: int, img_h: int, *, keep_empty: bool = False) -> None:
    """Write all of an image's annotations to its per-image JSON file.

    Empty list: ``keep_empty`` writes ``{"annotations": []}`` (unannotated until a human confirms it),
    else removes the file — writing an empty file never manufactures a negative.
    """
    records = [r for r in (_annotation_record(a) for a in annotations) if r is not None]
    p = str(path)
    if not records and not keep_empty:
        if os.path.exists(p):
            os.remove(p)
        return
    _atomic_write_json(
        p, {"image": Path(p).stem, "width": int(img_w), "height": int(img_h),
            ANNOTATIONS_KEY: records}
    )


# ── the one target-membership decision (shared by assembly and the loader) ───


def target_class_id(a: Annotation, subject: str, attribute: str | None,
                    id_map: dict[str, int]) -> int | None:
    """The 0-indexed class id ``a`` trains as for ``(subject, attribute)``, or ``None`` if it is not a
    detection/segmentation target for this scope (a different subject, or a geometry-less label).

    The single membership+id decision: :func:`to_coco_dataset` (training assembly) and the loader's
    per-image target reader both call this, so the assembled COCO and the calibration/eval GT can
    never disagree about which annotation is a target or which class it is. Raises when ``a`` *is* a
    target whose class ``id_map`` cannot decode — a real annotation read as nothing is a measurement
    bug, not something to drop silently.
    """
    if a.subject != subject or a.geometry is None:
        return None
    key = a.attributes.get(attribute) if attribute else subject
    if key is None or key not in id_map:
        raise ValueError(
            f"annotation of subject {subject!r} has class key {key!r} not in the run's id map "
            f"(known: {sorted(id_map)}) — the registry cannot decode its own labels")
    return id_map[key]


# ── dataset-COCO assembly (for training / export) ────────────────────────────


def to_coco_dataset(
    entries: list[tuple[str, str]],
    *,
    subject: str,
    id_map: dict[str, int],
    attribute: str | None = None,
    confirmed_negative_names: set[str] | None = None,
) -> dict:
    """Concatenate per-image JSON files into one COCO dataset dict, **scoped to one subject**.

    ``entries``: ``[(label_json_path, image_file_name), ...]``. ``id_map`` is the run's name→id
    assignment (``tcip_mcp.class_registry.assign_class_ids``): keyed by the ``attribute``'s value names
    when ``attribute`` is set, else by the ``subject`` itself. Each kept annotation's ``category_id`` is
    ``id_map[key]`` where ``key`` is its attribute value (``attribute`` set) or the subject.

    Only annotations of ``subject`` that carry a geometry become COCO annotations (a geometry-less
    annotation has no training destination this slice, but its presence still marks the image
    *annotated*, keeping it out of the negative bucket). A **missing** label file is skipped; an image
    with no annotations of ``subject`` is included only as a negative when its ``file_name`` is in
    ``confirmed_negative_names`` (a human marked it Complete-with-nothing). ``categories`` are emitted
    from ``id_map``. An annotation whose class key is not in ``id_map`` raises — a real annotation the
    registry cannot decode is a measurement bug, not something to drop silently.
    """
    categories = [{"id": cid, "name": name} for name, cid in sorted(id_map.items(), key=lambda kv: kv[1])]
    coco: dict = {"images": [], "annotations": [], "categories": categories}
    negatives = confirmed_negative_names or set()
    ann_id = 1
    img_id = 0
    for label_path, file_name in entries:
        if not os.path.exists(str(label_path)):
            continue  # unannotated — not part of the training set
        data = _load(str(label_path)) or {}
        scoped = [a for a in _annotations_of(data) if a.subject == subject]
        if not scoped and file_name not in negatives:
            continue  # no annotations of this subject and not a confirmed negative — skip
        img_id += 1
        coco["images"].append({
            "id": img_id, "file_name": file_name,
            "width": int(data.get("width", 0) or 0), "height": int(data.get("height", 0) or 0),
        })
        for a in scoped:
            cid = target_class_id(a, subject, attribute, id_map)
            if cid is None or a.geometry is None:
                continue  # image-level label: counts the image as annotated, no detection/seg target
            box = bbox_of(a.geometry)
            rec: dict = {
                "id": ann_id, "image_id": img_id, "category_id": cid, "iscrowd": 0,
                "bbox": [round(box.x1, 2), round(box.y1, 2),
                         round(box.x2 - box.x1, 2), round(box.y2 - box.y1, 2)],
                "area": round((box.x2 - box.x1) * (box.y2 - box.y1), 2),
            }
            if isinstance(a.geometry, Polygon):
                rec["segmentation"] = [[round(float(c), 2) for xy in a.geometry.points for c in xy]]
            if a.score is not None:
                rec["score"] = _safe_score(a.score)
            for k in _PROV_KEYS:
                v = getattr(a, k, None)
                if v is not None:
                    rec[k] = v
            coco["annotations"].append(rec)
            ann_id += 1
    return coco

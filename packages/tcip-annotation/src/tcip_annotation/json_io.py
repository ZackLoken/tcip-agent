"""Per-image JSON: the canonical on-disk label format (ground truth + predictions).

One JSON file per image, holding every subject's annotations by name. Each annotation carries its
``subject``, an optional geometry (``bbox`` xywh, ``segmentation`` polygon, ``point`` [x,y], or none
for an image/plant-level label), its attribute values by name, an optional ``score`` (predictions),
and provenance (``created_by/at``, ``accepted_by/at``), so a prediction's origin travels with it into
ground truth on accept, with no sidecar.

Schema::

    { "image": "<stem>", "width": W, "height": H,
      "annotations": [
        { "subject": "<subject>",
          "bbox": [x, y, w, h],                 # COCO xywh, pixel      (optional)
          "segmentation": [[x1,y1, ...], ...],  # pixel polygon, one or more rings (optional)
          "point": [x, y],                      # pixel point, a prompt or keypoint (optional)
          "attributes": {"<attribute>": "<value>"},   # attr name -> value name
          "score": 0.91,                        # predictions only
          "created_by": "sam", "created_at": "...",
          "accepted_by": "user:zack", "accepted_at": "..." } ] }

Integer class ids never appear on disk; a name→id assignment is a per-training-run artifact
(:mod:`tcip_mcp.class_registry`). :func:`to_coco_dataset` takes that run's ``id_map`` (a plain dict,
so this package never imports tcip-mcp) and assigns COCO ``category_id`` from it.

Negative invariant: a missing file is unannotated, and a present file with ``"annotations": []`` is
*still* unannotated until a human marks that image Complete, recorded as ``"negative"`` in
``.tcip/state/image_status.json``, scoped to the subject. Only that confirmation makes it a training
negative; :func:`to_coco_dataset` skips an unconfirmed empty. An annotation with a subject but no
geometry (an image-level label) is a real annotation: it keeps the image out of the empty-negative
bucket and never collapses to nothing.

Readers never raise on malformed input: a bad file / annotation is skipped and the reader returns what
it could parse. Writers and readers are symmetric, and a degenerate polygon (<3 points) is skipped on
write so it can never masquerade as an empty record.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

from tcip_annotation.state import Annotation, BBox, Point, Polygon, bbox_of

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


def _rings_from_segmentation(seg) -> list[list[tuple[float, float]]] | None:
    """Every valid polygon ring of a ``segmentation`` → pixel points per ring, or None if none usable.

    Each entry needs an even coord count and >=3 points (6 coords); a ring that fails this is dropped
    individually, not the whole annotation, the same treatment an RLE dict or other non-ring entry gets.
    Returns None only when no ring survives (an all-degenerate or empty ``segmentation``).
    """
    if not isinstance(seg, list) or not seg:
        return None
    rings: list[list[tuple[float, float]]] = []
    for ring in seg:
        if not isinstance(ring, list) or len(ring) < 6 or len(ring) % 2 != 0:
            continue
        try:
            rings.append([(float(ring[i]), float(ring[i + 1])) for i in range(0, len(ring), 2)])
        except (TypeError, ValueError):
            continue
    return rings or None


def _coerce_point(pt) -> tuple[float, float] | None:
    """A 2-list of floats as an (x, y) pair, or None (wrong shape / non-numeric)."""
    if not isinstance(pt, list) or len(pt) != 2:
        return None
    try:
        return (float(pt[0]), float(pt[1]))
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
    label). An entry with no ``subject`` is skipped, since a name-based label is undecodable without it.
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
        geometry: BBox | Polygon | Point | None = None
        rings = _rings_from_segmentation(o.get("segmentation"))
        if rings is not None:
            geometry = Polygon(rings)
        else:
            bb = _coerce_bbox(o.get("bbox"))
            if bb is not None:
                x, y, w, h = bb
                geometry = BBox(x, y, x + w, y + h)
            else:
                pt = _coerce_point(o.get("point"))
                if pt is not None:
                    geometry = Point(pt[0], pt[1])
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
        valid_rings = [r for r in geom.rings if len(r) >= 3]
        if not valid_rings:
            return None  # no ring is a real shape; skip so write<->read stays symmetric
        rec["segmentation"] = [[round(float(c), 2) for xy in ring for c in xy] for ring in valid_rings]
        # The polygon's box travels with it (COCO-style record). Derived from the rings here and
        # never authored or trusted as input: the polygon stays the sole source of truth, so the two
        # can't diverge; every reader re-derives via bbox_of rather than reading this stored value.
        b = bbox_of(Polygon(valid_rings))
        rec["bbox"] = [round(b.x1, 2), round(b.y1, 2), round(b.x2 - b.x1, 2), round(b.y2 - b.y1, 2)]
    elif isinstance(geom, BBox):
        rec["bbox"] = [round(geom.x1, 2), round(geom.y1, 2),
                       round(geom.x2 - geom.x1, 2), round(geom.y2 - geom.y1, 2)]
    elif isinstance(geom, Point):
        rec["point"] = [round(geom.x, 2), round(geom.y, 2)]
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
    else removes the file; writing an empty file never manufactures a negative.
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


UNLABELED = "unlabeled"  # a real target this scope covers, but not yet assessed for `attribute`


def target_class_id(a: Annotation, subject: str, attribute: str | None,
                    id_map: dict[str, int], *, allow_unlabeled: bool = False
                    ) -> int | None | str:
    """The 0-indexed class id ``a`` trains as for ``(subject, attribute)``.

    Returns ``None`` if ``a`` is not a detection/segmentation target for this scope at all (a
    different subject, a geometry-less label, or a :class:`~tcip_annotation.state.Point`, since a point
    has no box/area and is never a detection/segmentation training target). For a genuine
    target, two different failure shapes exist and must not be conflated: the instance was
    never assessed for ``attribute`` at all (``a.attributes.get(attribute) is None``, the annotator
    hasn't gotten to it yet, a soft/expected gap), versus the instance was assessed but with a value
    the registry cannot decode (a real decode bug, the registry and the labels disagree). The first
    case returns the distinguishable sentinel ``UNLABELED`` when ``allow_unlabeled=True`` (opt-in,
    default ``False`` preserves this function's original all-undecodable-cases-raise behavior for any
    caller that hasn't been updated to handle the three-way split); the second always raises,
    regardless of ``allow_unlabeled``, since a real annotation read as nothing is a measurement bug,
    never something to drop silently.

    The single membership+id decision: :func:`to_coco_dataset` (training assembly) and the loader's
    per-image target reader both call this, so the assembled COCO and the calibration/eval GT can
    never disagree about which annotation is a target or which class it is.
    """
    if a.subject != subject or a.geometry is None or isinstance(a.geometry, Point):
        return None
    key = a.attributes.get(attribute) if attribute else subject
    if key is None:
        if allow_unlabeled:
            return UNLABELED
        raise ValueError(
            f"annotation of subject {subject!r} has no value for attribute {attribute!r}: "
            "the registry cannot decode its own labels")
    if key not in id_map:
        raise ValueError(
            f"annotation of subject {subject!r} has class key {key!r} not in the run's id map "
            f"(known: {sorted(id_map)}): the registry cannot decode its own labels")
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
    """Concatenate per-image JSON files into one COCO dataset dict, scoped to one subject.

    ``entries``: ``[(label_json_path, image_file_name), ...]``. ``id_map`` is the run's name→id
    assignment (``tcip_mcp.class_registry.assign_class_ids``): keyed by the ``attribute``'s value names
    when ``attribute`` is set, else by the ``subject`` itself. Each kept annotation's ``category_id`` is
    ``id_map[key]`` where ``key`` is its attribute value (``attribute`` set) or the subject.

    Only annotations of ``subject`` that carry a geometry become COCO annotations (a geometry-less
    annotation has no training destination this slice, but its presence still marks the image
    *annotated*, keeping it out of the negative bucket). A missing label file is skipped; an image
    with no annotations of ``subject`` is included only as a negative when its ``file_name`` is in
    ``confirmed_negative_names`` (a human marked it Complete-with-nothing). ``categories`` are emitted
    from ``id_map``. An annotation whose class key is not in ``id_map`` raises, since a real annotation
    the registry cannot decode is a measurement bug, not something to drop silently.

    When ``attribute`` is set, an image with any instance never assessed for it is excluded
    wholesale, not trained on its labeled subset alone: silently narrowing to the labeled instances
    would leave the image's other real, unlabeled objects to train as background noise. The excluded
    ``file_name``s are reported in the returned dict's own ``excluded_incomplete_attribute`` list,
    alongside ``images``/``annotations``/``categories``, so a downstream partition
    (``trainable_stems``) can attribute the drop to its real reason rather than re-deriving one from
    the image's mere absence, which reads identically to an empty label file nobody confirmed.
    """
    categories = [{"id": cid, "name": name} for name, cid in sorted(id_map.items(), key=lambda kv: kv[1])]
    coco: dict = {"images": [], "annotations": [], "categories": categories,
                 "excluded_incomplete_attribute": []}
    negatives = confirmed_negative_names or set()
    ann_id = 1
    img_id = 0
    for label_path, file_name in entries:
        if not os.path.exists(str(label_path)):
            continue  # unannotated, not part of the training set
        data = _load(str(label_path)) or {}
        scoped = [a for a in _annotations_of(data) if a.subject == subject]
        if not scoped and file_name not in negatives:
            continue  # no annotations of this subject and not a confirmed negative, skip
        # allow_unlabeled=True: an instance never assessed for `attribute` is a soft, expected gap,
        # not a decode bug, and must not abort the whole assembly; computed once per instance and reused below.
        cids = [target_class_id(a, subject, attribute, id_map, allow_unlabeled=True) for a in scoped]
        if attribute is not None and UNLABELED in cids:
            coco["excluded_incomplete_attribute"].append(file_name)
            continue  # incomplete GT for this scope: the whole image, not just the gap, is excluded
        img_id += 1
        coco["images"].append({
            "id": img_id, "file_name": file_name,
            "width": int(data.get("width", 0) or 0), "height": int(data.get("height", 0) or 0),
        })
        for a, cid in zip(scoped, cids):
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
                rec["segmentation"] = [[round(float(c), 2) for xy in ring for c in xy]
                                       for ring in a.geometry.rings if len(ring) >= 3]
            if a.score is not None:
                rec["score"] = _safe_score(a.score)
            for k in _PROV_KEYS:
                v = getattr(a, k, None)
                if v is not None:
                    rec[k] = v
            coco["annotations"].append(rec)
            ann_id += 1
    return coco

"""Annotation I/O for the two on-disk formats: the canonical per-image JSON and a single-file COCO.

The internal representation is always the name-based :class:`~tcip_annotation.state.Annotation`
(pixel-coordinate geometry, ``subject`` + attribute values by name); format only matters at the file
I/O boundary.

  - json: one ``.json`` per image (the canonical ``json_io`` schema, an ``"annotations"`` key of
    name-based records)
  - coco: a single dataset-level ``.json`` (an ``"images"`` / ``"annotations"`` / ``"categories"``
    key); a genuine interop format, so its numeric ``category_id`` is decoded through the file's own
    ``categories`` back to names on read and encoded from a name→id map on write. A record's
    ``attributes`` (a classified prediction's decoded value) round-trips through the file's own
    ``attributes`` key; outside the freeze, this is a behavior change stated where it is made.

Usage::

    annotations = load_annotations(path)                     # -> list[Annotation]
    save_annotations(path, annotations, img_w, img_h)        # canonical per-image JSON
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

import tcip_store
from tcip_store import Key, StoreDescriptor, register_store
from tcip_store.file_backend import RootedFileLocator

from tcip_annotation.json_io import (
    ANNOTATIONS_KEY,
    UnreadableLabelDocument,
    geometry_extent_ok,
    read_annotations,
    write_annotations,
)
from tcip_annotation.state import Annotation, BBox, Point, Polygon, bbox_of

AnnotFormat = Literal["coco", "json"]

_PROV_KEYS = ("created_by", "created_at", "accepted_by", "accepted_at")

COCO_DOCUMENTS_STORE = "coco_documents"
_COCO_DOCUMENT_LOCATOR = RootedFileLocator(suffix=".json")
register_store(
    StoreDescriptor(
        name=COCO_DOCUMENTS_STORE,
        kind="blob",
        key_fields=("stem",),
        frozen=False,
        locator=_COCO_DOCUMENT_LOCATOR,
    )
)


def coco_document_key(directory: str | Path, stem: str) -> Key:
    """One assembled COCO document, addressed by the directory that holds it.

    A blob on the same terms as a per-image label document: it is written whole, from a view
    the caller already assembled. This package receives a location and never learns a layout,
    so an export bucket and a dataset root are addressed the same way.
    """
    return Key(COCO_DOCUMENTS_STORE, str(Path(directory).absolute()), (str(stem),))


# ── Format detection ────────────────────────────────────────────────────────


def detect_format(path: str) -> AnnotFormat:
    """The annotation format of a file or directory, from its own contents.

    ``"json"`` is the canonical per-image label file (``json_io`` schema, keyed on
    :data:`~tcip_annotation.json_io.ANNOTATIONS_KEY`); ``"coco"`` is an assembled dataset-level COCO
    (keyed on ``images`` / ``categories``). The old ``objects`` schema is not sniffed; it raises a
    ``ValueError``. Old files are converted once, never read in place: reading an unconverted file
    as the new schema would silently yield zero annotations and train on fabricated empty negatives.

    A present file that will not decode as JSON, or that decodes to something other than a dict,
    raises :class:`~tcip_annotation.json_io.UnreadableLabelDocument` instead of being tried as an
    unrecognized shape: an undecodable candidate is not evidence the directory holds no label
    format, it is a document nobody can read, and a caller must not learn that fact only as
    "cannot determine the annotation format".
    """
    from tcip_annotation.json_io import prediction_documents

    p = Path(path)
    candidates = prediction_documents(p) if p.is_dir() else [p]
    for candidate in candidates:
        fmt = detect_json_format(candidate)
        if fmt is not None:
            return fmt
    raise ValueError(
        f"Cannot determine the annotation format of {path}: expected the canonical per-image JSON "
        f"(an '{ANNOTATIONS_KEY}' key) or an assembled COCO (an 'images'/'categories' key)."
    )


def _shape_markers(data: dict) -> AnnotFormat | None:
    """``"coco"`` / ``"json"`` from a document's own keys, or ``None`` if it carries neither.

    A dataset-level COCO carries an ``images`` list and/or ``categories``; a per-image file
    carries a singular ``image`` string. Both use ``annotations``, so the COCO markers are
    checked first. Read only through :func:`detect_json_format`, the one classification a
    format-free detection and a caller's stated-``fmt`` claim both go through, so the two can
    never disagree about what shape a document is.
    """
    if "images" in data or "categories" in data:
        return "coco"
    if ANNOTATIONS_KEY in data:
        return "json"
    return None


def detect_json_format(path: Path) -> AnnotFormat | None:
    """``"json"`` / ``"coco"`` from a file's keys, or ``None`` if the path is missing or is a
    decodable dict that is neither. Raises on the old ``objects`` shape rather than sniffing it
    (it is converted, not read), and raises :class:`~tcip_annotation.json_io.UnreadableLabelDocument`,
    through the shared loader, for a *present* file that will not decode as a dict at all: a
    missing candidate is not evidence of a broken document, only an absent one, and stays
    ``detect_format``'s own "cannot determine" refusal rather than an unreadable-document one.
    Public: a caller with a single file already in hand and no directory to search (this
    module's own ``load_annotations`` claim check, ``label_queries.dir_label_format``) reaches
    this single-file classification directly rather than through ``detect_format``'s own
    directory-or-raise contract, which does not fit either caller's shape.

    Reads through :func:`~tcip_annotation.json_io.load_json_document`, not the version-checked
    ``load_label_document``: the shape is not yet known when this decode happens, and this
    platform's own version ceiling belongs to the per-image ``json`` shape alone, never to a
    document that turns out COCO-shaped (interop, ``frozen=False``) or unrecognized. Only once
    the shape resolves to ``"json"`` is :func:`~tcip_annotation.json_io.check_annotation_record_version`
    applied.
    """
    if not path.is_file():
        return None
    from tcip_annotation.json_io import check_annotation_record_version, load_json_document

    data = load_json_document(path)
    if "objects" in data:
        raise ValueError(
            f"{path} is the old 'objects' label schema, which is not read in place. Convert it "
            f"to the name-based per-image schema first; reading it as-is would yield zero "
            f"annotations and train on fabricated empty negatives."
        )
    shape = _shape_markers(data)
    if shape == "json":
        check_annotation_record_version(data, source=str(path))
    return shape


# ── COCO JSON parsing (names) ───────────────────────────────────────────────


def _parse_coco_json(path: str) -> dict:
    """A COCO-format JSON file's parsed dict, through the one decode every label reader shares.

    Raises :class:`~tcip_annotation.json_io.UnreadableLabelDocument` for a document that will not
    decode or does not parse as a dict, rather than a raw exception :func:`detect_format` (which
    already admitted the same bytes through the same decode) would not have raised. Never applies
    this platform's own per-image ``schema_version`` ceiling: COCO is interop (``frozen=False``
    by its own row), so a legitimate external COCO document carrying its own ``schema_version``
    key is read, not refused against a store it never claimed to be.
    """
    from tcip_annotation.json_io import load_json_document

    return load_json_document(path)


def _coco_categories(coco: dict) -> dict[int, str]:
    """``{category_id: name}`` from a COCO ``categories`` list, the file's own name map."""
    out: dict[int, str] = {}
    for c in coco.get("categories", []) if isinstance(coco, dict) else []:
        if isinstance(c, dict) and "id" in c and c.get("name"):
            try:
                out[int(c["id"])] = str(c["name"])
            except (TypeError, ValueError):
                continue
    return out


def _coco_image_annotations(
    coco: dict, image_id: int | None = None, file_name: str | None = None,
) -> tuple[list[dict], int, int]:
    """Annotations for a single image from a COCO dict, plus ``(img_w, img_h)``.

    ``file_name`` matches exactly first. When it names a ``.bandgroup`` manifest (whose own
    on-disk name never appears verbatim in an externally authored COCO document), it also ties by
    stem to the one recorded ``file_name`` that shares that stem; more than one recorded image
    sharing the stem is an unresolvable ambiguity and raises rather than picking one. An empty or
    absent ``file_name``, and a recorded image with an empty ``file_name``, never take part in the
    stem tie.
    """
    img_record = None
    for img in coco.get("images", []):
        if image_id is not None and img.get("id") == image_id:
            img_record = img
            break
        if file_name is not None and img.get("file_name") == file_name:
            img_record = img
            break
    if img_record is None and file_name and file_name.endswith(".bandgroup"):
        stem = Path(file_name).stem
        stem_matches = [
            img for img in coco.get("images", [])
            if img.get("file_name") and Path(img["file_name"]).stem == stem
        ]
        if len(stem_matches) > 1:
            raise ValueError(
                f"{file_name}: {len(stem_matches)} recorded images share the stem {stem!r}, an "
                "unresolvable ambiguity for a .bandgroup manifest lookup."
            )
        if stem_matches:
            img_record = stem_matches[0]
    if img_record is None:
        return [], 0, 0
    img_id = img_record["id"]
    w = int(img_record.get("width", 0) or 0)
    h = int(img_record.get("height", 0) or 0)
    anns = [a for a in coco.get("annotations", []) if a.get("image_id") == img_id]
    return anns, w, h


def _coco_prov(ann: dict) -> dict:
    """Provenance (and, for a prediction, ``score``) extension keys of a COCO annotation record."""
    out = {k: ann[k] for k in _PROV_KEYS if ann.get(k)}
    s = ann.get("score")
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        out["score"] = float(s)
    return out


def parse_coco_annotations(
    coco: dict, image_id: int | None = None, file_name: str | None = None,
) -> list[Annotation]:
    """Parse one image's COCO annotations into name-based :class:`Annotation` records.

    ``subject`` is the ``category_id``'s name from the file's own ``categories``, the same contract
    the per-image reader holds for a record it cannot coerce: a record whose ``category_id`` will
    not coerce to ``int``, or whose ``category_id`` has no name in this document's ``categories``,
    raises :class:`~tcip_annotation.json_io.UnreadableLabelDocument` naming the record's index,
    rather than reading the document short. A polygon geometry wins over a box when both are
    present (the polygon is the source of truth). A record's ``attributes``, when present, is
    restored when it is a mapping of strings to strings (the shape :func:`write_coco` emits), and
    raises the same error, naming the record's index, for any other shape: a classified
    prediction's value lives there, and losing it silently on a COCO round trip would launder a
    classified record back into a bare object-class one.
    """
    anns, _, _ = _coco_image_annotations(coco, image_id, file_name)
    id2name = _coco_categories(coco)
    out: list[Annotation] = []
    for i, ann in enumerate(anns):
        raw_category_id = cast(Any, ann.get("category_id"))
        try:
            cid = int(raw_category_id)
        except (TypeError, ValueError):
            raise UnreadableLabelDocument(
                f"record {i}'s category_id {raw_category_id!r} will not coerce to int"
            ) from None
        subject = id2name.get(cid)
        if not subject:
            raise UnreadableLabelDocument(
                f"record {i}'s category_id {cid} has no name in this document's categories"
            )
        geometry: BBox | Polygon | None = None
        segs = ann.get("segmentation")
        rings: list[list[tuple[float, float]]] = []
        if isinstance(segs, list):
            for coords in segs:
                if isinstance(coords, list) and len(coords) >= 6:
                    rings.append([(float(coords[i]), float(coords[i + 1]))
                                 for i in range(0, len(coords) - 1, 2)])
        if rings:
            geometry = Polygon(rings)
        else:
            bb = ann.get("bbox")
            if isinstance(bb, list) and len(bb) == 4:
                x, y, bw, bh = (float(v) for v in bb)
                geometry = BBox(x, y, x + bw, y + bh)
        attributes: dict[str, str] = {}
        raw_attributes = ann.get("attributes")
        if raw_attributes is not None:
            if not (isinstance(raw_attributes, dict)
                    and all(isinstance(k, str) and isinstance(v, str)
                            for k, v in raw_attributes.items())):
                raise UnreadableLabelDocument(
                    f"record {i}'s attributes {raw_attributes!r} is not a mapping of strings to "
                    "strings"
                )
            attributes = dict(raw_attributes)
        out.append(Annotation(
            subject=subject, geometry=geometry, attributes=attributes, **_coco_prov(ann)))
    return out


# ── COCO JSON writing (names → ids via a supplied map) ──────────────────────


def write_coco(
    path: str,
    images_annotations: dict[str, tuple[list[Annotation], int, int]],
    *,
    id_map: dict[str, int] | None = None,
) -> None:
    """Write name-based annotations to a single-file COCO JSON.

    ``images_annotations`` maps ``file_name -> (annotations, img_w, img_h)``. ``id_map`` is a
    ``subject -> category_id`` map; when omitted it is enumerated from the distinct subjects present
    (COCO carries its category names in the file, so the enumeration travels with the data; this is
    interop export, not the training id assignment, which is ``class_registry.assign_class_ids``).

    Every emitted record is a box (plus a polygon's ``segmentation``), so a geometry-less annotation
    and a :class:`~tcip_annotation.state.Point` are both skipped: COCO's own box/segmentation record
    has no honest representation for a point, and fabricating a zero-area one would export it as a
    detection target.
    """
    if id_map is None:
        subjects: list[str] = []
        for anns, _, _ in images_annotations.values():
            for a in anns:
                if a.subject not in subjects:
                    subjects.append(a.subject)
        id_map = {name: i for i, name in enumerate(subjects)}
    categories = [{"id": cid, "name": name} for name, cid in sorted(id_map.items(), key=lambda kv: kv[1])]
    coco: dict = {"images": [], "annotations": [], "categories": categories}
    ann_id = 1
    for img_id, (file_name, (anns, img_w, img_h)) in enumerate(images_annotations.items(), start=1):
        coco["images"].append({"id": img_id, "file_name": file_name, "width": img_w, "height": img_h})
        for a in anns:
            if a.geometry is None or isinstance(a.geometry, Point) or a.subject not in id_map:
                continue
            if not geometry_extent_ok(a.geometry):
                # The same drop the per-image export applies: a geometry the stored 2-decimal
                # grid collapses would emit a bbox claiming an extent its own segmentation lacks.
                continue
            box = bbox_of(a.geometry)
            bw, bh = box.x2 - box.x1, box.y2 - box.y1
            rec: dict = {
                "id": ann_id, "image_id": img_id, "category_id": id_map[a.subject], "iscrowd": 0,
                "bbox": [round(box.x1, 2), round(box.y1, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2),
            }
            if isinstance(a.geometry, Polygon):
                rec["segmentation"] = [[round(float(c), 2) for xy in ring for c in xy]
                                       for ring in a.geometry.rings if len(ring) >= 3]
            if a.attributes:
                rec["attributes"] = dict(a.attributes)
            if a.score is not None:
                rec["score"] = float(a.score)
            for k in _PROV_KEYS:
                v = getattr(a, k, None)
                if v is not None:
                    rec[k] = v
            coco["annotations"].append(rec)
            ann_id += 1
    target = Path(path)
    tcip_store.put_blob(
        coco_document_key(target.parent, target.stem),
        json.dumps(coco, indent=2).encode("utf-8"),
    )


# ── dispatch ────────────────────────────────────────────────────────────────


_SHAPE_DESCRIPTIONS: dict[AnnotFormat | None, str] = {
    "coco": "an images/categories key",
    "json": "a bare annotations key",
    None: "neither an images/categories key nor an annotations key",
}


def load_annotations(
    path: str,
    *,
    fmt: AnnotFormat | None = None,
    image_id: int | None = None,
    file_name: str | None = None,
) -> list[Annotation]:
    """Load an image's annotations as name-based records. ``fmt`` of ``None`` detects it.

    For COCO, either ``image_id`` or ``file_name`` must identify the target image. A present,
    unreadable document raises :class:`~tcip_annotation.json_io.UnreadableLabelDocument`, from
    ``detect_format`` when ``fmt`` is omitted, from the claim check when it is stated, or from
    ``read_annotations``/``parse_coco_annotations`` below either way.

    A caller-supplied ``fmt`` is a claim the document must satisfy, not a bypass of detection,
    checked through the same :func:`detect_json_format` a format-free call resolves through (the
    old ``objects`` schema included, refused there rather than sniffed): a present document whose
    keys carry the other format's markers, or neither format's, is refused, naming the format
    asked for and the shape its keys actually carry, rather than silently answering an empty
    read. A missing path is not checked against the claim at all; it stays ``detect_format``'s own
    absent-not-broken distinction and reaches ``read_annotations``/the COCO parser below exactly
    as an omitted ``fmt`` would.
    """
    if fmt is None:
        fmt = detect_format(path)
    elif Path(path).is_file():
        shape = detect_json_format(Path(path))
        if shape != fmt:
            raise ValueError(
                f"{path} was asked to be read as {fmt!r}, but its keys carry "
                f"{_SHAPE_DESCRIPTIONS[shape]}: a caller-supplied fmt must match what the "
                "document's own keys carry."
            )
    if fmt == "json":
        return read_annotations(path)
    if fmt == "coco":
        return parse_coco_annotations(_parse_coco_json(path), image_id=image_id, file_name=file_name)
    raise ValueError(f"Unsupported annotation format: {fmt}")


def save_annotations(
    path: str,
    annotations: list[Annotation],
    img_w: int,
    img_h: int,
    *,
    fmt: AnnotFormat = "json",
    file_name: str | None = None,
    keep_empty: bool = False,
    id_map: dict[str, int] | None = None,
) -> None:
    """Save name-based annotations.

    ``json`` writes the canonical per-image label file; ``coco`` writes a single-file COCO for the
    image (pass ``file_name`` to key it). ``keep_empty`` (json only): an empty list writes an
    ``"annotations": []`` record instead of deleting the label; an empty record is not a negative
    until a human confirms it.
    """
    if fmt == "json":
        write_annotations(path, annotations, img_w, img_h, keep_empty=keep_empty)
    elif fmt == "coco":
        fname = file_name or Path(path).stem
        write_coco(path, {fname: (annotations, img_w, img_h)}, id_map=id_map)
    else:
        raise ValueError(f"Unsupported annotation format: {fmt}")

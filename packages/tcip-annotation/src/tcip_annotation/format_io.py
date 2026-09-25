"""The reader of an external dataset-level COCO document (an ``"images"`` / ``"annotations"`` /
``"categories"`` key), on its way into per-image documents.

Its records carry the per-image document's own field spellings (``bbox`` as ``[x, y, w, h]``,
``segmentation`` rings, ``iscrowd``, ``attributes``, ``score``, provenance); this reader names each
record's subject from the document's own ``categories``, turns a run-length mask into rings through
:func:`~tcip_annotation.mask_contours.mask_to_polygon_rings`, and hands each record to the
per-record decoder.
"""

from __future__ import annotations

from typing import Any, Callable

from tcip_annotation.json_io import (
    ANNOTATIONS_KEY,
    annotation_object,
    annotation_of_record,
    is_dataset_level_document,
)
from tcip_annotation.mask_contours import mask_to_polygon_rings
from tcip_annotation.state import Annotation


def is_coco_id(value: Any) -> bool:
    """Whether ``value`` is a COCO identity: an integer, never a float, a string or a bool."""
    return isinstance(value, int) and not isinstance(value, bool)


def coco_categories(coco: dict, *, problems: list[str]) -> dict[int, str]:
    """``{category_id: name}`` from the document's own ``categories``, checked without coercion.

    A declaration that is not an object with an integer ``id``, and a second declaration of one id,
    is appended to ``problems`` by index rather than read.
    """
    out: dict[int, str] = {}
    declared = coco.get("categories", [])
    if not isinstance(declared, list):
        problems.append(f"'categories' is {declared!r}, not a list")
        return out
    for i, c in enumerate(declared):
        cid: Any = c.get("id") if isinstance(c, dict) else None
        name: Any = c.get("name") if isinstance(c, dict) else None
        if not is_coco_id(cid):
            problems.append(f"category {i} ({c!r}) is not an object with an integer id")
        elif cid in out:
            problems.append(f"category id {cid} is declared twice ({out[cid]!r} and {name!r})")
        else:
            out[cid] = name
    return out


def _named_annotation(
    ann: dict, categories: dict[int, str], decode_rle: Callable[[dict], Any],
) -> Annotation:
    """One COCO record as the :class:`Annotation` its ``category_id`` names, or ``ValueError``."""
    cid: Any = ann.get("category_id")
    if not is_coco_id(cid) or cid not in categories:
        raise ValueError(f"names category_id {cid!r}, which no valid declaration of this "
                         "document names")
    segmentation = ann.get("segmentation")
    if isinstance(segmentation, dict):
        rings = mask_to_polygon_rings(decode_rle(segmentation))
        segmentation = [[c for point in ring for c in point] for ring in rings]
    return annotation_of_record({**ann, "subject": categories[cid], "segmentation": segmentation})


def parse_coco_annotations(
    coco: dict, *, decode_rle: Callable[[dict], Any],
) -> tuple[dict[int, str], dict[int, list[Annotation]], list[str]]:
    """The document's valid category declarations, every record it holds as a name-based
    :class:`Annotation` keyed by the integer ``image_id`` its record names, and every fault found.

    A document without a dataset-level COCO's keys raises ``ValueError``. Every other fault is
    returned rather than raised, each naming its declaration or record index: a malformed or
    repeated category declaration (:func:`coco_categories`), a record whose ``image_id`` is not an
    integer, and a record the decoding refuses (a ``category_id`` no valid declaration names, then
    anything :func:`~tcip_annotation.json_io.annotation_of_record` refuses). A record's identity
    and its decoding are checked independently, so one record can report both, and only a record
    whose ``image_id`` is valid is associated with an image. A run-length ``segmentation`` (a dict
    with ``counts`` and ``size``) is decoded to a mask by ``decode_rle`` and converted to rings by
    :func:`~tcip_annotation.mask_contours.mask_to_polygon_rings`; a mask that yields no ring is
    refused by the decoder like any other segmentation with no ring.
    """
    if not is_dataset_level_document(coco):
        raise ValueError(
            "the document is not a dataset-level COCO document: it carries no 'images' or "
            "'categories' key"
        )
    problems: list[str] = []
    categories = coco_categories(coco, problems=problems)
    by_image: dict[int, list[Annotation]] = {}
    records = coco.get(ANNOTATIONS_KEY)
    if not isinstance(records, list):
        problems.append(f"{ANNOTATIONS_KEY!r} is {records!r}, not a list")
        return categories, by_image, problems
    for i, ann in enumerate(records):
        try:
            ann = annotation_object(ann)
        except ValueError as exc:
            problems.append(f"record {i} {exc}")
            continue
        image_id: Any = ann.get("image_id")
        identified = is_coco_id(image_id)
        if not identified:
            problems.append(f"record {i} names image_id {image_id!r}, not an integer image id")
        try:
            annotation = _named_annotation(ann, categories, decode_rle)
        except ValueError as exc:
            problems.append(f"record {i} {exc}")
            continue
        if identified:
            by_image.setdefault(image_id, []).append(annotation)
    return categories, by_image, problems

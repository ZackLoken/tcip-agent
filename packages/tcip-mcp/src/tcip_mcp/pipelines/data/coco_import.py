"""An external dataset-level COCO document, converted into the dataset's per-image label documents.

Object annotations train and calibrate from per-image documents, never from a dataset-level COCO:
a COCO export is read here once, on its way in, and never again.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _rle_mask(segmentation: dict) -> Any:
    """A COCO run-length ``segmentation``, compressed or uncompressed ``counts``, as a mask."""
    from pycocotools import mask as mask_utils

    try:
        height, width = segmentation["size"]
        counts = segmentation["counts"]
        rle = (mask_utils.frPyObjects(segmentation, height, width) if isinstance(counts, list)
               else {"size": [height, width], "counts": counts.encode("ascii")})
        return mask_utils.decode(rle)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"carries a run-length mask that does not decode ({exc!r})") from exc


def import_coco_document(document: str | Path, dataset_root: str | Path, *, date: str) -> dict:
    """Write one per-image label document for every image ``document`` lists, under
    ``annotations/<date>/`` of ``dataset_root``, and record the import on the dataset's audit log
    with the document's path and the digest of the one read of its bytes, once a document has
    been written: an import that wrote nothing changed nothing and leaves no line.

    ``date`` names the capture: the dated bucket ``images/<date>/`` when the dataset has dated
    buckets, the flat ``images/`` root only when it has none. Each declared category must be a
    subject the dataset's registry declares, and names the subject its records carry. An ordinary
    image is the logical image whose own file name the document's ``file_name`` is; a
    ``.bandgroup`` capture, whose manifest name no external document carries, is tied by stem.
    Each document is encoded at the frame its image decodes to, and a width or height the
    document states must match it. An image the document gives no annotation writes nothing: an
    empty document is not a negative until a person confirms one. Records keep the provenance and
    the crowd flag the document carried and gain none; a run-length mask becomes the rings its
    mask yields.

    Every fault the reader and one pass over the listed images find is reported together, and any
    of them refuses the import before anything is written: a malformed or repeated category
    declaration, an unregistered category, a record the reader or the writer's encoder refuses, an
    image id that is not an integer or is listed twice, an image not in the capture, two records
    naming one capture, an annotation naming an unlisted image, a stated frame the image does not
    have, and a per-image document already present. The writes are then create-only, one document
    at a time: a label placed after validation raises on that document and leaves the documents
    written before it, which the audit event names beside the one that failed.
    """
    import tcip_store
    from tcip_annotation.format_io import is_coco_id, parse_coco_annotations
    from tcip_annotation.json_io import (
        decode_document_bytes, encode_annotations, parse_json_document, read_document_bytes,
    )

    from tcip_mcp import dataset_layout
    from tcip_mcp.audit import record_event_or_raise
    from tcip_mcp.pipelines.image_utils import (
        BandGroupRef, image_dimensions, list_logical_images, refuse_incomplete_band_group,
    )
    from tcip_mcp.pipelines.resolution import digest_bytes
    from tcip_mcp.subject_registry import registry_for_dataset_root

    document, root = Path(document).resolve(), Path(dataset_root).resolve()
    source = str(document)
    raw = read_document_bytes(document)
    coco = parse_json_document(decode_document_bytes(raw, source=source), source=source)
    categories, by_image, problems = parse_coco_annotations(coco, decode_rle=_rle_mask)

    dates = dataset_layout.list_dates(root)
    if not dataset_layout.is_bucket_name(date) or (dates and date not in dates):
        raise ValueError(f"{date!r} names no capture under {dataset_layout.image_root(root)} "
                         f"(its buckets: {dates})")
    images = coco.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError(f"{source} lists no image, so there is nothing to import")
    registry = registry_for_dataset_root(root)
    if registry is None:
        raise ValueError(f"{root} has no subject registry; write one with write_subject_registry "
                         "before importing labels into it")

    declared = {repr(s.name) for s in registry.subjects}
    problems += [f"category {name} is not a subject the registry declares"
                 for name in sorted(set(map(repr, categories.values())) - declared)]
    images_dir = dataset_layout.image_dir(root, date if dates else None)
    logical = list_logical_images(images_dir)
    writes: list[tuple[tcip_store.Key, bytes]] = []
    ids: set[int] = set()
    stems: set[str] = set()
    for i, record in enumerate(images):
        if not isinstance(record, dict):
            problems.append(f"image record {i} ({record!r}) is not an object")
            continue
        image_id: Any = record.get("id")
        name = str(record.get("file_name") or "")
        identified = is_coco_id(image_id)
        if not identified:
            problems.append(f"image record {i} names id {image_id!r}, not an integer image id")
        elif image_id in ids:
            problems.append(f"image id {image_id!r} is listed twice")
        else:
            ids.add(image_id)
        found = logical.get(Path(name).stem)
        if not (isinstance(found, BandGroupRef) or (found is not None and found.name == name)):
            problems.append(f"image {name!r} is not in {images_dir}")
            found = None
        stem = found.stem if found is not None else Path(name).stem
        target = dataset_layout.annotation_path(root, date, stem)
        if target.exists():
            problems.append(f"{target} already exists")
        if found is None:
            continue
        if stem in stems:
            problems.append(f"image {name!r} is the capture {stem!r} another record already names")
        stems.add(stem)
        width, height = image_dimensions(refuse_incomplete_band_group(found))
        stated = (record.get("width"), record.get("height"))
        if stated != (None, None) and stated != (width, height):
            problems.append(f"image {name!r} is stated as {stated}, but decodes to "
                            f"{(width, height)}")
        if not identified:
            continue
        try:
            key, data = encode_annotations(str(target), by_image.get(image_id, []), width, height)
        except ValueError as exc:
            problems.append(f"image {name!r}: {exc}")
            continue
        if data is not None:
            writes.append((key, data))
    problems += [f"annotations name image id {image_id!r}, which the document does not list"
                 for image_id in sorted(set(by_image) - ids)]
    if problems:
        raise ValueError(f"{source} was not imported, nothing written: " + "; ".join(problems))

    written: list[str] = []

    def record_import(failed: str | None) -> None:
        if not written:
            return  # nothing committed, so nothing to record
        record_event_or_raise(
            "coco_document_imported",
            {"document": source, "date": date, "written": written, "failed": failed},
            status="ok" if failed is None else "failed", scope=root,
            document_digest=digest_bytes(raw))

    for key, data in writes:
        path = str(tcip_store.blob_path(key))
        try:
            tcip_store.put_blob(key, data, expect=tcip_store.Version.ABSENT)
        except Exception:
            record_import(path)
            raise
        written.append(path)
    record_import(None)
    return {"document": source, "date": date, "written": written}

"""An external dataset-level COCO document, converted into the dataset's per-image label
documents."""

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


def _read_coco(document: Path) -> tuple[dict, str]:
    """The external COCO file's parsed object and the digest of the one read of its bytes
    (:func:`tcip_store.read_blob_versioned`'s version). A file that is absent, not UTF-8 (a
    leading byte-order mark accepted), not JSON or not an object refuses (``ValueError``) naming
    it."""
    import json

    import tcip_store

    try:
        stored = tcip_store.read_blob_versioned(document)
        coco = json.loads(stored.value.decode("utf-8-sig"))
    except (tcip_store.NotFoundError, OSError, ValueError) as exc:
        raise ValueError(f"{document} does not read as a JSON document: {exc}") from exc
    if not isinstance(coco, dict):
        raise ValueError(f"{document} decodes to a {type(coco).__name__}, not the object a COCO "
                         "document is")
    return coco, stored.version.token


def import_coco_document(document: str | Path, dataset_root: str | Path, *, date: str,
                         project: Path) -> dict:
    """Write one per-image label document for every image ``document`` lists, keyed under the
    capture ``date`` of ``dataset_root`` (both :func:`~tcip_mcp.registry_paths.located` against
    ``project``), and record the import on the dataset's audit log with
    the document's path and the digest of the one read of its bytes, in one commit with the
    documents: an import that writes nothing leaves no line.

    ``date`` names the capture ``images/<date>/``; one :func:`~tcip_mcp.dataset_layout.list_dates`
    does not list refuses. Each declared category must be a
    subject the dataset's registry declares, and names the subject its records carry. An ordinary
    image is the logical image whose own file name the document's ``file_name`` is; a
    ``.bandgroup`` capture is tied by stem. Each document is encoded at the frame its image decodes
    to, and a width or height the document states must match it. An image the document gives no
    annotation writes nothing. Records keep the provenance and the crowd flag the document carried
    and gain none; a run-length mask becomes the rings its mask yields.

    Every fault the reader and one pass over the listed images find is reported together, and any
    of them refuses the import with nothing written: a malformed or repeated category declaration,
    an unregistered category, a record the reader or the writer's encoder refuses, an image id
    that is not an integer or is listed twice, an image not in the capture, two records naming one
    capture, an annotation naming an unlisted image, a stated frame the image does not have, and a
    per-image document already present, checked inside the one transaction that writes them.
    """
    import tcip_store
    from tcip_annotation.format_io import is_coco_id, parse_coco_annotations
    from tcip_annotation.json_io import document_payload

    from tcip_mcp import dataset_layout
    from tcip_mcp.audit import audit_entry, audit_log_key
    from tcip_mcp.pipelines.image_utils import (
        BandGroupRef, list_logical_images, refuse_incomplete_band_group,
    )
    from tcip_mcp.pipelines.raster_source import SourceHeader
    from tcip_mcp.registry_paths import located
    from tcip_mcp.subject_registry import registry_for_dataset_root

    document, root = located(document, project), located(dataset_root, project)
    source = str(document)
    coco, digest = _read_coco(document)
    categories, by_image, problems = parse_coco_annotations(coco, decode_rle=_rle_mask)

    dates = dataset_layout.list_dates(root)
    if date not in dates:
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
    images_dir = dataset_layout.image_dir(root, date)
    logical = list_logical_images(images_dir)
    writes: dict[tcip_store.Key, dict] = {}
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
        if found is None:
            continue
        if stem in stems:
            problems.append(f"image {name!r} is the capture {stem!r} another record already names")
        stems.add(stem)
        width, height = SourceHeader(refuse_incomplete_band_group(found)).display_frame
        stated = (record.get("width"), record.get("height"))
        if stated != (None, None) and stated != (width, height):
            problems.append(f"image {name!r} is stated as {stated}, but decodes to "
                            f"{(width, height)}")
        if not identified:
            continue
        try:
            data = document_payload(by_image.get(image_id, []), width, height)
        except ValueError as exc:
            problems.append(f"image {name!r}: {exc}")
            continue
        if data is not None:
            writes[dataset_layout.label_key(root, date, stem)] = data
    problems += [f"annotations name image id {image_id!r}, which the document does not list"
                 for image_id in sorted(set(by_image) - ids)]
    if problems:
        raise ValueError(f"{source} was not imported, nothing written: " + "; ".join(problems))

    arguments = {"document": source, "capture": date,
                 "written": sorted(key.parts[-1] for key in writes)}
    if writes:
        audit = audit_log_key(root)
        with tcip_store.transaction(audit, *writes) as txn:
            present = sorted(key.parts[-1] for key in writes if txn.read_versioned(
                key, default=None).version != tcip_store.Version.ABSENT)
            if present:
                raise ValueError(f"{source} was not imported, nothing written: the capture "
                                 f"already holds a label document for {present}")
            for key, data in writes.items():
                txn.write(key, data)
            txn.append(audit, audit_entry("coco_document_imported", arguments, None, "ok",
                                          {"document_digest": digest}))
    return arguments

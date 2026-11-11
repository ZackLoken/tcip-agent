"""An external dataset-level COCO document, converted into per-image label documents."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def _rle_mask(segmentation: dict) -> NDArray[np.uint8]:
    """Decode a COCO run-length ``segmentation``, compressed or uncompressed, as a mask.

    A segmentation that does not decode refuses (``ValueError``).
    """
    from pycocotools import mask as mask_utils

    try:
        height, width = segmentation["size"]
        counts = segmentation["counts"]
        rle = (
            mask_utils.frPyObjects(segmentation, height, width)
            if isinstance(counts, list)
            else {"size": [height, width], "counts": counts.encode("ascii")}
        )
        return mask_utils.decode(rle)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError(
            f"carries a run-length mask that does not decode ({exc!r})"
        ) from exc


def _read_coco(document: Path) -> tuple[dict, str]:
    """Return the external COCO file's parsed object and the digest of the one read of its bytes.

    The digest is :func:`tcip_store.read_blob_versioned`'s version. A file that is absent, not
    UTF-8 (a leading byte-order mark accepted), not JSON or not an object refuses
    (``ValueError``) naming it.
    """
    import json

    import tcip_store

    try:
        stored = tcip_store.read_blob_versioned(document)
        coco = json.loads(stored.value.decode("utf-8-sig"))
    except (tcip_store.NotFoundError, OSError, ValueError) as exc:
        raise ValueError(
            f"{document} does not read as a JSON document: {exc}"
        ) from exc
    if not isinstance(coco, dict):
        raise ValueError(
            f"{document} decodes to a {type(coco).__name__}, not the object a COCO "
            "document is"
        )
    return coco, stored.version.token


def import_coco_document(
    document: str | Path,
    dataset_root: str | Path,
    *,
    date: str,
    vertex_spacing: float,
    rationale: str,
    project: Path,
    actor: str | None,
) -> dict:
    """Write one per-image label document for every annotated image ``document`` lists.

    The documents are keyed under the capture ``date`` of ``dataset_root`` (both
    :func:`~tcip_mcp.registry_paths.located` against ``project``) and written by
    :func:`~tcip_mcp.dataset_layout.write_label_records` as ``actor``'s, its audit line recording
    the document's path, the digest of the one read of its bytes, the ``vertex_spacing`` a
    run-length mask's rings are simplified at and the ``rationale`` for it. The answer is that
    line's arguments.

    ``date`` names the capture ``images/<date>/``; one :func:`~tcip_mcp.dataset_layout.list_dates`
    does not list refuses. Each declared category must be a
    subject the dataset's registry declares, and names the subject its records carry. Every
    listed image is claimed by the document's ``file_name`` at the width and height the document
    states for it, an annotated one with its document encoded at that frame and one the document
    gives no annotation at the frame alone (:class:`~tcip_mcp.dataset_layout.ImageFrame`), which
    writes nothing. The writer's one admission of them all
    (:func:`~tcip_mcp.dataset_layout.admitted`) resolves each to the capture's image or names it
    absent and admits it against the frame the image decodes to. Records keep the
    provenance and the crowd flag the document carried and gain none; a run-length mask becomes
    the rings its mask yields.

    Every fault the reader and one pass over the listed images find is reported together with the
    writer's, and any of them refuses the import with nothing written: a malformed or repeated
    category declaration, an unregistered category, a record the reader or the encoder refuses,
    an image id that is not an integer or is listed twice, two records naming one file name and
    an annotation naming an unlisted image.
    """
    from tcip_annotation.format_io import is_coco_id, parse_coco_annotations
    from tcip_annotation.json_io import document_payload
    from tcip_mcp import dataset_layout
    from tcip_mcp.registry_paths import located
    from tcip_mcp.subject_registry import registry_for_dataset_root

    document, root = located(document, project), located(dataset_root, project)
    source = str(document)
    coco, digest = _read_coco(document)
    categories, by_image, problems = parse_coco_annotations(
        coco, decode_rle=_rle_mask, vertex_spacing=vertex_spacing
    )

    dataset_layout.require_listed_captures(root, [date])
    images = coco.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError(
            f"{source} lists no image, so there is nothing to import"
        )
    registry = registry_for_dataset_root(root)
    if registry is None:
        raise ValueError(
            f"{root} has no subject registry; write one with write_subject_registry "
            "before importing labels into it"
        )

    declared = {repr(s.name) for s in registry.subjects}
    problems += [
        f"category {name} is not a subject the registry declares"
        for name in sorted(set(map(repr, categories.values())) - declared)
    ]
    listed: dict[str, dataset_layout.ImageClaim] = {}
    ids: set[int] = set()
    for i, record in enumerate(images):
        if not isinstance(record, dict):
            problems.append(f"image record {i} ({record!r}) is not an object")
            continue
        image_id: Any = record.get("id")
        name = str(record.get("file_name") or "")
        identified = is_coco_id(image_id)
        if not identified:
            problems.append(
                f"image record {i} names id {image_id!r}, not an integer image id"
            )
        elif image_id in ids:
            problems.append(f"image id {image_id!r} is listed twice")
        else:
            ids.add(image_id)
        if name in listed:
            problems.append(f"image {name!r} is listed twice")
        annotations = by_image.get(image_id) if identified else None
        try:
            frame = dataset_layout.ImageFrame.model_validate(
                {"width": record.get("width"), "height": record.get("height")}
            )
            listed[name] = (
                frame
                if annotations is None
                else dataset_layout.ImageLabels(
                    document=document_payload(
                        annotations, frame.width, frame.height
                    ),
                    flags=None,
                )
            )
        except ValueError as exc:
            problems.append(f"image {name!r}: {exc}")
    problems += [
        f"annotations name image id {image_id!r}, which the document does not list"
        for image_id in sorted(set(by_image) - ids)
    ]
    return dataset_layout.write_label_records(
        root,
        {date: listed},
        registry=registry,
        faults=problems,
        source=source,
        digest=digest,
        facts={"vertex_spacing": vertex_spacing, "rationale": rationale},
        actor=actor,
    )

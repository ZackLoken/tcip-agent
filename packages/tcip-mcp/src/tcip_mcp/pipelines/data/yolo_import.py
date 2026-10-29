"""YOLO label files, converted to dataset-level COCO data for import."""

from __future__ import annotations

import math
from pathlib import Path


def _yolo_line(line: str, names: list[str], width: int, height: int) -> dict:
    """Return a YOLO box or polygon as a pixel COCO annotation without an image id.

    Class ids are integers covered by ``names``. Coordinates must be finite and in [0, 1],
    with absolute slack of 1e-9 clipped to the boundary. Box extents must be positive.
    A malformed line raises ``ValueError``.
    """
    from tcip_annotation.json_io import xywh
    from tcip_annotation.state import BBox

    fields = line.split()
    if len(fields) != 5 and (len(fields) < 7 or len(fields) % 2 != 1):
        raise ValueError(
            "expected a box or a polygon of at least three points"
        )
    try:
        category = int(fields[0])
    except ValueError as exc:
        raise ValueError(f"class id {fields[0]!r} is not an integer") from exc
    if not 0 <= category < len(names):
        raise ValueError(f"class id {category} is not covered by names")
    try:
        coordinates = [float(value) for value in fields[1:]]
    except ValueError as exc:
        raise ValueError("coordinates must be numbers") from exc
    if any(
        not math.isfinite(value) or not -1e-9 <= value <= 1 + 1e-9
        for value in coordinates
    ):
        raise ValueError("coordinates must be finite and in [0, 1]")
    coordinates = [min(1.0, max(0.0, value)) for value in coordinates]
    annotation: dict = {"category_id": category}
    if len(coordinates) == 4:
        x, y, w, h = coordinates
        if w <= 0 or h <= 0:
            raise ValueError("box width and height must be positive")
        box = BBox.from_normalized_center((x, y, w, h), width, height)
        annotation["bbox"] = xywh(
            box.x1, box.y1, box.x2, box.y2, on_grid=False
        )
    else:
        annotation["segmentation"] = [
            [
                value * (width if index % 2 == 0 else height)
                for index, value in enumerate(coordinates)
            ]
        ]
    return annotation


def import_yolo_labels(
    labels_dir: str | Path,
    names: list[str],
    dataset_root: str | Path,
    *,
    date: str,
    project: Path,
    actor: str | None,
) -> dict:
    """Import ``labels_dir``'s YOLO files over the capture's images and audit the documents.

    Paths are located against ``project``. Each ``<stem>.txt`` names one logical image of
    ``date``; its normalized coordinates use that image's display frame. A malformed line
    refuses naming the file and line. Empty files write nothing and their stems are returned
    under ``empty``. Images without a label file are untouched.

    The COCO conversion and transaction refuse unregistered subjects, invalid documents and
    existing labels. The answer is the audit arguments: the folder, ``names`` in class-id order,
    capture, empty stems and written stems. The event's ``labels_digest`` is the
    :func:`~tcip_annotation.json_io.canonical_digest` of the pairs of file name and store version
    token, sorted by file name, from each file's one versioned read. Documents and audit commit
    together; no documents means no audit entry.
    """
    import tcip_store
    from tcip_annotation.json_io import canonical_digest
    from tcip_mcp.pipelines.data.coco_import import (
        capture_images_dir,
        import_coco_data,
    )
    from tcip_mcp.pipelines.image_utils import (
        BandGroupRef,
        list_logical_images,
        refuse_incomplete_band_group,
    )
    from tcip_mcp.pipelines.raster_source import SourceHeader
    from tcip_mcp.registry_paths import located

    folder, root = located(labels_dir, project), located(dataset_root, project)
    images_dir = capture_images_dir(root, date)
    logical = list_logical_images(images_dir)
    if not folder.is_dir():
        raise ValueError(f"{folder} is not a label folder")
    images: list[dict] = []
    annotations: list[dict] = []
    versions: list[tuple[str, str]] = []
    empty: list[str] = []
    for image_id, label in enumerate(sorted(folder.glob("*.txt"))):
        try:
            stored = tcip_store.read_blob_versioned(label)
            lines = stored.value.decode("utf-8-sig").splitlines()
        except (tcip_store.NotFoundError, OSError, ValueError) as exc:
            raise ValueError(
                f"{label} does not read as YOLO labels: {exc}"
            ) from exc
        versions.append((label.name, stored.version.token))
        found = logical.get(label.stem)
        if found is None:
            raise ValueError(f"{label} names no image in {images_dir}")
        width, height = SourceHeader(
            refuse_incomplete_band_group(found)
        ).display_frame
        images.append(
            {
                "id": image_id,
                "file_name": found.manifest_path.name
                if isinstance(found, BandGroupRef)
                else found.name,
                "width": width,
                "height": height,
            }
        )
        populated = False
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                annotation = _yolo_line(line, names, width, height)
            except ValueError as exc:
                raise ValueError(f"{label}, line {number}: {exc}") from exc
            annotations.append({**annotation, "image_id": image_id})
            populated = True
        if not populated:
            empty.append(label.stem)
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": category, "name": names[category]}
            for category in sorted({ann["category_id"] for ann in annotations})
        ],
    }
    digest = canonical_digest(versions)
    return import_coco_data(
        coco,
        root,
        date=date,
        source=str(folder),
        vertex_spacing=0,
        event="yolo_labels_imported",
        arguments={
            "labels_dir": str(folder),
            "names": names,
            "capture": date,
            "empty": empty,
        },
        extra={"labels_digest": digest},
        actor=actor,
    )

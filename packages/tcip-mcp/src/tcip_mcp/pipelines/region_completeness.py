"""Per-cell content digest for the region-completeness store
(:func:`tcip_mcp.dataset_layout.region_completeness_path`): detects an annotation edited or
deleted inside an attested cell after attestation.

One digest per ``(subject, stem, cell)``: a hash of the subject's annotations whose geometry
centers inside that cell's own rect, canonically serialized the same way
``class_registry.attribute_schema_digest`` hashes an attribute vocabulary. Recomputing it from the
label file currently on disk and comparing against the stamp taken at attestation time
(:func:`tcip_mcp.dataset_layout.region_completeness_digest_path`) is the store's staleness check.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tcip_annotation.state import Annotation, BBox, Point, Polygon

from tcip_mcp.pipelines.reference_grid import Cell


def _annotation_center(a: Annotation) -> tuple[float, float] | None:
    """The point used to assign ``a`` to a cell, or ``None`` for a geometry-less annotation
    (an image/plant-level rating has no cell to belong to, so it never enters a digest)."""
    g = a.geometry
    if g is None:
        return None
    if isinstance(g, BBox):
        return (g.x1 + g.x2) / 2.0, (g.y1 + g.y2) / 2.0
    if isinstance(g, Point):
        return g.x, g.y
    xs = [pt[0] for ring in g.rings for pt in ring]
    ys = [pt[1] for ring in g.rings for pt in ring]
    return (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0


def _in_cell(point: tuple[float, float], cell: Cell) -> bool:
    x, y = point
    return cell.x0 <= x < cell.x1 and cell.y0 <= y < cell.y1


def _annotation_record(a: Annotation) -> dict:
    """Canonical content of one annotation for digest purposes: subject, geometry, and attribute
    values. Provenance (``created_by``/``created_at``/...) is deliberately excluded, the same
    reasoning ``attribute_schema_digest`` applies to a subject's description text: who authored a
    shape, or when, says nothing about whether the shape itself changed."""
    g = a.geometry
    if isinstance(g, BBox):
        geom = ["bbox", round(g.x1, 2), round(g.y1, 2), round(g.x2, 2), round(g.y2, 2)]
    elif isinstance(g, Point):
        geom = ["point", round(g.x, 2), round(g.y, 2)]
    elif isinstance(g, Polygon):
        geom = ["polygon", [[[round(x, 2), round(y, 2)] for x, y in ring] for ring in g.rings]]
    else:
        geom = None
    return {"subject": a.subject, "geometry": geom, "attributes": dict(sorted(a.attributes.items()))}


def cell_annotation_digest(annotations: list[Annotation], subject: str, cell: Cell) -> str:
    """Content digest of ``subject``'s annotations centered inside ``cell``.

    Deterministic in the annotation content alone (subject, geometry, attribute values): the same
    canonical-json + sha256[:16] recipe ``class_registry.attribute_schema_digest`` uses. An empty
    cell still gets a real, stable digest, of an empty list.
    """
    records = []
    for a in annotations:
        if a.subject != subject:
            continue
        center = _annotation_center(a)
        if center is None or not _in_cell(center, cell):
            continue
        records.append(_annotation_record(a))
    records.sort(key=lambda r: json.dumps(r, sort_keys=True))
    canonical = json.dumps(records, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def stale_cells(
    dataset_root: str | Path,
    record: dict,
    stamped_digests: dict[str, str],
    subject: str,
) -> list[str]:
    """Names, from ``record['cells_complete']``, whose current annotation content disagrees with
    the digest stamped at attestation time: a cell edited or deleted since it was attested.

    Reads the label file the record's own ``stem``/``date`` resolve to
    (``dataset_layout.annotation_path``) fresh, recomputes each attested cell's digest via
    :func:`cell_annotation_digest`, and compares. A cell with no stamp at all (an attestation made
    before the digest sidecar existed, or the sidecar itself lost) is reported stale: this gate has
    no escape hatch by design, and there is no legitimate no-digest case to preserve (no real
    attestation predates the digest sidecar, since the platform carries no user data yet). A cell
    absent from the record's own recomputed grid (``cell is None``, a different, structurally-
    impossible-in-normal-operation condition) is still skipped, unchanged.
    """
    from tcip_annotation.json_io import read_annotations

    from tcip_mcp.dataset_layout import annotation_path
    from tcip_mcp.pipelines.reference_grid import reference_cells

    stem = record.get("stem")
    grid = record.get("grid") or {}
    cells_complete = record.get("cells_complete") or []
    if not stem or not cells_complete:
        return []
    try:
        cells_by_name = {
            c.name: c
            for c in reference_cells(
                grid["width"], grid["height"], grid["tile_size"], grid.get("overlap", 0.0),
                clamp=True,
            )
        }
    except (KeyError, TypeError, ValueError):
        return []
    label_path = annotation_path(dataset_root, record.get("date"), stem)
    annotations = read_annotations(str(label_path)) if label_path.is_file() else []

    stale: list[str] = []
    for name in cells_complete:
        cell = cells_by_name.get(name)
        if cell is None:
            continue
        stamped = stamped_digests.get(name)
        if stamped is None or cell_annotation_digest(annotations, subject, cell) != stamped:
            stale.append(name)
    return stale


def incomplete_cells_for_rect(
    dataset_root: str | Path, subject: str, stem: str, rect: tuple[int, int, int, int],
) -> list[str] | None:
    """Cell names inside ``rect`` (a half-open pixel rect, full-mosaic coordinates) that are not
    attested complete for ``subject`` on the raster ``stem``, or are stale (edited since
    attestation): the exact list a completeness gate (block calibration's own hard door) names in
    its refusal. ``None`` when no completeness record exists at all for this ``(subject, stem)``
    bucket, distinct from an attested-but-gapped record: a caller phrases "never attested" and
    "attested but incomplete" differently.

    Reads the record's own recorded grid, whatever lattice the breeder actually attested cells
    against, never a caller's own block/tile geometry: the cells that matter here are the ones a
    human toggled, not this call's own tiling choice.
    """
    from tcip_mcp.dataset_layout import (
        normalize_region_completeness_store, region_completeness_digest_path,
        region_completeness_path, status_bucket,
    )
    from tcip_mcp.pipelines.data.tiling import rects_overlap
    from tcip_mcp.pipelines.reference_grid import reference_cells
    from tcip_mcp.utils.atomic_io import read_json

    bucket = status_bucket(subject, stem)
    store = normalize_region_completeness_store(
        read_json(region_completeness_path(dataset_root), default={}))
    record = store.get(bucket)
    if record is None:
        return None
    grid = record.get("grid") or {}
    try:
        cells = reference_cells(
            int(grid["width"]), int(grid["height"]), int(grid["tile_size"]),
            float(grid.get("overlap", 0.0)), clamp=True,
        )
    except (KeyError, TypeError, ValueError):
        return None
    rx0, ry0, rx1, ry1 = rect
    intersecting = [c for c in cells if rects_overlap((c.x0, c.y0, c.x1, c.y1), (rx0, ry0, rx1, ry1))]
    complete = set(record.get("cells_complete") or [])
    digests = read_json(region_completeness_digest_path(dataset_root), default={})
    stamped = digests.get(bucket) if isinstance(digests, dict) else None
    stale = set(stale_cells(dataset_root, record, stamped if isinstance(stamped, dict) else {}, subject))
    return sorted({c.name for c in intersecting if c.name not in complete or c.name in stale})

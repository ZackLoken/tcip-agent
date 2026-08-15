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

import tcip_store
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


def _digest_of_records(records: list[dict]) -> str:
    """Canonical-json + sha256[:16] recipe ``class_registry.attribute_schema_digest`` also uses,
    shared by :func:`cell_annotation_digest` and :func:`cell_annotation_digests` so the two never
    drift into computing "the same" digest two different ways."""
    ordered = sorted(records, key=lambda r: json.dumps(r, sort_keys=True))
    canonical = json.dumps(ordered, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def cell_annotation_digest(annotations: list[Annotation], subject: str, cell: Cell) -> str:
    """Content digest of ``subject``'s annotations centered inside ``cell``.

    Deterministic in the annotation content alone (subject, geometry, attribute values). An empty
    cell still gets a real, stable digest, of an empty list. Rescans ``annotations`` in full;
    a caller checking many cells at once (:func:`stale_cells`) should use
    :func:`cell_annotation_digests` instead, one pass over ``annotations`` for every cell rather
    than one pass per cell.
    """
    records = []
    for a in annotations:
        if a.subject != subject:
            continue
        center = _annotation_center(a)
        if center is None or not _in_cell(center, cell):
            continue
        records.append(_annotation_record(a))
    return _digest_of_records(records)


def cell_annotation_digests(
    annotations: list[Annotation], subject: str, cells: list[Cell], tile_size: int,
    overlap: float = 0.0,
) -> dict[str, str]:
    """:func:`cell_annotation_digest` for every cell in ``cells`` at once, one pass over
    ``annotations`` rather than one pass per cell -- O(annotations + cells) instead of
    O(annotations x cells), the real cost at real orthomosaic scale (thousands of annotations,
    up to hundreds of reserved-region cells).

    Requires ``overlap == 0.0`` (every real region-completeness/coverage grid caller constructs
    one) to bin by direct ``tile_size`` floor-division of each annotation's center, the same
    origin math :func:`~tcip_mcp.pipelines.reference_grid.reference_cells` itself uses -- a
    non-zero overlap would put some points in more than one cell, which floor-division alone
    cannot resolve, so that case falls back to :func:`cell_annotation_digest` per cell instead of
    silently computing a wrong digest.
    """
    if overlap != 0.0:
        return {c.name: cell_annotation_digest(annotations, subject, c) for c in cells}
    by_colrow = {(c.col, c.row): c.name for c in cells}
    buckets: dict[str, list[dict]] = {c.name: [] for c in cells}
    for a in annotations:
        if a.subject != subject:
            continue
        center = _annotation_center(a)
        if center is None:
            continue
        x, y = center
        name = by_colrow.get((int(x // tile_size), int(y // tile_size)))
        if name is not None:
            buckets[name].append(_annotation_record(a))
    return {name: _digest_of_records(records) for name, records in buckets.items()}


def stale_cells(
    dataset_root: str | Path,
    record: dict,
    stamped_digests: dict[str, str],
    subject: str,
) -> list[str]:
    """Names, from ``record['cells_complete']``, whose current annotation content disagrees with
    the digest stamped at attestation time: a cell edited or deleted since it was attested.

    Reads the label file the record's own ``stem``/``date`` resolve to
    (``dataset_layout.annotation_path``) fresh, recomputes every attested cell's digest in one
    pass via :func:`cell_annotation_digests`, and compares. A cell with no stamp at all (an
    attestation made
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

    complete_cells = [cells_by_name[name] for name in cells_complete if name in cells_by_name]
    digests = cell_annotation_digests(
        annotations, subject, complete_cells, int(grid["tile_size"]),
        float(grid.get("overlap", 0.0)))

    stale: list[str] = []
    for name in cells_complete:
        if name not in cells_by_name:
            continue
        stamped = stamped_digests.get(name)
        if stamped is None or digests.get(name) != stamped:
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
        normalize_region_completeness_store, region_completeness_digest_key,
        region_completeness_key, status_bucket,
    )
    from tcip_mcp.pipelines.data.tiling import rects_overlap
    from tcip_mcp.pipelines.reference_grid import reference_cells

    bucket = status_bucket(subject, stem)
    store = normalize_region_completeness_store(
        tcip_store.read(region_completeness_key(dataset_root), default={}))
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
    digests = tcip_store.read(region_completeness_digest_key(dataset_root), default={})
    stamped = digests.get(bucket) if isinstance(digests, dict) else None
    stale = set(stale_cells(dataset_root, record, stamped if isinstance(stamped, dict) else {}, subject))
    return sorted({c.name for c in intersecting if c.name not in complete or c.name in stale})

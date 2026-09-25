"""Per-cell content digest for the region-completeness store
(:func:`tcip_mcp.dataset_layout.region_completeness_path`): detects an annotation edited or
deleted inside an attested cell after attestation.

One digest per ``(subject, stem, cell)``: a hash of the subject's annotations whose geometry
centers inside that cell's own rect, each as the label writer stores it on its grid with
provenance aside (``json_io.stored_content``), canonically serialized the same way
``subject_registry.attribute_schema_digest`` hashes an attribute vocabulary. Recomputing it from the
label file currently on disk and comparing against the stamp taken at attestation time
(:func:`tcip_mcp.dataset_layout.region_completeness_digest_path`) is the store's staleness check.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import tcip_store
from tcip_annotation.json_io import stored_content
from tcip_annotation.state import Annotation, Point, bbox_of

from tcip_mcp.pipelines.reference_grid import Cell


def _annotation_center(a: Annotation) -> tuple[float, float] | None:
    """The point used to assign ``a`` to a cell, or ``None`` for a geometry-less annotation
    (an image/plant-level rating has no cell to belong to, so it never enters a digest)."""
    g = a.geometry
    if g is None:
        return None
    if isinstance(g, Point):
        return g.x, g.y
    b = bbox_of(g)
    return (b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0


def _in_cell(point: tuple[float, float], cell: Cell) -> bool:
    x, y = point
    return cell.x0 <= x < cell.x1 and cell.y0 <= y < cell.y1


def _digest_of_records(records: list[dict]) -> str:
    """Canonical-json + sha256[:16], the recipe ``subject_registry.attribute_schema_digest`` also
    uses.
    """
    ordered = sorted(records, key=lambda r: json.dumps(r, sort_keys=True))
    canonical = json.dumps(ordered, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def cell_annotation_digest(annotations: list[Annotation], subject: str, cell: Cell) -> str:
    """Content digest of ``subject``'s annotations centered inside ``cell``.

    Deterministic in the annotation content alone (subject, geometry, attribute values). An empty
    cell gets the digest of an empty list. Rescans ``annotations`` in full;
    :func:`cell_annotation_digests` digests many cells in one pass.
    """
    records = []
    for a in annotations:
        if a.subject != subject:
            continue
        center = _annotation_center(a)
        if center is None or not _in_cell(center, cell):
            continue
        records.append(stored_content(a))
    return _digest_of_records(records)


def _bin_annotations(
    annotations: list[Annotation], cells: list[Cell], tile_size: int, overlap: float,
) -> dict[str, list[Annotation]]:
    """Every one of ``annotations`` whose center falls in one of ``cells``, in one pass over
    ``annotations``: O(annotations + cells). No subject filter.

    ``overlap == 0.0`` bins by direct ``tile_size`` floor-division of each annotation's center, the
    origin math :func:`~tcip_mcp.pipelines.reference_grid.reference_cells` uses. A non-zero overlap
    falls back to per-cell containment.
    """
    buckets: dict[str, list[Annotation]] = {c.name: [] for c in cells}
    if overlap != 0.0:
        for a in annotations:
            center = _annotation_center(a)
            if center is None:
                continue
            for c in cells:
                if _in_cell(center, c):
                    buckets[c.name].append(a)
        return buckets
    by_colrow = {(c.col, c.row): c.name for c in cells}
    for a in annotations:
        center = _annotation_center(a)
        if center is None:
            continue
        x, y = center
        name = by_colrow.get((int(x // tile_size), int(y // tile_size)))
        if name is not None:
            buckets[name].append(a)
    return buckets


def annotations_by_cell(
    annotations: list[Annotation], subject: str, cells: list[Cell], tile_size: int,
    overlap: float = 0.0,
) -> dict[str, list[Annotation]]:
    """``subject``'s annotations from ``annotations``, binned by which of ``cells`` each one's
    center falls in (:func:`_bin_annotations`).
    """
    by_cell = _bin_annotations(annotations, cells, tile_size, overlap)
    return {name: [a for a in anns if a.subject == subject] for name, anns in by_cell.items()}


def annotation_counts_by_cell(
    annotations: list[Annotation], cells: list[Cell], tile_size: int, overlap: float = 0.0,
) -> dict[str, dict[str, int]]:
    """Every subject's per-cell annotation count over ``cells``, one pass over ``annotations``
    (:func:`_bin_annotations`) regardless of how many subjects are present.
    """
    by_cell = _bin_annotations(annotations, cells, tile_size, overlap)
    counts: dict[str, dict[str, int]] = {}
    for cell_name, anns in by_cell.items():
        for a in anns:
            subject_counts = counts.setdefault(a.subject, {})
            subject_counts[cell_name] = subject_counts.get(cell_name, 0) + 1
    return counts


def cell_annotation_digests(
    annotations: list[Annotation], subject: str, cells: list[Cell], tile_size: int,
    overlap: float = 0.0,
) -> dict[str, str]:
    """:func:`cell_annotation_digest` for every cell in ``cells`` at once, via
    :func:`annotations_by_cell`'s shared one-pass binning rather than one digest computation per
    cell.
    """
    by_cell = annotations_by_cell(annotations, subject, cells, tile_size, overlap)
    return {name: _digest_of_records([stored_content(a) for a in anns])
            for name, anns in by_cell.items()}


def record_annotations(dataset_root: str | Path, record: dict) -> list:
    """The current annotations of the label file a completeness record's own ``stem``/``date``
    resolve to (``dataset_layout.annotation_path``), ``[]`` when there is no such file; an
    unreadable file raises ``UnreadableLabelDocument``."""
    from tcip_annotation.json_io import read_annotations

    from tcip_mcp.dataset_layout import annotation_path

    label_path = annotation_path(dataset_root, record["date"], record["stem"])
    return read_annotations(str(label_path)) if label_path.is_file() else []


def stale_cells(
    record: dict,
    annotations: list,
    stamped_digests: dict[str, str],
    subject: str,
) -> list[str]:
    """Names, from ``record['cells_complete']``, whose current annotation content disagrees with
    the digest stamped at attestation time: a cell edited or deleted since it was attested.

    ``annotations`` is the record's label as it reads now (:func:`record_annotations` reads it
    for a caller holding only the dataset root). Recomputes every attested cell's digest in one
    pass via :func:`cell_annotation_digests`, and compares. A cell with no stamp at all is
    reported stale. A cell absent from the record's own recomputed grid is skipped.
    """
    from tcip_mcp.pipelines.reference_grid import reference_cells

    grid = record["grid"]
    cells_complete = record["cells_complete"]
    if not cells_complete:
        return []
    cells_by_name = {
        c.name: c
        for c in reference_cells(
            grid["width"], grid["height"], grid["tile_size"], grid["overlap"], clamp=True,
        )
    }
    complete_cells = [cells_by_name[name] for name in cells_complete if name in cells_by_name]
    digests = cell_annotation_digests(
        annotations, subject, complete_cells, int(grid["tile_size"]), float(grid["overlap"]))

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
    attestation). ``None`` when no completeness record exists at all for this ``(subject, stem)``
    bucket, distinct from an attested-but-gapped record.

    Reads the record's own recorded grid, the lattice the breeder attested cells against, never a
    caller's own block/tile geometry.
    """
    from tcip_mcp.dataset_layout import (
        region_completeness_digest_key, region_completeness_key, status_bucket,
    )
    from tcip_mcp.pipelines.data.tiling import rects_overlap
    from tcip_mcp.pipelines.reference_grid import reference_cells

    bucket = status_bucket(subject, stem)
    record = tcip_store.read(region_completeness_key(dataset_root), default={}).get(bucket)
    if record is None:
        return None
    grid = record["grid"]
    cells = reference_cells(
        grid["width"], grid["height"], grid["tile_size"], grid["overlap"], clamp=True)
    rx0, ry0, rx1, ry1 = rect
    intersecting = [c for c in cells if rects_overlap((c.x0, c.y0, c.x1, c.y1), (rx0, ry0, rx1, ry1))]
    complete = set(record["cells_complete"])
    digests = tcip_store.read(region_completeness_digest_key(dataset_root), default={})
    stale = set(stale_cells(record, record_annotations(dataset_root, record),
                            digests.get(bucket, {}), subject))
    return sorted({c.name for c in intersecting if c.name not in complete or c.name in stale})

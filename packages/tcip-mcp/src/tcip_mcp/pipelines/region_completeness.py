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
import statistics
from dataclasses import dataclass
from pathlib import Path

import tcip_store
from tcip_annotation.state import Annotation, BBox, Point, Polygon, bbox_of

from tcip_mcp.pipelines import pixel_size as pixel_size_module
from tcip_mcp.pipelines.pixel_size import PixelSize
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


def _bin_annotations(
    annotations: list[Annotation], cells: list[Cell], tile_size: int, overlap: float,
) -> dict[str, list[Annotation]]:
    """Every one of ``annotations`` whose center falls in one of ``cells``, one pass over
    ``annotations`` -- O(annotations + cells) instead of O(annotations x cells), the real cost at
    real orthomosaic scale (thousands of annotations, up to hundreds of reserved-region cells).
    No subject filter: :func:`annotations_by_cell` and :func:`annotation_counts_by_cell` apply
    theirs after, so the one pass over ``annotations`` here serves either.

    ``overlap == 0.0`` (every real region-completeness/coverage grid carries it) bins by direct
    ``tile_size`` floor-division of each annotation's center, the same origin math
    :func:`~tcip_mcp.pipelines.reference_grid.reference_cells` itself uses. A non-zero overlap
    would put some points in more than one cell, which floor-division alone cannot resolve, so
    that case falls back to per-cell containment instead of silently computing a wrong bin.
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
    center falls in (:func:`_bin_annotations`): the shared binning
    :func:`cell_annotation_digests` and the completeness route's per-cell saved-annotation counts
    (:func:`annotation_counts_by_cell`) both read off, so the two can never drift into computing
    "which cell" two different ways.
    """
    by_cell = _bin_annotations(annotations, cells, tile_size, overlap)
    return {name: [a for a in anns if a.subject == subject] for name, anns in by_cell.items()}


def annotation_counts_by_cell(
    annotations: list[Annotation], cells: list[Cell], tile_size: int, overlap: float = 0.0,
) -> dict[str, dict[str, int]]:
    """Every subject's per-cell annotation count over ``cells``, one pass over ``annotations``
    (:func:`_bin_annotations`) regardless of how many subjects are present -- the completeness
    route's ``annotation_counts`` field, read once per raster rather than once per subject.
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
    return {name: _digest_of_records([_annotation_record(a) for a in anns])
            for name, anns in by_cell.items()}


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


def default_working_scale_source(judged_span_px: int) -> str:
    """The one sentence :func:`working_scale_bar`'s ``source`` field carries wherever a route
    serves or stores a bar: states plainly that ``judged_span_px`` is a documented default, not
    a measurement of this subject's own legibility, so a caller on either side of the route
    boundary reads the same disclosure rather than two independently-worded ones."""
    return (
        f"a documented default: {judged_span_px}px is the span a typical annotated object is "
        "taken to read at on screen, not a measurement of this subject's own legibility"
    )


def saved_extents(annotations: list[Annotation], subject: str) -> list[float]:
    """The longer bounding-box side, in native pixels, of every ``subject`` annotation in
    ``annotations`` that carries a box or polygon geometry: the raw material the working-scale
    bar (:func:`working_scale_bar`) takes its median over.

    A :class:`Point` and a geometry-less annotation contribute nothing: neither has a bounding
    box, and a point's tiny nominal extent would pull the median toward a scale no box or polygon
    on the image was actually judged at. The longer side is what "spans" names: the box or
    polygon's longer edge is what a breeder's eye has to take in to judge the object legible, not
    its narrower one.
    """
    extents: list[float] = []
    for a in annotations:
        if a.subject != subject:
            continue
        g = a.geometry
        if not isinstance(g, (BBox, Polygon)):
            continue
        box = bbox_of(g)
        extents.append(max(box.x2 - box.x1, box.y2 - box.y1))
    return extents


def working_scale_bar(
    extents: list[float], *, judged_span_px: int, source: str, from_this_image: bool,
) -> dict | None:
    """The view scale at which the median of ``extents`` spans ``judged_span_px`` screen pixels,
    or ``None`` for no extents at all.

    One median over every extent handed in, per image and per subject: a per-cell bar would be a
    different measure over too few annotations per cell to be a real median, and is not computed
    here. The value is not clamped to the viewer's zoom ladder or to the image's own fit scale; a
    caller that wants to state what an out-of-range bar means does so on the served value, not by
    silently bending it back into range. A single whole-frame annotation on an otherwise
    unannotated image yields a bar every ordinary view meets, an accepted consequence for a large
    object: the median moves as more annotations are saved, never adjusted here to soften that.

    ``from_this_image`` is set on the built dict directly, a required keyword rather than a
    caller-side mutation after the call: true for this image's own saved annotations, false for
    the dataset-derived branch (:func:`dataset_working_scale_bar`), so the served and stored
    shapes always carry it consistently.
    """
    if not extents:
        return None
    median_extent = statistics.median(extents)
    return {
        "value": judged_span_px / median_extent,
        "median_extent_native_px": median_extent,
        "annotation_count": len(extents),
        "judged_span_px": judged_span_px,
        "source": source,
        "from_this_image": from_this_image,
    }


def _format_length_m(value_m: float) -> str:
    """One length in metres, formatted for a breeder-facing sentence at a unit chosen by
    magnitude: millimetres under a metre, metres at or above it, so a leaf-scale pixel size and a
    canopy-scale median both read at a sensible number of digits rather than one fixed unit
    forcing ``0.0003 m`` or ``1.5e6 mm`` on the other."""
    if abs(value_m) < 1.0:
        return f"{value_m * 1000:.3g} mm"
    return f"{value_m:.3g} m"


@dataclass(frozen=True)
class DatasetExtent:
    """The dataset-wide physical extent of one subject's saved box/polygon annotations, pooled
    across every georeferenced raster whose pixel size
    :func:`tcip_mcp.pipelines.pixel_size.raster_pixel_size` can resolve
    (:func:`dataset_physical_extent`): the material a raster or a whole dataset that carries none
    of its own annotations of the subject has no other way to derive a working-scale bar from.

    ``dates`` are the distinct capture buckets (``None`` for the dateless one) that contributed
    at least one annotation, sorted with the dateless bucket first.
    """

    median_extent_m: float
    annotation_count: int
    image_count: int
    metres_per_px_min: float
    metres_per_px_max: float
    dates: tuple[str | None, ...]


def dataset_physical_extent(
    dataset_root: str | Path, subject: str, *,
    pixel_sizes: dict[Path, tuple[PixelSize | None, str]],
    label_cache: dict[Path, list[Annotation]] | None = None,
) -> DatasetExtent | None:
    """``subject``'s saved box/polygon annotations, pooled in physical units across every
    georeferenced raster in ``dataset_root`` whose pixel size is known, or ``None`` when none
    exists (no annotation of ``subject`` sits on a raster with a resolvable pixel size).

    Walks every date bucket under ``images/`` (the dateless bucket alone when there are none),
    lists each bucket's logical images once (an ``AmbiguousImageStem`` under any date ends the
    derivation with that reason, since the dataset does not enumerate), reads every label file in
    the matching ``annotations/<date>/`` directory once, and resolves each labelled image's own
    raster source's pixel size once. ``pixel_sizes`` and ``label_cache`` are caches the caller
    owns and shares across every subject one request derives a bar for, and ``pixel_sizes``
    carries the ``(pixel_size, reason)`` pair
    :func:`tcip_mcp.pipelines.pixel_size.resolve_pixel_size` returns (the walk itself
    only ever needs the size, but the pair is the one cache format a caller resolving this
    image's own pixel size for a served reason -- ``get_completeness``, ``post_completeness`` --
    shares with the walk, so a raster is opened at most once per request regardless of which
    caller needed it first). A caller deriving for one subject alone passes neither and gets a
    fresh cache of each. A raster whose pixel size cannot be resolved is cached as ``(None,
    reason)`` and skipped, never ending the walk; a photographic capture or a band group is
    never opened.

    Each annotation's physical extent is its native-pixel extent (:func:`saved_extents`) times
    its own raster's ``metres_per_px``: the metres conversion is the admission rule for which
    annotations may contribute, not an arithmetic step applied uniformly after the fact, so a
    dataset drawn entirely from one pixel size and one where sizes vary are never silently pooled
    the same way. The median is over every kept annotation, annotation-weighted the same way
    :func:`working_scale_bar` weights a single image's own annotations.
    """
    from tcip_annotation import json_io

    from tcip_mcp.dataset_layout import annotation_dir, image_dir, list_dates
    from tcip_mcp.pipelines.image_utils import capture_kind, list_logical_images

    root = Path(dataset_root)
    dates: list[str | None] = list(list_dates(root)) or [None]
    labels = {} if label_cache is None else label_cache

    extents_m: list[float] = []
    metres_per_px_values: list[float] = []
    contributing_dates: set[str | None] = set()
    contributing_images = 0

    for date in dates:
        sources = list_logical_images(image_dir(root, date))
        label_dir = annotation_dir(root, date)
        if not label_dir.is_dir():
            continue
        for label_path in json_io.prediction_documents(label_dir):
            if label_path not in labels:
                labels[label_path] = json_io.read_annotations(str(label_path))
            image_extents = saved_extents(labels[label_path], subject)
            if not image_extents:
                continue
            source = sources.get(label_path.stem)
            if source is None or capture_kind(source) != "raster":
                continue
            assert isinstance(source, Path)
            if source not in pixel_sizes:
                pixel_sizes[source] = pixel_size_module.resolve_pixel_size(source)
            resolved_size, _reason = pixel_sizes[source]
            if resolved_size is None:
                continue
            for extent_px in image_extents:
                extents_m.append(extent_px * resolved_size.metres_per_px)
            metres_per_px_values.append(resolved_size.metres_per_px)
            contributing_dates.add(date)
            contributing_images += 1

    if not extents_m:
        return None
    return DatasetExtent(
        median_extent_m=statistics.median(extents_m),
        annotation_count=len(extents_m),
        image_count=contributing_images,
        metres_per_px_min=min(metres_per_px_values),
        metres_per_px_max=max(metres_per_px_values),
        dates=tuple(sorted(contributing_dates, key=lambda d: (d is not None, d))),
    )


def _format_dates(dates: tuple[str | None, ...]) -> str:
    """``dates`` as a plain English list ("2026-02-11 and 2026-03-02",
    "2026-02-11, 2026-03-02 and 2026-04-01"): the contributing capture dates a dataset-derived
    bar's source sentence names, so the breeder sees which captures the pooled median came from.
    The dateless bucket (``None``) reads as "an undated capture", the one case a real dataset
    walk can produce alongside real dates (:func:`dataset_physical_extent` always sorts it
    first)."""
    labels = [d if d is not None else "an undated capture" for d in dates]
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def dataset_extent_source(
    subject: str, extent: DatasetExtent, pixel_size: PixelSize, judged_span_px: int,
) -> str:
    """The one sentence a dataset-derived working-scale bar's ``source`` field carries, wherever a
    route serves or stores one, so a caller on either side of the route boundary reads the same
    disclosure :func:`default_working_scale_source` gives the own-annotations case: the dataset
    median in physical units, the georeferenced population and capture dates it was drawn from,
    and this image's own pixel size (``pixel_size``, its ``source_clause`` named) it was
    expressed through.
    """
    return (
        f"the median of {extent.annotation_count} saved {subject} annotations across "
        f"{extent.image_count} georeferenced images under {_format_dates(extent.dates)} "
        f"({_format_length_m(extent.median_extent_m)}; pixel sizes from "
        f"{_format_length_m(extent.metres_per_px_min)} to "
        f"{_format_length_m(extent.metres_per_px_max)} per px), expressed through this image's "
        f"{_format_length_m(pixel_size.metres_per_px)} per px ({pixel_size.source_clause}); the "
        f"{judged_span_px} px span is a documented default"
    )


def dataset_working_scale_bar(
    subject: str, extent: DatasetExtent, pixel_size: PixelSize, judged_span_px: int,
) -> dict:
    """The dataset-derived working-scale bar for ``subject``: ``extent``'s pooled physical median
    expressed through ``pixel_size`` (this image's own), with its ``source`` sentence
    (:func:`dataset_extent_source`) and its ``annotation_count`` overridden to the dataset's own
    count rather than the single-value list :func:`working_scale_bar` was called over.

    The one recipe both coverage routes call for this branch, so the divide, the bar, the source
    sentence and the annotation-count override live in exactly one place instead of being
    duplicated at each call site; ``from_this_image`` is always ``False`` here.
    """
    native_extent_px = extent.median_extent_m / pixel_size.metres_per_px
    bar = working_scale_bar(
        [native_extent_px], judged_span_px=judged_span_px,
        source=dataset_extent_source(subject, extent, pixel_size, judged_span_px),
        from_this_image=False)
    assert bar is not None
    bar["annotation_count"] = extent.annotation_count
    return bar


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

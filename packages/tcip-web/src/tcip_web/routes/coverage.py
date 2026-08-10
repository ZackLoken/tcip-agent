"""View-coverage routes: the reference grid over a raster and the per-image record of
two per-cell facts: which cells were served to the browser at native resolution (a
delivery fact) and which cells were swept in the viewport at or above the breeder's own
working scale (a sweep fact). Neither is an attention claim.

The grid geometry has one implementation (``tcip_mcp.pipelines.reference_grid``): the
grid route serves cells computed there and the frontend consumes them verbatim, never
re-deriving them. The record store is ``view_coverage.json``
(``dataset_layout.view_coverage_path``), bucketed like ``image_status.json``
(``status_bucket(subject, date)``, then image name). The store is advisory: training
never reads it, and unviewed cells warn rather than block a Complete.

Also here: region-completeness routes, a different store on the same grid. An attestation ("I
found every instance of this subject in these cells") gates a scientific claim (block
calibration's completeness check), so unlike view coverage it is written with the same discipline
as ``routes/classes.py``'s image-status store, and a stale attestation (a cell's annotation
content edited or deleted since it was attested) is detected on every read, not trusted forever.
See ``dataset_layout.region_completeness_path`` and ``tcip_mcp.pipelines.region_completeness``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from tcip_mcp.utils.atomic_io import atomic_write_json, file_transaction, read_json
from tcip_web.routes.classes import _audit_dataset_write, _guard_dataset_root
from tcip_web.routes.images import _checked

router = APIRouter(prefix="/api/coverage", tags=["coverage"])

GRID_KEYS = ("width", "height", "tile_size", "overlap", "cols", "rows")

# One audit entry per image per backend process: the first coverage write for an image
# audits, the debounced merges that follow coalesce into it.
_audited_coverage_keys: set[tuple[str, str, str]] = set()


def _normalized_grid(grid: dict) -> dict:
    """The grid geometry dict reduced to its defining keys with canonical numeric types,
    so a stored grid and a posted grid compare by value, not by JSON accident."""
    try:
        normalized = {
            "width": int(grid["width"]),
            "height": int(grid["height"]),
            "tile_size": int(grid["tile_size"]),
            "overlap": float(grid["overlap"]),
            "cols": int(grid["cols"]),
            "rows": int(grid["rows"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            400, f"grid must carry {list(GRID_KEYS)} as numbers: {exc}") from exc
    if normalized["overlap"] != 0.0:
        raise HTTPException(
            400, "a coverage record's grid requires overlap 0: the exact-partition contract "
                 "puts every native pixel in exactly one cell, which overlapping cells break")
    return normalized


def _resolve_root(image_path: str, dataset_root: Optional[str]) -> str:
    """The dataset root the record belongs to: the explicit one, else derived from the
    image's canonical path. Either way confined to the allowed image roots."""
    if dataset_root:
        return _guard_dataset_root(dataset_root)
    from tcip_mcp.dataset_layout import parse_image_path

    try:
        root, _date, _stem = parse_image_path(image_path)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _guard_dataset_root(str(root))


def _require_subject(subject: Optional[str]) -> str:
    if not subject:
        raise HTTPException(400, "view coverage is scoped to a subject and date, the same "
                                 "bucket image status uses; pass a subject")
    return subject


def _require_completeness_subject(subject: Optional[str]) -> str:
    if not subject:
        raise HTTPException(400, "region completeness is scoped to a subject, the same way "
                                 "every other confirmation in this platform is; a bare 'mark "
                                 "region done' with no subject would silently clear a different "
                                 "subject's calibration; pass a subject")
    return subject


@router.get("/grid")
def get_grid(
    path: str = Query(..., description="Absolute path to the image file"),
    tile_size: int | None = Query(None, ge=1, description="Cell edge in native pixels; "
                                  "omitted derives the coverage lattice edge"),
    overlap: float = Query(0.0),
) -> dict:
    """The reference grid over ``path``'s native frame: geometry plus the full cell list.

    Native dims come off the raster header (``image_dimensions``), never a decode.
    ``overlap`` other than 0 is refused: the coverage record's exact-partition contract
    puts every native pixel in exactly one cell, which overlapping cells break. The
    frontend consumes these cells verbatim and never re-derives them.
    """
    from tcip_mcp.pipelines.data.band_groups import BandGroupIncomplete
    from tcip_mcp.pipelines.image_utils import image_dimensions, resolve_image_source
    from tcip_mcp.pipelines.raster_source import is_georeferenced, opens_windowed
    from tcip_mcp.pipelines.reference_grid import (
        derive_coverage_tile_size,
        derive_large_raster_grid_tile_size,
        grid_geometry,
        reference_cells,
    )

    src = _checked(path)
    if overlap != 0.0:
        raise HTTPException(
            400,
            f"the coverage grid requires overlap 0: its exact-partition contract puts every "
            f"native pixel in exactly one cell, which overlapping cells break; got {overlap}")
    try:
        source = resolve_image_source(src.parent, src.stem)
    except BandGroupIncomplete as exc:
        raise HTTPException(409, str(exc)) from exc
    width, height = image_dimensions(source)
    # A stitched orthomosaic carries real per-pixel georeferencing; an ordinary drone/ground
    # capture (any pixel size) carries at most a single EXIF GPS point, never that.
    if tile_size is not None:
        edge = tile_size
    elif opens_windowed(source, 3) and is_georeferenced(source):
        edge = derive_large_raster_grid_tile_size(width, height)
    else:
        edge = derive_coverage_tile_size(width, height)
    cells = reference_cells(width, height, edge, 0.0, clamp=True)
    return {
        **grid_geometry(width, height, edge, 0.0),
        "cells": [{"name": c.name, "x0": c.x0, "y0": c.y0, "x1": c.x1, "y1": c.y1}
                  for c in cells],
    }


@router.get("")
def get_coverage(
    path: str = Query(..., description="Absolute path to the image file"),
    subject: str | None = Query(None),
    date: str | None = Query(None),
    dataset_root: str | None = Query(None),
) -> dict:
    """The stored coverage record for one image under one subject/date bucket, or
    ``{"coverage": null}`` when nothing has been recorded."""
    from tcip_mcp.dataset_layout import status_bucket, view_coverage_path

    _require_subject(subject)
    root = _resolve_root(path, dataset_root)
    store = read_json(view_coverage_path(root), default={})
    if not isinstance(store, dict):
        store = {}
    bucket = store.get(status_bucket(subject, date))
    record = bucket.get(Path(path).name) if isinstance(bucket, dict) else None
    return {"coverage": record}


class CoveragePayload(BaseModel):
    """One coverage post from the browser: the session's accumulated served-at-native and
    swept cell lists (either may be empty; the server union-merges, so resending is
    harmless), with the grid they were accumulated against and the viewing context they
    were served under (bands, stretch, stats_source, base_served_size, display_bounds,
    and working_scale_bar as ``{value, source}`` where the source states how the bar was
    measured)."""

    image_path: str
    subject: Optional[str] = None
    date: Optional[str] = None
    dataset_root: Optional[str] = None
    grid: dict
    cells_served_at_native: list[str] = []
    cells_swept: list[str] = []
    viewing: dict = {}


@router.post("")
def post_coverage(payload: CoveragePayload) -> dict:
    """Merge a coverage delta into the per-image record.

    Union-merge when the stored record's grid matches the posted one; a mismatched grid
    replaces the record wholesale and the response flags it (``replaced``), since the
    record carries the grid it was accumulated against precisely so a derivation change
    can never silently misread old cell names. Cell names are validated against the
    posted grid's own cells; unknown names are refused. ``date`` must be explicit:
    a non-dated dataset passes ``null``, so an image under a date bucket can never
    silently land in the dateless one.
    """
    from tcip_mcp.dataset_layout import status_bucket, view_coverage_path
    from tcip_mcp.pipelines.reference_grid import reference_cells

    subject = _require_subject(payload.subject)
    if "date" not in payload.model_fields_set:
        raise HTTPException(400, "view coverage is bucketed by subject and date; pass date "
                                 "(null for a non-dated dataset)")
    root = _resolve_root(payload.image_path, payload.dataset_root)
    grid = _normalized_grid(payload.grid)
    valid_names = {c.name for c in reference_cells(
        grid["width"], grid["height"], grid["tile_size"], grid["overlap"], clamp=True)}
    unknown = sorted(
        (set(payload.cells_served_at_native) | set(payload.cells_swept)) - valid_names)
    if unknown:
        raise HTTPException(
            400, f"cells not in this grid: {unknown}; the grid has {len(valid_names)} cells")

    bucket = status_bucket(subject, payload.date)
    image_name = Path(payload.image_path).name
    path = view_coverage_path(root)
    replaced = False
    with file_transaction(path):
        store = read_json(path, default={})
        if not isinstance(store, dict):
            store = {}
        records = store.setdefault(bucket, {})
        if not isinstance(records, dict):
            records = store[bucket] = {}
        existing = records.get(image_name)
        served = set(payload.cells_served_at_native)
        swept = set(payload.cells_swept)
        if isinstance(existing, dict) and existing.get("grid") == grid:
            served |= set(existing.get("cells_served_at_native") or [])
            swept |= set(existing.get("cells_swept") or [])
        else:
            replaced = existing is not None
        records[image_name] = {
            "grid": grid,
            "cells_served_at_native": sorted(served),
            "cells_swept": sorted(swept),
            "viewing": payload.viewing,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(path, store)

    audit_key = (root, bucket, image_name)
    # A replaced grid re-audits even within one process: the record's identity changed.
    if audit_key not in _audited_coverage_keys or replaced:
        _audited_coverage_keys.add(audit_key)
        _audit_dataset_write(
            root, "gui_view_coverage",
            {"image_name": image_name, "subject": subject, "date": payload.date,
             "grid": grid},
        )
    return {
        "status": "ok",
        "replaced": replaced,
        "cells_served_at_native": len(served),
        "cells_swept": len(swept),
        "total_cells": grid["cols"] * grid["rows"],
    }


# Region completeness gates a scientific claim rather than warning advisory, so (unlike view
# coverage above) it is written with classes.py's image-status discipline (file_transaction + atomic_write_json + _audit_dataset_write).


class CompletenessTogglePayload(BaseModel):
    """One double-click on the minimap: toggle one cell's completeness for a subject."""

    image_path: str
    subject: str
    dataset_root: Optional[str] = None
    grid: dict
    cell: str
    # GUI-set identity; stamped as attested_by ("user:<name>"), mirroring annotate.py/review.py.
    user: Optional[str] = None


@router.get("/completeness")
def get_completeness(
    path: str = Query(..., description="Absolute path to the image file"),
    dataset_root: Optional[str] = Query(None),
) -> dict:
    """Every subject's region-completeness record for the raster at ``path`` (its own stem), not
    just one subject's: lets the minimap render a cell attested complete for a subject other than
    the one currently active, distinguishably from the active subject's own attestations.

    Each record carries ``stale_cells``: attested cells whose annotation content has changed since
    attestation (:mod:`tcip_mcp.pipelines.region_completeness`), recomputed fresh on every read so
    a stale attestation is never served as if it still held.
    """
    from tcip_mcp.dataset_layout import (
        normalize_region_completeness_store,
        region_completeness_digest_path,
        region_completeness_path,
    )
    from tcip_mcp.pipelines.region_completeness import stale_cells

    root = _resolve_root(path, dataset_root)
    stem = Path(path).stem
    store = normalize_region_completeness_store(
        read_json(region_completeness_path(root), default={}))
    digests = read_json(region_completeness_digest_path(root), default={})
    if not isinstance(digests, dict):
        digests = {}

    by_subject: dict[str, dict] = {}
    for bucket, record in store.items():
        if record.get("stem") != stem:
            continue
        subject = record.get("subject")
        if not isinstance(subject, str) or not subject:
            continue
        stamped = digests.get(bucket)
        stale = stale_cells(root, record, stamped if isinstance(stamped, dict) else {}, subject)
        by_subject[subject] = {**record, "stale_cells": stale}
    return {"by_subject": by_subject}


@router.post("/completeness")
def post_completeness(payload: CompletenessTogglePayload) -> dict:
    """Toggle one cell's completeness for a subject: not-complete -> complete (stamping the
    cell's current annotation-content digest) or complete -> not-complete (clearing its stamp).

    Cell names are validated against the posted grid's own cells, same as ``post_coverage``. A
    stored record whose grid disagrees with the posted one replaces wholesale (cells and digest
    stamps alike), rather than trusting a same-named cell across two different lattices.
    """
    from tcip_annotation.json_io import read_annotations

    from tcip_mcp.dataset_layout import (
        annotation_path,
        normalize_region_completeness_store,
        parse_image_path,
        region_completeness_digest_path,
        region_completeness_path,
        status_bucket,
    )
    from tcip_mcp.pipelines.reference_grid import reference_cells
    from tcip_mcp.pipelines.region_completeness import cell_annotation_digest
    from tcip_web.identity import resolve_user, user_id

    subject = _require_completeness_subject(payload.subject)
    root = _resolve_root(payload.image_path, payload.dataset_root)
    try:
        _root_from_image, date, stem = parse_image_path(payload.image_path)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    grid = _normalized_grid(payload.grid)
    cells_by_name = {c.name: c for c in reference_cells(
        grid["width"], grid["height"], grid["tile_size"], grid["overlap"], clamp=True)}
    if payload.cell not in cells_by_name:
        raise HTTPException(
            400, f"cell not in this grid: {payload.cell!r}; the grid has {len(cells_by_name)} "
                 f"cells")

    bucket = status_bucket(subject, stem)
    completeness_path = region_completeness_path(root)
    digest_path = region_completeness_digest_path(root)
    author = user_id(resolve_user(payload.user))
    now_iso = datetime.now(timezone.utc).isoformat()

    # The digest write nests inside the completeness-store transaction, before the store's own
    # write: stale_cells fails closed on a missing digest, so it must exist before an attestation.
    with file_transaction(completeness_path):
        store = normalize_region_completeness_store(read_json(completeness_path, default={}))
        existing = store.get(bucket)
        grid_matches = existing is not None and existing.get("grid") == grid
        cells_complete = set(existing.get("cells_complete") or []) if grid_matches else set()
        replaced = existing is not None and not grid_matches
        complete = payload.cell not in cells_complete
        if complete:
            cells_complete.add(payload.cell)
        else:
            cells_complete.discard(payload.cell)
        with file_transaction(digest_path):
            digests = read_json(digest_path, default={})
            if not isinstance(digests, dict):
                digests = {}
            bucket_digests = digests.get(bucket)
            bucket_digests = dict(bucket_digests) if isinstance(bucket_digests, dict) else {}
            if replaced:
                bucket_digests = {}
            if complete:
                label_path = annotation_path(root, date, stem)
                annotations = read_annotations(str(label_path)) if label_path.is_file() else []
                bucket_digests[payload.cell] = cell_annotation_digest(
                    annotations, subject, cells_by_name[payload.cell])
            else:
                bucket_digests.pop(payload.cell, None)
            digests[bucket] = bucket_digests
            atomic_write_json(digest_path, digests)

        store[bucket] = {
            "grid": grid,
            "cells_complete": sorted(cells_complete),
            "attested_by": author,
            "attested_at": now_iso,
            "stem": stem,
            "date": date,
            "subject": subject,
        }
        atomic_write_json(completeness_path, store)

    _audit_dataset_write(
        root, "gui_set_region_completeness",
        {"image_name": Path(payload.image_path).name, "subject": subject, "cell": payload.cell,
         "complete": complete, "stem": stem, "date": date},
    )
    return {"status": "ok", "complete": complete, "cells_complete": sorted(cells_complete)}

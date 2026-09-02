"""View-coverage routes: the reference grid over a raster and the per-image record of
two per-cell facts: which cells were served to the browser at native resolution (a
delivery fact) and which cells were swept in the viewport at or above the breeder's own
working scale (a sweep fact). Neither is an attention claim. This "sweep" is the viewport-scan
fact alone, distinct from both a persisted calibration curve and an HPO run's own sweep.

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
from pydantic import BaseModel, ConfigDict, ValidationError

import tcip_store

from tcip_web.paths import assert_path_allowed
from tcip_web.routes._coverage_models import CoverageRecord, CoverageViewing, GridGeometry
from tcip_web.routes.classes import _audit_dataset_write, _guard_dataset_root
from tcip_web.routes.images import _checked

router = APIRouter(prefix="/api/coverage", tags=["coverage"])

# One audit entry per image per backend process: the first coverage write for an image
# audits, the debounced merges that follow coalesce into it.
_audited_coverage_keys: set[tuple[str, str, str]] = set()

_CONFORM_HINT = (
    "run scripts/conform_view_coverage_viewing.py against this dataset to bring it to the "
    "current shape (--plan first to preview the change)"
)


def _validated_record(image_name: str, record: dict) -> None:
    """Refuse rather than serve or merge into a stored record that no longer validates as
    ``CoverageRecord``, naming the conform script instead of silently coercing or dropping it."""
    try:
        CoverageRecord.model_validate(record)
    except ValidationError as exc:
        raise HTTPException(
            400,
            f"{image_name}'s stored view-coverage record does not validate against the "
            f"current shape: {exc}; {_CONFORM_HINT}",
        ) from exc


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


def _grid_for_raster(src: Path, tile_size: int | None) -> tuple[dict, str]:
    """The reference-grid geometry for the raster at ``src`` and the one-line explanation of how
    its tile size was chosen, shared by ``get_grid`` and ``get_completeness`` (the latter's own
    ``annotation_counts`` field) so the grid a breeder sees and the grid a saved-annotation count
    is binned against can never be two different lattices.

    Native dims come off the raster header (``image_dimensions``), never a decode. Raises
    ``tcip_mcp.pipelines.data.band_groups.BandGroupIncomplete`` for a band group missing a
    member; ``get_grid`` turns that into a 409, ``get_completeness`` into its ``counts_error``
    field rather than blocking the read.
    """
    from tcip_mcp.pipelines.image_utils import image_dimensions, resolve_image_source
    from tcip_mcp.pipelines.raster_source import is_georeferenced, opens_windowed
    from tcip_mcp.pipelines.reference_grid import (
        derive_coverage_tile_size,
        derive_large_raster_grid_tile_size,
        grid_geometry,
    )

    source = resolve_image_source(src.parent, src.stem)
    width, height = image_dimensions(source)
    # A stitched orthomosaic carries real per-pixel georeferencing; an ordinary drone/ground
    # capture (any pixel size) carries at most a single EXIF GPS point, never that.
    if tile_size is not None:
        edge = tile_size
        derivation = f"a chosen cell edge of {tile_size} px"
    elif opens_windowed(source, 3) and is_georeferenced(source):
        edge = derive_large_raster_grid_tile_size(width, height)
        derivation = "the long edge in 16 equal divisions"
    else:
        edge = derive_coverage_tile_size(width, height)
        derivation = "cells sized to one full-resolution screenful"
    return grid_geometry(width, height, edge, 0.0), derivation


@router.get("/grid")
def get_grid(
    path: str = Query(..., description="Absolute path to the image file"),
    tile_size: int | None = Query(None, ge=1, description="Cell edge in native pixels; "
                                  "omitted derives the coverage lattice edge"),
    overlap: float = Query(0.0),
) -> dict:
    """The reference grid over ``path``'s native frame: geometry plus the full cell list and the
    ``derivation`` line naming how the tile size was chosen ("the long edge in 16 equal
    divisions", "cells sized to one full-resolution screenful", or "a chosen cell edge of <n>
    px" for an explicit ``tile_size``); a cell is never a training tile, a different lattice with
    a different origin and (by default) a training overlap.

    ``overlap`` other than 0 is refused: the coverage record's exact-partition contract
    puts every native pixel in exactly one cell, which overlapping cells break. The
    frontend consumes these cells verbatim and never re-derives them; ``derivation`` is not
    part of ``GridGeometry`` and is stripped back off before a grid round-trips into a
    coverage or completeness payload.
    """
    from tcip_mcp.pipelines.data.band_groups import BandGroupIncomplete
    from tcip_mcp.pipelines.reference_grid import reference_cells

    src = _checked(path)
    if overlap != 0.0:
        raise HTTPException(
            400,
            f"the coverage grid requires overlap 0: its exact-partition contract puts every "
            f"native pixel in exactly one cell, which overlapping cells break; got {overlap}")
    try:
        geometry, derivation = _grid_for_raster(src, tile_size)
    except BandGroupIncomplete as exc:
        raise HTTPException(409, str(exc)) from exc
    cells = reference_cells(
        geometry["width"], geometry["height"], geometry["tile_size"], 0.0, clamp=True)
    return {
        **geometry,
        "derivation": derivation,
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
    ``{"coverage": null}`` when nothing has been recorded. A record stored in an old shape refuses
    (400), naming ``scripts/conform_view_coverage_viewing.py`` rather than serving it as-is."""
    from tcip_mcp.dataset_layout import status_bucket, view_coverage_key

    _require_subject(subject)
    root = _resolve_root(path, dataset_root)
    store = tcip_store.read(view_coverage_key(root), default={})
    if not isinstance(store, dict):
        store = {}
    bucket = store.get(status_bucket(subject, date))
    image_name = Path(path).name
    record = bucket.get(image_name) if isinstance(bucket, dict) else None
    if isinstance(record, dict):
        _validated_record(image_name, record)
    return {"coverage": record}


class CoveragePayload(BaseModel):
    """One coverage post from the browser: the session's accumulated served-at-native and
    swept cell lists (either may be empty; the server union-merges, so resending is
    harmless), with the grid they were accumulated against and the viewing context they
    were served under. ``date`` and ``viewing`` carry no default: a non-dated dataset must
    still pass ``date: null`` explicitly, so an image under a date bucket can never silently
    land in the dateless one, and every post states the viewing context it was served under
    rather than the model quietly filling in one no browser ever chose."""

    model_config = ConfigDict(extra="forbid")

    image_path: str
    subject: Optional[str] = None
    date: Optional[str]
    dataset_root: Optional[str] = None
    grid: GridGeometry
    cells_served_at_native: list[str] = []
    cells_swept: list[str] = []
    viewing: CoverageViewing


@router.post("")
def post_coverage(payload: CoveragePayload) -> dict:
    """Merge a coverage delta into the per-image record.

    Union-merge when the stored record's grid matches the posted one; a mismatched grid
    replaces the record wholesale and the response flags it (``replaced``), since the
    record carries the grid it was accumulated against precisely so a derivation change
    can never silently misread old cell names. Cell names are validated against the
    posted grid's own cells; unknown names are refused. On the merge path the stored
    record is validated against ``CoverageRecord`` before its cells are folded in, naming
    the conform script when it no longer holds that shape; the replace path overwrites it
    wholesale, since there is nothing to merge into.
    """
    from tcip_mcp.dataset_layout import status_bucket, view_coverage_key
    from tcip_mcp.pipelines.reference_grid import reference_cells

    subject = _require_subject(payload.subject)
    root = _resolve_root(payload.image_path, payload.dataset_root)
    grid = payload.grid.model_dump()
    valid_names = {c.name for c in reference_cells(
        grid["width"], grid["height"], grid["tile_size"], grid["overlap"], clamp=True)}
    unknown = sorted(
        (set(payload.cells_served_at_native) | set(payload.cells_swept)) - valid_names)
    if unknown:
        raise HTTPException(
            400, f"cells not in this grid: {unknown}; the grid has {len(valid_names)} cells")

    bucket = status_bucket(subject, payload.date)
    image_name = Path(payload.image_path).name
    key = view_coverage_key(root)
    replaced = False
    with tcip_store.transaction(key) as txn:
        store = txn.read(key, default={})
        if not isinstance(store, dict):
            store = {}
        records = store.setdefault(bucket, {})
        if not isinstance(records, dict):
            records = store[bucket] = {}
        existing = records.get(image_name)
        served = set(payload.cells_served_at_native)
        swept = set(payload.cells_swept)
        if isinstance(existing, dict) and existing.get("grid") == grid:
            _validated_record(image_name, existing)
            served |= set(existing.get("cells_served_at_native") or [])
            swept |= set(existing.get("cells_swept") or [])
        else:
            replaced = existing is not None
        records[image_name] = {
            "grid": grid,
            "cells_served_at_native": sorted(served),
            "cells_swept": sorted(swept),
            "viewing": payload.viewing.model_dump(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        txn.write(key, store)

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
# coverage above) every write is audited as well as transacted.


class CompletenessSetPayload(BaseModel):
    """One explicit attestation write: set one cell's completeness for a subject to
    ``complete``, never a toggle a caller infers from the stored state. ``complete=True``
    stamps the cell's current annotation-content digest (whether or not the cell was already
    complete, so a re-attest restamps and clears staleness); ``complete=False`` clears the
    stamp. The control that posts this states the direction in its own label, so the write it
    performs always matches what the label promised."""

    image_path: str
    subject: str
    dataset_root: Optional[str] = None
    grid: GridGeometry
    cell: str
    complete: bool
    # GUI-set identity; stamped as attested_by ("user:<name>"), mirroring annotate.py/review.py.
    user: Optional[str] = None


@router.get("/completeness")
def get_completeness(
    path: str = Query(..., description="Absolute path to the image file"),
    dataset_root: Optional[str] = Query(None),
) -> dict:
    """Every subject's region-completeness record for the raster at ``path`` (its own stem), not
    just one subject's: lets the coverage overlay render a cell attested complete for a subject
    other than the one currently active, distinguishably from the active subject's own
    attestations.

    Each record carries ``stale_cells``: attested cells whose annotation content has changed since
    attestation (:mod:`tcip_mcp.pipelines.region_completeness`), recomputed fresh on every read so
    a stale attestation is never served as if it still held. A label document that will not decode
    for a subject that already holds a record here still refuses (400): staleness cannot be
    computed without it.

    ``annotation_counts`` (subject -> cell name -> count) is every subject's saved-annotation count
    per cell, one entry per subject present in the raster's label file; the hook selects the active
    subject's entry client-side. ``counts_grid`` is the six-field lattice the counts were binned
    against (:func:`_grid_for_raster`, the same derivation ``get_grid`` uses), served beside
    ``annotation_counts`` rather than folded into it: a count belongs to the grid, not to any one
    attestation record. The whole counts computation -- deriving the grid, reading the label file,
    binning -- is best-effort beside ``by_subject``, which needs no raster at all: a band group
    missing a member, a raster gone from disk, or any other raster-read failure reports the reason
    in ``counts_error`` (with ``annotation_counts`` empty and ``counts_grid`` null) rather than
    blanking the read.
    """
    from tcip_annotation.json_io import UnreadableLabelDocument, read_annotations
    from tcip_mcp.dataset_layout import (
        annotation_path,
        normalize_region_completeness_store,
        parse_image_path,
        region_completeness_digest_key,
        region_completeness_key,
    )
    from tcip_mcp.pipelines.data.band_groups import BandGroupIncomplete
    from tcip_mcp.pipelines.reference_grid import reference_cells
    from tcip_mcp.pipelines.region_completeness import annotation_counts_by_cell, stale_cells

    try:
        src = assert_path_allowed(path)
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc

    root = _resolve_root(path, dataset_root)
    stem = Path(path).stem
    store = normalize_region_completeness_store(
        tcip_store.read(region_completeness_key(root), default={}))
    digests = tcip_store.read(region_completeness_digest_key(root), default={})
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
        try:
            stale = stale_cells(root, record, stamped if isinstance(stamped, dict) else {}, subject)
        except UnreadableLabelDocument as exc:
            raise HTTPException(400, str(exc)) from exc
        by_subject[subject] = {**record, "stale_cells": stale}

    annotation_counts: dict[str, dict[str, int]] = {}
    counts_grid: Optional[dict] = None
    counts_error: Optional[str] = None
    try:
        geometry, _derivation = _grid_for_raster(src, None)
        _root_from_image, date, image_stem = parse_image_path(path)
        label_path = annotation_path(root, date, image_stem)
        annotations = read_annotations(str(label_path)) if label_path.is_file() else []
        cells = reference_cells(
            geometry["width"], geometry["height"], geometry["tile_size"], 0.0, clamp=True)
        annotation_counts = annotation_counts_by_cell(annotations, cells, geometry["tile_size"])
        counts_grid = geometry
    except (BandGroupIncomplete, FileNotFoundError, OSError, ValueError,
            UnreadableLabelDocument) as exc:
        counts_error = str(exc)

    return {"by_subject": by_subject, "annotation_counts": annotation_counts,
            "counts_grid": counts_grid, "counts_error": counts_error}


@router.post("/completeness")
def post_completeness(payload: CompletenessSetPayload) -> dict:
    """Set one cell's completeness for a subject to ``payload.complete``: never a toggle, so a
    control's label ("Attest", "Unattest", "Re-attest") always states the write it performs.
    ``complete=True`` stamps the cell's current annotation-content digest, whether or not the
    cell was already complete (a re-attest restamps and clears staleness); ``complete=False``
    clears the stamp.

    Cell names are validated against the posted grid's own cells, same as ``post_coverage``. A
    stored record whose grid disagrees with the posted one replaces wholesale (cells and digest
    stamps alike), rather than trusting a same-named cell across two different lattices; the
    response and the audit line both carry the discarded record's grid and cells (``replaced``,
    null when nothing was discarded), the way ``post_coverage`` states its own replacement.
    """
    from tcip_annotation.json_io import UnreadableLabelDocument, read_annotations

    from tcip_mcp.dataset_layout import (
        annotation_path,
        normalize_region_completeness_store,
        parse_image_path,
        region_completeness_digest_key,
        region_completeness_key,
        status_bucket,
        unreadable_completeness_entries,
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
    grid = payload.grid.model_dump()
    cells_by_name = {c.name: c for c in reference_cells(
        grid["width"], grid["height"], grid["tile_size"], grid["overlap"], clamp=True)}
    if payload.cell not in cells_by_name:
        raise HTTPException(
            400, f"cell not in this grid: {payload.cell!r}; the grid has {len(cells_by_name)} "
                 f"cells")

    bucket = status_bucket(subject, stem)
    completeness_key = region_completeness_key(root)
    digest_key = region_completeness_digest_key(root)
    author = user_id(resolve_user(payload.user))
    now_iso = datetime.now(timezone.utc).isoformat()
    complete = payload.complete

    # Digest key named first, so it is applied first: stale_cells fails closed on a missing
    # digest, so a stamp must never land after the attestation that points at it.
    with tcip_store.transaction(digest_key, completeness_key) as txn:
        raw_store = txn.read(completeness_key, default={})
        if not isinstance(raw_store, dict):
            raise HTTPException(
                400,
                f"the region-completeness store under {root} is a "
                f"{type(raw_store).__name__}, not a dict; refusing to write over it")
        unreadable = unreadable_completeness_entries(raw_store)
        if unreadable:
            raise HTTPException(
                400,
                f"the region-completeness store under {root} holds {len(unreadable)} entries in a "
                f"shape this reader does not recognize, starting with {unreadable[:3]}; merging a "
                f"write into it would drop them. Conform the store to the recognized record shape "
                f"first")
        store = normalize_region_completeness_store(raw_store)
        existing = store.get(bucket)
        grid_matches = existing is not None and existing.get("grid") == grid
        cells_complete = set(existing.get("cells_complete") or []) if grid_matches else set()

        if not complete and payload.cell not in cells_complete:
            # Nothing to unattest here, whether from no record, a different lattice, or a
            # same-lattice record that never held this cell: an idempotent no-op, not a write.
            return {"status": "ok", "complete": False, "cells_complete": sorted(cells_complete),
                    "replaced": None}

        replaced = existing is not None and not grid_matches
        replaced_info = None
        if replaced and isinstance(existing, dict):
            replaced_info = {"grid": existing.get("grid"),
                              "cells_complete": sorted(existing.get("cells_complete") or [])}
        if complete:
            cells_complete.add(payload.cell)
        else:
            cells_complete.discard(payload.cell)
        digests = txn.read(digest_key, default={})
        if not isinstance(digests, dict):
            digests = {}
        bucket_digests = digests.get(bucket)
        bucket_digests = dict(bucket_digests) if isinstance(bucket_digests, dict) else {}
        if replaced:
            bucket_digests = {}
        if complete:
            label_path = annotation_path(root, date, stem)
            try:
                annotations = read_annotations(str(label_path)) if label_path.is_file() else []
            except UnreadableLabelDocument as exc:
                raise HTTPException(400, str(exc)) from exc
            bucket_digests[payload.cell] = cell_annotation_digest(
                annotations, subject, cells_by_name[payload.cell])
        else:
            bucket_digests.pop(payload.cell, None)
        digests[bucket] = bucket_digests
        txn.write(digest_key, digests)

        store[bucket] = {
            "grid": grid,
            "cells_complete": sorted(cells_complete),
            "attested_by": author,
            "attested_at": now_iso,
            "stem": stem,
            "date": date,
            "subject": subject,
        }
        txn.write(completeness_key, store)

    _audit_dataset_write(
        root, "gui_set_region_completeness",
        {"image_name": Path(payload.image_path).name, "subject": subject, "cell": payload.cell,
         "complete": complete, "stem": stem, "date": date, "replaced": replaced_info},
    )
    return {"status": "ok", "complete": complete, "cells_complete": sorted(cells_complete),
            "replaced": replaced_info}

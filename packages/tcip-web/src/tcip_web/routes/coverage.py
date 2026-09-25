"""View-coverage routes: the coverage lattice over a raster, the grid-zoom setting it is derived
from, the per-image view-coverage record, and region-completeness attestations.

The view-coverage record (``view_coverage.json``, ``dataset_layout.view_coverage_path``, bucketed
by ``status_bucket(subject, date)`` then image name) holds two per-cell facts: which cells were
served to the browser at native resolution, and the tightest scale at which each cell has sat fully
on screen (``cells_seen_at_scale``, a bound). Neither is an attention claim; whether a seen cell
counts as "swept" is never stored. The store is advisory: training never reads it, and unswept
cells warn rather than block a Complete.

A coverage cell is one screenful of native pixels at the breeder's set zoom
(``coverage_grid_zoom.json``, one entry per subject;
:func:`tcip_mcp.pipelines.reference_grid.derive_lattice_tile_size`). There is no default zoom: a
subject with none set has no lattice until the breeder states one (``POST
/api/coverage/grid_zoom``). Region serving uses its own display-derived tiling
(:func:`tcip_mcp.pipelines.reference_grid.derive_serving_tile_size`), independent of the lattice.

A region-completeness attestation ("I found every instance of this subject in these cells") is
written create-checked like the image-status store; a stale attestation (a cell's annotation
content edited or deleted since it was attested) is detected on every read. An attestation records
its own scale provenance (``cells_attested_view``). See ``dataset_layout.region_completeness_path``
and ``tcip_mcp.pipelines.region_completeness``.

Every write that changes a record is audited after its transaction commits, through
``record_event_or_raise``; a failed append answers a 500 marked ``AUDIT_ENTRY_NOT_WRITTEN``: the
change landed, its line did not, and a retry of the same payload changes nothing and records
nothing.
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
from tcip_web.routes.audit_gap import AUDIT_ENTRY_NOT_WRITTEN
from tcip_web.routes.subjects import _guard_dataset_root
from tcip_web.routes.images import _checked

router = APIRouter(prefix="/api/coverage", tags=["coverage"])

# The stable marker post_coverage's 409 body carries for a grid mismatch with no replace flag;
# coverageTracker.ts reads it to set the tracker's replace hold, never to retry or drop the push.
COVERAGE_LATTICE_MISMATCH = "coverage_lattice_mismatch"


def _require_date_matches_path(image_path: str, date: Optional[str]) -> None:
    """Refuse a posted ``date`` that disagrees with the date :func:`parse_image_path` reads off
    ``image_path`` itself (an explicit ``null`` against a dated image, or a date against a dateless
    one). A path outside the recognized ``<root>/images/[<date>/]<stem>`` tree is not compared.
    """
    from tcip_mcp.dataset_layout import parse_image_path

    try:
        _root, parsed_date, _stem = parse_image_path(image_path)
    except ValueError:
        return
    if parsed_date != date:
        raise HTTPException(
            400,
            f"date {date!r} disagrees with the date {parsed_date!r} parse_image_path reads off "
            f"{image_path!r}; the two must agree so one image's coverage can never split across "
            f"two buckets")


def _validated_record(image_name: str, record: object) -> None:
    """Refuse (400) a stored record of any type that does not validate as ``CoverageRecord``."""
    try:
        CoverageRecord.model_validate(record)
    except ValidationError as exc:
        raise HTTPException(
            400,
            f"{image_name}'s stored view-coverage record does not validate against the "
            f"current shape: {exc}",
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


def _audit_or_answer_500(tool: str, arguments: dict, root: str) -> None:
    """Emit one audit line through ``record_event_or_raise`` after a write has already committed,
    and turn a failed append into the marked 500 (``AUDIT_ENTRY_NOT_WRITTEN``).
    """
    from tcip_mcp.audit import AuditEntryNotWritten, record_event_or_raise

    try:
        record_event_or_raise(tool, arguments, source="gui", scope=root)
    except AuditEntryNotWritten as exc:
        raise HTTPException(500, {"error": AUDIT_ENTRY_NOT_WRITTEN, "message": str(exc)}) from exc


def _grid_for_raster(src: Path, tile_size: int | None) -> tuple[dict, str]:
    """The display-derived reference-grid geometry for the raster at ``src`` and the one-line
    explanation of how its tile size was chosen. ``tile_size`` explicit bypasses the derivation;
    this function never derives the coverage lattice (:func:`_lattice_for_raster`).

    Native dims come off the raster header (``image_dimensions``), never a decode. Raises
    ``tcip_mcp.pipelines.data.band_groups.BandGroupIncomplete`` for a band group missing a member.
    """
    from tcip_mcp.pipelines.image_utils import image_dimensions, resolve_image_source
    from tcip_mcp.pipelines.reference_grid import derive_serving_tile_size, grid_geometry

    source = resolve_image_source(src.parent, src.stem)
    width, height = image_dimensions(source)
    if tile_size is not None:
        edge = tile_size
        derivation = f"a chosen cell edge of {tile_size} px"
    else:
        edge = derive_serving_tile_size(width, height)
        derivation = "cells sized to one full-resolution screenful"
    return grid_geometry(width, height, edge, 0.0), derivation


def _rendered_grid(geometry: dict, derivation: str) -> dict:
    """One grid response block: ``get_grid``'s own top-level shape, minus its
    ``cells``/``derivation``.
    """
    from tcip_mcp.pipelines.reference_grid import reference_cells

    cells = reference_cells(
        geometry["width"], geometry["height"], geometry["tile_size"], 0.0, clamp=True)
    return {
        **geometry,
        "derivation": derivation,
        "cells": [{"name": c.name, "x0": c.x0, "y0": c.y0, "x1": c.x1, "y1": c.y1}
                  for c in cells],
    }


def _subject_zoom(root: str, subject: str) -> Optional[dict]:
    """``subject``'s stored grid-zoom entry (``{zoom, set_by, set_at}``) for the dataset at
    ``root``, or ``None`` when the breeder has not set one. Refuses a store that will not decode or
    an entry whose ``zoom`` is not a positive number.
    """
    from tcip_mcp.dataset_layout import coverage_grid_zoom_key

    try:
        store = tcip_store.read(coverage_grid_zoom_key(root), default={})
    except tcip_store.DecodeError as exc:
        raise HTTPException(
            400, f"{subject}'s coverage grid zoom store under {root} will not decode: {exc}"
        ) from exc
    if not isinstance(store, dict):
        return None
    entry = store.get(subject)
    if not isinstance(entry, dict):
        return None
    zoom = entry.get("zoom")
    if isinstance(zoom, bool) or not isinstance(zoom, (int, float)) or zoom <= 0:
        raise HTTPException(
            400,
            f"{subject}'s stored grid-zoom entry does not carry a positive zoom: {entry!r}")
    return entry


def _no_zoom_reason(subject: str) -> str:
    """The sentence naming why ``subject`` has no coverage lattice yet."""
    return f"set the grid zoom to derive a coverage lattice for {subject}"


def _working_scale_of(entry: Optional[dict]) -> Optional[dict]:
    """``entry`` (a stored grid-zoom record) rendered as the ``WorkingScale`` shape (``{zoom,
    source}``), or ``None`` when there is no entry.
    """
    if entry is None:
        return None
    return {"zoom": entry.get("zoom"),
            "source": f"set by {entry.get('set_by')} at {entry.get('set_at')}"}


def _lattice_for_raster(
    src: Path, viewport_w: int, viewport_h: int, zoom: float,
) -> tuple[dict, str]:
    """The coverage-lattice geometry for the raster at ``src`` at ``zoom``, one screenful of native
    pixels at that zoom on a ``viewport_w`` x ``viewport_h`` canvas host
    (:func:`tcip_mcp.pipelines.reference_grid.derive_lattice_tile_size`).
    """
    from tcip_mcp.pipelines.image_utils import image_dimensions, resolve_image_source
    from tcip_mcp.pipelines.reference_grid import derive_lattice_tile_size, grid_geometry

    source = resolve_image_source(src.parent, src.stem)
    width, height = image_dimensions(source)
    edge = derive_lattice_tile_size(viewport_w, viewport_h, zoom)
    return grid_geometry(width, height, edge, 0.0), f"one screenful at {zoom}x zoom"


@router.get("/grid")
def get_grid(
    path: str = Query(..., description="Absolute path to the image file"),
    subject: str | None = Query(
        None, description="Whose set grid zoom derives the coverage lattice; omitted when only "
                          "an explicit tile_size or the serving grid is wanted"),
    date: str | None = Query(None),
    dataset_root: str | None = Query(None),
    viewport_w: int | None = Query(
        None, ge=1, description="Canvas host width at fetch time, native-pixel lattice sizing"),
    viewport_h: int | None = Query(
        None, ge=1, description="Canvas host height at fetch time, native-pixel lattice sizing"),
    tile_size: int | None = Query(
        None, ge=1, description="Explicit cell edge in native pixels; bypasses the set-zoom "
                                "lattice and the already-worked lattice alike, for a caller "
                                "(the agent) that wants a specific grid regardless of either"),
    rederive: bool = Query(
        False, description="Ignore this image's already-worked lattice and derive fresh at the "
                           "current zoom, even when a view_coverage record exists"),
    overlap: float = Query(0.0),
) -> dict:
    """The coverage lattice over ``path`` at ``subject``'s set grid zoom, plus the zoom-independent
    region-serving grid (``serving``) and, when a coverage lattice cannot be derived, the one-line
    ``reason`` why.

    With ``tile_size`` given, ``grid`` is that explicit lattice regardless of any set zoom or
    stored record. Without it: when a ``view_coverage`` record already exists for this image and
    ``subject`` and ``rederive`` is not set, ``grid`` is that record's own lattice (``derivation``
    "the lattice this image's coverage was recorded on"); ``fresh_derivation_differs`` then says
    whether the zoom currently in effect would derive a different tile size. Otherwise ``grid`` is
    derived fresh from ``subject``'s set zoom and the ``viewport_w``/``viewport_h`` the canvas host
    measured at fetch time; with no zoom set for ``subject``, or no viewport supplied yet, ``grid``
    is ``null`` and ``reason`` names why.

    ``serving`` is always the display-derived region-serving grid
    (:func:`tcip_mcp.pipelines.reference_grid.derive_serving_tile_size`).

    ``overlap`` other than 0 is refused: the coverage record puts every native pixel in exactly one
    cell. ``derivation`` is not part of ``GridGeometry``.
    """
    from tcip_mcp.dataset_layout import parse_image_path, status_bucket, view_coverage_key
    from tcip_mcp.pipelines.data.band_groups import BandGroupIncomplete
    from tcip_mcp.pipelines.image_utils import AmbiguousImageStem
    from tcip_mcp.pipelines.reference_grid import derive_lattice_tile_size

    src = _checked(path)
    if overlap != 0.0:
        raise HTTPException(
            400,
            f"the coverage grid requires overlap 0: its exact-partition contract puts every "
            f"native pixel in exactly one cell, which overlapping cells break; got {overlap}")

    try:
        serving_geometry, serving_derivation = _grid_for_raster(src, None)
    except BandGroupIncomplete as exc:
        raise HTTPException(409, str(exc)) from exc
    except AmbiguousImageStem as exc:
        raise HTTPException(400, str(exc)) from exc
    serving = _rendered_grid(serving_geometry, serving_derivation)

    if tile_size is not None:
        geometry, derivation = _grid_for_raster(src, tile_size)
        return {"grid": _rendered_grid(geometry, derivation), "reason": None,
                "fresh_derivation_differs": None, "serving": serving}

    if not subject:
        return {"grid": None,
                "reason": "the coverage lattice is scoped to a subject's own set grid zoom; "
                          "pass a subject",
                "fresh_derivation_differs": None, "serving": serving}

    root = _resolve_root(path, dataset_root)
    zoom_entry = _subject_zoom(root, subject)

    existing_record: Optional[dict] = None
    if not rederive:
        try:
            _root_from_image, parsed_date, _stem = parse_image_path(path)
        except ValueError:
            parsed_date = date
        store = tcip_store.read(view_coverage_key(root), default={})
        if isinstance(store, dict):
            bucket = store.get(status_bucket(subject, parsed_date))
            record = bucket.get(Path(path).name) if isinstance(bucket, dict) else None
            if isinstance(record, dict) and isinstance(record.get("grid"), dict):
                existing_record = record

    if existing_record is not None:
        grid = _rendered_grid(
            existing_record["grid"], "the lattice this image's coverage was recorded on")
        fresh_derivation_differs: Optional[bool] = None
        if zoom_entry is not None and viewport_w is not None and viewport_h is not None:
            fresh_edge = derive_lattice_tile_size(viewport_w, viewport_h, zoom_entry["zoom"])
            fresh_derivation_differs = fresh_edge != existing_record["grid"].get("tile_size")
        return {"grid": grid, "reason": None,
                "fresh_derivation_differs": fresh_derivation_differs, "serving": serving}

    if zoom_entry is None:
        return {"grid": None, "reason": _no_zoom_reason(subject),
                "fresh_derivation_differs": None, "serving": serving}
    if viewport_w is None or viewport_h is None:
        return {"grid": None,
                "reason": "the canvas host has not been measured yet; the coverage lattice is "
                          "derived from that measurement",
                "fresh_derivation_differs": None, "serving": serving}

    geometry, derivation = _lattice_for_raster(src, viewport_w, viewport_h, zoom_entry["zoom"])
    return {"grid": _rendered_grid(geometry, derivation), "reason": None,
            "fresh_derivation_differs": None, "serving": serving}


class GridZoomPayload(BaseModel):
    """One breeder-set grid zoom for one subject: screen pixels per native pixel, the same number
    the status bar shows as a percentage. ``zoom`` must be positive.
    """

    model_config = ConfigDict(extra="forbid")

    subject: str
    zoom: float
    dataset_root: Optional[str] = None
    user: Optional[str] = None


@router.post("/grid_zoom")
def post_grid_zoom(payload: GridZoomPayload) -> dict:
    """Set ``payload.subject``'s coverage-lattice zoom for this dataset, replacing any previous
    value. Never touches any stored coverage or completeness record.

    ``zoom`` at or below 0 is refused by name. A failed audit append answers the marked 500
    (``AUDIT_ENTRY_NOT_WRITTEN``) after the write has already committed.
    """
    from tcip_mcp.dataset_layout import coverage_grid_zoom_key
    from tcip_web.identity import resolve_user, user_id

    subject = _require_subject(payload.subject)
    if payload.zoom <= 0:
        raise HTTPException(
            400, f"zoom must be positive (screen px per native px), got {payload.zoom}")
    if not payload.dataset_root:
        raise HTTPException(400, "the coverage grid zoom is scoped to a dataset; pass "
                                 "dataset_root")
    root = _guard_dataset_root(payload.dataset_root)
    author = user_id(resolve_user(payload.user))
    now_iso = datetime.now(timezone.utc).isoformat()
    key = coverage_grid_zoom_key(root)

    with tcip_store.transaction(key) as txn:
        store = txn.read(key, default={})
        if not isinstance(store, dict):
            store = {}
        store = dict(store)
        store[subject] = {"zoom": payload.zoom, "set_by": author, "set_at": now_iso}
        txn.write(key, store)

    _audit_or_answer_500(
        "gui_set_coverage_grid_zoom",
        {"subject": subject, "zoom": payload.zoom, "set_by": author},
        root,
    )
    return {"status": "ok", "subject": subject, "zoom": payload.zoom, "set_by": author,
            "set_at": now_iso}


@router.get("")
def get_coverage(
    path: str = Query(..., description="Absolute path to the image file"),
    subject: str | None = Query(None),
    date: str | None = Query(None),
    dataset_root: str | None = Query(None),
) -> dict:
    """The stored coverage record for one image under one subject/date bucket, or
    ``{"coverage": null}`` when nothing has been recorded. A stored record that does not
    validate as ``CoverageRecord`` refuses (400)."""
    from tcip_mcp.dataset_layout import status_bucket, view_coverage_key

    subject = _require_subject(subject)
    root = _resolve_root(path, dataset_root)
    store = tcip_store.read(view_coverage_key(root), default={})
    image_name = Path(path).name
    record = store.get(status_bucket(subject, date), {}).get(image_name)
    if record is not None:
        _validated_record(image_name, record)
    return {"coverage": record}


class CoveragePayload(BaseModel):
    """One coverage post from the browser: the session's accumulated served-at-native cells and
    per-cell containment scales (either may be empty; the server merges, so resending is harmless),
    with the grid they were accumulated against and the viewing context they were served under.
    ``date`` and ``viewing`` carry no default: a non-dated dataset passes ``date: null``
    explicitly. ``replace`` confirms a grid mismatch's wholesale replace of the stored record; a
    first record needs no flag, and a mismatched grid without one refuses with 409.
    """

    model_config = ConfigDict(extra="forbid")

    image_path: str
    subject: Optional[str] = None
    date: Optional[str]
    dataset_root: Optional[str] = None
    grid: GridGeometry
    cells_served_at_native: list[str] = []
    cells_seen_at_scale: dict[str, float] = {}
    viewing: CoverageViewing
    replace: bool = False


@router.post("")
def post_coverage(payload: CoveragePayload) -> dict:
    """Merge a coverage delta into the per-image record.

    Merge when the stored record's grid matches the posted one: served cells union, and
    ``cells_seen_at_scale`` keeps the greater value per cell. A mismatched grid against an existing
    record refuses with 409 (``COVERAGE_LATTICE_MISMATCH``, naming the stored grid and its
    ``cells_seen`` count) unless ``payload.replace`` is true; with the flag the record is replaced
    wholesale and the response flags it (``replaced``), and the audit line carries
    ``replace_confirmed``. A first record needs no flag. Cell names are validated against the
    posted grid's own cells; unknown names are refused. On the merge path the stored record is
    validated against ``CoverageRecord`` before its cells are folded in.

    A write happens, and is audited, only when the merge actually changes something: a raised or
    newly-seen cell, a newly served cell, a changed ``viewing``, or a replace. The audit line is
    appended after the transaction commits; a failed append answers the marked 500.
    """
    from tcip_mcp.dataset_layout import status_bucket, view_coverage_key
    from tcip_mcp.pipelines.reference_grid import reference_cells

    subject = _require_subject(payload.subject)
    _require_date_matches_path(payload.image_path, payload.date)
    root = _resolve_root(payload.image_path, payload.dataset_root)
    grid = payload.grid.model_dump()
    valid_names = {c.name for c in reference_cells(
        grid["width"], grid["height"], grid["tile_size"], grid["overlap"], clamp=True)}
    unknown = sorted(
        (set(payload.cells_served_at_native) | set(payload.cells_seen_at_scale)) - valid_names)
    if unknown:
        raise HTTPException(
            400, f"cells not in this grid: {unknown}; the grid has {len(valid_names)} cells")

    bucket = status_bucket(subject, payload.date)
    image_name = Path(payload.image_path).name
    key = view_coverage_key(root)
    viewing_dict = payload.viewing.model_dump()

    with tcip_store.transaction(key) as txn:
        store = txn.read(key, default={})
        records = store.setdefault(bucket, {})
        existing = records.get(image_name)
        served = set(payload.cells_served_at_native)
        seen = dict(payload.cells_seen_at_scale)
        replaced = False
        served_added = sorted(served)
        seen_added = dict(seen)
        viewing_changed = True
        grid_matches = existing is not None and existing["grid"] == grid
        # Only a confirmed replace across a grid mismatch has nothing to merge into.
        if existing is not None and not (payload.replace and not grid_matches):
            _validated_record(image_name, existing)
        if not grid_matches and existing is not None and not payload.replace:
            stored_grid = existing["grid"]
            raise HTTPException(409, {
                "error": COVERAGE_LATTICE_MISMATCH,
                "message": (
                    f"{image_name}'s stored coverage record was accumulated against a grid "
                    f"this push's grid disagrees with; pass replace: true to discard it and "
                    f"record on the new lattice"),
                "stored_grid": {"cols": stored_grid["cols"], "rows": stored_grid["rows"],
                                 "tile_size": stored_grid["tile_size"]},
                "cells_seen": len(existing["cells_seen_at_scale"]),
            })
        if grid_matches:
            prior_served = set(existing["cells_served_at_native"])
            prior_seen = dict(existing["cells_seen_at_scale"])
            served_added = sorted(served - prior_served)
            served = served | prior_served
            merged_seen = dict(prior_seen)
            seen_added = {}
            for name, scale in seen.items():
                if scale > merged_seen.get(name, float("-inf")):
                    merged_seen[name] = scale
                    seen_added[name] = scale
            seen = merged_seen
            viewing_changed = existing["viewing"] != viewing_dict
        else:
            replaced = existing is not None

        changed = replaced or bool(served_added) or bool(seen_added) or viewing_changed
        record_body = {"cells_seen_at_scale": seen, "cells_served_at_native": sorted(served)}
        if not changed:
            return {
                "status": "ok",
                "replaced": False,
                "cells_served_at_native": len(served),
                "total_cells": grid["cols"] * grid["rows"],
                "record": record_body,
            }

        records[image_name] = {
            "grid": grid,
            "cells_served_at_native": sorted(served),
            "cells_seen_at_scale": seen,
            "viewing": viewing_dict,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        txn.write(key, store)

    # A log append cannot join a record transaction; see the docstring for what a failed
    # append means at this position, after the write above has already committed.
    _audit_or_answer_500(
        "gui_view_coverage",
        {"image_name": image_name, "subject": subject, "date": payload.date, "grid": grid,
         "cells_seen_added": seen_added, "cells_served_at_native_added": served_added,
         "viewing_changed": viewing_changed,
         "viewing": viewing_dict if viewing_changed else None, "replaced": replaced,
         "replace_confirmed": payload.replace},
        root,
    )

    return {
        "status": "ok",
        "replaced": replaced,
        "cells_served_at_native": len(served),
        "total_cells": grid["cols"] * grid["rows"],
        "record": record_body,
    }


# Region completeness gates a scientific claim: its write below audits through the same
# raising emitter post_coverage uses (_audit_or_answer_500), never the best-effort one.


class CompletenessSetPayload(BaseModel):
    """One explicit attestation write: set one cell's completeness for a subject to ``complete``.
    ``complete=True`` stamps the cell's current annotation-content digest (a re-attest restamps and
    clears staleness) and its scale provenance (``cells_attested_view`` on the stored record);
    ``complete=False`` clears both.

    ``view_scale`` carries no default: a caller with no view states ``null``.
    """

    image_path: str
    subject: str
    dataset_root: Optional[str] = None
    grid: GridGeometry
    cell: str
    complete: bool
    view_scale: Optional[float]
    # GUI-set identity; stamped as attested_by ("user:<name>"), mirroring annotate.py/review.py.
    user: Optional[str] = None


@router.get("/completeness")
def get_completeness(
    path: str = Query(..., description="Absolute path to the image file"),
    dataset_root: Optional[str] = Query(None),
    subject: Optional[str] = Query(
        None, description="Include this subject's working scale even when it has no "
                          "completeness record on this image, so a negative or unannotated "
                          "image still answers for the active subject rather than omitting it"),
) -> dict:
    """Every subject's region-completeness record for the raster at ``path`` (its own stem).

    Each record carries ``stale_cells`` (recomputed fresh on every read) and, where an attestation
    stamped it, ``cells_attested_view``. A label document that will not decode for a subject that
    already holds a record here refuses (400).

    ``working_scale`` (subject -> ``WorkingScale`` or ``null``) is every subject with a
    completeness record on this raster, plus the requested ``subject`` when it has none, each read
    fresh through :func:`_subject_zoom`. Where no zoom is set, ``working_scale[subject]`` stays
    null and ``working_scale_reason: dict[str, str]`` names why, per subject. Where
    ``_subject_zoom`` itself refuses for a subject, ``working_scale[subject]`` stays null and
    ``working_scale_error`` carries the refusal's own message instead, without blocking
    ``by_subject``.

    ``annotation_counts`` (subject -> cell name -> count) is every subject's saved-annotation count
    per cell, binned against ``counts_grid`` (:func:`_grid_for_raster`). A label-read failure
    empties it, named in ``counts_error``; a raster-read failure (a missing file, an incomplete
    band group) costs only ``counts_error`` too.
    """
    from tcip_annotation.json_io import UnreadableLabelDocument, read_annotations
    from tcip_mcp.dataset_layout import (
        annotation_path,
        parse_image_path,
        region_completeness_digest_key,
        region_completeness_key,
    )
    from tcip_mcp.pipelines.data.band_groups import BandGroupIncomplete
    from tcip_mcp.pipelines.reference_grid import reference_cells
    from tcip_mcp.pipelines.region_completeness import (
        annotation_counts_by_cell, record_annotations, stale_cells,
    )

    try:
        src = assert_path_allowed(path)
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc

    root = _resolve_root(path, dataset_root)
    stem = Path(path).stem
    store = tcip_store.read(region_completeness_key(root), default={})
    digests = tcip_store.read(region_completeness_digest_key(root), default={})

    by_subject: dict[str, dict] = {}
    for bucket, record in store.items():
        if record["stem"] != stem:
            continue
        record_subject = record["subject"]
        try:
            stale = stale_cells(record, record_annotations(root, record),
                                digests.get(bucket, {}), record_subject)
        except UnreadableLabelDocument as exc:
            raise HTTPException(400, str(exc)) from exc
        by_subject[record_subject] = {**record, "stale_cells": stale}

    subjects = set(by_subject)
    if subject:
        subjects.add(subject)
    working_scale: dict[str, Optional[dict]] = {}
    working_scale_reason: dict[str, str] = {}
    working_scale_error: Optional[str] = None
    for subj in sorted(subjects):
        try:
            entry = _subject_zoom(root, subj)
        except HTTPException as exc:
            working_scale[subj] = None
            working_scale_error = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            continue
        if entry is not None:
            working_scale[subj] = _working_scale_of(entry)
        else:
            working_scale[subj] = None
            working_scale_reason[subj] = _no_zoom_reason(subj)

    label_error: Optional[str] = None
    annotations: list = []
    try:
        _root_from_image, date, image_stem = parse_image_path(path)
        label_path = annotation_path(root, date, image_stem)
        annotations = read_annotations(str(label_path)) if label_path.is_file() else []
    except (UnreadableLabelDocument, ValueError, FileNotFoundError, OSError) as exc:
        label_error = str(exc)

    annotation_counts: dict[str, dict[str, int]] = {}
    counts_grid: Optional[dict] = None
    counts_error: Optional[str] = label_error
    if label_error is None:
        try:
            geometry, _derivation = _grid_for_raster(src, None)
            cells = reference_cells(
                geometry["width"], geometry["height"], geometry["tile_size"], 0.0, clamp=True)
            annotation_counts = annotation_counts_by_cell(annotations, cells, geometry["tile_size"])
            counts_grid = geometry
        except (BandGroupIncomplete, FileNotFoundError, OSError, ValueError) as exc:
            counts_error = str(exc)

    return {"by_subject": by_subject, "annotation_counts": annotation_counts,
            "counts_grid": counts_grid, "counts_error": counts_error,
            "working_scale": working_scale, "working_scale_error": working_scale_error,
            "working_scale_reason": working_scale_reason}


@router.post("/completeness")
def post_completeness(payload: CompletenessSetPayload) -> dict:
    """Set one cell's completeness for a subject to ``payload.complete``.

    ``complete=True`` stamps the cell's current annotation-content digest, whether or not the cell
    was already complete, and its scale provenance in ``cells_attested_view`` (the pressed
    ``view_scale``, the subject's working scale in effect at write time, and whether this image's
    own view-coverage record, read under ``status_bucket(subject, date)`` by the image's file name
    and on a matching grid only, shows the cell already seen); ``complete=False`` clears both the
    digest stamp and the ``cells_attested_view`` entry.

    Cell names are validated against the posted grid's own cells. A stored record whose grid
    disagrees with the posted one replaces wholesale (cells, digest stamps and scale provenance
    alike); the response and the audit line both carry the discarded record's grid and cells
    (``replaced``, null when nothing was discarded).

    The working scale is the subject's own set grid zoom, read once ahead of the transaction
    through :func:`_subject_zoom`. An absent zoom stamps a null working scale, while an entry
    :func:`_subject_zoom` refuses refuses this write outright.

    A failed audit append answers the marked 500 (``AUDIT_ENTRY_NOT_WRITTEN``).
    """
    from tcip_annotation.json_io import UnreadableLabelDocument, read_annotations

    from tcip_mcp.dataset_layout import (
        annotation_path,
        parse_image_path,
        region_completeness_digest_key,
        region_completeness_key,
        status_bucket,
        view_coverage_key,
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

    complete = payload.complete
    if complete:
        try:
            assert_path_allowed(payload.image_path)
        except ValueError as exc:
            raise HTTPException(403, str(exc)) from exc

    bucket = status_bucket(subject, stem)
    completeness_key = region_completeness_key(root)
    digest_key = region_completeness_digest_key(root)
    author = user_id(resolve_user(payload.user))
    now_iso = datetime.now(timezone.utc).isoformat()
    image_name = Path(payload.image_path).name
    # The subject's set zoom, never this image's own label file or pixel size: this can't fail.
    working_scale_at_write = _working_scale_of(_subject_zoom(root, subject)) if complete else None

    # Digest key named first, so it is applied first: stale_cells fails closed on a missing
    # digest, so a stamp must never land after the attestation that points at it.
    with tcip_store.transaction(digest_key, completeness_key) as txn:
        store = txn.read(completeness_key, default={})
        existing = store.get(bucket)
        grid_matches = existing is not None and existing["grid"] == grid
        cells_complete = (
            set(existing["cells_complete"])
            if grid_matches and existing is not None else set())
        cells_attested_view = (
            dict(existing["cells_attested_view"])
            if grid_matches and existing is not None else {})

        if not complete and payload.cell not in cells_complete:
            # Nothing to unattest here, whether from no record, a different lattice, or a
            # same-lattice record that never held this cell: an idempotent no-op, not a write.
            return {"status": "ok", "complete": False, "cells_complete": sorted(cells_complete),
                    "replaced": None}

        replaced = existing is not None and not grid_matches
        replaced_info = None
        if replaced and isinstance(existing, dict):
            replaced_info = {"grid": existing["grid"],
                              "cells_complete": sorted(existing["cells_complete"])}
        if complete:
            cells_complete.add(payload.cell)
        else:
            cells_complete.discard(payload.cell)
        digests = txn.read(digest_key, default={})
        bucket_digests = dict(digests.get(bucket, {}))
        if replaced:
            bucket_digests = {}
        annotations: list = []
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

        attested_view_entry: Optional[dict] = None
        if complete:
            view_bucket = status_bucket(subject, date)
            view_store = tcip_store.read(view_coverage_key(root), default={})
            view_record = view_store.get(view_bucket, {}).get(image_name)
            grid_matched = view_record is not None and view_record["grid"] == grid
            at_scale = None
            if grid_matched:
                at_scale = view_record["cells_seen_at_scale"].get(payload.cell)
            attested_view_entry = {
                "view_scale": payload.view_scale,
                "working_scale_at_write": working_scale_at_write,
                "seen_on_record": {"at_scale": at_scale, "grid_matched": grid_matched},
            }
            cells_attested_view[payload.cell] = attested_view_entry
        else:
            cells_attested_view.pop(payload.cell, None)

        store[bucket] = {
            "grid": grid,
            "cells_complete": sorted(cells_complete),
            "attested_by": author,
            "attested_at": now_iso,
            "stem": stem,
            "date": date,
            "subject": subject,
            "cells_attested_view": cells_attested_view,
        }
        txn.write(completeness_key, store)

    _audit_or_answer_500(
        "gui_set_region_completeness",
        {"image_name": image_name, "subject": subject, "cell": payload.cell,
         "complete": complete, "stem": stem, "date": date, "replaced": replaced_info,
         "cells_attested_view": attested_view_entry},
        root,
    )
    return {"status": "ok", "complete": complete, "cells_complete": sorted(cells_complete),
            "replaced": replaced_info}

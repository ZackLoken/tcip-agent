"""View-coverage routes: the reference grid over a raster and the per-image record of two
per-cell facts: which cells were served to the browser at native resolution (a delivery fact)
and which cells have, at some point this session or an earlier one, sat fully on screen at some
recorded scale (``cells_seen_at_scale``, a bound). Neither is an attention claim. Whether a seen
cell counts as "swept" is derived against a subject's working-scale bar -- the view scale at
which the median saved annotation of that subject on this image spans a documented default
number of screen pixels (:data:`tcip_mcp.pipelines.reference_grid.JUDGED_SPAN_PX`), served by
``get_completeness`` below and never stored on this record. The one comparison
(``lib/coverage.ts``'s ``meetsBar``) lives in the browser; this route never compares a recorded
scale against a bar.

The grid geometry has one implementation (``tcip_mcp.pipelines.reference_grid``): the
grid route serves cells computed there and the frontend consumes them verbatim, never
re-deriving them. The record store is ``view_coverage.json``
(``dataset_layout.view_coverage_path``), bucketed like ``image_status.json``
(``status_bucket(subject, date)``, then image name). The store is advisory: training
never reads it, and unswept cells warn rather than block a Complete. Every write that changes
the record is audited (``record_event_or_raise``, after the transaction commits, since a log
append cannot join a record transaction): an append that cannot land still raises
(``AuditEntryNotWritten``, named by a stable marker in the 500 body) rather than warning, but by
then the record is already committed, so the missing line stays missing -- a retry of the same
payload merges to no change and writes and audits nothing. The 500 tells the caller the true
guarantee (the change landed, its line did not), never that its own retry is what recovers it.

Also here: region-completeness routes, a different store on the same grid. An attestation ("I
found every instance of this subject in these cells") gates a scientific claim (block
calibration's completeness check), so unlike view coverage it is written with the same discipline
as ``routes/classes.py``'s image-status store, and a stale attestation (a cell's annotation
content edited or deleted since it was attested) is detected on every read, not trusted forever.
An attestation also records its own scale provenance (``cells_attested_view``): the view scale
the breeder pressed at, the working-scale bar derived from the label file at write time, and
whether this image's own coverage record shows the cell seen on a matching lattice -- facts
only, no verdict, since whether an unswept cell should have blocked the attestation stays
advisory. See ``dataset_layout.region_completeness_path`` and
``tcip_mcp.pipelines.region_completeness``.
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

_CONFORM_HINT = (
    "run scripts/conform_view_coverage_viewing.py against this dataset to bring it to the "
    "current shape (--plan first to preview the change)"
)

# The stable marker post_coverage's 500 body carries for AuditEntryNotWritten (api/http.ts's
# decodeRefusal parses detail as an object; coverageTracker.ts's outbox reads it as terminal).
AUDIT_ENTRY_NOT_WRITTEN = "audit_entry_not_written"

# The stable marker post_coverage's 409 body carries for a grid mismatch with no replace flag;
# coverageTracker.ts reads it to set the tracker's replace hold, never to retry or drop the push.
COVERAGE_LATTICE_MISMATCH = "coverage_lattice_mismatch"


def _require_date_matches_path(image_path: str, date: Optional[str]) -> None:
    """Refuse a posted ``date`` that disagrees with the date :func:`parse_image_path` reads off
    ``image_path`` itself (an explicit ``null`` against a dated image, or a date against a
    dateless one): the two must agree by construction, since a record filed under a bucket the
    image's own path never carries would silently split one image's coverage across two buckets.
    A path outside the recognized ``<root>/images/[<date>/]<stem>`` tree carries nothing to
    compare against and is left to whatever check already applies to it (``_resolve_root``, on
    the branch that itself calls ``parse_image_path``)."""
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
    """Refuse rather than serve or merge into a stored record that no longer validates as
    ``CoverageRecord``, naming the conform script instead of silently coercing or dropping it.
    ``record`` need not be a dict at all: a stored document in some other shape is exactly as
    unmergeable as a dict missing a required key, and validates the same way."""
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

    subject = _require_subject(subject)
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
    """One coverage post from the browser: the session's accumulated served-at-native cells and
    per-cell containment scales (either may be empty; the server merges, so resending is
    harmless), with the grid they were accumulated against and the viewing context they were
    served under. ``date`` and ``viewing`` carry no default: a non-dated dataset must still pass
    ``date: null`` explicitly, so an image under a date bucket can never silently land in the
    dateless one, and every post states the viewing context it was served under rather than the
    model quietly filling in one no browser ever chose. ``replace`` confirms a grid mismatch's
    wholesale replace of the stored record; a first record (no prior record at all) needs no
    flag, and a mismatched grid without one refuses with 409 rather than silently discarding a
    previous lattice's sweeps."""

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
    ``cells_seen_at_scale`` keeps the greater value per cell (a cell's recorded scale only ever
    rises). A mismatched grid against an existing record refuses with 409
    (``COVERAGE_LATTICE_MISMATCH``, naming the stored grid and its ``cells_seen`` count) unless
    ``payload.replace`` is true, since the record carries the grid it was accumulated against
    precisely so a derivation change can never silently discard a previous lattice's sweeps; with
    the flag the record is replaced wholesale and the response flags it (``replaced``), and the
    audit line carries ``replace_confirmed``. A first record (no prior record at all) needs no
    flag. Cell names are validated against the posted grid's own cells; unknown names are
    refused. On the merge path the stored record is validated against ``CoverageRecord`` before
    its cells are folded in, naming the conform script when it no longer holds that shape; the
    replace path overwrites it wholesale, since there is nothing to merge into.

    A write happens, and is audited, only when the merge actually changes something: a raised or
    newly-seen cell, a newly served cell, a changed ``viewing``, or a replace. An unchanged push
    neither writes nor stamps ``updated_at`` nor audits, so a tracker's own retry of an
    already-acknowledged payload costs nothing.

    The audit line is appended after the transaction commits, never inside it: ``tcip_store``
    refuses an ``append`` while a transaction is open (a log cannot join a record transaction),
    the same constraint ``plant_mapping.persist_mapping`` states for its own receipt. A failed
    append still raises (``record_event_or_raise``, not the best-effort ``record_event``) rather
    than warning: the write above has already landed, so the route answers 500, naming
    ``AuditEntryNotWritten`` by a stable marker in the body, and tells the caller the true
    guarantee: the record is committed, its line is not, and a retry of this same payload merges
    to no change and writes and audits nothing, never recovering it.
    """
    from tcip_mcp.audit import AuditEntryNotWritten, record_event_or_raise
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
        if not isinstance(store, dict):
            store = {}
        records = store.setdefault(bucket, {})
        if not isinstance(records, dict):
            records = store[bucket] = {}
        existing = records.get(image_name)
        served = set(payload.cells_served_at_native)
        seen = dict(payload.cells_seen_at_scale)
        replaced = False
        served_added = sorted(served)
        seen_added = dict(seen)
        viewing_changed = True
        grid_matches = isinstance(existing, dict) and existing.get("grid") == grid
        # Only a confirmed replace across a grid mismatch has nothing to merge into; every other
        # existing record, dict or not, must validate before the mismatch branch below.
        if existing is not None and not (payload.replace and not grid_matches):
            _validated_record(image_name, existing)
        if not grid_matches and existing is not None and not payload.replace:
            stored_grid = existing.get("grid") or {}
            raise HTTPException(409, {
                "error": COVERAGE_LATTICE_MISMATCH,
                "message": (
                    f"{image_name}'s stored coverage record was accumulated against a grid "
                    f"this push's grid disagrees with; pass replace: true to discard it and "
                    f"record on the new lattice"),
                "stored_grid": {"cols": stored_grid.get("cols"), "rows": stored_grid.get("rows"),
                                 "tile_size": stored_grid.get("tile_size")},
                "cells_seen": len(existing.get("cells_seen_at_scale") or {}),
            })
        if grid_matches and isinstance(existing, dict):
            prior_served = set(existing.get("cells_served_at_native") or [])
            prior_seen = dict(existing.get("cells_seen_at_scale") or {})
            served_added = sorted(served - prior_served)
            served = served | prior_served
            merged_seen = dict(prior_seen)
            seen_added = {}
            for name, scale in seen.items():
                if scale > merged_seen.get(name, float("-inf")):
                    merged_seen[name] = scale
                    seen_added[name] = scale
            seen = merged_seen
            viewing_changed = existing.get("viewing") != viewing_dict
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
    try:
        record_event_or_raise(
            "gui_view_coverage",
            {"image_name": image_name, "subject": subject, "date": payload.date, "grid": grid,
             "cells_seen_added": seen_added, "cells_served_at_native_added": served_added,
             "viewing_changed": viewing_changed,
             "viewing": viewing_dict if viewing_changed else None, "replaced": replaced,
             "replace_confirmed": payload.replace},
            source="gui", scope=root,
        )
    except AuditEntryNotWritten as exc:
        raise HTTPException(
            500, {"error": AUDIT_ENTRY_NOT_WRITTEN, "message": str(exc)}) from exc

    return {
        "status": "ok",
        "replaced": replaced,
        "cells_served_at_native": len(served),
        "total_cells": grid["cols"] * grid["rows"],
        "record": record_body,
    }


# Region completeness gates a scientific claim; its write also audits after the commit, above,
# through the best-effort record_event emitter, so a failed append there warns, never raises.


class CompletenessSetPayload(BaseModel):
    """One explicit attestation write: set one cell's completeness for a subject to
    ``complete``, never a toggle a caller infers from the stored state. ``complete=True``
    stamps the cell's current annotation-content digest (whether or not the cell was already
    complete, so a re-attest restamps and clears staleness) and its scale provenance
    (``cells_attested_view`` on the stored record); ``complete=False`` clears both. The control
    that posts this states the direction in its own label, so the write it performs always
    matches what the label promised.

    ``view_scale`` carries no default, the ``CoveragePayload.date`` precedent: the canvas states
    the view scale at the press explicitly, and a non-GUI caller with no view states ``null``
    rather than the model quietly filling in a scale nobody chose.
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
        None, description="Include this subject's working-scale bar even when it saved no "
                          "annotation on this image, so a negative or unannotated image still "
                          "answers for the active subject rather than omitting it"),
) -> dict:
    """Every subject's region-completeness record for the raster at ``path`` (its own stem), not
    just one subject's: lets the coverage overlay render a cell attested complete for a subject
    other than the one currently active, distinguishably from the active subject's own
    attestations.

    Each record carries ``stale_cells`` (recomputed fresh on every read) and, where an
    attestation stamped it, ``cells_attested_view`` (the scale provenance ``post_completeness``
    records: facts only, no verdict). A label document that will not decode for a subject that
    already holds a record here still refuses (400): staleness cannot be computed without it.

    ``working_scale`` (subject -> bar or ``null``) is every subject present in the raster's
    label file, plus the requested ``subject`` when absent from it, each derived fresh from that
    file (:func:`tcip_mcp.pipelines.region_completeness.working_scale_bar`) -- never a value the
    browser echoes back. When a subject saved no box or polygon annotation of its own on this
    image, its bar falls back to the dataset's physical median for that subject
    (:func:`tcip_mcp.pipelines.region_completeness.dataset_physical_extent`), expressed through
    this image's own pixel size (:func:`tcip_mcp.pipelines.region_completeness.raster_pixel_size`)
    when both are known; the served/stored dict's ``from_this_image`` says which branch produced
    it. Where neither exists, ``working_scale[subject]`` stays null and
    ``working_scale_reason: dict[str, str]`` names why, per subject. The label file is read once,
    ahead of and outside the raster-grid block, and the same annotations feed
    ``annotation_counts``: a raster-read failure (a missing file, an incomplete band group) leaves
    the bar standing and only costs ``counts_error``, while a label-read failure empties both
    ``working_scale`` and ``annotation_counts``, named in ``working_scale_error``/``counts_error``
    alike; a failure in the pixel-size/dataset-extent derivation alone (an ``AmbiguousImageStem``,
    an unreadable label elsewhere in the dataset, an ``OSError``) costs only the bars it would have
    produced and is named the same way, in ``working_scale_error``.

    ``annotation_counts`` (subject -> cell name -> count) is every subject's saved-annotation
    count per cell, binned against ``counts_grid`` (:func:`_grid_for_raster`, the same
    derivation ``get_grid`` uses and served beside ``annotation_counts`` rather than folded into
    it: a count belongs to the grid, not to any one attestation record.
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
    from tcip_mcp.pipelines.reference_grid import JUDGED_SPAN_PX, reference_cells
    from tcip_mcp.pipelines.region_completeness import (
        PixelSize,
        annotation_counts_by_cell,
        dataset_physical_extent,
        dataset_working_scale_bar,
        default_working_scale_source,
        resolve_pixel_size,
        saved_extents,
        stale_cells,
        working_scale_bar,
    )

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
        record_subject = record.get("subject")
        if not isinstance(record_subject, str) or not record_subject:
            continue
        stamped = digests.get(bucket)
        try:
            stale = stale_cells(
                root, record, stamped if isinstance(stamped, dict) else {}, record_subject)
        except UnreadableLabelDocument as exc:
            raise HTTPException(400, str(exc)) from exc
        by_subject[record_subject] = {**record, "stale_cells": stale}

    label_error: Optional[str] = None
    annotations: list = []
    try:
        _root_from_image, date, image_stem = parse_image_path(path)
        label_path = annotation_path(root, date, image_stem)
        annotations = read_annotations(str(label_path)) if label_path.is_file() else []
    except (UnreadableLabelDocument, ValueError, FileNotFoundError, OSError) as exc:
        label_error = str(exc)

    working_scale_reason: dict[str, str] = {}
    dataset_error: Optional[str] = None
    if label_error is not None:
        working_scale: dict[str, Optional[dict]] = {}
    else:
        subjects = {a.subject for a in annotations if a.subject}
        if subject:
            subjects.add(subject)
        own_source = default_working_scale_source(JUDGED_SPAN_PX)
        working_scale = {}
        pixel_sizes: dict[Path, tuple[Optional[PixelSize], str]] = {}
        label_cache: dict[Path, list] = {}
        this_image_pixel_size: Optional[PixelSize] = None
        this_image_pixel_size_reason: Optional[str] = None
        this_image_pixel_size_checked = False
        for subj in sorted(subjects):
            own_bar = working_scale_bar(
                saved_extents(annotations, subj), judged_span_px=JUDGED_SPAN_PX,
                source=own_source, from_this_image=True)
            if own_bar is not None:
                working_scale[subj] = own_bar
                continue
            if not this_image_pixel_size_checked:
                this_image_pixel_size_checked = True
                if src not in pixel_sizes:
                    pixel_sizes[src] = resolve_pixel_size(src)
                this_image_pixel_size, this_image_pixel_size_reason = pixel_sizes[src]
            if this_image_pixel_size is None:
                working_scale[subj] = None
                working_scale_reason[subj] = (
                    f"no saved box or polygon annotation of {subj} on this image and no known "
                    f"pixel size ({this_image_pixel_size_reason})")
                continue
            if dataset_error is not None:
                working_scale[subj] = None
                continue
            try:
                extent = dataset_physical_extent(
                    root, subj, pixel_sizes=pixel_sizes, label_cache=label_cache)
            except (OSError, ValueError, UnreadableLabelDocument,
                    tcip_store.SchemaVersionRefused) as exc:
                dataset_error = str(exc)
                working_scale[subj] = None
                continue
            if extent is None:
                working_scale[subj] = None
                working_scale_reason[subj] = (
                    f"no saved {subj} annotation on any georeferenced image with a known pixel "
                    "size")
                continue
            working_scale[subj] = dataset_working_scale_bar(
                subj, extent, this_image_pixel_size, JUDGED_SPAN_PX)

    working_scale_error = label_error if label_error is not None else dataset_error

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
    """Set one cell's completeness for a subject to ``payload.complete``: never a toggle, so a
    control's label ("Attest", "Unattest", "Re-attest") always states the write it performs.
    ``complete=True`` stamps the cell's current annotation-content digest, whether or not the
    cell was already complete (a re-attest restamps and clears staleness), and its scale
    provenance in ``cells_attested_view`` (the pressed ``view_scale``, the working-scale bar
    derived from the same label-file read at write time, and whether this image's own
    view-coverage record -- read under ``status_bucket(subject, date)`` by the image's file
    name, on a matching grid only -- shows the cell already seen); ``complete=False`` clears
    both the digest stamp and the ``cells_attested_view`` entry.

    Cell names are validated against the posted grid's own cells, same as ``post_coverage``. A
    stored record whose grid disagrees with the posted one replaces wholesale (cells, digest
    stamps and scale provenance alike), rather than trusting a same-named cell across two
    different lattices; the response and the audit line both carry the discarded record's grid
    and cells (``replaced``, null when nothing was discarded), the way ``post_coverage`` states
    its own replacement.

    An attestation stamps the bar the breeder actually swept against: this image's own saved
    annotations first, the dataset's physical median (:func:`~tcip_mcp.pipelines.
    region_completeness.dataset_physical_extent`) expressed through this image's own pixel size
    when the image saved none of its own. Both are resolved once, before the transaction opens
    (neither reads this image's label file, which stays a single read inside it); a failure there
    (an ``AmbiguousImageStem``, a bad manifest, an ``OSError``) never refuses the cell -- the
    attestation proceeds and stamps a null bar, and the failure is named in the response's
    ``working_scale_error`` (present only when one occurred).
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
        view_coverage_key,
    )
    from tcip_mcp.pipelines.reference_grid import JUDGED_SPAN_PX, reference_cells
    from tcip_mcp.pipelines.region_completeness import (
        DatasetExtent,
        PixelSize,
        cell_annotation_digest,
        dataset_physical_extent,
        dataset_working_scale_bar,
        default_working_scale_source,
        resolve_pixel_size,
        saved_extents,
        working_scale_bar,
    )
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
    image_name = Path(payload.image_path).name

    # Resolved before the transaction opens: neither reads this image's own label file, so the
    # lock below holds no directory walk and a failure here never refuses the cell.
    dataset_pixel_size: Optional[PixelSize] = None
    dataset_extent: Optional[DatasetExtent] = None
    working_scale_derivation_error: Optional[str] = None
    if complete:
        try:
            img_path = assert_path_allowed(payload.image_path)
        except ValueError as exc:
            raise HTTPException(403, str(exc)) from exc
        try:
            pixel_sizes: dict[Path, tuple[Optional[PixelSize], str]] = {
                img_path: resolve_pixel_size(img_path)}
            dataset_pixel_size, _reason = pixel_sizes[img_path]
            if dataset_pixel_size is not None:
                dataset_extent = dataset_physical_extent(
                    root, subject, pixel_sizes=pixel_sizes)
        except (OSError, ValueError, UnreadableLabelDocument,
                tcip_store.SchemaVersionRefused) as exc:
            working_scale_derivation_error = str(exc)
            dataset_pixel_size = None
            dataset_extent = None

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
        cells_complete = (
            set(existing.get("cells_complete") or [])
            if grid_matches and existing is not None else set())
        cells_attested_view = (
            dict(existing.get("cells_attested_view") or {})
            if grid_matches and existing is not None else {})

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
            own_extents = saved_extents(annotations, subject)
            if own_extents:
                bar = working_scale_bar(
                    own_extents, judged_span_px=JUDGED_SPAN_PX,
                    source=default_working_scale_source(JUDGED_SPAN_PX), from_this_image=True)
            elif dataset_pixel_size is not None and dataset_extent is not None:
                bar = dataset_working_scale_bar(
                    subject, dataset_extent, dataset_pixel_size, JUDGED_SPAN_PX)
            else:
                bar = None
            view_bucket = status_bucket(subject, date)
            view_store = tcip_store.read(view_coverage_key(root), default={})
            view_records = view_store.get(view_bucket) if isinstance(view_store, dict) else None
            view_record = (
                view_records.get(image_name) if isinstance(view_records, dict) else None)
            grid_matched = isinstance(view_record, dict) and view_record.get("grid") == grid
            at_scale = None
            if grid_matched and isinstance(view_record, dict):
                at_scale = (view_record.get("cells_seen_at_scale") or {}).get(payload.cell)
            attested_view_entry = {
                "view_scale": payload.view_scale,
                "working_scale_bar_at_write": bar,
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

    _audit_dataset_write(
        root, "gui_set_region_completeness",
        {"image_name": image_name, "subject": subject, "cell": payload.cell,
         "complete": complete, "stem": stem, "date": date, "replaced": replaced_info,
         "cells_attested_view": attested_view_entry},
    )
    response = {"status": "ok", "complete": complete, "cells_complete": sorted(cells_complete),
                "replaced": replaced_info}
    if working_scale_derivation_error is not None:
        response["working_scale_error"] = working_scale_derivation_error
    return response

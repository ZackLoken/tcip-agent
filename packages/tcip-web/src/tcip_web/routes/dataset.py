"""Dataset discovery + selection routes.

The frontend hits these to:
  * discover what's available under a project root,
  * read and persist the ``GuiState.dataset`` selection,
  * persist the browser's current image position within that selection.

Convention: the canonical layout (see :mod:`tcip_mcp.dataset_layout`):

    <dataset_root>/
        images/<date>/*.JPG
        annotations/<date>/<stem>.json          # ground truth, one file per image (all subjects)
        predictions/<model>/<date>/<stem>.json  # model outputs
        classes.json                            # the nested subject/attribute registry
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import tcip_store as ts

from tcip_mcp import workspace
from tcip_mcp.dataset_layout import (
    annotation_dir,
    annotation_root,
    classes_path,
    image_dir,
    image_root,
    list_dates,
    list_models,
    list_subjects,
    models_with_predictions,
    prediction_dir,
    prediction_root,
    subjects_with_labels,
)
from tcip_mcp.pipelines.image_utils import list_logical_images, logical_image_name
from tcip_mcp.web_client import canvas_open_binding_key
from tcip_web.label_annotations_cache import cached_label_annotations
from tcip_web.paths import assert_path_allowed
from tcip_web.state import DatasetSelection, store

router = APIRouter(prefix="/api/dataset", tags=["dataset"])


def _guarded(path: str) -> Path:
    """Confine a client-supplied root and hand back the resolved path every later read uses."""
    try:
        return assert_path_allowed(path)
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc


# Whether this server process has selected a dataset yet. Resuming the persisted image index
# is helpful within a running session, but a fresh process opening a project should start at
# the first image: resuming a prior session's position across a restart reads as stale state.
_selected_this_session = False


class DatasetTree(BaseModel):
    dataset_root: str
    dates_with_images: list[str]
    # Every subject the dataset's registry (``classes.json``) declares, e.g.
    # ["bush", "leaf"]. This is *what a label set is about*, not the shape kind; a label file
    # names its subject on each annotation rather than in the path.
    subjects: list[str]
    model_names: list[str]       # every model present anywhere, e.g. ["baseline"]
    # Per-date availability: the subjects that actually have labels / models that actually
    # have predictions on each date. The GUI's subject/model pickers filter to these so a
    # date with no labels for a subject doesn't offer it (which would open an empty canvas).
    subjects_by_date: dict[str, list[str]]
    models_by_date: dict[str, list[str]]
    # Where each (date, model)'s predictions live, straight from dataset_layout.prediction_dir,
    # so a client never points a delivery at a path no writer produces.
    prediction_dirs: dict[str, dict[str, str]]
    # The first date whose labels would not read, naming the file (mirrors ProjectSummary's own
    # site_problem). The tree still lists every other date; a corrupt label costs one date.
    label_problem: Optional[str] = None


# ── /tree cache ────────────────────────────────────────────────────────────
# subjects_with_labels/models_with_predictions each re-list annotations/ or predictions/ and
# scan every per-image label file per date, so a naive /tree is an iterdir storm on a dataset
# with many dates. Cache the built tree per dataset_root, keyed by a signature of every
# directory the computation reads (stat-only, no listing): a write inside any of those date
# dirs bumps its own mtime_ns and invalidates the entry. Bounded to a handful of recent roots.
_TREE_CACHE_MAX = 64
_tree_cache: "OrderedDict[str, tuple[tuple, DatasetTree]]" = OrderedDict()


def _dir_mtime_ns(p: Path) -> int:
    try:
        return p.stat().st_mtime_ns
    except OSError:
        return -1


def _subjects_by_date(root: Path, dates: list[str]) -> tuple[dict[str, list[str]], Optional[str]]:
    """``subjects_with_labels`` per date, and the first date's problem when one won't read.

    A date whose labels won't read, or whose annotations directory the path guard refuses,
    reports an empty subject list for that date rather than aborting the scan: every other
    date's own labels are unaffected by one date's corrupt file or disallowed storage. The guard
    checked here is the one ``routes/classes.py``'s ``load_classes`` applies to the same
    directory, so a dataset the class registry route 403s never lists its subjects here instead.
    """
    from tcip_annotation.json_io import UnreadableLabelDocument

    by_date: dict[str, list[str]] = {}
    problem: Optional[str] = None
    for d in dates:
        try:
            assert_path_allowed(str(annotation_dir(root, d)))
        except ValueError as exc:
            by_date[d] = []
            if problem is None:
                problem = str(exc)
            continue
        try:
            by_date[d] = subjects_with_labels(root, d, reader=cached_label_annotations)
        except UnreadableLabelDocument as exc:
            by_date[d] = []
            if problem is None:
                problem = str(exc)
    return by_date, problem


def _tree_signature(root: Path, dates: list[str], models: list[str]) -> tuple:
    sig = [
        _dir_mtime_ns(image_root(root)),
        _dir_mtime_ns(annotation_root(root)),
        _dir_mtime_ns(prediction_root(root)),
        _dir_mtime_ns(classes_path(root)),
    ]
    for d in dates:
        sig.append(_dir_mtime_ns(annotation_dir(root, d)))
        for model in models:
            sig.append(_dir_mtime_ns(prediction_dir(root, model, d)))
    return tuple(sig)


@router.get("/tree")
def get_dataset_tree(dataset_root: str) -> DatasetTree:
    """Return the high-level tree (dates, subjects, models) for a dataset."""
    root = _guarded(dataset_root)
    if not root.is_dir():
        raise HTTPException(404, f"dataset_root not found: {dataset_root}")

    dates = list_dates(root)
    # Subjects come from the dataset registry, not from listing annotations/: that dir now holds
    # date buckets, not subject dirs.
    subjects = list_subjects(root)
    model_names = list_models(root)

    # Read every call, not from the cache below: a label edited in place leaves the
    # directory's own mtime untouched, and label_problem must never answer from stale content.
    subjects_by_date, label_problem = _subjects_by_date(root, dates)

    key = str(root)
    signature = _tree_signature(root, dates, model_names)
    cached = _tree_cache.get(key)
    if cached is not None and cached[0] == signature:
        _tree_cache.move_to_end(key)
        return cached[1].model_copy(
            update={"subjects_by_date": subjects_by_date, "label_problem": label_problem}
        )

    tree = DatasetTree(
        dataset_root=str(root),
        dates_with_images=dates,
        subjects=subjects,
        # A model is selectable once it has a predictions bucket under this dataset.
        model_names=model_names,
        subjects_by_date=subjects_by_date,
        models_by_date={d: models_with_predictions(root, d) for d in dates},
        prediction_dirs={
            d: {m: str(prediction_dir(root, m, d)) for m in model_names} for d in dates
        },
        label_problem=label_problem,
    )
    _tree_cache[key] = (signature, tree)
    _tree_cache.move_to_end(key)
    if len(_tree_cache) > _TREE_CACHE_MAX:
        _tree_cache.popitem(last=False)
    return tree


class SelectionRequest(BaseModel):
    project_root: str
    dataset_root: str
    subject: Optional[str] = None
    date: Optional[str] = None
    model_name: Optional[str] = None


def _write_canvas_binding(root: Path) -> int:
    """Record ``root`` as the GUI's open root; return the generation now in force.

    Read-modify-write inside a transaction, the store's own ``cas`` policy: the current record
    is read to decide whether ``root`` actually changed (generation bumps only then, so a
    same-project re-select or ordinary navigation never supersedes a sibling tab), and the write
    is staged in the same transaction so a concurrent select cannot land between the read and the
    write and have its own bump silently dropped.
    """
    key = canvas_open_binding_key()
    root_str = str(root)
    with ts.transaction(key) as txn:
        current = txn.read(key, default=None)
        if current is not None and ts.canonical_path(current["root"]) == ts.canonical_path(root_str):
            generation = current["generation"]
        else:
            generation = (current["generation"] + 1) if current is not None else 1
        txn.write(key, {
            "generation": generation,
            "root": root_str,
            "project_name": workspace.workspace_project_name(root),
            "issued_at": datetime.now(timezone.utc).isoformat(),
        })
    return generation


@router.post("/select")
async def select_dataset(req: SelectionRequest) -> dict:
    """Set the active dataset for the GUI; broadcasts a state delta."""
    project_root = _guarded(req.project_root)
    root = _guarded(req.dataset_root)
    if not root.is_dir():
        raise HTTPException(404, f"dataset_root not found: {req.dataset_root}")

    # Recorded before anything else, off the event loop (a cross-process lock, a possible
    # fsync), so a busy binding store refuses the whole select rather than half-adopting it.
    try:
        generation = await asyncio.to_thread(_write_canvas_binding, project_root)
    except ts.StoreBusy as exc:
        raise HTTPException(
            503, f"could not record which project the GUI has open: {exc}"
        ) from exc

    # Rehydrate any persisted GUI state for this project first (so backend state
    # survives a restart), then apply the fresh selection on top via mutate().
    store.open_project(project_root)

    image_list: list[str] = []
    if req.date:
        date_dir = image_dir(root, req.date)
        if date_dir.is_dir():
            # Sorted display names, band groups folded to one entry per capture, through the
            # one naming primitive gui_tools' own by-name callers resolve the same capture under.
            image_list = sorted(
                logical_image_name(src) for src in list_logical_images(date_dir).values()
            )

    # Canonical layout (see tcip_mcp.dataset_layout): the single source of truth shared with the
    # agent tools, so agent writes land where the GUI reads. The browser composes none of these.
    images_dir = str(image_dir(root, req.date)) if req.date else None
    annotations_dir = str(annotation_dir(root, req.date)) if req.date else None
    predictions_dir = (
        str(prediction_dir(root, req.model_name, req.date)) if req.model_name and req.date else None
    )

    # Re-selecting the same (root, subject, date) within a session resumes at the persisted
    # position instead of clobbering it back to image 0. The first select of a fresh process
    # (the auto-open on app load) starts at image 0 rather than resurfacing a prior session's
    # position.
    global _selected_this_session
    prev = store.state.dataset
    same_identity = (
        _selected_this_session
        and prev.dataset_root == str(root)
        and prev.date == req.date
        and prev.subject == req.subject
    )
    index = prev.current_image_index if same_identity else 0
    index = max(0, min(index, len(image_list) - 1)) if image_list else 0
    _selected_this_session = True

    selection = DatasetSelection(
        project_root=str(project_root),
        dataset_root=str(root),
        subject=req.subject,
        date=req.date,
        image_list=image_list,
        current_image_index=index,
        images_dir=images_dir,
        annotations_dir=annotations_dir,
        predictions_dir=predictions_dir,
    )
    # Adopted here, immediately before the mutate it names and with no await between: an
    # exception raised while gathering the selection above leaves both still naming the old root.
    store.set_binding_generation(generation)
    await store.mutate({"dataset": selection})

    # Advisory only (never rejects): does the resolved (subject, date) actually have any labels /
    # the (model, date) any predictions? Empty label files count as present (confirmed
    # negatives), and starting a brand-new annotation on an unlabelled date is still allowed,
    # so we don't block; we just tell the caller (agent or GUI) the canvas will start empty
    # instead of leaving a silent blank canvas.
    annotations_present = False
    label_problem: Optional[str] = None
    if req.date:
        from tcip_annotation.json_io import UnreadableLabelDocument

        labels_this_date: list[str] = []
        try:
            # The one guard load_classes and the dataset tree apply to this directory: a
            # directory they refuse is reported here, never scanned.
            assert_path_allowed(annotations_dir or "")
        except ValueError as exc:
            label_problem = str(exc)
        else:
            try:
                labels_this_date = subjects_with_labels(
                    root, req.date, reader=cached_label_annotations)
            except UnreadableLabelDocument as exc:
                # Advisory only, stated above: an unreadable label must not block a selection.
                label_problem = str(exc)
        if req.subject:
            annotations_present = req.subject in labels_this_date
    predictions_present = bool(
        req.model_name and req.date and req.model_name in models_with_predictions(root, req.date)
    )
    return {
        "status": "ok",
        "selection": selection.model_dump(mode="json"),
        # The canvas-open binding's current generation, for the client to adopt in the same
        # store update as the selection: see CanvasStatePayload.binding_generation.
        "generation": generation,
        "annotations_present": annotations_present,
        "predictions_present": predictions_present,
        "label_problem": label_problem,
    }


class NavRequest(BaseModel):
    current_image_index: int


@router.post("/nav")
async def set_current_image(req: NavRequest) -> dict:
    """Persist the browser's current image position into ``GuiState.dataset``.

    The frontend debounces this so rapid arrow-key nav doesn't flood the store; the
    agent reads the resulting index via ``view_gui_state`` (last image the human
    looked at). Merges into the live dataset so the other selection fields survive.
    """
    dataset = store.state.dataset
    n = len(dataset.image_list)
    index = req.current_image_index
    if n and not (0 <= index < n):
        raise HTTPException(400, f"index {index} out of range for {n} images")
    await store.mutate({"dataset": dataset.model_copy(update={"current_image_index": index})})
    return {"status": "ok", "current_image_index": index}

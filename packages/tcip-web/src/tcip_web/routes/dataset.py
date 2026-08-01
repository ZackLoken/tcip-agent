"""Dataset discovery + selection routes.

The frontend hits these to:
  * discover what's available under a project root,
  * list images for a given (dataset_root, subject, date),
  * read and persist the ``GuiState.dataset`` selection.

Convention — the canonical layout (see :mod:`tcip_mcp.dataset_layout`):

    <dataset_root>/
        images/<date>/*.JPG
        annotations/<date>/<stem>.json          # ground truth, one file per image (all subjects)
        predictions/<model>/<date>/<stem>.json  # model outputs
        classes.json                            # the nested subject/attribute registry
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tcip_mcp.dataset_layout import (
    annotation_dir,
    list_subjects,
    models_with_predictions,
    prediction_dir,
    subjects_with_labels,
)
from tcip_mcp.pipelines.image_utils import BandGroupRef, list_logical_images
from tcip_web.paths import safe_join
from tcip_web.state import DatasetSelection, store

router = APIRouter(prefix="/api/dataset", tags=["dataset"])


def _logical_image_names(date_dir: Path) -> list[str]:
    """Every logical image's display name under ``date_dir`` — a plain file's own name, or (for a
    ``.bandgroup``-grouped capture) its manifest's filename. Folds sibling band files into one
    grouped entry per capture so the GUI's gallery shows 16 composites, not 64 grayscale frames."""
    return sorted(
        src.manifest_path.name if isinstance(src, BandGroupRef) else src.name
        for src in list_logical_images(date_dir).values()
    )

# Whether this server process has selected a dataset yet. Resuming the persisted image index
# is helpful within a running session, but a fresh process opening a project should start at
# the first image — resuming a prior session's position across a restart reads as stale state.
_selected_this_session = False


class DatasetTree(BaseModel):
    dataset_root: str
    dates_with_images: list[str]
    # Every subject the dataset's registry (``classes.json``) declares, e.g.
    # ["catkin", "bush"]. This is *what a label set is about*, not the shape kind; a label file
    # names its subject on each annotation rather than in the path.
    subjects: list[str]
    model_names: list[str]       # every model present anywhere, e.g. ["baseline"]
    # Per-date availability: the subjects that actually have labels / models that actually
    # have predictions on each date. The GUI's subject/model pickers filter to these so a
    # date with no catkin labels doesn't offer "catkin" (which would open an empty canvas).
    subjects_by_date: dict[str, list[str]]
    models_by_date: dict[str, list[str]]


def _list_children(p: Path) -> list[str]:
    if not p.is_dir():
        return []
    return sorted(e.name for e in p.iterdir() if e.is_dir() and not e.name.startswith("."))


# ── /tree cache ────────────────────────────────────────────────────────────
# subjects_with_labels/models_with_predictions each re-list annotations/ or predictions/ and
# scan every per-image label file per date, so a naive /tree is an iterdir storm on a dataset
# with many dates. Cache the built tree per dataset_root, keyed by a signature of every
# directory the computation reads (stat-only, no listing) — a write inside any of those date
# dirs bumps its own mtime_ns and invalidates the entry. Bounded to a handful of recent roots.
_TREE_CACHE_MAX = 64
_tree_cache: "OrderedDict[str, tuple[tuple, DatasetTree]]" = OrderedDict()


def _dir_mtime_ns(p: Path) -> int:
    try:
        return p.stat().st_mtime_ns
    except OSError:
        return -1


def _tree_signature(root: Path, dates: list[str], models: list[str]) -> tuple:
    sig = [
        _dir_mtime_ns(root / "images"),
        _dir_mtime_ns(root / "annotations"),
        _dir_mtime_ns(root / "models"),
        _dir_mtime_ns(root / "predictions"),
        _dir_mtime_ns(root / "classes.json"),
    ]
    for d in dates:
        sig.append(_dir_mtime_ns(annotation_dir(root, d)))
        for model in models:
            sig.append(_dir_mtime_ns(prediction_dir(root, model, d)))
    return tuple(sig)


@router.get("/tree")
def get_dataset_tree(dataset_root: str) -> DatasetTree:
    """Return the high-level tree (dates, subjects, models) for a dataset."""
    root = Path(dataset_root)
    if not root.is_dir():
        raise HTTPException(404, f"dataset_root not found: {dataset_root}")

    dates = _list_children(root / "images")
    # Subjects come from the dataset registry, not from listing annotations/ — that dir now holds
    # date buckets, not subject dirs.
    subjects = list_subjects(root)
    model_names = sorted(
        set(_list_children(root / "models")) | set(_list_children(root / "predictions"))
    )

    key = str(root)
    signature = _tree_signature(root, dates, model_names)
    cached = _tree_cache.get(key)
    if cached is not None and cached[0] == signature:
        _tree_cache.move_to_end(key)
        return cached[1]

    tree = DatasetTree(
        dataset_root=str(root),
        dates_with_images=dates,
        subjects=subjects,
        # A model is selectable if it has a checkpoint dir and/or a predictions dir.
        model_names=model_names,
        subjects_by_date={d: subjects_with_labels(root, d) for d in dates},
        models_by_date={d: models_with_predictions(root, d) for d in dates},
    )
    _tree_cache[key] = (signature, tree)
    _tree_cache.move_to_end(key)
    if len(_tree_cache) > _TREE_CACHE_MAX:
        _tree_cache.popitem(last=False)
    return tree


@router.get("/images")
def list_images(dataset_root: str, date: str) -> dict:
    """List image files on a specific date."""
    try:
        date_dir = safe_join(dataset_root, "images", date)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not date_dir.is_dir():
        raise HTTPException(404, f"images/{date} not found under {dataset_root}")
    items = _logical_image_names(date_dir)
    return {"dataset_root": dataset_root, "date": date, "images": items, "count": len(items)}


class SelectionRequest(BaseModel):
    project_root: str
    dataset_root: str
    subject: Optional[str] = None
    date: Optional[str] = None
    model_name: Optional[str] = None


@router.post("/select")
async def select_dataset(req: SelectionRequest) -> dict:
    """Set the active dataset for the GUI; broadcasts a state delta."""
    root = Path(req.dataset_root)
    if not root.is_dir():
        raise HTTPException(404, f"dataset_root not found: {req.dataset_root}")

    # Rehydrate any persisted GUI state for this project first (so backend state
    # survives a restart), then apply the fresh selection on top via mutate().
    store.load_from_disk(Path(req.project_root))

    image_list: list[str] = []
    if req.date:
        date_dir = root / "images" / req.date
        if date_dir.is_dir():
            image_list = _logical_image_names(date_dir)

    # Canonical layout (see tcip_mcp.dataset_layout) — the single source of truth shared with the
    # agent tools, so agent writes land where the GUI reads. One file per image now holds every
    # subject, so the label/prediction dirs carry no subject or task segment.
    annotations_dir = str(annotation_dir(root, req.date)) if req.date else None
    predictions_dir = (
        str(prediction_dir(root, req.model_name, req.date)) if req.model_name and req.date else None
    )

    # Re-selecting the same (root, subject, date) within a session resumes at the persisted
    # position instead of clobbering it back to image 0. The first select of a fresh process
    # (the auto-open on app load) starts at image 0 rather than resurfacing a prior session's
    # position — that stale resume is bug #1 (opening a project landed on image 3/112).
    global _selected_this_session
    prev = store.state.dataset
    same_identity = (
        _selected_this_session
        and prev.dataset_root == req.dataset_root
        and prev.date == req.date
        and prev.subject == req.subject
    )
    index = prev.current_image_index if same_identity else 0
    index = max(0, min(index, len(image_list) - 1)) if image_list else 0
    _selected_this_session = True

    selection = DatasetSelection(
        project_root=req.project_root,
        dataset_root=req.dataset_root,
        subject=req.subject,
        date=req.date,
        image_list=image_list,
        current_image_index=index,
        annotations_dir=annotations_dir,
        predictions_dir=predictions_dir,
    )
    await store.mutate({"dataset": selection})

    # Advisory only (never rejects): does the resolved (subject, date) actually have any labels /
    # the (model, date) any predictions? Empty label files count as present (confirmed
    # negatives), and starting a brand-new annotation on an unlabelled date is still allowed —
    # so we don't block; we just tell the caller (agent or GUI) the canvas will start empty
    # instead of leaving a silent blank canvas.
    annotations_present = bool(
        req.subject
        and req.date
        and req.subject in subjects_with_labels(root, req.date)
    )
    predictions_present = bool(
        req.model_name and req.date and req.model_name in models_with_predictions(root, req.date)
    )
    return {
        "status": "ok",
        "selection": selection.model_dump(mode="json"),
        "annotations_present": annotations_present,
        "predictions_present": predictions_present,
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


@router.get("/state")
def get_state_snapshot() -> dict:
    """Return a JSON snapshot of the full :class:`GuiState` (debugging / replay)."""
    return store.snapshot()

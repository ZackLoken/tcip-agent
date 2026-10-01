"""Dataset routes: what a dataset root holds (its dates and subjects, through
:mod:`tcip_mcp.dataset_layout`, and its published buckets, through :mod:`tcip_mcp.buckets`), the ``GuiState.dataset`` selection, and the current image position
within it."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tcip_mcp.buckets import buckets_by_date
from tcip_mcp.dataset_layout import (
    annotation_dir,
    annotation_root,
    image_root,
    list_dates,
    list_subjects,
    prediction_root,
    subjects_path,
    subjects_with_labels,
)
from tcip_web.label_annotations_cache import cached_label_annotations
from tcip_mcp.web_client import selection_for
from tcip_web.paths import allowed_path, assert_path_allowed
from tcip_web.state import store

router = APIRouter(prefix="/api/dataset", tags=["dataset"])


# Whether this server process has selected a dataset yet. Resuming the persisted image index
# is helpful within a running session, but a fresh process opening a project should start at
# the first image: resuming a prior session's position across a restart reads as stale state.
_selected_this_session = False


class DatasetTree(BaseModel):
    dataset_root: str
    dates_with_images: list[str]
    # Every subject the dataset's registry (``subjects.json``) declares, e.g. ["tree", "fruit"].
    subjects: list[str]
    # The subjects that have labels on each date, so the picker never opens an empty canvas.
    subjects_by_date: dict[str, list[str]]
    # Each date's published buckets: name (its path under predictions/) to directory.
    prediction_dirs: dict[str, dict[str, str]]
    # The first date whose labels would not read, naming the file (mirrors ProjectSummary's own
    # site_problem). The tree still lists every other date; a corrupt label costs one date.
    label_problem: Optional[str] = None


# ── /tree cache ────────────────────────────────────────────────────────────
# subjects_with_labels and the bucket walk each re-list annotations/ or predictions/ and
# scan every per-image file per date, so a naive /tree is an iterdir storm on a dataset
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
    """``subjects_with_labels`` per date, and the first date's problem when one won't read. A date
    whose labels won't read, or whose annotations directory the path guard refuses, lists an empty
    subject list.
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


def _tree_signature(root: Path, dates: list[str], buckets: list[Path]) -> tuple:
    sig = [
        _dir_mtime_ns(image_root(root)),
        _dir_mtime_ns(annotation_root(root)),
        _dir_mtime_ns(prediction_root(root)),
        _dir_mtime_ns(subjects_path(root)),
    ]
    sig.extend(_dir_mtime_ns(annotation_dir(root, d)) for d in dates)
    sig.extend(_dir_mtime_ns(b) for b in buckets)
    return tuple(sig)


@router.get("/tree")
def get_dataset_tree(dataset_root: str) -> DatasetTree:
    """Return the high-level tree (dates, subjects, buckets) for a dataset."""
    root = allowed_path(dataset_root)
    if not root.is_dir():
        raise HTTPException(404, f"dataset_root not found: {dataset_root}")

    dates = list_dates(root)
    # Subjects come from the dataset registry, not from listing annotations/: that dir now holds
    # date buckets, not subject dirs.
    subjects = list_subjects(root)
    by_date = buckets_by_date(root, dates)

    # Read every call, not from the cache below: a label edited in place leaves the
    # directory's own mtime untouched, and label_problem must never answer from stale content.
    subjects_by_date, label_problem = _subjects_by_date(root, dates)

    key = str(root)
    signature = _tree_signature(root, dates, [Path(p) for named in by_date.values()
                                              for p in named.values()])
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
        subjects_by_date=subjects_by_date,
        prediction_dirs=by_date,
        label_problem=label_problem,
    )
    _tree_cache[key] = (signature, tree)
    _tree_cache.move_to_end(key)
    if len(_tree_cache) > _TREE_CACHE_MAX:
        _tree_cache.popitem(last=False)
    return tree


class SelectionRequest(BaseModel):
    dataset_root: str
    subject: Optional[str] = None
    date: Optional[str] = None
    # The bucket to review: a directory the tree's prediction_dirs serves, or staged documents.
    predictions_dir: Optional[str] = None


@router.post("/select")
async def select_dataset(req: SelectionRequest) -> dict:
    """Set the dataset the GUI looks at inside the open project; broadcasts a state delta.
    Answers 409 while no project is open."""
    from tcip_mcp.pipelines.image_utils import AmbiguousImageStem

    store.open_root()
    root = allowed_path(req.dataset_root)
    if not root.is_dir():
        raise HTTPException(404, f"dataset_root not found: {req.dataset_root}")

    # Re-selecting the same (root, subject, date) in a session resumes at the held position;
    # the first select of a fresh process starts at image 0.
    global _selected_this_session
    prev = store.state.dataset
    same_identity = (
        _selected_this_session
        and prev.dataset_root == str(root)
        and prev.date == req.date
        and prev.subject == req.subject
    )
    _selected_this_session = True
    try:
        predictions_dir = str(allowed_path(req.predictions_dir)) if req.predictions_dir else None
        selection = selection_for(root, req.subject, req.date, predictions_dir,
                                  prev.current_image_index if same_identity else 0)
    except AmbiguousImageStem as exc:
        raise HTTPException(400, str(exc)) from exc
    await store.mutate({"dataset": selection})
    annotations_dir = selection.annotations_dir

    # Advisory only (never rejects): does the resolved (subject, date) actually have any labels /
    # the (model, date) any predictions? Empty label files count as present (confirmed
    # negatives), and starting a brand-new annotation on an unlabeled date is still allowed,
    # so we don't block; we just tell the caller (agent or GUI) the canvas will start empty
    # instead of leaving a silent blank canvas.
    annotations_present = False
    label_problem: Optional[str] = None
    if req.date:
        from tcip_annotation.json_io import UnreadableLabelDocument

        labels_this_date: list[str] = []
        try:
            # The one guard load_subjects and the dataset tree apply to this directory: a
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
    predictions_present = bool(selection.prediction_paths)
    return {
        "status": "ok",
        "selection": selection.model_dump(mode="json"),
        "annotations_present": annotations_present,
        "predictions_present": predictions_present,
        "label_problem": label_problem,
    }


class NavRequest(BaseModel):
    current_image_index: int


@router.post("/nav")
async def set_current_image(req: NavRequest) -> dict:
    """Persist the browser's current image position into ``GuiState.dataset``, merging into the
    live dataset so the other selection fields survive.
    """
    dataset = store.state.dataset
    n = len(dataset.image_list)
    index = req.current_image_index
    if n and not (0 <= index < n):
        raise HTTPException(400, f"index {index} out of range for {n} images")
    await store.mutate({"dataset": dataset.model_copy(update={"current_image_index": index})})
    return {"status": "ok", "current_image_index": index}

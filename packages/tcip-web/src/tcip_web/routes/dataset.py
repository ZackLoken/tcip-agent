"""Dataset routes: what a dataset root holds (its dates and subjects, through
:mod:`tcip_mcp.dataset_layout`, and its published buckets, through :mod:`tcip_mcp.buckets`), the
``GuiState.dataset`` selection, and the current image position within it."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tcip_mcp.buckets import buckets_by_date
from tcip_mcp.dataset_layout import capture_subjects, list_dates, list_subjects
from tcip_mcp.web_client import selection_for
from tcip_web.paths import allowed_path
from tcip_web.state import store

router = APIRouter(prefix="/api/dataset", tags=["dataset"])


# Whether this process has selected a dataset yet: a fresh process opening a project starts at
# the first image, since resuming a prior session's position across a restart reads as stale.
_selected_this_session = False


class DatasetTree(BaseModel):
    dataset_root: str
    dates_with_images: list[str]
    # Every subject the dataset's registry (``subjects.json``) declares, e.g. ["tree", "fruit"].
    subjects: list[str]
    # The subjects that have labels on each date, so the picker never opens an empty canvas.
    subjects_by_date: dict[str, list[str]]
    # Each date's published buckets, by name.
    buckets_by_date: dict[str, list[str]]
    # The first date whose labels would not read, naming the documents; every other document's
    # subjects are still listed.
    label_problem: Optional[str] = None


def _label_problem(capture: str, unreadable: list[str]) -> Optional[str]:
    """The line naming ``capture``'s label documents that will not read, if any."""
    if not unreadable:
        return None
    return f"capture {capture}: label document(s) {', '.join(unreadable)} will not read"


def _subjects_by_date(root, dates: list[str]) -> tuple[dict[str, list[str]], Optional[str]]:
    """:func:`~tcip_mcp.dataset_layout.capture_subjects` per date, and the first date's
    :func:`_label_problem`."""
    by_date: dict[str, list[str]] = {}
    problem: Optional[str] = None
    for d in dates:
        by_date[d], unreadable = capture_subjects(root, d)
        problem = problem or _label_problem(d, unreadable)
    return by_date, problem


@router.get("/tree")
def get_dataset_tree(dataset_root: str) -> DatasetTree:
    """Return the high-level tree (dates, subjects, buckets) for a dataset."""
    root = allowed_path(dataset_root)
    if not root.is_dir():
        raise HTTPException(404, f"dataset_root not found: {dataset_root}")
    dates = list_dates(root)
    subjects_by_date, label_problem = _subjects_by_date(root, dates)
    return DatasetTree(
        dataset_root=str(root), dates_with_images=dates, subjects=list_subjects(root),
        subjects_by_date=subjects_by_date, buckets_by_date=buckets_by_date(root, dates),
        label_problem=label_problem)


class SelectionRequest(BaseModel):
    dataset_root: str
    subject: Optional[str] = None
    # The capture: a date folder, a literal bucket or ``UNDATED_BUCKET``.
    date: str
    # The name of the bucket to review, published under dataset_root.
    bucket: Optional[str] = None


@router.post("/select")
async def select_dataset(req: SelectionRequest) -> dict:
    """Set the dataset the GUI looks at inside the open project; broadcasts a state delta.
    Answers 409 while no project is open, and 400 for a bucket name no bucket is published
    under, a ``date`` that is no capture name or a capture holding two images of one stem.
    Whether the (subject, date) has labels and the bucket a document for these images is
    advisory, never a refusal: a brand-new annotation on an unlabeled date starts empty."""
    from tcip_mcp.buckets import read_bucket

    project, held = store.held_state()
    root = allowed_path(req.dataset_root)
    if not root.is_dir():
        raise HTTPException(404, f"dataset_root not found: {req.dataset_root}")

    # Re-selecting the same (root, subject, date) in a session resumes at the held position;
    # the first select of a fresh process starts at image 0.
    global _selected_this_session
    prev = held.dataset
    same_identity = (
        _selected_this_session
        and prev.dataset_root == str(root)
        and prev.date == req.date
        and prev.subject == req.subject
    )
    _selected_this_session = True
    documents = read_bucket(root, req.bucket).documents if req.bucket else {}
    try:
        selection = selection_for(root, req.subject, req.date, req.bucket,
                                  prev.current_image_index if same_identity else 0)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    await store.mutate({"dataset": selection}, project=project)

    found, unreadable = capture_subjects(root, req.date)
    return {
        "status": "ok",
        "selection": selection.model_dump(mode="json"),
        "annotations_present": req.subject in found,
        "predictions_present": bool(set(documents.values()) & set(selection.image_list)),
        "label_problem": _label_problem(req.date, unreadable),
    }


class NavRequest(BaseModel):
    current_image_index: int


@router.post("/nav")
async def set_current_image(req: NavRequest) -> dict:
    """Persist the browser's current image position into ``GuiState.dataset``: the open project
    and its selection read as one value (:meth:`~tcip_web.state.StateStore.held_state`), the
    index replaced, held once :meth:`~tcip_web.state.StateStore.mutate` admits that project
    under the lock."""
    project, held = store.held_state()
    dataset = held.dataset
    n = len(dataset.image_list)
    index = req.current_image_index
    if n and not (0 <= index < n):
        raise HTTPException(400, f"index {index} out of range for {n} images")
    await store.mutate({"dataset": dataset.model_copy(update={"current_image_index": index})},
                       project=project)
    return {"status": "ok", "current_image_index": index}

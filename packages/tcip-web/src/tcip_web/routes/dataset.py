"""Dataset routes: what a dataset root holds, the dataset selection and the image position.

A dataset root holds its dates and subjects, through :mod:`tcip_mcp.dataset_layout`, and its
published buckets, through :mod:`tcip_mcp.buckets`; the selection is ``GuiState.dataset``, and
the position is the current image's within it.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from tcip_mcp.buckets import buckets_by_date
from tcip_mcp.dataset_layout import capture_subjects, list_dates, list_subjects
from tcip_mcp.web_client import (
    CaptureChoice,
    DatasetChoice,
    DatasetSelection,
    GuiState,
    selection_for,
)
from tcip_web.paths import allowed_path
from tcip_web.state import selected, store

router = APIRouter(prefix="/api/dataset", tags=["dataset"])


# Whether this process has selected a dataset yet: a fresh process opening a project starts at
# the first image, since resuming a prior session's position across a restart reads as stale.
_selected_this_session = False


class DatasetTree(BaseModel):
    """A dataset root's dates, subjects, subjects per date and buckets per date."""

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
    label_problem: str | None


def _label_problem(capture: str, unreadable: list[str]) -> str | None:
    """Return the line naming ``capture``'s label documents that will not read, if any."""
    if not unreadable:
        return None
    return f"capture {capture}: label document(s) {', '.join(unreadable)} will not read"


def _subjects_by_date(
    root: Path, dates: list[str]
) -> tuple[dict[str, list[str]], str | None]:
    """Return each date's labeled subjects and the first label problem encountered.

    The subjects are :func:`~tcip_mcp.dataset_layout.capture_subjects` per date; the problem is
    the first label problem encountered (:func:`_label_problem`), in date order.
    """
    by_date: dict[str, list[str]] = {}
    problem: str | None = None
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
        dataset_root=str(root),
        dates_with_images=dates,
        subjects=list_subjects(root),
        subjects_by_date=subjects_by_date,
        buckets_by_date=buckets_by_date(root, dates),
        label_problem=label_problem,
    )


class SelectAnswer(BaseModel):
    """A selection made: the selection held, and what its capture's documents advise of it.

    ``annotations_present`` is whether the capture's labels hold the subject,
    ``predictions_present`` whether the bucket holds a document for one of its images, and
    ``label_problem`` the line naming the label documents that will not read, if any.
    """

    model_config = ConfigDict(json_schema_mode_override="serialization")

    selection: DatasetSelection
    annotations_present: bool
    predictions_present: bool
    label_problem: str | None


@router.post("/select")
async def select_dataset(req: CaptureChoice) -> SelectAnswer:
    """Set the open project's ``GuiState.dataset`` selection; broadcasts a state delta.

    Answers 409 while no project is open, and 400 for a bucket name no bucket is published
    under, a ``date`` that is no capture name or a capture holding two images of one stem.
    Whether the (subject, date) has labels and the bucket a document for these images is
    advisory, never a refusal: a brand-new annotation on an unlabeled date starts empty.
    """
    from tcip_mcp.buckets import read_bucket

    project, held = store.held_state()
    root = allowed_path(req.dataset_root)
    if not root.is_dir():
        raise HTTPException(404, f"dataset_root not found: {req.dataset_root}")

    # Re-selecting the same (root, subject, date) in a session resumes at the held position;
    # the first select of a fresh process starts at image 0.
    global _selected_this_session
    prev = held.dataset
    resumed = (
        prev.current_image_index
        if _selected_this_session
        and prev is not None
        and (prev.dataset_root, prev.date, prev.subject)
        == (str(root), req.date, req.subject)
        else 0
    )
    _selected_this_session = True
    documents = read_bucket(root, req.bucket).documents if req.bucket else {}
    try:
        selection = selection_for(
            DatasetChoice.model_validate(
                {
                    **req.model_dump(),
                    "dataset_root": str(root),
                    "current_image_index": resumed,
                }
            )
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    await store.mutate(lambda _held: {"dataset": selection}, project=project)

    found, unreadable = capture_subjects(root, req.date)
    return SelectAnswer(
        selection=selection,
        annotations_present=req.subject in found,
        predictions_present=bool(
            set(documents.values())
            & {image.name for image in selection.images}
        ),
        label_problem=_label_problem(req.date, unreadable),
    )


class NavRequest(BaseModel):
    """The current image position within the selection."""

    current_image_index: int


@router.post("/nav")
async def set_current_image(req: NavRequest) -> NavRequest:
    """Persist the current image position into ``GuiState.dataset`` and answer it.

    The open project is read once (:meth:`~tcip_web.state.StateStore.held`) and the index
    replaced in the selection :meth:`~tcip_web.state.StateStore.mutate` holds once it admits
    that project under the lock, so the project's refusal answers first; then no capture
    selected (:func:`~tcip_web.state.selected`) or an index outside its images answers 400.
    """
    index = req.current_image_index

    def navigated(state: GuiState) -> dict[str, DatasetSelection]:
        dataset = selected(state)
        n = len(dataset.images)
        if n and not (0 <= index < n):
            raise HTTPException(
                400, f"index {index} out of range for {n} images"
            )
        return {
            "dataset": dataset.model_copy(
                update={"current_image_index": index}
            )
        }

    await store.mutate(navigated, project=store.held())
    return req

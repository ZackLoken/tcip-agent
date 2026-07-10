"""Dataset discovery + selection routes.

The frontend hits these to:
  * discover what's available under a project root,
  * list images for a given (dataset_root, annotation_type, date),
  * read and persist the ``GuiState.dataset`` selection.

Convention — the canonical layout (see :mod:`tcip_mcp.dataset_layout`):

    <dataset_root>/
        images/<date>/*.JPG
        annotations/<type>/<date>/{detect,segment}/*.txt
        predictions/<model>/<date>/{detect,segment}/*.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tcip_mcp.dataset_layout import (
    annotation_dir,
    models_with_predictions,
    prediction_dir,
    traits_with_labels,
)
from tcip_web.paths import safe_join
from tcip_web.state import DatasetSelection, store

router = APIRouter(prefix="/api/dataset", tags=["dataset"])

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff", ".bmp")


class DatasetTree(BaseModel):
    dataset_root: str
    dates_with_images: list[str]
    annotation_types: list[str]  # every trait present anywhere, e.g. ["catkin", "bush"]
    model_names: list[str]       # every model present anywhere, e.g. ["baseline"]
    # Per-date availability: the traits that actually have labels / models that actually
    # have predictions on each date. The GUI's trait/model pickers filter to these so a
    # date with no catkin labels doesn't offer "catkin" (which would open an empty canvas).
    traits_by_date: dict[str, list[str]]
    models_by_date: dict[str, list[str]]


def _list_children(p: Path) -> list[str]:
    if not p.is_dir():
        return []
    return sorted(e.name for e in p.iterdir() if e.is_dir() and not e.name.startswith("."))


@router.get("/tree")
def get_dataset_tree(dataset_root: str) -> DatasetTree:
    """Return the high-level tree (dates, annotation types, models) for a dataset."""
    root = Path(dataset_root)
    if not root.is_dir():
        raise HTTPException(404, f"dataset_root not found: {dataset_root}")
    dates = _list_children(root / "images")
    return DatasetTree(
        dataset_root=str(root),
        dates_with_images=dates,
        annotation_types=_list_children(root / "annotations"),
        # A model is selectable if it has a checkpoint dir and/or a predictions dir.
        model_names=sorted(
            set(_list_children(root / "models")) | set(_list_children(root / "predictions"))
        ),
        traits_by_date={d: traits_with_labels(root, d) for d in dates},
        models_by_date={d: models_with_predictions(root, d) for d in dates},
    )


@router.get("/images")
def list_images(dataset_root: str, date: str) -> dict:
    """List image files on a specific date."""
    try:
        date_dir = safe_join(dataset_root, "images", date)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not date_dir.is_dir():
        raise HTTPException(404, f"images/{date} not found under {dataset_root}")
    items = sorted(
        p.name
        for p in date_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return {"dataset_root": dataset_root, "date": date, "images": items, "count": len(items)}


class SelectionRequest(BaseModel):
    project_root: str
    dataset_root: str
    annotation_type: Optional[str] = None
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
            image_list = sorted(
                p.name
                for p in date_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )

    # Canonical layout (see tcip_mcp.dataset_layout) — the single source of truth
    # shared with the agent tools, so agent writes land where the GUI reads.
    ann_detect = (
        str(annotation_dir(root, req.annotation_type, req.date, "detect"))
        if req.annotation_type and req.date
        else None
    )
    ann_segment = (
        str(annotation_dir(root, req.annotation_type, req.date, "segment"))
        if req.annotation_type and req.date
        else None
    )
    pred_detect = (
        str(prediction_dir(root, req.model_name, req.date, "detect")) if req.model_name else None
    )
    pred_segment = (
        str(prediction_dir(root, req.model_name, req.date, "segment")) if req.model_name else None
    )

    selection = DatasetSelection(
        project_root=req.project_root,
        dataset_root=req.dataset_root,
        annotation_type=req.annotation_type,
        date=req.date,
        image_list=image_list,
        current_image_index=0,
        annotations_detect_dir=ann_detect,
        annotations_segment_dir=ann_segment,
        predictions_detect_dir=pred_detect,
        predictions_segment_dir=pred_segment,
    )
    await store.mutate({"dataset": selection})

    # Advisory only (never rejects): does the resolved (trait, date) actually have any labels /
    # the (model, date) any predictions? Empty label files count as present (confirmed
    # negatives), and starting a brand-new annotation on an unlabelled date is still allowed —
    # so we don't block; we just tell the caller (agent or GUI) the canvas will start empty
    # instead of leaving a silent blank canvas.
    annotations_present = bool(
        req.annotation_type
        and req.date
        and req.annotation_type in traits_with_labels(root, req.date)
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


@router.get("/state")
def get_state_snapshot() -> dict:
    """Return a JSON snapshot of the full :class:`GuiState` (debugging / replay)."""
    return store.snapshot()

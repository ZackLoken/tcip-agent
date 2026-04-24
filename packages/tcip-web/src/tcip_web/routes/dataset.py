"""Dataset discovery + selection routes.

The frontend hits these to:
  * discover what's available under a project root,
  * list images for a given (dataset_root, annotation_type, date),
  * read and persist the ``GuiState.dataset`` selection.

Convention (from the Valley_Farm catkin layout):

    <dataset_root>/
        images/<date>/*.JPG
        annotations/<type>/<date>/{detect,segment}/*.txt
        models/<name>/predictions/{detect,segment}/*.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tcip_web.paths import safe_join
from tcip_web.state import DatasetSelection, store

router = APIRouter(prefix="/api/dataset", tags=["dataset"])

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff", ".bmp")


class DatasetTree(BaseModel):
    dataset_root: str
    dates_with_images: list[str]
    annotation_types: list[str]  # e.g. ["catkin", "bush"]
    model_names: list[str]       # e.g. ["baseline"]


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
    return DatasetTree(
        dataset_root=str(root),
        dates_with_images=_list_children(root / "images"),
        annotation_types=_list_children(root / "annotations"),
        model_names=_list_children(root / "models"),
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

    image_list: list[str] = []
    if req.date:
        date_dir = root / "images" / req.date
        if date_dir.is_dir():
            image_list = sorted(
                p.name
                for p in date_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )

    ann_detect = (
        str(root / "annotations" / req.annotation_type / req.date / "detect")
        if req.annotation_type and req.date
        else None
    )
    ann_segment = (
        str(root / "annotations" / req.annotation_type / req.date / "segment")
        if req.annotation_type and req.date
        else None
    )
    pred_detect = (
        str(root / "models" / req.model_name / "predictions" / "detect")
        if req.model_name
        else None
    )
    pred_segment = (
        str(root / "models" / req.model_name / "predictions" / "segment")
        if req.model_name
        else None
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
    return {"status": "ok", "selection": selection.model_dump(mode="json")}


@router.get("/state")
def get_state_snapshot() -> dict:
    """Return a JSON snapshot of the full :class:`GuiState` (debugging / replay)."""
    return store.snapshot()

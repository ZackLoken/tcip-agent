"""Review routes: compute matches, walk detections, record actions, save GT.

Uses the shared :class:`tcip_annotation.ReviewEngine`; one engine instance
lives in memory per project (keyed by project_root). Review state is
persisted via the engine to ``<project_root>/.tcip/state/review_stats.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from PIL import Image
from pydantic import BaseModel

from tcip_annotation import (
    BBox,
    PredBBox,
    PredPolygon,
    Polygon,
    ReviewContext,
    ReviewDetection,
    ReviewEngine,
    compute_matches,
    parse_detect_labels,
    parse_detect_predictions,
    parse_segment_labels,
    parse_segment_predictions,
)
from tcip_annotation.utils import auto_orient_image
from tcip_web.paths import assert_path_allowed

router = APIRouter(prefix="/api/review", tags=["review"])


# ── Engine cache ──────────────────────────────────────────────────────────

_engines: dict[str, ReviewEngine] = {}


def _get_engine(project_root: str, class_names: Optional[dict[int, str]] = None) -> ReviewEngine:
    key = str(Path(project_root).resolve())
    if key not in _engines:
        state_dir = Path(project_root) / ".tcip" / "state"
        _engines[key] = ReviewEngine(state_dir=state_dir, class_names=class_names or {})
    elif class_names:
        _engines[key].class_names = class_names
    return _engines[key]


def _image_dims(path: str) -> tuple[int, int]:
    try:
        p = assert_path_allowed(path)
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc
    if not p.is_file():
        raise HTTPException(404, f"image not found: {path}")
    with Image.open(p) as raw:
        return auto_orient_image(raw).size


def _load_ctx(
    image_name: str,
    image_path: str,
    *,
    gt_detect_path: Optional[str],
    gt_segment_path: Optional[str],
    pred_detect_path: Optional[str],
    pred_segment_path: Optional[str],
) -> ReviewContext:
    w, h = _image_dims(image_path)
    ctx = ReviewContext(img_name=image_name, img_width=w, img_height=h)

    if gt_detect_path:
        boxes, _ = parse_detect_labels(gt_detect_path, w, h)
        ctx.gt_boxes = boxes
    if gt_segment_path:
        polys, _ = parse_segment_labels(gt_segment_path, w, h)
        ctx.gt_polygons = polys
    if pred_detect_path:
        pred_boxes, _ = parse_detect_predictions(pred_detect_path, w, h)
        ctx.pred_boxes = pred_boxes
    if pred_segment_path:
        pred_polys, _ = parse_segment_predictions(pred_segment_path, w, h)
        ctx.pred_polygons = pred_polys
    return ctx


# ── Request/response schemas ──────────────────────────────────────────────


class MatchesRequest(BaseModel):
    project_root: str
    image_name: str
    image_path: str
    gt_detect_path: Optional[str] = None
    gt_segment_path: Optional[str] = None
    pred_detect_path: Optional[str] = None
    pred_segment_path: Optional[str] = None
    iou_threshold: float = 0.5
    conf_threshold: float = 0.25
    filter_type: str = "all"
    filter_class: str | int = "all"
    status_filter: str = "all"


class Detection(BaseModel):
    det_type: str
    class_id: int
    conf: Optional[float]
    iou: Optional[float]
    gt_type: Optional[str]
    gt_idx: Optional[int]
    pred_type: Optional[str]
    pred_idx: Optional[int]
    bbox: tuple[float, float, float, float]
    reviewed: bool = False
    reviewed_action: Optional[str] = None


class MatchesResponse(BaseModel):
    img_width: int
    img_height: int
    n_tp: int
    n_fp: int
    n_fn: int
    detections: list[Detection]
    gt_boxes: list[dict]
    gt_polygons: list[dict]
    pred_boxes: list[dict]
    pred_polygons: list[dict]
    image_status: str  # "not_started" | "started" | "completed"


@router.post("/matches")
def compute_image_matches(req: MatchesRequest) -> MatchesResponse:
    """Compute TP/FP/FN, decorate with review status, and return everything the canvas needs."""
    ctx = _load_ctx(
        image_name=req.image_name,
        image_path=req.image_path,
        gt_detect_path=req.gt_detect_path,
        gt_segment_path=req.gt_segment_path,
        pred_detect_path=req.pred_detect_path,
        pred_segment_path=req.pred_segment_path,
    )
    engine = _get_engine(req.project_root)

    matches = compute_matches(
        gt_boxes=ctx.gt_boxes,
        gt_polygons=ctx.gt_polygons,
        pred_boxes=ctx.pred_boxes,
        pred_polygons=ctx.pred_polygons,
        iou_threshold=req.iou_threshold,
        conf_threshold=req.conf_threshold,
    )
    dets = engine.build_detection_list(
        ctx,
        matches,
        filter_type=req.filter_type,
        filter_class=req.filter_class,
        status_filter=req.status_filter,
    )
    # Decorate with review status
    out_dets: list[Detection] = []
    for d in dets:
        entry = engine.find_reviewed_entry(d, ctx)
        out_dets.append(Detection(
            det_type=d.det_type,
            class_id=d.class_id,
            conf=d.conf,
            iou=d.iou,
            gt_type=d.gt_type,
            gt_idx=d.gt_idx,
            pred_type=d.pred_type,
            pred_idx=d.pred_idx,
            bbox=d.bbox,
            reviewed=entry is not None,
            reviewed_action=entry.get("action") if entry else None,
        ))

    def _box_dict(b: BBox) -> dict:
        return {"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2, "class_id": b.class_id}

    def _poly_dict(p: Polygon) -> dict:
        return {"points": [list(pt) for pt in p.points], "class_id": p.class_id}

    def _pred_box_dict(b: PredBBox) -> dict:
        return {
            "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2,
            "class_id": b.class_id, "confidence": b.confidence,
        }

    def _pred_poly_dict(p: PredPolygon) -> dict:
        return {
            "points": [list(pt) for pt in p.points],
            "class_id": p.class_id, "confidence": p.confidence,
        }

    return MatchesResponse(
        img_width=ctx.img_width,
        img_height=ctx.img_height,
        n_tp=len(matches["tp"]),
        n_fp=len(matches["fp"]),
        n_fn=len(matches["fn"]),
        detections=out_dets,
        gt_boxes=[_box_dict(b) for b in ctx.gt_boxes],
        gt_polygons=[_poly_dict(p) for p in ctx.gt_polygons],
        pred_boxes=[_pred_box_dict(b) for b in ctx.pred_boxes],
        pred_polygons=[_pred_poly_dict(p) for p in ctx.pred_polygons],
        image_status=engine.get_image_review_status(req.image_name),
    )


class ActionPayload(BaseModel):
    project_root: str
    image_name: str
    image_path: str
    gt_detect_path: Optional[str] = None
    gt_segment_path: Optional[str] = None
    pred_detect_path: Optional[str] = None
    pred_segment_path: Optional[str] = None
    # The detection being acted on (same shape as the Detection response)
    det_type: str
    class_id: int
    conf: Optional[float] = None
    iou: Optional[float] = None
    gt_type: Optional[str] = None
    gt_idx: Optional[int] = None
    pred_type: Optional[str] = None
    pred_idx: Optional[int] = None
    bbox: tuple[float, float, float, float]
    action: str  # "accepted" | "rejected" | "edited"


@router.post("/action")
def record_action(payload: ActionPayload) -> dict:
    """Record a user's accept/reject/edit decision for one detection."""
    ctx = _load_ctx(
        image_name=payload.image_name,
        image_path=payload.image_path,
        gt_detect_path=payload.gt_detect_path,
        gt_segment_path=payload.gt_segment_path,
        pred_detect_path=payload.pred_detect_path,
        pred_segment_path=payload.pred_segment_path,
    )
    engine = _get_engine(payload.project_root)
    det = ReviewDetection(
        det_type=payload.det_type,
        class_id=payload.class_id,
        conf=payload.conf,
        iou=payload.iou,
        gt_type=payload.gt_type,
        gt_idx=payload.gt_idx,
        pred_type=payload.pred_type,
        pred_idx=payload.pred_idx,
        bbox=payload.bbox,
    )
    engine.record_detection_action(det, ctx, action=payload.action)
    return {"status": "ok"}


class BackupPayload(BaseModel):
    project_root: str
    label_dirs: list[str]


@router.post("/backup_labels")
def backup_labels(payload: BackupPayload) -> dict:
    """Backup label directories to ``<dir>/.original/`` (once per project)."""
    engine = _get_engine(payload.project_root)
    engine.backup_original_labels(*payload.label_dirs)
    return {"status": "ok", "labels_backed_up": engine.raw_state.get("labels_backed_up", False)}


class SaveGtPayload(BaseModel):
    project_root: str
    image_name: str
    image_path: str
    detect_path: Optional[str] = None
    segment_path: Optional[str] = None
    boxes: list[dict] = []        # [{x1,y1,x2,y2,class_id}]
    polygons: list[dict] = []     # [{points: [[x,y]...], class_id}]


@router.post("/save_gt")
def save_gt(payload: SaveGtPayload) -> dict:
    """Persist edited GT (post-review modification) for a single image."""
    w, h = _image_dims(payload.image_path)
    engine = _get_engine(payload.project_root)

    ctx = ReviewContext(
        img_name=payload.image_name,
        img_width=w,
        img_height=h,
        gt_boxes=[
            BBox(
                x1=b["x1"], y1=b["y1"], x2=b["x2"], y2=b["y2"],
                class_id=int(b.get("class_id", 0)),
            )
            for b in payload.boxes
        ],
        gt_polygons=[
            Polygon(
                points=[tuple(pt) for pt in p["points"]],
                class_id=int(p.get("class_id", 0)),
            )
            for p in payload.polygons
        ],
    )
    ok = engine.save_gt(
        ctx,
        detect_path=payload.detect_path,
        segment_path=payload.segment_path,
    )
    return {"status": "ok" if ok else "partial"}


@router.get("/image_status")
def get_image_status(project_root: str, image_name: str) -> dict:
    engine = _get_engine(project_root)
    return {"status": engine.get_image_review_status(image_name)}

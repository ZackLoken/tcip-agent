"""Review routes: compute matches, walk detections, record actions, save GT.

Uses the shared :class:`tcip_annotation.ReviewEngine`; one engine instance
lives in memory per project (keyed by project_root). Review state is
persisted via the engine to ``<project_root>/.tcip/state/review_stats.json``.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
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
    write_detect_labels,
    write_segment_labels,
)
from tcip_annotation.utils import auto_orient_image
from tcip_mcp.utils.atomic_io import append_jsonl, read_json
from tcip_web.paths import assert_path_allowed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/review", tags=["review"])


# ── Engine cache ──────────────────────────────────────────────────────────

_engines: dict[str, ReviewEngine] = {}


def _load_class_names(project_root: str) -> dict[int, str]:
    """Read id→name from ``<project_root>/.tcip/state/classes.json``.

    Without this the engine records ``class_{id}`` placeholders; loading the real
    names makes review_stats human-auditable (and refreshes when classes are added).
    """
    data = read_json(Path(project_root) / ".tcip" / "state" / "classes.json", default={})
    names: dict[int, str] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                cid = int(k)
            except (TypeError, ValueError):
                continue
            name = v.get("name") if isinstance(v, dict) else None
            if name:
                names[cid] = name
    return names


def _current_user() -> str:
    """Reviewer recorded on every verdict — ``TCIP_REVIEW_USER`` override else the OS user."""
    user = os.environ.get("TCIP_REVIEW_USER", "").strip()
    if user:
        return user
    try:
        import getpass

        return getpass.getuser() or "gui"
    except Exception:
        return "gui"


def _get_engine(project_root: str) -> ReviewEngine:
    key = str(Path(project_root).resolve())
    class_names = _load_class_names(project_root)
    if key not in _engines:
        state_dir = Path(project_root) / ".tcip" / "state"
        _engines[key] = ReviewEngine(
            state_dir=state_dir, class_names=class_names, current_user=_current_user()
        )
    else:
        # Refresh names so classes added mid-session appear in newly recorded entries.
        _engines[key].class_names = class_names
    return _engines[key]


def _audit(project_root: str, tool: str, arguments: dict) -> None:
    """Append a GUI review mutation to ``<project_root>/.tcip/audit.jsonl`` (best-effort).

    Review verdicts + GT writes change tracked state, so — like @audited MCP tools and
    the annotate save path — they belong in the tamper-evident log. Never fails the request.
    """
    if not project_root:
        return
    try:
        append_jsonl(
            os.path.join(project_root, ".tcip", "audit.jsonl"),
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tool": tool,
                "source": "gui",
                "arguments": arguments,
                "status": "ok",
            },
        )
    except Exception:
        pass


def _image_dims(path: str) -> tuple[int, int]:
    try:
        p = assert_path_allowed(path)
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc
    if not p.is_file():
        raise HTTPException(404, f"image not found: {path}")
    with Image.open(p) as raw:
        return auto_orient_image(raw).size


def _guard_path(path: Optional[str]) -> None:
    """403 if a client-supplied label/dir path escapes the configured image roots.

    ``save_gt`` / ``backup_labels`` write to caller-provided paths, so an exposed
    deployment (``TCIP_IMAGE_ROOTS`` set) must confine them like image serving.
    """
    if not path:
        return
    try:
        assert_path_allowed(path)
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc


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
    # Edited shape committed from the Review canvas (only for action="edited"): a box, or a
    # polygon's points. Accept/Reject don't carry these — they act on the loaded pred/gt by index.
    edited_box: Optional[tuple[float, float, float, float]] = None
    edited_polygon: Optional[list[list[float]]] = None
    # Review thresholds so the route can decide (at the same op point as the GUI) whether
    # this verdict was the last one and the image should flip to 'completed'.
    iou_threshold: float = 0.5
    conf_threshold: float = 0.25


def _apply_gt_mutation(
    ctx: ReviewContext, payload: "ActionPayload"
) -> tuple[Optional[str], Optional[int]]:
    """Author GT from a verdict; return (task changed: "detect"/"segment"/None, index the
    written shape landed at in ctx's GT list — edited/accepted writes only). Accept an FP adds
    the prediction; accept a TP/FN keeps GT; reject a TP/FN deletes that GT; reject an FP is a
    no-op; edit writes the edited shape (replacing the matched GT, or adding it)."""
    dt, act = payload.det_type, payload.action

    if act == "edited":
        if payload.edited_box is not None:
            x1, y1, x2, y2 = payload.edited_box
            nb = BBox(x1=x1, y1=y1, x2=x2, y2=y2, class_id=payload.class_id)
            if dt in ("tp", "fn") and payload.gt_type == "box" and payload.gt_idx is not None \
                    and 0 <= payload.gt_idx < len(ctx.gt_boxes):
                ctx.gt_boxes[payload.gt_idx] = nb
                return "detect", payload.gt_idx
            ctx.gt_boxes.append(nb)
            return "detect", len(ctx.gt_boxes) - 1
        if payload.edited_polygon is not None:
            npg = Polygon(points=[tuple(pt) for pt in payload.edited_polygon], class_id=payload.class_id)
            if dt in ("tp", "fn") and payload.gt_type == "polygon" and payload.gt_idx is not None \
                    and 0 <= payload.gt_idx < len(ctx.gt_polygons):
                ctx.gt_polygons[payload.gt_idx] = npg
                return "segment", payload.gt_idx
            ctx.gt_polygons.append(npg)
            return "segment", len(ctx.gt_polygons) - 1
        return None, None

    if act == "rejected" and dt in ("tp", "fn") and payload.gt_idx is not None:
        if payload.gt_type == "box" and 0 <= payload.gt_idx < len(ctx.gt_boxes):
            ctx.gt_boxes.pop(payload.gt_idx)
            return "detect", None
        if payload.gt_type == "polygon" and 0 <= payload.gt_idx < len(ctx.gt_polygons):
            ctx.gt_polygons.pop(payload.gt_idx)
            return "segment", None
        return None, None

    if act == "accepted" and dt == "fp" and payload.pred_idx is not None:
        if payload.pred_type == "box" and 0 <= payload.pred_idx < len(ctx.pred_boxes):
            pb = ctx.pred_boxes[payload.pred_idx]
            ctx.gt_boxes.append(BBox(x1=pb.x1, y1=pb.y1, x2=pb.x2, y2=pb.y2, class_id=pb.class_id))
            return "detect", len(ctx.gt_boxes) - 1
        if payload.pred_type == "polygon" and 0 <= payload.pred_idx < len(ctx.pred_polygons):
            pp = ctx.pred_polygons[payload.pred_idx]
            ctx.gt_polygons.append(Polygon(points=list(pp.points), class_id=pp.class_id))
            return "segment", len(ctx.gt_polygons) - 1
    return None, None  # accept TP/FN and reject FP leave GT untouched


@router.post("/action")
def record_action(payload: ActionPayload) -> dict:
    """Record a user's accept/reject/edit decision; auto-complete the image when done."""
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

    # Author GT on a copy so the guard can 400 before anything is recorded, and so the
    # verdict entry is recorded against the pristine ctx (its bbox lookups read gt_idx).
    work = replace(ctx, gt_boxes=list(ctx.gt_boxes), gt_polygons=list(ctx.gt_polygons))
    changed, landed_idx = _apply_gt_mutation(work, payload)
    if changed == "detect" and not payload.gt_detect_path:
        raise HTTPException(400, "this verdict writes detect ground truth, but no detect annotations path was provided")
    if changed == "segment" and not payload.gt_segment_path:
        raise HTTPException(400, "this verdict writes segment ground truth, but no segment annotations path was provided")

    # An edited verdict rewrites the GT bbox, so key the entry to the post-edit geometry —
    # otherwise the next reload's spatial lookup misses it and the detection reads unreviewed.
    norm_det = norm_ctx = None
    if payload.action == "edited" and changed is not None and landed_idx is not None:
        kind = "box" if changed == "detect" else "polygon"
        norm_det = replace(det, gt_type=kind, gt_idx=landed_idx)
        norm_ctx = work
    engine.record_detection_action(
        det, ctx, action=payload.action, norm_det=norm_det, norm_ctx=norm_ctx
    )

    # Write only the changed label file (keep_empty: an emptied GT stays a 0-byte file,
    # not deleted). accept-TP/FN and reject-FP are no-ops.
    if changed == "detect" and payload.gt_detect_path:
        _guard_path(payload.gt_detect_path)
        os.makedirs(os.path.dirname(payload.gt_detect_path) or ".", exist_ok=True)
        write_detect_labels(
            payload.gt_detect_path, work.gt_boxes, ctx.img_width, ctx.img_height, keep_empty=True
        )
    elif changed == "segment" and payload.gt_segment_path:
        _guard_path(payload.gt_segment_path)
        os.makedirs(os.path.dirname(payload.gt_segment_path) or ".", exist_ok=True)
        write_segment_labels(
            payload.gt_segment_path, work.gt_polygons, ctx.img_width, ctx.img_height, keep_empty=True
        )

    # Annotation status to sync client-side (only when GT changed); an emptied GT reads as
    # "unannotated" — a negative needs an explicit Complete, not just an empty file.
    annotation_status: Optional[str] = None
    if changed is not None:
        annotation_status = "partial" if (work.gt_boxes or work.gt_polygons) else "unannotated"

    # Promote to 'completed' once every detection at these thresholds is reviewed — the
    # only path by which a GUI review reaches 'completed'. Recompute against the (now-authored) GT.
    matches = compute_matches(
        gt_boxes=work.gt_boxes,
        gt_polygons=work.gt_polygons,
        pred_boxes=ctx.pred_boxes,
        pred_polygons=ctx.pred_polygons,
        iou_threshold=payload.iou_threshold,
        conf_threshold=payload.conf_threshold,
    )
    engine.check_image_review_complete(payload.image_name, matches)
    _audit(payload.project_root, "gui_review_action", {
        "image_name": payload.image_name,
        "det_type": payload.det_type,
        "class_id": payload.class_id,
        "action": payload.action,
        "gt_changed": changed,
    })
    return {
        "status": "ok",
        "image_status": engine.get_image_review_status(payload.image_name),
        "annotation_status": annotation_status,
    }


class MarkCompletePayload(BaseModel):
    project_root: str
    image_name: str


@router.post("/mark_complete")
def mark_complete(payload: MarkCompletePayload) -> dict:
    """Manually mark an image fully reviewed (covers negatives / bulk-accept cases)."""
    engine = _get_engine(payload.project_root)
    engine.mark_image_reviewed(payload.image_name)
    _audit(payload.project_root, "gui_review_mark_complete", {"image_name": payload.image_name})
    return {"status": "ok", "image_status": engine.get_image_review_status(payload.image_name)}


class BackupPayload(BaseModel):
    project_root: str
    label_dirs: list[str]


@router.post("/backup_labels")
def backup_labels(payload: BackupPayload) -> dict:
    """Top up ``<dir>/.original/`` — capture any label file that has no baseline yet."""
    for d in payload.label_dirs:
        _guard_path(d)
    engine = _get_engine(payload.project_root)
    n = engine.backup_original_labels(*payload.label_dirs)
    return {"status": "ok", "files_backed_up": n}


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
    _guard_path(payload.detect_path)
    _guard_path(payload.segment_path)
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
    _audit(payload.project_root, "gui_review_save_gt", {
        "image_name": payload.image_name,
        "detect_path": payload.detect_path,
        "segment_path": payload.segment_path,
        "n_boxes": len(payload.boxes),
        "n_polygons": len(payload.polygons),
    })
    return {"status": "ok" if ok else "partial"}


@router.get("/image_status")
def get_image_status(project_root: str, image_name: str) -> dict:
    engine = _get_engine(project_root)
    return {"status": engine.get_image_review_status(image_name)}


# ── Materialize a training set from review verdicts (synchronous file copy) ──


class MaterializePayload(BaseModel):
    project_root: str
    source_images_dir: str
    output_dir: str
    experiment_id: str = ""
    include_hard_negatives: bool = True
    only_completed: bool = False
    copy_files: bool = True


@router.post("/materialize")
def materialize_route(payload: MaterializePayload) -> dict:
    """Turn recorded review verdicts into a curated YOLO training set (chains into split/train)."""
    _guard_path(payload.source_images_dir)
    _guard_path(payload.output_dir)
    from tcip_mcp.tools.feedback_tools import materialize_review_dataset

    review_state_dir = str(Path(payload.project_root) / ".tcip" / "state")
    result = materialize_review_dataset(
        review_state_dir=review_state_dir,
        source_images_dir=payload.source_images_dir,
        output_dir=payload.output_dir,
        experiment_id=payload.experiment_id,
        include_hard_negatives=payload.include_hard_negatives,
        only_completed=payload.only_completed,
        copy_files=payload.copy_files,
    )
    if result.get("error"):
        raise HTTPException(400, result["error"])
    _audit(payload.project_root, "gui_materialize_review_dataset", {
        "output_dir": payload.output_dir,
        "only_completed": payload.only_completed,
        "positive": result.get("positive"),
        "hard_negative": result.get("hard_negative"),
    })
    return result


# ── Review queue: rank unreviewed images (async — loads a model + scores a dir) ──


@dataclass
class QueueJob:
    job_id: str
    status: str = "pending"  # pending | running | completed | failed
    error: Optional[str] = None
    result: dict = field(default_factory=dict)
    thread: Optional[threading.Thread] = field(default=None, repr=False)


_queue_jobs: dict[str, QueueJob] = {}
_queue_lock = threading.Lock()


class QueuePayload(BaseModel):
    project_root: str
    checkpoint_path: str
    images_dir: str
    method: str = "combined"  # uncertainty | diversity | combined
    task: str = "detection"
    budget: int = 50
    skip_reviewed: bool = True


def _queue_worker(job: QueueJob, payload: QueuePayload) -> None:
    try:
        job.status = "running"
        from tcip_mcp.tools.feedback_tools import prioritize_review_queue

        review_state_dir = str(Path(payload.project_root) / ".tcip" / "state")
        res = prioritize_review_queue(
            checkpoint_path=payload.checkpoint_path,
            images_dir=payload.images_dir,
            review_state_dir=review_state_dir,
            method=payload.method,
            task=payload.task,
            budget=payload.budget,
            skip_reviewed=payload.skip_reviewed,
        )
        if res.get("error"):
            job.status = "failed"
            job.error = res["error"]
        else:
            job.result = res
            job.status = "completed"
    except Exception as exc:
        logger.exception("review queue job %s failed", job.job_id)
        job.status = "failed"
        job.error = str(exc)


@router.post("/queue/launch")
def launch_queue(payload: QueuePayload) -> dict:
    """Kick off active-learning ranking of unreviewed images on a background thread."""
    _guard_path(payload.images_dir)
    _guard_path(payload.checkpoint_path)  # fed to build_predictor → torch.load; confine it too
    if not Path(payload.checkpoint_path).is_file():
        raise HTTPException(404, f"checkpoint not found: {payload.checkpoint_path}")
    if not Path(payload.images_dir).is_dir():
        raise HTTPException(404, f"images_dir not found: {payload.images_dir}")

    job = QueueJob(job_id=f"rq-{uuid.uuid4().hex[:8]}")
    with _queue_lock:
        from tcip_web import jobstore

        _queue_jobs[job.job_id] = job
        jobstore.evict_terminal(_queue_jobs)  # bound the registry (results are recomputable)
    t = threading.Thread(target=_queue_worker, args=(job, payload), daemon=True)
    job.thread = t
    t.start()
    return {"status": "launched", "job_id": job.job_id}


@router.get("/queue/{job_id}")
def get_queue(job_id: str) -> dict:
    with _queue_lock:
        job = _queue_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"queue job not found: {job_id}")
    return {"job_id": job.job_id, "status": job.status, "error": job.error, "result": job.result}

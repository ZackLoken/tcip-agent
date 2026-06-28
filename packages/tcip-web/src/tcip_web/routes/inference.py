"""Inference routes: async SAHI-tiled runs + live progress WebSocket.

Jobs run on a background thread. Each job writes YOLO-format predictions
(one ``<stem>.txt`` per image, ``class conf cx cy w h``) to ``output_dir``
so they plug straight into the Review tab and the per-plant curve pipeline.

The existing ``run_inference`` MCP tool does plain ultralytics predict; for
Phase 1 we need SAHI tiling to preserve small-object recall, so this module
runs SAHI directly when ``sahi=true``. Plain mode falls back to the MCP
tool.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/inference", tags=["inference"])

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


# ── Job registry ────────────────────────────────────────────────────────


@dataclass
class InferenceJob:
    job_id: str
    checkpoint_path: str
    images_dir: str
    output_dir: str
    sahi: bool
    conf: float
    iou: float
    slice_hw: tuple[int, int]
    overlap: float
    total: int = 0
    done: int = 0
    status: str = "pending"  # pending | running | completed | failed
    error: Optional[str] = None
    results: list[dict] = field(default_factory=list)  # [{image, n_detections}]
    thread: Optional[threading.Thread] = field(default=None, repr=False)


_jobs: dict[str, InferenceJob] = {}
_job_lock = threading.Lock()


def _register(job: InferenceJob) -> None:
    with _job_lock:
        _jobs[job.job_id] = job


def _get(job_id: str) -> Optional[InferenceJob]:
    with _job_lock:
        return _jobs.get(job_id)


def _list_jobs() -> list[InferenceJob]:
    with _job_lock:
        return list(_jobs.values())


# ── Worker ─────────────────────────────────────────────────────────────


def _list_images(images_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        return []
    return sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _run_sahi_on_image(det_model, image_path: Path, slice_h: int, slice_w: int, overlap: float):
    from PIL import Image

    from sahi.predict import get_sliced_prediction

    with Image.open(image_path) as im:
        w, h = im.size
    result = get_sliced_prediction(
        image=str(image_path),
        detection_model=det_model,
        slice_height=slice_h,
        slice_width=slice_w,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        verbose=0,
        postprocess_type="NMS",
        postprocess_match_threshold=0.5,
    )
    lines: list[str] = []
    for pred in result.object_prediction_list:
        bbox = pred.bbox
        x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        ww = (x2 - x1) / w
        hh = (y2 - y1) / h
        cls = int(pred.category.id)
        conf = float(pred.score.value)
        lines.append(f"{cls} {conf:.6f} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")
    return lines


def _worker(job: InferenceJob) -> None:
    try:
        job.status = "running"
        output_dir = Path(job.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        images = _list_images(Path(job.images_dir))
        job.total = len(images)

        if job.sahi:
            from sahi import AutoDetectionModel

            det_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=job.checkpoint_path,
                confidence_threshold=job.conf,
                device="cuda:0",
            )

            for img in images:
                lines = _run_sahi_on_image(
                    det_model, img, job.slice_hw[0], job.slice_hw[1], job.overlap
                )
                out = output_dir / f"{img.stem}.txt"
                out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                job.results.append({"image": img.name, "n_detections": len(lines)})
                job.done += 1
        else:
            from ultralytics import YOLO

            model = YOLO(job.checkpoint_path)
            for img in images:
                result = model.predict(source=str(img), conf=job.conf, iou=job.iou, verbose=False)[0]
                lines: list[str] = []
                if result.boxes is not None:
                    xywhn = result.boxes.xywhn.cpu().numpy()
                    conf = result.boxes.conf.cpu().numpy()
                    cls = result.boxes.cls.cpu().numpy().astype(int)
                    for (cx, cy, w, h), c, k in zip(xywhn, conf, cls):
                        lines.append(
                            f"{int(k)} {float(c):.6f} {float(cx):.6f} {float(cy):.6f} {float(w):.6f} {float(h):.6f}"
                        )
                out = output_dir / f"{img.stem}.txt"
                out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                job.results.append({"image": img.name, "n_detections": len(lines)})
                job.done += 1

        job.status = "completed"
    except Exception as exc:
        logger.exception("inference job %s failed", job.job_id)
        job.status = "failed"
        job.error = str(exc)


# ── Request/response ───────────────────────────────────────────────────


class LaunchInferencePayload(BaseModel):
    checkpoint_path: str
    images_dir: str
    output_dir: str
    sahi: bool = True
    conf: float = 0.25
    iou: float = 0.7
    slice_h: int = 640
    slice_w: int = 640
    overlap: float = 0.2


@router.post("/launch")
def launch_inference(payload: LaunchInferencePayload) -> dict:
    if not Path(payload.checkpoint_path).is_file():
        raise HTTPException(404, f"checkpoint not found: {payload.checkpoint_path}")
    if not Path(payload.images_dir).is_dir():
        raise HTTPException(404, f"images_dir not found: {payload.images_dir}")

    job = InferenceJob(
        job_id=f"inf-{uuid.uuid4().hex[:8]}",
        checkpoint_path=payload.checkpoint_path,
        images_dir=payload.images_dir,
        output_dir=payload.output_dir,
        sahi=payload.sahi,
        conf=payload.conf,
        iou=payload.iou,
        slice_hw=(payload.slice_h, payload.slice_w),
        overlap=payload.overlap,
    )
    _register(job)

    t = threading.Thread(target=_worker, args=(job,), daemon=True)
    job.thread = t
    t.start()

    return {"status": "launched", "job_id": job.job_id}


@router.get("/jobs")
def list_jobs() -> dict:
    return {
        "jobs": [
            {
                "job_id": j.job_id,
                "status": j.status,
                "done": j.done,
                "total": j.total,
                "images_dir": j.images_dir,
                "output_dir": j.output_dir,
                "error": j.error,
            }
            for j in _list_jobs()
        ]
    }


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    j = _get(job_id)
    if j is None:
        raise HTTPException(404, f"job not found: {job_id}")
    return {
        "job_id": j.job_id,
        "status": j.status,
        "done": j.done,
        "total": j.total,
        "images_dir": j.images_dir,
        "output_dir": j.output_dir,
        "error": j.error,
        "results_tail": j.results[-50:],
    }


@router.get("/jobs/{job_id}/preview")
def get_preview(job_id: str, limit: int = 12) -> dict:
    j = _get(job_id)
    if j is None:
        raise HTTPException(404, f"job not found: {job_id}")
    preview = j.results[:limit]
    return {
        "job_id": job_id,
        "images_dir": j.images_dir,
        "output_dir": j.output_dir,
        "preview": preview,
    }


@router.websocket("/jobs/{job_id}/stream")
async def stream_job(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    job = _get(job_id)
    if job is None:
        await websocket.send_json({"error": "job not found"})
        await websocket.close()
        return
    try:
        last_done = -1
        while True:
            if job.done != last_done:
                last_done = job.done
                await websocket.send_json({
                    "type": "progress",
                    "job_id": job.job_id,
                    "done": job.done,
                    "total": job.total,
                    "status": job.status,
                })
            if job.status in ("completed", "failed"):
                await websocket.send_json({
                    "type": "final",
                    "job_id": job.job_id,
                    "status": job.status,
                    "error": job.error,
                })
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass

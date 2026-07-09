"""Inference routes: async tiled runs + live progress WebSocket.

Jobs run on a background thread. Each job writes YOLO-format predictions
(one ``<stem>.txt`` per image, ``class conf cx cy w h``) to ``output_dir``
so they plug straight into the Review tab and the per-plant curve pipeline.

Inference goes through the tcip pipeline ``GenericPredictor`` — the same code path as
the MCP ``run_inference`` tool — with native SAHI-style tiling when ``sahi=true``. There
is no separate ultralytics/SAHI stack. Checkpoints are tcip composed-model checkpoints
(``model_spec`` + ``model_state_dict``) produced by the training pipeline.
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
    status: str = "pending"  # pending | running | completed | failed | cancelled
    error: Optional[str] = None
    results: list[dict] = field(default_factory=list)  # [{image, n_detections}]
    thread: Optional[threading.Thread] = field(default=None, repr=False)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)


_jobs: dict[str, InferenceJob] = {}
_job_lock = threading.Lock()


def _summary(job: InferenceJob) -> dict:
    return {
        "job_id": job.job_id, "status": job.status, "done": job.done, "total": job.total,
        "images_dir": job.images_dir, "output_dir": job.output_dir, "error": job.error,
    }


def _persist() -> None:
    from tcip_web import jobstore
    with _job_lock:
        summaries = [_summary(j) for j in _jobs.values()]
    jobstore.persist("inference_jobs", summaries)


def _register(job: InferenceJob) -> None:
    from tcip_web import jobstore
    with _job_lock:
        _jobs[job.job_id] = job
        jobstore.evict_terminal(_jobs)  # bound the registry (drop oldest terminal jobs)
    _persist()


def _get(job_id: str) -> Optional[InferenceJob]:
    with _job_lock:
        return _jobs.get(job_id)


def _list_jobs() -> list[InferenceJob]:
    with _job_lock:
        return list(_jobs.values())


def rehydrate() -> None:
    """Seed the registry from the last persisted summaries after a backend restart.

    The worker threads are gone, so a persisted non-terminal job is dead — it is
    surfaced as ``interrupted``. Only the fields the API exposes are restored (the
    per-image results list isn't persisted), so preview/results_tail come back empty.
    """
    from tcip_web import jobstore

    with _job_lock:
        if _jobs:
            return
        for s in jobstore.load("inference_jobs"):
            jid = s.get("job_id")
            if not jid:
                continue
            status = s.get("status", "interrupted")
            if status not in jobstore.TERMINAL_STATUSES:
                status = "interrupted"
            _jobs[jid] = InferenceJob(
                job_id=jid,
                checkpoint_path="",
                images_dir=s.get("images_dir", ""),
                output_dir=s.get("output_dir", ""),
                sahi=False,
                conf=0.0,
                iou=0.0,
                slice_hw=(0, 0),
                overlap=0.0,
                total=s.get("total", 0),
                done=s.get("done", 0),
                status=status,
                error=s.get("error"),
            )


# ── Worker ─────────────────────────────────────────────────────────────


def _list_images(images_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        return []
    return sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _worker(job: InferenceJob) -> None:
    try:
        job.status = "running"
        _persist()
        output_dir = Path(job.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        images = _list_images(Path(job.images_dir))
        job.total = len(images)

        # One inference code path: the tcip pipeline predictor (same as MCP run_inference),
        # with native sliding-window tiling when sahi=true. No ultralytics/SAHI stack.
        from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
        from tcip_mcp.pipelines.postprocessing.export import result_to_yolo_lines

        predictor = GenericPredictor(
            checkpoint_path=job.checkpoint_path,
            device=None,  # auto: cuda if available, else cpu
            score_threshold=job.conf,
        )

        for img in images:
            if job.cancel_event.is_set():
                break
            results = predictor.predict_batch(
                [str(img)],
                tile=job.sahi,
                tile_size=job.slice_hw[0],
                overlap=job.overlap,
                global_nms_iou=job.iou,
            )
            lines = result_to_yolo_lines(results[0])
            out = output_dir / f"{img.stem}.txt"
            out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            job.results.append({"image": img.name, "n_detections": len(lines)})
            job.done += 1

        job.status = "cancelled" if job.cancel_event.is_set() else "completed"
    except Exception as exc:
        logger.exception("inference job %s failed", job.job_id)
        job.status = "failed"
        job.error = str(exc)
    finally:
        _persist()


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


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    """Request graceful cancellation; the worker stops at the next image boundary."""
    j = _get(job_id)
    if j is None:
        raise HTTPException(404, f"job not found: {job_id}")
    j.cancel_event.set()
    return {"job_id": job_id, "status": j.status, "cancel_requested": True}


@router.websocket("/jobs/{job_id}/stream")
async def stream_job(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    job = _get(job_id)
    if job is None:
        await websocket.send_json({"error": "job not found"})
        await websocket.close()
        return
    try:
        from tcip_web import jobstore

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
            # Terminate on ANY terminal state — a cancelled/interrupted job never
            # reaches completed/failed, so keying only on those spun this loop forever.
            if job.status in jobstore.TERMINAL_STATUSES:
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

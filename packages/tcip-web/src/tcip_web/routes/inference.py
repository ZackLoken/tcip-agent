"""Inference routes: async runs + live progress WebSocket.

Each job runs ``run_inference``'s operation (:func:`~tcip_mcp.tools.inference_tools.infer`) on a
background thread, reporting each image and stopping at an image boundary once canceled. A launch
naming the bucket a live job is still writing under the same dataset root refuses, naming that
job; the publication itself refuses a bucket that already exists.
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

from tcip_mcp.identity import actor
from tcip_mcp.pipelines.execution import Stated
from tcip_web import jobstore
from tcip_web.paths import allowed_path
from tcip_web.routes._body_common import PersonPayload
from tcip_web.state import store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/inference", tags=["inference"])


# ── Job registry ────────────────────────────────────────────────────────


@dataclass
class InferenceJob:
    job_id: str
    # The project open when the job launched, which it runs for.
    project: str
    checkpoint_path: str
    dataset_root: str
    images_dir: str
    bucket: str
    # The person who launched the job, who publishes its bucket.
    actor: str
    # What the launch stated of the execution record, resolved as run_inference resolves it.
    stated: Stated = field(default_factory=Stated)
    # The assessment whose execution record the pass runs and the bucket names.
    assessment_id: Optional[str] = None
    total: int = 0
    done: int = 0
    status: str = "pending"  # one of jobstore.JOB_STATES
    error: Optional[str] = None
    # Detections dropped for a zero-extent box, as the published bucket's record counts them;
    # None until a bucket is published.
    dropped_boxes: Optional[int] = None
    thread: Optional[threading.Thread] = field(default=None, repr=False)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)


def _summary(job: InferenceJob) -> dict:
    """The job as every listing and stream frame serves it."""
    return {
        "job_id": job.job_id, "status": job.status, "done": job.done, "total": job.total,
        "images_dir": job.images_dir, "dataset_root": job.dataset_root, "bucket": job.bucket,
        "error": job.error, "dropped_boxes": job.dropped_boxes,
    }


_registry = jobstore.JobRegistry()
"""This route's own live jobs (``jobstore.JobRegistry``)."""


def _register(job: InferenceJob) -> None:
    _registry.register(job.job_id, job)


def _get(job_id: str) -> Optional[InferenceJob]:
    """A job by id, whichever project it runs for: opening another project must not make an
    in-flight job unreachable for canceling or streaming it."""
    return _registry.get(job_id)


# ── Worker ─────────────────────────────────────────────────────────────


def _worker(job: InferenceJob) -> None:
    # Held through try/except, assigned to job.status only in finally, after the publication is
    # attempted. "running" (never a terminal read) until a branch below names the real outcome.
    terminal_status = "running"
    try:
        job.status = "running"

        from tcip_mcp.pipelines.execution import DEFAULT_TILE_BATCH_SIZE
        from tcip_mcp.tools.inference_tools import infer

        def progress(done: int, total: int) -> None:
            job.done, job.total = done, total

        result = infer(
            Path(job.project), checkpoint_path=Path(job.checkpoint_path),
            images_dir=job.images_dir,
            raster_path=None, bucket=job.bucket, assessment_id=job.assessment_id,
            stated=job.stated, device=None, tile_batch_size=DEFAULT_TILE_BATCH_SIZE,
            dry_run=False, require_masks=True, resume=False, progress=progress,
            canceled=job.cancel_event.is_set, actor=job.actor)
        # The publication's own result names the error a failed pass met, and a published
        # bucket's dropped_boxes.
        job.error = result.get("error")
        job.dropped_boxes = result.get("dropped_boxes")
        terminal_status = ("failed" if job.error is not None
                           else "canceled" if job.cancel_event.is_set() else "completed")
    except Exception as exc:
        logger.exception("inference job %s failed", job.job_id)
        terminal_status = "failed"
        job.error = str(exc)
    finally:
        job.status = terminal_status


# ── Request/response ───────────────────────────────────────────────────


class LaunchInferencePayload(BaseModel):
    checkpoint_path: str
    # The run's images are named, not spelled: resolved server-side through dataset_layout.
    dataset_root: str
    # The capture whose images the pass runs over.
    date: str
    # The name to publish the bucket under, beneath dataset_root, as run_inference takes it.
    bucket: str
    # Unstated where omitted: the pass derives each value as run_inference does.
    stated: Stated = Stated()
    # The assessment whose execution record the pass runs, so the bucket is published under it.
    assessment_id: str | None = None
    user: str


@router.post("/launch")
def launch_inference(payload: LaunchInferencePayload) -> dict:
    """Launch a pass over the dataset's images on a background thread, by the person ``user``
    names, who publishes its bucket."""
    person = actor(payload.user)
    project = store.held().root
    # A caller must not name a file outside the allowed roots, registered checkpoint or not.
    checkpoint_path = allowed_path(payload.checkpoint_path)
    dataset_root = allowed_path(payload.dataset_root)

    from tcip_store import canonical_path

    from tcip_mcp.dataset_layout import image_dir

    try:
        images_dir = image_dir(dataset_root, payload.date)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not checkpoint_path.is_file():
        raise HTTPException(404, f"checkpoint not found: {checkpoint_path}")
    if not images_dir.is_dir():
        raise HTTPException(404, f"images_dir not found: {images_dir}")

    live_job = next((j for j in _registry.list(str(project)) if jobstore.live(j)
                     and j.bucket == payload.bucket
                     and canonical_path(j.dataset_root) == canonical_path(dataset_root)), None)
    if live_job is not None:
        raise HTTPException(409, {
            "kind": "bucket_exists",
            "message": f"bucket {payload.bucket!r} under {dataset_root} is the bucket job "
                       f"{live_job.job_id} is still writing.",
            "date": payload.date, "requested_bucket": payload.bucket,
            "job_id": live_job.job_id,
        })

    job = InferenceJob(
        job_id=f"inf-{uuid.uuid4().hex[:8]}", project=str(project),
        checkpoint_path=str(checkpoint_path), dataset_root=str(dataset_root),
        images_dir=str(images_dir), bucket=payload.bucket, actor=person, stated=payload.stated,
        assessment_id=payload.assessment_id)
    _register(job)

    t = threading.Thread(target=_worker, args=(job,), daemon=True)
    job.thread = t
    t.start()

    return {"status": "launched", "job_id": job.job_id, "images_dir": str(images_dir),
            "dataset_root": str(dataset_root), "bucket": payload.bucket}


@router.get("/jobs")
def list_jobs() -> dict:
    return {"jobs": [_summary(j) for j in _registry.list(str(store.held().root))]}


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, payload: PersonPayload) -> dict:
    """Request graceful cancellation by the person ``user`` names; the worker stops at the next
    image boundary and the request leaves one ``inference_canceled`` line in the job's project.
    Refuses (404) an unknown job, and answers 409 when the line cannot follow the request."""
    from tcip_mcp.audit import AuditEntryNotWrittenError, record_event_or_raise
    from tcip_web.routes.audit_gap import audit_gap_409

    person = actor(payload.user)
    j = _get(job_id)
    if j is None:
        raise HTTPException(404, f"job not found: {job_id}")
    j.cancel_event.set()
    answer = {"job_id": job_id, "status": j.status, "cancel_requested": True}
    try:
        record_event_or_raise("inference_canceled",
                              {"job_id": job_id, "dataset_root": j.dataset_root,
                               "bucket": j.bucket}, actor=person, scope=j.project)
    except AuditEntryNotWrittenError as exc:
        raise audit_gap_409(exc, answer) from exc
    return answer


@router.websocket("/jobs/{job_id}/stream")
async def stream_job(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    job = _get(job_id)
    if job is None:
        # Typed the same as a run's own terminal frame, so the client stops reconnecting instead
        # of retrying forever against a job that will never exist.
        await websocket.send_json({"type": "final", "error": "job not found"})
        await websocket.close()
        return
    try:
        last_done = -1
        while True:
            if job.done != last_done:
                last_done = job.done
                await websocket.send_json({"type": "progress", **_summary(job)})
            if not jobstore.live(job):
                await websocket.send_json({"type": "final", **_summary(job)})
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass

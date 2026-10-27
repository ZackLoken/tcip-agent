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
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, PrivateAttr

from tcip_mcp.dataset_layout import parse_capture_dir
from tcip_mcp.identity import actor
from tcip_mcp.pipelines.execution import Stated
from tcip_web import jobstore
from tcip_web.paths import allowed_path
from tcip_web.routes._body_common import PersonPayload
from tcip_web.state import store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/inference", tags=["inference"])


# ── Job registry ────────────────────────────────────────────────────────


class InferenceJob(BaseModel):
    """One inference pass this process launched: what it runs over, its progress and its outcome.

    Its thread and cancel event are private state, absent from every dump of it.
    """

    # Its lifecycle values are defaulted and every field is still sent, so the schema is the
    # answer's.
    model_config = ConfigDict(
        json_schema_mode_override="serialization",
        json_schema_serialization_defaults_required=True,
    )

    job_id: str
    # The project open when the job launched, which it runs for.
    project: str
    checkpoint_path: str
    images_dir: str
    bucket: str
    # The person who launched the job, who publishes its bucket.
    actor: str
    # What the launch stated of the execution record, resolved as run_inference resolves it.
    stated: Stated
    # The assessment whose execution record the pass runs and the bucket names.
    assessment_id: str | None = None
    total: int = 0
    done: int = 0
    status: jobstore.JobStatus = "pending"
    error: str | None = None
    _thread: threading.Thread | None = PrivateAttr(default=None)
    _cancel_event: threading.Event = PrivateAttr(
        default_factory=threading.Event
    )

    @property
    def thread(self) -> threading.Thread | None:
        """The thread the job runs on, ``None`` before its launch starts one."""
        return self._thread

    @property
    def cancel_event(self) -> threading.Event:
        """The event a cancel sets, asked before each image."""
        return self._cancel_event

    @property
    def dataset_root(self) -> Path:
        """The dataset root of the capture ``images_dir`` (:func:`parse_capture_dir`)."""
        return parse_capture_dir(self.images_dir)[0]


class InferenceJobs(BaseModel):
    """The open project's jobs this process launched."""

    jobs: list[InferenceJob]


class BucketExistsRefusal(BaseModel):
    """A launch refused for naming the bucket a live job under the same root is still writing."""

    kind: Literal["bucket_exists"]
    message: str
    job: InferenceJob


class InferenceJobFrame(BaseModel):
    """A job's stream frame: its record as the job stands, the last one ``final``."""

    type: Literal["progress", "final"]
    job: InferenceJob


class MissingJobFrame(BaseModel):
    """The one frame a stream to a job this process holds no record of sends."""

    type: Literal["final"]
    error: str


InferenceFrame = InferenceJobFrame | MissingJobFrame
"""Every frame a job's stream sends."""


_registry: jobstore.JobRegistry[InferenceJob] = jobstore.JobRegistry()
"""This route's own live jobs (``jobstore.JobRegistry``)."""


def _get(job_id: str) -> InferenceJob | None:
    """Return a job by id, whichever project it runs for."""
    return _registry.get(job_id)


# ── Worker ─────────────────────────────────────────────────────────────


def _worker(job: InferenceJob) -> None:
    # Held through try/except, assigned to job.status only in finally, after the publication is
    # attempted. "running" (never a terminal read) until a branch below names the real outcome.
    terminal_status: jobstore.JobStatus = "running"
    try:
        job.status = "running"

        from tcip_mcp.pipelines.execution import DEFAULT_TILE_BATCH_SIZE
        from tcip_mcp.tools.inference_tools import infer

        def progress(done: int, total: int) -> None:
            job.done, job.total = done, total

        result = infer(
            Path(job.project),
            checkpoint_path=Path(job.checkpoint_path),
            images_dir=job.images_dir,
            raster_path=None,
            bucket=job.bucket,
            assessment_id=job.assessment_id,
            stated=job.stated,
            device=None,
            tile_batch_size=DEFAULT_TILE_BATCH_SIZE,
            dry_run=False,
            resume=False,
            progress=progress,
            canceled=job.cancel_event.is_set,
            actor=job.actor,
        )
        # The publication's own result names the error a failed pass met.
        job.error = result.get("error")
        terminal_status = (
            "failed"
            if job.error is not None
            else "canceled"
            if job.cancel_event.is_set()
            else "completed"
        )
    except Exception as exc:
        logger.exception("inference job %s failed", job.job_id)
        terminal_status = "failed"
        job.error = str(exc)
    finally:
        job.status = terminal_status


# ── Request/response ───────────────────────────────────────────────────


class LaunchInferencePayload(BaseModel):
    """An inference launch: its checkpoint, capture, bucket, statement and person."""

    checkpoint_path: str
    # The run's images are named, not spelled: resolved server-side through dataset_layout.
    dataset_root: str
    # The capture whose images the pass runs over.
    date: str
    # The name to publish the bucket under, beneath dataset_root, as run_inference takes it.
    bucket: str
    # Each value the statement leaves out the pass derives as run_inference does.
    stated: Stated
    # The assessment whose execution record the pass runs, so the bucket is published under it.
    assessment_id: str | None = None
    user: str


@router.post("/launch")
def launch_inference(payload: LaunchInferencePayload) -> InferenceJob:
    """Launch a pass over the dataset's images on a background thread, answering its record.

    The person ``user`` names launches it and publishes its bucket. The record is the job's as
    it stands before the thread starts. A bucket a live job under the same dataset root is still
    writing refuses (409, :class:`BucketExistsRefusal`).
    """
    person = actor(payload.user)
    project = store.held().root
    # A caller must not name a file outside the allowed roots, registered checkpoint or not.
    checkpoint_path = allowed_path(payload.checkpoint_path)
    dataset_root = allowed_path(payload.dataset_root)

    from tcip_mcp.dataset_layout import image_dir
    from tcip_store import canonical_path

    try:
        images_dir = image_dir(dataset_root, payload.date)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not checkpoint_path.is_file():
        raise HTTPException(404, f"checkpoint not found: {checkpoint_path}")
    if not images_dir.is_dir():
        raise HTTPException(404, f"images_dir not found: {images_dir}")

    job, created = _registry.find_or_register(
        lambda j: (
            j.project == str(project)
            and jobstore.live(j)
            and j.bucket == payload.bucket
            and canonical_path(j.dataset_root) == canonical_path(dataset_root)
        ),
        lambda: InferenceJob(
            job_id=f"inf-{uuid.uuid4().hex[:8]}",
            project=str(project),
            checkpoint_path=str(checkpoint_path),
            images_dir=str(images_dir),
            bucket=payload.bucket,
            actor=person,
            stated=payload.stated,
            assessment_id=payload.assessment_id,
        ),
    )
    if not created:
        raise HTTPException(
            409,
            BucketExistsRefusal(
                kind="bucket_exists",
                message=f"bucket {payload.bucket!r} under {dataset_root} is the bucket job "
                f"{job.job_id} is still writing.",
                job=job,
            ).model_dump(mode="json"),
        )
    launched = job.model_copy()
    job._thread = threading.Thread(target=_worker, args=(job,), daemon=True)
    job._thread.start()
    return launched


@router.get("/jobs")
def list_jobs() -> InferenceJobs:
    """List this process's jobs for the open project."""
    return InferenceJobs(jobs=_registry.list(str(store.held().root)))


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, payload: PersonPayload) -> InferenceJob:
    """Request graceful cancellation by the person ``user`` names, answering the job's record.

    The job stops at the next image boundary and the request leaves one
    ``inference_canceled`` line in the job's project. Refuses (404) an unknown job, and answers
    409 when the line cannot follow the request.
    """
    from tcip_mcp.audit import AuditEntryNotWrittenError, record_event_or_raise
    from tcip_web.routes.audit_gap import audit_gap_409

    person = actor(payload.user)
    j = _get(job_id)
    if j is None:
        raise HTTPException(404, f"job not found: {job_id}")
    j.cancel_event.set()
    try:
        record_event_or_raise(
            "inference_canceled",
            {
                "job_id": job_id,
                "dataset_root": str(j.dataset_root),
                "bucket": j.bucket,
            },
            actor=person,
            scope=j.project,
        )
    except AuditEntryNotWrittenError as exc:
        raise audit_gap_409(exc, j) from exc
    return j


@router.websocket("/jobs/{job_id}/stream")
async def stream_job(websocket: WebSocket, job_id: str) -> None:
    """Stream a job's frames (:data:`InferenceFrame`) until a final frame once it ends.

    A progress frame is sent whenever the done count changes. An unknown job gets one
    :class:`MissingJobFrame` carrying ``job not found`` and the socket closes.
    """
    await websocket.accept()
    job = _get(job_id)
    if job is None:
        # Typed the same as a run's own terminal frame, so the client stops reconnecting instead
        # of retrying forever against a job that will never exist.
        missing = MissingJobFrame(type="final", error="job not found")
        await websocket.send_json(missing.model_dump(mode="json"))
        await websocket.close()
        return
    try:
        last_done = -1
        while True:
            ended = not jobstore.live(job)
            if job.done != last_done:
                last_done = job.done
                await websocket.send_json(
                    InferenceJobFrame(type="progress", job=job).model_dump(
                        mode="json"
                    )
                )
            if ended:
                await websocket.send_json(
                    InferenceJobFrame(type="final", job=job).model_dump(
                        mode="json"
                    )
                )
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass

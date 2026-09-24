"""Inference routes: async tiled runs + live progress WebSocket.

Jobs run on a background thread. Each job writes per-image JSON predictions
(one ``<stem>.json`` per image, pixel-xyxy boxes + per-object score) to ``output_dir``
so they plug straight into the Review tab and the per-plant curve pipeline.

Inference goes through ``build_predictor``, the same entry point as the MCP ``run_inference``
tool, which dispatches on the checkpoint's model kind and runs the tcip composed-model
checkpoint through its own native SAHI-style tiling.
The operating point (conf / NMS IoU / tiling / max_dets) is resolved through the same
``raw_operating_point`` bundle as the MCP door, and the run publishes through the MCP door's own
publisher (``inference_tools.publish_bucket``): its gates refuse before any image is predicted,
and the documents, the stamp and the publication's line are written as ``run_inference`` writes
them, one image at a time as the publisher consumes the predictions.

A launch into a bucket already holding another run's prediction documents is refused the way
``run_inference`` refuses one, naming a fresh bucket to write into instead; a bucket a live job of
this process is still writing is refused by that job's own identity, never by a resolved path a
second, concurrent pass could still be moving.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from tcip_mcp.web_client import INFERENCE_JOBS, current_root
from tcip_web import jobstore
from tcip_web.paths import assert_path_allowed
from tcip_web.routes._body_common import EmptyBodyPayload

if TYPE_CHECKING:
    from tcip_web.jobstore import JobStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/inference", tags=["inference"])

INFERENCE_REGISTRY = INFERENCE_JOBS
"""The job registry this module persists its jobs to."""


def _current_root() -> str:
    return current_root()


# ── Job registry ────────────────────────────────────────────────────────


@dataclass
class InferenceJob:
    job_id: str
    checkpoint_path: str
    images_dir: str
    output_dir: str
    # The launch's own stated values, ``None`` where it stated nothing: the pass resolves each
    # one exactly as run_inference resolves its own arguments (inference_tools._prepare_pass).
    conf: Optional[float] = None
    iou: Optional[float] = None
    tile: Optional[bool] = None
    tile_size: Optional[int] = None
    overlap: Optional[float] = None
    max_dets: Optional[int] = None
    postprocess: str = "nms"  # cross-tile merge: "nms" suppresses, "nmm" unions seam-split boxes
    total: int = 0
    done: int = 0
    status: JobStatus = "pending"
    error: Optional[str] = None
    warning: Optional[str] = None
    # Set when a line the publishing library writes for this run could not be written: a distinct
    # fact from warning, never reused for it. The predictions and their stamp are on disk regardless.
    audit_warning: Optional[str] = None
    # Detections dropped for a zero-extent box: no detection, so dropped rather than failing the
    # run. A rehydrated job's count is whatever the last persist wrote, never a live measurement.
    dropped_boxes: int = 0
    thread: Optional[threading.Thread] = field(default=None, repr=False)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    # The platform root this job launched under, resolved on whichever thread constructs it
    # (the request thread for a real launch); a rehydrated job restates the persisted value.
    platform_root: str = field(default_factory=_current_root)
    # The launch this job answers, held only for the in-flight refusal: never in _summary, so
    # a rehydrated job (whose worker is gone) never answers an in-flight match.
    requested_dataset_root: str = ""
    requested_model_name: str = ""
    date: Optional[str] = None


def _summary(job: InferenceJob) -> dict:
    return {
        "job_id": job.job_id, "status": job.status, "done": job.done, "total": job.total,
        "images_dir": job.images_dir, "output_dir": job.output_dir, "error": job.error,
        "warning": job.warning, "audit_warning": job.audit_warning,
        "dropped_nonpositive_boxes": job.dropped_boxes,
        "platform_root": job.platform_root,
    }


def _from_summary(s: dict, root: str) -> InferenceJob:
    """A persisted summary, rehydrated: only the fields the API exposes are restored. An
    interrupted job's ``done`` and ``dropped_boxes`` are whatever the last persist wrote, not a
    live measurement: the worker only persists at launch and once more when it finishes or dies,
    so a crash mid-run restores the counts as of launch rather than the work actually done
    before the crash."""
    return InferenceJob(
        job_id=s["job_id"],
        checkpoint_path="",
        images_dir=s.get("images_dir", ""),
        output_dir=s.get("output_dir", ""),
        total=s.get("total", 0),
        done=s.get("done", 0),
        platform_root=jobstore.require_platform_root(s, name=INFERENCE_REGISTRY, root=root),
        status=jobstore.rehydrated_status(s),
        error=s.get("error"),
        warning=s.get("warning"),
        audit_warning=s.get("audit_warning"),
        dropped_boxes=s.get("dropped_nonpositive_boxes", 0),
    )


_registry = jobstore.JobRegistry(
    INFERENCE_REGISTRY, to_summary=_summary, from_summary=_from_summary,
)
"""The dict-plus-lock live registry for this route's own jobs (see ``jobstore.JobRegistry``),
the shared home review.py's priority queue and tuning.py's sweeps adopt too."""


def _persist() -> None:
    _registry.persist()


def _register(job: InferenceJob) -> None:
    _registry.register(job.job_id, job, job_root=job.platform_root)


def _get(job_id: str) -> Optional[InferenceJob]:
    """A job by id, from any root this process holds: a repin to another project must not
    make an in-flight job unreachable for cancelling or streaming it."""
    return _registry.get(job_id)


def _list_jobs() -> list[InferenceJob]:
    return _registry.list(current_root())


def rehydrate_for_current_root() -> None:
    """Merge this root's persisted jobs, not already live, into memory via :func:`_from_summary`.

    Called at startup and again after this process repins to another root: the worker
    threads behind a persisted non-terminal job are gone, so it is surfaced as
    ``interrupted``. Merges by job id rather than requiring an empty registry first, so it
    never displaces a job still live from another root. Bounds the dict afterwards the same
    way registering a job does, so adopting N roots without ever registering a job here still
    keeps this process's memory bounded rather than growing by ``MAX_JOBS`` for every root
    adopted.
    """
    _registry.rehydrate()


# ── Worker ─────────────────────────────────────────────────────────────


def _worker(job: InferenceJob) -> None:
    # Held through try/except, assigned to job.status only in finally, after the publication is
    # attempted. "running" (never a terminal read) until a branch below names the real outcome.
    terminal_status: JobStatus = "running"
    try:
        job.status = "running"
        _persist()

        from tcip_mcp.audit import AuditEntryNotWritten
        from tcip_mcp.dataset_layout import bucket_dataset_root
        from tcip_mcp.model_registry import UnregisteredCheckpoint, load_registered_checkpoint
        from tcip_mcp.pipelines.resolution import DEFAULT_TILE_BATCH_SIZE
        from tcip_mcp.tools.inference_tools import _prepare_pass, publish_bucket

        try:
            checkpoint = load_registered_checkpoint(
                job.checkpoint_path, project_path=job.platform_root)
        except UnregisteredCheckpoint as exc:
            terminal_status = "failed"
            job.error = str(exc)
            logger.warning("inference job %s refused: %s", job.job_id, job.error)
            return
        # The pass run_inference prepares, from the launch's own stated values; device auto.
        prepared = _prepare_pass(
            checkpoint, images_dir=job.images_dir, conf_threshold=job.conf, device=None,
            tile=job.tile, tile_size=job.tile_size, overlap=job.overlap,
            global_nms_iou=job.iou, max_dets=job.max_dets,
            postprocess=job.postprocess, experiment_id=None,
            tile_batch_size=DEFAULT_TILE_BATCH_SIZE)
        if isinstance(prepared, str):
            terminal_status = "failed"
            job.error = prepared
            return
        job.total = len(prepared.paths)

        def predictions():
            """Predict one image at a time as the publisher consumes them, counting each written
            document, and stop at the next image boundary once a cancel is requested."""
            for img in prepared.paths:
                if job.cancel_event.is_set():
                    return
                yield prepared.predict([img])[0]
                job.done += 1

        run = prepared.raw_result()
        run["results"] = predictions()
        try:
            pub = publish_bucket(
                run, out=Path(job.output_dir), checkpoint_path=job.checkpoint_path, trait=None,
                images_dir=job.images_dir, dataset_root=bucket_dataset_root(job.output_dir),
                allow_unvalidated_staging=False)
        except AuditEntryNotWritten as exc:
            # The publisher committed an act whose line it could not write; a failed pass's own
            # line carries the error the pass failed on.
            job.audit_warning = str(exc)
            job.error = exc.arguments.get("error")
        else:
            if pub["refusal"] is not None:
                job.error = pub["refusal"]["error"]
            job.dropped_boxes = pub["dropped_boxes"]
        terminal_status = ("failed" if job.error is not None
                           else "cancelled" if job.cancel_event.is_set() else "completed")
    except Exception as exc:
        logger.exception("inference job %s failed", job.job_id)
        terminal_status = "failed"
        job.error = str(exc)
    finally:
        job.status = terminal_status
        _persist()


# ── Request/response ───────────────────────────────────────────────────


class LaunchInferencePayload(BaseModel):
    checkpoint_path: str
    # The run's images and its prediction bucket are named, not spelled: both dirs are resolved
    # server-side through dataset_layout / prediction_buckets, so no caller reimplements the layout.
    dataset_root: str
    # A bucket name, not a model identity: it may name a variant (e.g. an @r2 suggestion) no registered model bears, which
    # nothing downstream reads as a model (the stamp takes the checkpoint's own stem, the publication's line records only the bucket).
    model_name: str
    date: str | None = None
    # None (default) derives tiling from the checkpoint's own training geometry in the worker,
    # distinct from an explicit caller choice, so the job's provenance can say which happened.
    tile: bool | None = None
    # conf/iou/slice_h/overlap are all None by default: an omitted
    # field is a real "let the platform derive it" request, distinguished from an explicit choice
    # that happens to match the default, the same way `tile` already works. A frozen literal
    # transmitted on every launch would permanently shadow resolve_tile_geometry's
    # checkpoint-derived tile_size/overlap and the shared conf/iou defaults below.
    conf: float | None = None
    iou: float | None = None
    slice_h: int | None = None
    overlap: float | None = None
    max_dets: int | None = None
    postprocess: str = "nms"
    # Never overrides the document refusal below (a bucket with no verdict but a document refuses regardless); its one live effect is turning a verdict redirect
    # into a 409 instead of the default auto-redirect, including on exhaustion of every @r<n> variant to the ceiling. The browser client never sends this field.
    overwrite: bool = False


@router.post("/launch")
def launch_inference(payload: LaunchInferencePayload) -> dict:
    # Confine client-supplied paths to the allowed roots (TCIP_IMAGE_ROOTS): a caller must not
    # name a file outside them, registered checkpoint or not. No-op when unset.
    try:
        for p in (payload.checkpoint_path, payload.dataset_root):
            assert_path_allowed(p)
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc

    from tcip_mcp.dataset_layout import image_dir, prediction_dir
    from tcip_mcp.workspace import is_valid_name

    # model_name and date become path segments of the prediction bucket: the same check every
    # other writer of one applies (see stage_proposals), refused here rather than resolving a
    # path that escapes the dataset.
    for label, value in (("model_name", payload.model_name), ("date", payload.date)):
        if value is not None and not is_valid_name(value):
            raise HTTPException(
                400,
                f"{label} must be a single safe path segment (no separators/'..'), got {value!r}",
            )

    if not Path(payload.checkpoint_path).is_file():
        raise HTTPException(404, f"checkpoint not found: {payload.checkpoint_path}")
    images_dir = image_dir(payload.dataset_root, payload.date)
    if not images_dir.is_dir():
        raise HTTPException(404, f"images_dir not found: {images_dir}")

    # Prediction-bucket immutability: never silently overwrite a bucket with review verdicts, and
    # never begin a second publish beside one already writing or one a prior run already filled.
    from tcip_mcp.prediction_buckets import (
        BucketHasVerdicts,
        BucketHoldsDocuments,
        resolve_prediction_bucket,
        review_state_dir_of,
    )

    # Keyed on what was requested, not a resolved path a moving document count could shift: a
    # second launch of this (dataset, model, date) while the first still writes names that job.
    requested_dataset_root = Path(payload.dataset_root).resolve()  # two spellings, one match
    live_job = next(
        (
            j for j in _list_jobs()
            if j.status in ("pending", "running")
            and Path(j.requested_dataset_root).resolve() == requested_dataset_root
            and j.requested_model_name == payload.model_name
            and j.date == payload.date
        ),
        None,
    )
    if live_job is not None:
        raise HTTPException(409, {
            "kind": "bucket_in_flight",
            "message": (
                f"a job for model {payload.model_name!r} on date {payload.date!r} in "
                f"{payload.dataset_root!r} is already writing to {live_job.output_dir!r} "
                f"(job {live_job.job_id!r}); wait for it or watch it rather than launching a "
                "second job over the same images."
            ),
            "date": payload.date,
            "requested_output_dir": live_job.output_dir,
            "job_id": live_job.job_id,
        })

    review_state_dir = review_state_dir_of(payload.dataset_root)
    try:
        bucket_dir, resolution = resolve_prediction_bucket(
            payload.dataset_root,
            payload.model_name,
            payload.date,
            review_state_dir=review_state_dir,
            overwrite=payload.overwrite,
            refuse_documents=True,
        )
    except BucketHasVerdicts as exc:
        raise HTTPException(409, str(exc)) from exc
    except BucketHoldsDocuments as exc:
        requested_output_dir = str(
            prediction_dir(payload.dataset_root, payload.model_name, payload.date))
        suggested_output_dir = (
            str(prediction_dir(payload.dataset_root, exc.suggested, payload.date))
            if exc.suggested is not None else None
        )
        raise HTTPException(409, {
            "kind": "bucket_holds_documents",
            "message": str(exc),
            "date": payload.date,
            "requested_model_name": payload.model_name,
            "requested_output_dir": requested_output_dir,
            "document_stem_count": exc.document_stem_count,
            "suggested_model_name": exc.suggested,
            "suggested_output_dir": suggested_output_dir,
        }) from exc
    resolved_output_dir = str(bucket_dir)

    # Every tuning value travels as stated, None where omitted; the worker's pass resolves each.
    job = InferenceJob(
        job_id=f"inf-{uuid.uuid4().hex[:8]}",
        checkpoint_path=payload.checkpoint_path,
        images_dir=str(images_dir),
        output_dir=resolved_output_dir,
        requested_dataset_root=payload.dataset_root,
        requested_model_name=payload.model_name,
        date=payload.date,
        tile=payload.tile,
        conf=payload.conf,
        iou=payload.iou,
        tile_size=payload.slice_h,
        overlap=payload.overlap,
        max_dets=payload.max_dets,
        postprocess=payload.postprocess,
    )
    _register(job)

    t = threading.Thread(target=_worker, args=(job,), daemon=True)
    job.thread = t
    t.start()

    requested_dir = (
        str(prediction_dir(payload.dataset_root, resolution.requested, payload.date))
        if resolution.redirected
        else None
    )
    return {"status": "launched", "job_id": job.job_id, "images_dir": str(images_dir),
            "output_dir": resolved_output_dir,
            "bucket_redirected": resolution.redirected,
            "requested_output_dir": requested_dir}


@router.get("/jobs")
def list_jobs() -> dict:
    return {"jobs": [_summary(j) for j in _list_jobs()]}


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, payload: EmptyBodyPayload) -> dict:
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
        # Typed the same as a run's own terminal frame, so the client stops reconnecting instead
        # of retrying forever against a job that will never exist.
        await websocket.send_json({"type": "final", "error": "job not found"})
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
                    "warning": job.warning,
                    "audit_warning": job.audit_warning,
                })
            # Terminate on any terminal state: a cancelled/interrupted job never
            # reaches completed/failed, so keying only on those spun this loop forever.
            if job.status in jobstore.TERMINAL_STATUSES:
                await websocket.send_json({
                    "type": "final",
                    "job_id": job.job_id,
                    "status": job.status,
                    "error": job.error,
                    "warning": job.warning,
                    "audit_warning": job.audit_warning,
                    "dropped_nonpositive_boxes": job.dropped_boxes,
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

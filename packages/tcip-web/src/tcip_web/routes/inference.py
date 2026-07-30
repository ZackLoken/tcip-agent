"""Inference routes: async tiled runs + live progress WebSocket.

Jobs run on a background thread. Each job writes per-image COCO/JSON predictions
(one ``<stem>.json`` per image, pixel-xyxy boxes + per-object score) to ``output_dir``
so they plug straight into the Review tab and the per-plant curve pipeline.

Inference goes through ``build_predictor`` — the same entry point as the MCP ``run_inference``
tool — which dispatches on the checkpoint's model kind and runs the tcip composed-model
checkpoint through its own native SAHI-style tiling.
The operating point (conf / NMS IoU / tiling / max_dets) is resolved through the same
``raw_operating_point`` bundle as the MCP door and its provenance is stamped alongside the
predictions, so a GUI run and an agent run can't diverge on the count or hide an unvalidated
operating point.
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

from tcip_mcp.pipelines.resolution import (
    DEFAULT_CONF,
    DEFAULT_MAX_DETS,
    DEFAULT_NMS_IOU,
    DEFAULT_OVERLAP,
    DEFAULT_TILED,
)
from tcip_web.paths import assert_path_allowed

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
    tile: bool
    conf: float
    iou: float
    slice_hw: tuple[int, int]
    overlap: float
    max_dets: int = DEFAULT_MAX_DETS
    postprocess: str = "nms"  # cross-tile merge: "nms" suppresses, "nmm" unions seam-split boxes
    # K10 finding 3 (K21: renamed from sahi/sahi_source — this predates ultralytics removal and
    # is the generic tile toggle, not a SAHI-specific one): whether the caller explicitly chose to
    # tile, or it fell back to DEFAULT_TILED — threaded into raw_operating_point's tiled_source so
    # the sidecar's provenance can tell the two apart, same as the MCP door's run_inference already
    # does for its own `tile` param.
    tile_source: str = "default"
    # TRAP 4 step 2 (K6): whether slice_hw was the breeder's explicit override or should be
    # re-derived from the checkpoint's own persisted training geometry (resolve_tile_geometry).
    slice_source: str = "default"
    total: int = 0
    done: int = 0
    status: str = "pending"  # pending | running | completed | failed | cancelled
    error: Optional[str] = None
    warning: Optional[str] = None
    results: list[dict] = field(default_factory=list)  # [{image, n_detections}]
    thread: Optional[threading.Thread] = field(default=None, repr=False)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)


_jobs: dict[str, InferenceJob] = {}
_job_lock = threading.Lock()


def _summary(job: InferenceJob) -> dict:
    return {
        "job_id": job.job_id, "status": job.status, "done": job.done, "total": job.total,
        "images_dir": job.images_dir, "output_dir": job.output_dir, "error": job.error,
        "warning": job.warning,
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
                tile=False,
                conf=0.0,
                iou=0.0,
                slice_hw=(0, 0),
                overlap=0.0,
                total=s.get("total", 0),
                done=s.get("done", 0),
                status=status,
                error=s.get("error"),
                warning=s.get("warning"),
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

        # One inference entry point (same as MCP run_inference): build_predictor dispatches on
        # the checkpoint's model kind, sniffed from the checkpoint itself.
        from tcip_mcp.pipelines.inference.predictor import build_predictor
        from tcip_mcp.pipelines.postprocessing.export import write_predictions_json
        from tcip_mcp.pipelines.resolution import raw_operating_point
        from tcip_mcp.utils.atomic_io import atomic_write_json

        predictor = build_predictor(
            checkpoint_path=job.checkpoint_path,
            device=None,  # auto: cuda if available, else cpu
            score_threshold=job.conf,  # conf is an unvalidated documented default either way —
            nms_iou=job.iou,           # raw_operating_point below wraps the same raw value, never
            max_dets=job.max_dets,     # transforms it (see its own docstring).
        )

        # instance_seg tiled inference can't carry masks through the cross-tile merge (same rail
        # run_inference enforces — see inference_tools.py). The MCP door can tell an explicit
        # tile=True request apart from an unset default and refuse only the explicit case; the
        # GUI's tile checkbox is a controlled input with no "unset" state (K10 finding 3 above), so
        # job.tile_source is "explicit" on every real launch and refusing on that basis here would
        # refuse every GUI instance_seg run, not just the ones that mean it. So this door always
        # forces untiled for instance_seg instead of ever refusing (masks survive; a rail must admit
        # valid work, not only reject invalid work) and records the forced value's source as
        # "default" rather than "explicit" — the platform, not the breeder, decided this run's
        # tiling, and the provenance should say so.
        if getattr(predictor, "task", None) == "instance_seg" and job.tile:
            job.tile = False
            job.tile_source = "default"
            job.warning = (
                "instance_seg checkpoint: tiled inference cannot carry masks through the "
                "cross-tile merge yet, so this run forced untiled (masks survive) regardless of "
                "the 'Tiled inference' checkbox. Small dense objects may be under-detected at "
                "full resolution."
            )
            logger.warning(job.warning)

        # Derive tile_size/overlap from the checkpoint's own persisted training geometry (K6/TRAP 4
        # step 2) — the same resolver run_inference uses, so the GUI door can't silently diverge
        # from the MCP door on the object count's scale for the same checkpoint. An explicit
        # caller-chosen tile size (job.tile_source == "explicit") still wins over the derivation.
        from tcip_mcp.pipelines.inference.predictor import resolve_tile_geometry

        resolved_tile, tile_size_source, resolved_overlap, overlap_source = resolve_tile_geometry(
            predictor, tile_size=job.slice_hw[0] if job.slice_source == "explicit" else None,
            overlap=job.overlap if job.slice_source == "explicit" else None,
        )

        # Resolve the operating point through the SAME firewalled bundle as the MCP door: conf is a
        # documented default with no per-dataset GT, so it is unvalidated and stamped validated=false.
        op_bundle = raw_operating_point(
            conf=job.conf, cross_tile_nms=job.iou, tiled=job.tile, tiled_source=job.tile_source,
            tile_size=resolved_tile, tile_size_source=tile_size_source, max_dets=job.max_dets,
        )

        # Stamp the operating point next to the predictions so a GUI-produced set carries the same
        # provenance (and validated=false) the MCP door records — a phenotype's numbers are only as
        # trustworthy as the operating point that produced them.
        #
        # checkpoint_sha256/experiment_id (stage-6 review, K2 Fix G): the SAME producing-model
        # identity resolver the MCP door uses (model_registry.resolve_model_identity — never a
        # second implementation), so a bucket the GUI's own Inference tab produces carries the same
        # identity fact Fix G's review-verdict scoping matches against. Without this, a GUI-produced
        # bucket could never be validated via the review-confirmation route: its sidecar carried
        # neither field, so producer-identity matching failed closed on every review session no
        # matter how thoroughly it was reviewed.
        from tcip_mcp.model_registry import resolve_model_identity

        identity = resolve_model_identity(job.checkpoint_path)

        # This run's name->id map (K6/K3): resolved the same way the MCP door's run_inference does
        # (predictor.config["data"]["subject"]/["attribute"] -> the inference dataset's registry),
        # never a second implementation — without this, every GUI-produced bucket decodes to raw
        # index-string subjects and permanently fails the coverage/classifier-validity mechanism.
        id_map = None
        try:
            data_cfg = (getattr(predictor, "config", {}) or {}).get("data") or {}
            subject = data_cfg.get("subject")
            if subject and job.images_dir:
                from tcip_mcp.pipelines.data.datasets import _resolve_registry_id_map

                _reg, id_map = _resolve_registry_id_map(job.images_dir, subject, data_cfg.get("attribute"))
        except Exception:  # noqa: BLE001 — no run scope for the map; predictions decode by raw id
            id_map = None

        provenance = op_bundle.to_provenance()
        provenance["validated"] = op_bundle.is_shippable
        provenance["checkpoint_sha256"] = identity["sha256"]
        provenance["experiment_id"] = identity["experiment_id"]
        provenance["id_map"] = id_map
        # overlap has no home in ResolvedBundle's tracked params (only conf/cross_tile_nms/tiled/
        # tile_size/max_dets are) — surface the value + source this run actually used directly,
        # matching the MCP door's own run_inference (inference_tools.py).
        provenance["overlap"] = resolved_overlap
        provenance["overlap_source"] = overlap_source
        if getattr(predictor, "task", None) == "instance_seg":
            # The unvalidated mask-binarize threshold write_predictions_json will use for every mask
            # in this run — a run constant, so it travels once here (see export.py's
            # mask_binarize_provenance docstring), never per-annotation.
            from tcip_mcp.pipelines.postprocessing.export import mask_binarize_provenance

            provenance["mask_binarize"] = mask_binarize_provenance()
        atomic_write_json(output_dir / "operating_point.json", provenance)

        for img in images:
            if job.cancel_event.is_set():
                break
            results = predictor.predict_batch(
                [str(img)],
                tile=job.tile,
                tile_size=resolved_tile,
                overlap=resolved_overlap,
                global_nms_iou=job.iou,
                postprocess=job.postprocess,
            )
            write_predictions_json(output_dir / f"{img.stem}.json", results[0],
                                   created_by=f"model:{Path(job.checkpoint_path).stem}", id_map=id_map)
            job.results.append({"image": img.name, "n_detections": results[0]["count"]})
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
    # tiling + conf/iou default to the ONE shared source so the GUI and the MCP agent produce the
    # SAME count off the same checkpoint (they used to diverge: this field, named `sahi` before
    # K21 removed ultralytics/SAHI support, at 640 + 0.25/0.7 here vs tile=False/224 + 0.5/0.3 in
    # run_inference — tiling drives the count most of all).
    # K10 finding 3: None (default) is a documented fallback, not an implicit True — distinguished
    # from an explicit caller choice so the job's provenance can say which one happened.
    tile: bool | None = None
    # TRAP 4 step 1 (K6): conf/iou/slice_h/slice_w/overlap are all None-by-default now — an omitted
    # field is a real "let the platform derive it" request, distinguished from an explicit choice
    # that happens to match the default, the same way `tile` already works. Without this, the GUI
    # transmitted a frozen literal on every launch, permanently shadowing resolve_tile_geometry's
    # checkpoint-derived tile_size/overlap and the shared conf/iou defaults below.
    conf: float | None = None
    iou: float | None = None
    slice_h: int | None = None
    slice_w: int | None = None
    overlap: float | None = None
    max_dets: int = DEFAULT_MAX_DETS
    postprocess: str = "nms"
    # Write into output_dir even if it exists. Refused if the bucket has review verdicts; the
    # default (False) auto-redirects to a fresh bucket so a re-run never orphans recorded verdicts.
    overwrite: bool = False


@router.post("/launch")
def launch_inference(payload: LaunchInferencePayload) -> dict:
    # Confine client-supplied paths to the allowed roots when the server is locked down
    # (TCIP_IMAGE_ROOTS) — the checkpoint is fed to torch.load(weights_only=False), an
    # arbitrary-pickle sink, so an unconfined path is the sharpest edge here. No-op when unset.
    try:
        for p in (payload.checkpoint_path, payload.images_dir, payload.output_dir):
            assert_path_allowed(p)
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc
    if not Path(payload.checkpoint_path).is_file():
        raise HTTPException(404, f"checkpoint not found: {payload.checkpoint_path}")
    if not Path(payload.images_dir).is_dir():
        raise HTTPException(404, f"images_dir not found: {payload.images_dir}")

    # Prediction-bucket immutability: never silently overwrite a bucket with review verdicts.
    from tcip_mcp.prediction_buckets import BucketHasVerdicts, resolve_writable_bucket
    from tcip_mcp.project_paths import resolve_state

    out_path = Path(payload.output_dir)
    parent, base_name = out_path.parent, out_path.name
    review_state_dir = resolve_state(Path(".tcip") / "state")
    try:
        resolution = resolve_writable_bucket(
            review_state_dir, base_name, lambda n: [parent / n], overwrite=payload.overwrite)
    except BucketHasVerdicts as exc:
        raise HTTPException(409, str(exc)) from exc
    resolved_output_dir = str(parent / resolution.name)

    # K10 finding 3: resolve the None sentinel ONCE, here — InferenceJob.tile stays a concrete
    # bool everywhere else (behavior + provenance both read it), with tile_source carrying which
    # case this was. Note the meaning of "explicit" differs by door on purpose: the MCP tool's
    # `tile` distinguishes an agent that supplied the kwarg from one that omitted it; here it
    # distinguishes a request body that carried the field from one that didn't — since the GUI's
    # checkbox is a controlled input with no "unset" state, every real launch IS the breeder's
    # explicit choice, even when it matches the default. Stage-6 review (K10): this is a
    # deliberate difference in what "explicit" means per door, not a labeling bug.
    resolved_tile = DEFAULT_TILED if payload.tile is None else payload.tile
    tile_source = "explicit" if payload.tile is not None else "default"
    # conf/iou: no checkpoint-derivation concept applies (unlike tile_size) — an omitted value
    # falls back to the same shared defaults resolution.py names, matching the MCP door.
    resolved_conf = DEFAULT_CONF if payload.conf is None else payload.conf
    resolved_iou = DEFAULT_NMS_IOU if payload.iou is None else payload.iou
    # slice_h/slice_w/overlap: the "explicit" signal resolve_tile_geometry needs (TRAP 4 step 2) —
    # None here means "derive from the checkpoint's training geometry," resolved in the worker
    # once the predictor is built. slice_hw stays a concrete tuple everywhere else (0 sentinel is
    # never read as a real value; the worker only branches on slice_source).
    slice_source = "explicit" if payload.slice_h is not None else "default"

    job = InferenceJob(
        job_id=f"inf-{uuid.uuid4().hex[:8]}",
        checkpoint_path=payload.checkpoint_path,
        images_dir=payload.images_dir,
        output_dir=resolved_output_dir,
        tile=resolved_tile,
        tile_source=tile_source,
        conf=resolved_conf,
        iou=resolved_iou,
        slice_hw=(payload.slice_h or 0, payload.slice_w or 0),
        slice_source=slice_source,
        overlap=payload.overlap if payload.overlap is not None else DEFAULT_OVERLAP,
        max_dets=payload.max_dets,
        postprocess=payload.postprocess,
    )
    _register(job)

    t = threading.Thread(target=_worker, args=(job,), daemon=True)
    job.thread = t
    t.start()

    return {"status": "launched", "job_id": job.job_id, "output_dir": resolved_output_dir,
            "bucket_redirected": resolution.redirected,
            "requested_output_dir": payload.output_dir if resolution.redirected else None}


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
    from tcip_web.paths import origin_allowed

    if not origin_allowed(websocket.headers.get("origin")):
        await websocket.close(code=1008, reason="origin not allowed")
        return
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
                    "warning": job.warning,
                })
            # Terminate on ANY terminal state — a cancelled/interrupted job never
            # reaches completed/failed, so keying only on those spun this loop forever.
            if job.status in jobstore.TERMINAL_STATUSES:
                await websocket.send_json({
                    "type": "final",
                    "job_id": job.job_id,
                    "status": job.status,
                    "error": job.error,
                    "warning": job.warning,
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

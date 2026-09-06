"""Inference routes: async tiled runs + live progress WebSocket.

Jobs run on a background thread. Each job writes per-image COCO/JSON predictions
(one ``<stem>.json`` per image, pixel-xyxy boxes + per-object score) to ``output_dir``
so they plug straight into the Review tab and the per-plant curve pipeline.

Inference goes through ``build_predictor``, the same entry point as the MCP ``run_inference``
tool, which dispatches on the checkpoint's model kind and runs the tcip composed-model
checkpoint through its own native SAHI-style tiling.
The operating point (conf / NMS IoU / tiling / max_dets) is resolved through the same
``raw_operating_point`` bundle as the MCP door and its provenance is stamped alongside the
predictions, so a GUI run and an agent run can't diverge on the count or hide an unvalidated
operating point. A run whose tile scale has no real basis at all is refused by the shared delivery
gate before anything is written, the same refusal ``run_inference`` makes.

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
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from tcip_mcp.pipelines.resolution import (
    DEFAULT_CONF,
    DEFAULT_MAX_DETS,
    DEFAULT_NMS_IOU,
    DEFAULT_OVERLAP,
)
from tcip_web import jobstore
from tcip_web.paths import assert_path_allowed
from tcip_web.routes._body_common import EmptyBodyPayload

if TYPE_CHECKING:
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_web.jobstore import JobStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/inference", tags=["inference"])

INFERENCE_REGISTRY = jobstore.INFERENCE_JOBS
"""The job registry this module persists its jobs to."""


def _current_root() -> str:
    from tcip_web import jobstore
    return jobstore.current_root()


# ── Job registry ────────────────────────────────────────────────────────


@dataclass
class InferenceJob:
    job_id: str
    checkpoint_path: str
    images_dir: str
    output_dir: str
    conf: float
    iou: float
    slice_hw: tuple[int, int]
    overlap: float
    max_dets: int = DEFAULT_MAX_DETS
    postprocess: str = "nms"  # cross-tile merge: "nms" suppresses, "nmm" unions seam-split boxes
    # The caller's own tile choice, unresolved: None means "derive from the checkpoint's own
    # training geometry", resolved in the worker once the predictor is built (never here).
    tile: Optional[bool] = None
    # Whether the caller explicitly chose `tile`, threaded into raw_operating_point's tiled_source.
    tile_source: str = "default"
    # Whether the caller explicitly chose conf/max_dets, fed to raw_operating_point's own
    # conf_stated/max_dets_stated: a value equal to the platform default still stamps explicit.
    conf_stated: bool = False
    max_dets_stated: bool = False
    # Whether slice_hw was the breeder's explicit override or should be re-derived from the
    # checkpoint's own recorded training geometry, tiled or native-frame (resolve_tile_geometry).
    slice_source: str = "default"
    total: int = 0
    done: int = 0
    status: JobStatus = "pending"
    error: Optional[str] = None
    warning: Optional[str] = None
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
        "warning": job.warning, "dropped_nonpositive_boxes": job.dropped_boxes,
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
        tile=None,  # a dead job's own tile choice is never read again; honest, not fabricated
        conf=0.0,
        iou=0.0,
        slice_hw=(0, 0),
        overlap=0.0,
        total=s.get("total", 0),
        done=s.get("done", 0),
        platform_root=jobstore.require_platform_root(s, name=INFERENCE_REGISTRY, root=root),
        status=jobstore.rehydrated_status(s),
        error=s.get("error"),
        warning=s.get("warning"),
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
    from tcip_web import jobstore
    return _registry.list(jobstore.current_root())


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


def _list_images(images_dir: Path) -> list[Path | BandGroupRef]:
    """Every logical image in ``images_dir``: a ``.bandgroup``-grouped multi-band capture folds
    into one entry here (see ``image_utils.list_logical_images``), the same enumeration every other
    reader in this platform shares, instead of this route's own raw sibling-file listing enumerating
    each band file as its own (spurious) image."""
    from tcip_mcp.pipelines.image_utils import list_logical_images

    logical = list_logical_images(images_dir)
    return [logical[stem] for stem in sorted(logical)]


def _worker(job: InferenceJob) -> None:
    try:
        job.status = "running"
        _persist()
        output_dir = Path(job.output_dir)

        images = _list_images(Path(job.images_dir))
        job.total = len(images)

        # One inference entry point (same as MCP run_inference): build_predictor dispatches on
        # the checkpoint's model kind, sniffed from the checkpoint itself.
        from tcip_mcp.model_registry import UnregisteredCheckpoint, load_registered_checkpoint
        from tcip_mcp.pipelines.inference.predictor import build_predictor
        from tcip_mcp.pipelines.postprocessing.export import write_predictions_json
        from tcip_mcp.pipelines.resolution import raw_operating_point

        try:
            checkpoint = load_registered_checkpoint(
                job.checkpoint_path, project_path=job.platform_root)
        except UnregisteredCheckpoint as exc:
            job.status = "failed"
            job.error = str(exc)
            logger.warning("inference job %s refused: %s", job.job_id, job.error)
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        predictor = build_predictor(
            checkpoint,
            device=None,  # auto: cuda if available, else cpu
            score_threshold=job.conf,  # conf is an unvalidated documented default either way,
            nms_iou=job.iou,           # raw_operating_point below wraps the same raw value, never
            max_dets=job.max_dets,     # transforms it (see its own docstring).
        )

        # Resolve the tiled bool now the checkpoint's own training geometry is in hand: an unset
        # job.tile gets the checkpoint's own tiled-or-not regime, never a fixed platform default.
        resolved_tile_bool = (
            getattr(predictor, "train_tile_size", None) is not None
            if job.tile is None else job.tile
        )

        # Derive tile_size/overlap/resize the same way run_inference does; job.slice_source ==
        # "explicit" still wins. A contradicting edge raises, caught by this worker's own try/except.
        from tcip_mcp.pipelines.inference.predictor import (
            explicit_edge_provenance, resolve_tile_regime,
        )

        resolved_tile, tile_size_source, resolved_overlap, overlap_source, tile_resize = (
            resolve_tile_regime(
                predictor, tiled=resolved_tile_bool,
                tile_size=job.slice_hw[0] if job.slice_source == "explicit" else None,
                overlap=job.overlap if job.slice_source == "explicit" else None,
            )
        )
        tile_size_derived_from = (
            explicit_edge_provenance(predictor, resolved_tile)
            if tile_size_source == "explicit" and resolved_tile is not None else None)

        # Resolve the operating point through the same firewalled bundle as the MCP door: conf is a
        # documented default with no per-dataset GT, so it is unvalidated and stamped validated=false.
        op_bundle = raw_operating_point(
            conf=job.conf, cross_tile_nms=job.iou, tiled=resolved_tile_bool,
            tiled_source=job.tile_source, tile_size=resolved_tile,
            tile_size_source=tile_size_source, tile_size_derived_from=tile_size_derived_from,
            max_dets=job.max_dets,
            conf_stated=job.conf_stated, max_dets_stated=job.max_dets_stated,
        )

        # Refuse an ungrounded tile scale before the pass, the same gate run_inference applies.
        from tcip_mcp.pipelines.resolution import check_delivery_gate, tile_size_gate_flag

        tile_ref = tile_size_gate_flag(op_bundle.to_provenance()["operating_point"])
        gate = check_delivery_gate({"tile_size": tile_ref} if tile_ref is not None else {})
        if not gate.ok:
            job.status = "failed"
            tile_desc = f"{resolved_tile}px" if resolved_tile is not None else "no resolvable size"
            job.error = (
                f"inference refused: unvalidated dimension(s) {list(gate.unvalidated)}. "
                f"This run's tile scale ({tile_desc}) has no real basis: the checkpoint records no "
                "training tile geometry and no tile size was stated for this run, so the counts it "
                "would produce rest on nothing that justifies them. Run untiled, or use a checkpoint "
                "whose training tile geometry was persisted, or pass an explicit tile size."
            )
            logger.warning("inference job %s refused by the delivery gate: %s",
                           job.job_id, job.error)
            return

        # Stamp the operating point next to the predictions so a GUI-produced set carries the same
        # provenance (and validated=false) the MCP door records: a phenotype's numbers are only as
        # trustworthy as the operating point that produced them.
        #
        # checkpoint_sha256/experiment_id: the same producing-model identity resolver the MCP door
        # uses (model_registry.resolve_model_identity, never a second implementation), so a bucket
        # the GUI's own Inference tab produces carries the same identity fact the review-verdict
        # scoping matches against. Without this, a GUI-produced bucket could never be validated via
        # the review-confirmation route: its sidecar carried neither field, so producer-identity
        # matching failed closed on every review session no matter how thoroughly it was reviewed.
        from tcip_mcp.model_registry import resolve_model_identity

        identity = resolve_model_identity(checkpoint)

        # This run's name->id map and scope: the same resolvers run_inference calls, never a
        # second implementation; a registry read failing for a real reason fails the job outright.
        from tcip_mcp.tools.inference_tools import (
            resolve_decode_id_map, run_scope, unmapped_classified_run,
        )

        subject, attribute = run_scope(predictor)
        id_map = resolve_decode_id_map(predictor, job.images_dir)
        # A classified run resolving no map fails with the composed remedy, never a silent decode.
        classified_refusal = unmapped_classified_run(
            {"subject": subject, "attribute": attribute}, id_map, images_dir=job.images_dir)
        if classified_refusal is not None:
            job.status = "failed"
            job.error = classified_refusal
            return

        from tcip_mcp.pipelines import image_utils
        from tcip_mcp.pipelines.resolution import (
            operating_point_stamp, prediction_producer, write_sidecar,
        )

        # overlap has no home in ResolvedBundle's tracked params (only conf/cross_tile_nms/tiled/
        # tile_size/max_dets are), so the value and source this run actually used travel directly.
        provenance = operating_point_stamp(
            op_bundle.to_provenance()["operating_point"],
            validated=op_bundle.is_shippable,
            # This worker stamps from a raw operating point and is never validated at write time.
            validated_by=None,
            tile_size_validated=gate.stamp.get("tile_size"),
            shippable_issues=op_bundle.shippable_issues(),
            id_map=id_map,
            subject=subject,
            attribute=attribute,
            trait=None,
            dataset_hash=op_bundle.dataset_hash,
            checkpoint=Path(job.checkpoint_path).stem,
            checkpoint_sha256=identity["sha256"],
            experiment_id=identity["experiment_id"],
            images_dir=job.images_dir,
            raster_path=None,
            produced_at=datetime.now(timezone.utc).isoformat(),
            overlap=resolved_overlap,
            overlap_source=overlap_source,
            image_filenames={img.stem: image_utils.logical_image_name(img) for img in images},
        )
        if getattr(predictor, "task", None) == "instance_seg":
            # The unvalidated mask-binarize threshold write_predictions_json will use for every mask
            # in this run: a run constant, so it travels once here, never per-annotation.
            from tcip_mcp.pipelines.postprocessing.export import mask_binarize_provenance

            provenance["mask_binarize"] = mask_binarize_provenance()

        for img in images:
            if job.cancel_event.is_set():
                break
            results = predictor.predict_batch(
                [img],
                tile=resolved_tile_bool,
                tile_size=resolved_tile,
                overlap=resolved_overlap,
                global_nms_iou=job.iou,
                postprocess=job.postprocess,
                tile_resize=tile_resize,
            )
            if attribute is not None:
                from tcip_mcp.pipelines.postprocessing.export import unmapped_label_ids

                unmapped = unmapped_label_ids(results, id_map)
                if unmapped:
                    job.status = "failed"
                    job.error = (
                        f"{img.stem}: this classified run decoded to id(s) {unmapped}, not keys "
                        f"of its recorded id_map ({sorted((id_map or {}).values())})."
                    )
                    break
            job.dropped_boxes += write_predictions_json(
                output_dir / f"{img.stem}.json", results[0],
                created_by=prediction_producer(job.checkpoint_path, identity["sha256"]),
                id_map=id_map, subject=subject, attribute=attribute)
            job.done += 1

        if job.status != "failed":
            # Last, never beside where it is built, so a partway-dead pass leaves a bucket no reader mistakes for certified;
            # image_filenames names every enumerated image whether written or not, including one cancelled before its first write.
            write_sidecar(output_dir, provenance)
            job.status = "cancelled" if job.cancel_event.is_set() else "completed"
    except Exception as exc:
        logger.exception("inference job %s failed", job.job_id)
        job.status = "failed"
        job.error = str(exc)
    finally:
        from tcip_mcp.dataset_layout import dataset_root_of
        from tcip_web.routes.classes import _audit_dataset_write

        dataset_root = dataset_root_of(job.output_dir)
        if dataset_root is not None:
            _audit_dataset_write(
                str(dataset_root),
                "gui_inference_run",
                {
                    "job_id": job.job_id,
                    "checkpoint_path": job.checkpoint_path,
                    "images_dir": job.images_dir,
                    "output_dir": job.output_dir,
                    "status": job.status,
                    "images_written": job.done,
                    "total": job.total,
                    "error": job.error,
                    "dropped_nonpositive_boxes": job.dropped_boxes,
                },
            )
        _persist()


# ── Request/response ───────────────────────────────────────────────────


class LaunchInferencePayload(BaseModel):
    checkpoint_path: str
    # The run's images and its prediction bucket are named, not spelled: both dirs are resolved
    # server-side through dataset_layout / prediction_buckets, so no caller reimplements the layout.
    dataset_root: str
    # A bucket name, not a model identity: it may name a variant (e.g. an @r2 suggestion) no registered model bears, which
    # nothing downstream reads as a model (the stamp takes the checkpoint's own stem, the audit line records only output_dir).
    model_name: str
    date: str | None = None
    # None (default) derives tiling from the checkpoint's own training geometry in the worker,
    # distinct from an explicit caller choice, so the job's provenance can say which happened.
    tile: bool | None = None
    # conf/iou/slice_h/slice_w/overlap are all None by default: an omitted
    # field is a real "let the platform derive it" request, distinguished from an explicit choice
    # that happens to match the default, the same way `tile` already works. A frozen literal
    # transmitted on every launch would permanently shadow resolve_tile_geometry's
    # checkpoint-derived tile_size/overlap and the shared conf/iou defaults below.
    conf: float | None = None
    iou: float | None = None
    slice_h: int | None = None
    slice_w: int | None = None
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

    # job.tile carries the caller's raw choice (None = derive from the checkpoint's training
    # geometry, resolved in the worker), never resolved here; the GUI checkbox has no "unset" state.
    tile_source = "explicit" if payload.tile is not None else "default"
    # conf/iou: no checkpoint-derivation concept applies (unlike tile_size): an omitted value
    # falls back to the same shared defaults resolution.py names, matching the MCP door.
    resolved_conf = DEFAULT_CONF if payload.conf is None else payload.conf
    conf_stated = payload.conf is not None
    resolved_iou = DEFAULT_NMS_IOU if payload.iou is None else payload.iou
    resolved_max_dets = DEFAULT_MAX_DETS if payload.max_dets is None else payload.max_dets
    max_dets_stated = payload.max_dets is not None
    # slice_h/slice_w/overlap: the "explicit" signal resolve_tile_geometry needs.
    # None here means "derive from the checkpoint's training geometry," resolved in the worker
    # once the predictor is built. slice_hw stays a concrete tuple everywhere else (0 sentinel is
    # never read as a real value; the worker only branches on slice_source).
    slice_source = "explicit" if payload.slice_h is not None else "default"

    job = InferenceJob(
        job_id=f"inf-{uuid.uuid4().hex[:8]}",
        checkpoint_path=payload.checkpoint_path,
        images_dir=str(images_dir),
        output_dir=resolved_output_dir,
        requested_dataset_root=payload.dataset_root,
        requested_model_name=payload.model_name,
        date=payload.date,
        tile=payload.tile,
        tile_source=tile_source,
        conf=resolved_conf,
        conf_stated=conf_stated,
        iou=resolved_iou,
        slice_hw=(payload.slice_h or 0, payload.slice_w or 0),
        slice_source=slice_source,
        overlap=payload.overlap if payload.overlap is not None else DEFAULT_OVERLAP,
        max_dets=resolved_max_dets,
        max_dets_stated=max_dets_stated,
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

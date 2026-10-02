"""Inference MCP tools: running a checkpoint and publishing its predictions as a bucket, and
delivering a bucket's per-image counts."""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tcip_store import (
    RECORD_JSON, Key, StoreDescriptor, Version, VersionConflict, register_store, store,
)
from tcip_store.file_backend import RootedFileLocator

from tcip_mcp.pipelines.execution import DEFAULT_TILE_BATCH_SIZE, Stated
from tcip_mcp.server import tool

logger = logging.getLogger(__name__)

RASTER_PASS_PROGRESS_STORE = "raster_pass_progress"
register_store(
    StoreDescriptor(
        name=RASTER_PASS_PROGRESS_STORE,
        kind="record",
        key_fields=("bucket", "segment"),
        frozen=False,
        codec=RECORD_JSON,
        concurrency="cas",
        enumerable=True,
        locator=RootedFileLocator(prefix=(".tcip", "raster_pass_progress"), suffix=".json"),
    )
)
"""One tiled raster pass's resume state, kept under the project rather than in the bucket it will
publish: an ``identity`` record naming the pass, plus one ``batch-<index>`` record per tile batch
already predicted, under ``<project>/.tcip/raster_pass_progress/<bucket digest>/``."""


def infer(
    project: Path, *, checkpoint_path: str, images_dir: str | None, raster_path: str | None,
    output_dir: str, assessment_id: str | None, stated: Stated | None, device: str | None,
    tile_batch_size: int, dry_run: bool, require_masks: bool, resume: bool,
    progress: Callable[[int, int], None] | None = None,
    canceled: Callable[[], bool] | None = None,
) -> dict:
    """Run a registered checkpoint's pass over ``images_dir`` or ``raster_path`` and publish its
    predictions as the bucket ``output_dir`` (the arguments are ``run_inference``'s), answering
    the publication's result or the error dict naming a refusal, with ``progress`` called with ``(done, total)`` images once the pass is prepared and after each
    image, and ``canceled`` asked before each image: once it answers true the pass stops at that
    image boundary and publishes the documents written."""
    from tcip_mcp.assessment import read_assessment
    from tcip_mcp.buckets import BucketExists, pass_documents, publish
    from tcip_mcp.model_registry import UnregisteredCheckpoint, load_registered_checkpoint
    from tcip_mcp.pipelines.execution import Execution, ExecutionRefused, prepare_pass

    if not output_dir:
        return {"error": "output_dir is required"}
    if (images_dir is None) == (raster_path is None):
        return {"error": "Provide exactly one of images_dir or raster_path."}
    if resume and raster_path is None:
        return {"error": "resume applies only to a raster_path pass: a pass over images writes "
                         "its whole bucket at once."}
    if raster_path is not None and not Path(raster_path).is_file():
        return {"error": f"raster_path not found: {raster_path}"}
    out = Path(project, output_dir)
    try:
        checkpoint = load_registered_checkpoint(checkpoint_path, project=project)
    except (UnregisteredCheckpoint, FileNotFoundError) as exc:
        return {"error": str(exc)}

    recorded = (store.read(_progress_key(project, out, "identity"), default=None)
                if resume else None)
    if resume and recorded is None:
        return {"error": f"resume=True but no raster-pass progress toward {out} is recorded."}
    stated = stated or Stated()
    if raster_path is not None:
        stated = stated.model_copy(update={"tile": True})
    try:
        restored = (Execution.of(recorded.pop("execution")) if recorded is not None
                    else read_assessment(project, assessment_id).execution
                    if assessment_id is not None else None)
        p = prepare_pass(checkpoint, stated, images_dir=images_dir, device=device,
                         tile_batch_size=tile_batch_size, restored=restored)
    except (ExecutionRefused, ValueError) as exc:
        return {"error": str(exc)}

    raster_identity = None
    if raster_path is not None:
        from tcip_mcp.pipelines.data.split_construction import raster_identity as identity_of

        try:
            raster_identity = RECORD_JSON.decode(RECORD_JSON.encode(identity_of(raster_path)))
        except ValueError as exc:
            return {"error": f"raster content identity could not be computed for "
                             f"{raster_path}: {exc}"}
    pass_identity = {"raster_identity": raster_identity, **checkpoint.producer,
                     "assessment_id": assessment_id, "tile_batch_size": tile_batch_size,
                     "require_masks": require_masks}
    if recorded is not None:
        differing = sorted(k for k in recorded.keys() | pass_identity.keys()
                           if recorded.get(k) != pass_identity.get(k))
        if differing:
            return {"error": f"resume=True but the recorded pass toward {out} differs from this "
                             f"call in {differing}: a resumed pass is the identical pass."}
    if dry_run:
        return {"dry_run": True, "output_dir": str(out), "bucket_exists": out.exists(),
                "execution": p.execution.record(), "assessment_id": assessment_id}

    results: Any
    if raster_path is not None:
        try:
            results = [_raster_pass(project, out, p, raster_path,
                                    {**pass_identity, "execution": p.execution.record()},
                                    require_masks=require_masks, resumed=recorded is not None)]
        except VersionConflict:
            return {"error": f"a raster pass toward {out} is already recorded: resume it "
                             "(resume=True), or name a bucket no pass has been recorded toward."}
    else:
        results = _image_results(p, progress, canceled)
    try:
        bucket = publish(project, out, pass_documents(p, results), producer=p.checkpoint.producer,
                         scope=p.scope, execution=p.execution, raster_path=raster_path,
                         raster_identity=raster_identity, assessment_id=assessment_id)
    except BucketExists as exc:
        return {"error": str(exc)}
    if raster_path is not None:
        _clear_raster_pass_progress(project, out)
    from tcip_mcp.buckets import detection_rows

    rows = detection_rows(bucket) if p.execution.conf is not None else None
    return {
        "output_dir": str(out), "image_count": len(bucket.documents),
        "total_detections": sum(r["count"] for r in rows) if rows is not None else None,
        "execution": p.execution.record(), "assessment_id": bucket.assessment_id,
        **bucket.producer, "date": bucket.date, "dropped_boxes": bucket.dropped_boxes,
    }


def _image_results(p: Any, progress: Callable[[int, int], None] | None,
                   canceled: Callable[[], bool] | None):
    """The pass's prediction per image, one image at a time as the publication consumes them,
    reporting each and stopping at the image boundary a cancel is first seen at."""
    total = len(p.paths)
    if progress is not None:
        progress(0, total)
    for done, path in enumerate(p.paths, start=1):
        if canceled is not None and canceled():
            return
        yield p.predict([path])[0]
        if progress is not None:
            progress(done, total)


@tool()
def run_inference(
    project: Path,
    checkpoint_path: str,
    images_dir: str | None = None,
    raster_path: str | None = None,
    output_dir: str = "",
    assessment_id: str | None = None,
    stated: Stated | None = None,
    device: str | None = None,
    tile_batch_size: int = DEFAULT_TILE_BATCH_SIZE,
    dry_run: bool = False,
    require_masks: bool = True,
    resume: bool = False,
) -> dict:
    """Run a trained model over images or a raster, and publish the predictions as a bucket.

    Provide exactly one of ``images_dir`` (an ordinary directory of per-image captures) or
    ``raster_path`` (a single raster, potentially too large to decode whole, always tiled). The
    bucket is ``output_dir``: a new directory holding one ``<stem>.json`` per image (one for the
    raster, in full-raster pixels) and ``bucket.json``, the record of the checkpoint, class scope,
    execution record, capture and assessment behind them. A bucket is published once: an existing
    ``output_dir`` refuses, before an image pass predicts anything and once a raster pass has
    predicted its raster, and a new run names a new bucket.

    With ``assessment_id`` the pass runs exactly that assessment's execution record (its conf,
    cap, tile edge, overlap, merge and threshold) and the bucket names the assessment, which is
    what a delivery reads to call its numbers validated; a stated execution value it records
    differently refuses by name. Without one, every execution value is the stated one, else the
    checkpoint's own recorded geometry, else a documented default, and the bucket is unassessed.

    Args:
        checkpoint_path: A checkpoint registered in this project.
        images_dir: Directory of input images (exclusive with ``raster_path``).
        raster_path: A single raster (exclusive with ``images_dir``).
        output_dir: The bucket directory to publish; a relative path is under the project.
        assessment_id: The assessment whose execution record the pass runs and whose id the
            bucket records.
        stated: The execution values to state rather than derive (``execution.Stated``):
            ``tile`` (tiled inference over ``images_dir``; unstated follows the checkpoint's own
            training regime), ``tile_size``, ``overlap``, ``postprocess`` (one of
            ``execution.CROSS_TILE_MERGES``), ``cross_tile_nms``, ``conf`` and ``max_dets``.
        device: cuda / cpu (auto if omitted).
        tile_batch_size: Tiles per forward batch.
        dry_run: Resolve the execution record and the bucket and report them, with whether the
            bucket already exists, predicting and publishing nothing.
        require_masks: Collect masks for an ``instance_seg`` checkpoint over a raster.
        resume: ``raster_path`` only: continue a raster pass toward ``output_dir`` whose progress
            the project records, under the execution record that progress recorded; a stated
            value it records differently refuses by name, and so does a different checkpoint,
            raster, assessment, tile batch size or mask choice.
    """
    return infer(project, checkpoint_path=checkpoint_path, images_dir=images_dir,
                 raster_path=raster_path, output_dir=output_dir, assessment_id=assessment_id,
                 stated=stated, device=device, tile_batch_size=tile_batch_size, dry_run=dry_run,
                 require_masks=require_masks, resume=resume)


def _progress_key(project: Path, bucket: Path, segment: str) -> Key:
    """One raster pass's progress record toward ``bucket``, kept under ``project``: the identity
    (``segment="identity"``) or one flushed tile batch (``segment=f"batch-{index:06d}"``)."""
    return Key(RASTER_PASS_PROGRESS_STORE, str(project), (_bucket_digest(bucket), segment))


def _bucket_digest(bucket: Path) -> str:
    """The progress store's name for the bucket a raster pass will publish: a digest of its
    resolved path."""
    return hashlib.sha256(os.path.normcase(str(bucket.resolve())).encode("utf-8")).hexdigest()[:16]


def _progress_keys(project: Path, bucket: Path) -> list[Key]:
    """Every progress record a raster pass toward ``bucket`` left under ``project``."""
    name = _bucket_digest(bucket)
    return [key for key in store.keys(RASTER_PASS_PROGRESS_STORE, str(project))
            if key.parts[0] == name]


def _raster_pass(project: Path, out: Path, p: Any, raster_path: str, pass_identity: dict, *,
                 require_masks: bool, resumed: bool) -> dict:
    """The one tiled pass over the raster: its identity recorded first on a fresh pass, each
    flushed tile batch recorded as it lands, the batches already recorded fed back in on a
    resumed one."""
    from tcip_mcp.pipelines.raster_source import open_raster

    prior: dict[str, list] | None = None
    if resumed:
        indexed = sorted((int(key.parts[1][len("batch-"):]), key)
                         for key in _progress_keys(project, out)
                         if key.parts[1].startswith("batch-"))
        prior = {"slices": [], "predictions": []}
        for _index, key in indexed:
            batch = store.read(key)
            prior["slices"].extend(batch["slices"])
            prior["predictions"].extend(batch["predictions"])
    else:
        store.replace(_progress_key(project, out, "identity"), pass_identity,
                      expect=Version.ABSENT)

    def record(start: int, _end: int, batch: dict) -> None:
        store.replace(_progress_key(project, out, f"batch-{start:06d}"), batch,
                      expect=Version.ABSENT)

    with open_raster(raster_path, p.predictor.in_chans) as reader:
        return p.predictor.predict_sliced(
            reader, execution=p.execution, tile_batch_size=p.tile_batch_size,
            require_masks=require_masks, source_label=str(raster_path), prior=prior,
            progress=record)


def _clear_raster_pass_progress(project: Path, bucket: Path) -> None:
    """Delete every progress record a raster pass toward ``bucket`` left, in one transaction: a
    published pass has nothing left to resume."""
    keys = _progress_keys(project, bucket)
    if not keys:
        return
    with store.transaction(*keys) as txn:
        for key in keys:
            txn.delete(key)


@tool()
def deliver_per_image_counts(project: Path, predictions_dir: str, output_path: str, *,
                             trait: str, acknowledgment_id: str | None = None) -> dict:
    """Deliver a published bucket's per-image detection counts as a CSV.

    Each row is one of the bucket's documents, in stem order: its source image's file name, its
    detection count and mean confidence, and the delivery's ``validated``, ``trait_revision`` and
    ``delivery_event_id``. The bucket clears the one delivery gate when the assessment it was
    published under answers for a ``per_image_count`` delivery of ``trait``'s latest confirmed
    revision; an unvalidated bucket ships only under ``acknowledgment_id``, a breeder's recorded
    acknowledgment of exactly this result, which this door executes and never records. Every
    delivery appends one delivery event.

    Args:
        predictions_dir: The published bucket; a relative path is under the project.
        output_path: The CSV to write; a relative path is under the project.
        trait: The trait whose confirmed per-image-count operationalization this rests on.
        acknowledgment_id: A breeder's recorded acknowledgment of this unvalidated result.
    """
    from tcip_mcp.delivery import DeliveryRefused
    from tcip_mcp.operationalization import OperationalizationRefused
    from tcip_mcp.pipelines.postprocessing.export import deliver_per_image_counts_csv
    from tcip_mcp.traits import TraitUnknownError

    try:
        return deliver_per_image_counts_csv(
            project, Path(project, predictions_dir), str(Path(project, output_path)),
            trait=trait, acknowledgment_id=acknowledgment_id, door="deliver_per_image_counts")
    except (DeliveryRefused, OperationalizationRefused, TraitUnknownError, ValueError) as exc:
        return {"error": str(exc)}

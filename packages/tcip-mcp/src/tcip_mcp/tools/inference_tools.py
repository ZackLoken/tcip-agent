"""Inference MCP tools: running a checkpoint and publishing its predictions as a bucket, and
delivering a bucket's per-image counts."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from tcip_store import (
    Key, Version, VersionConflictError, canonical_path, decode_value, encode_record, store,
)

from tcip_mcp.pipelines.execution import DEFAULT_TILE_BATCH_SIZE, Stated
from tcip_mcp.server import tool

logger = logging.getLogger(__name__)

RASTER_PASS_PROGRESS_STORE = "raster_pass_progress"
"""One tiled raster pass's resume state under the project: an ``identity`` record naming the
pass, plus one ``batch-<index>`` record per tile batch already predicted, both keyed under the
bucket's dataset root and name."""


def infer(
    project: Path, *, checkpoint_path: str, images_dir: str | None, raster_path: str | None,
    bucket: str, assessment_id: str | None, stated: Stated | None, device: str | None,
    tile_batch_size: int, dry_run: bool, require_masks: bool, resume: bool,
    progress: Callable[[int, int], None] | None = None,
    canceled: Callable[[], bool] | None = None, actor: str | None,
) -> dict:
    """Run a registered checkpoint's pass over the capture ``images_dir``
    (:func:`~tcip_mcp.dataset_layout.parse_capture_dir`, which refuses any other directory) or
    ``raster_path`` and publish its predictions as the bucket named ``bucket`` under their dataset
    root (:func:`~tcip_mcp.buckets.source_root`) by ``actor``, answering the publication's result or
    the error dict naming a refusal. ``progress`` is called with ``(done, total)`` images once the
    pass is prepared and after each image, and ``canceled`` asked before each image: once it
    answers true the pass stops at that image boundary and publishes the documents predicted, or
    publishes nothing when it predicted none."""
    from tcip_mcp.assessment import read_assessment
    from tcip_mcp.buckets import pass_documents, publish, source_root
    from tcip_mcp.dataset_layout import bucket_key, parse_capture_dir
    from tcip_mcp.model_registry import UnregisteredCheckpointError, load_registered_checkpoint
    from tcip_mcp.pipelines.execution import Execution, ExecutionRefusedError, prepare_pass

    if not bucket:
        return {"error": "bucket is required: the name the predictions are published under"}
    if (images_dir is None) == (raster_path is None):
        return {"error": "Provide exactly one of images_dir or raster_path."}
    if resume and raster_path is None:
        return {"error": "resume applies only to a raster_path pass: a pass over images writes "
                         "its whole bucket at once."}
    if raster_path is not None and not Path(raster_path).is_file():
        return {"error": f"raster_path not found: {raster_path}"}
    try:
        root = (source_root([raster_path]) if raster_path is not None
                else parse_capture_dir(cast(str, images_dir))[0])
        checkpoint = load_registered_checkpoint(checkpoint_path, project=project)
    except (UnregisteredCheckpointError, FileNotFoundError, ValueError) as exc:
        return {"error": str(exc)}

    recorded = (store.read(_progress_key(project, root, bucket, "identity"), default=None)
                if resume else None)
    if resume and recorded is None:
        return {"error": f"resume=True but no raster-pass progress toward bucket {bucket!r} "
                         f"under {root} is recorded."}
    stated = stated or Stated()
    if raster_path is not None:
        stated = stated.model_copy(update={"tile": True})
    try:
        restored = (Execution.of(recorded.pop("execution")) if recorded is not None
                    else read_assessment(project, assessment_id).execution
                    if assessment_id is not None else None)
        p = prepare_pass(checkpoint, stated, images_dir=images_dir, device=device,
                         tile_batch_size=tile_batch_size, restored=restored)
    except (ExecutionRefusedError, ValueError) as exc:
        return {"error": str(exc)}

    raster_identity = None
    if raster_path is not None:
        from tcip_mcp.pipelines.data.split_construction import raster_identity as identity_of
        from tcip_mcp.pipelines.image_utils import resolve_image_path

        try:
            raster_identity = decode_value(encode_record(identity_of(
                resolve_image_path(raster_path))))
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
            return {"error": f"resume=True but the recorded pass toward bucket {bucket!r} differs "
                             f"from this call in {differing}: a resumed pass is the identical "
                             "pass."}
    if dry_run:
        return {"dry_run": True, "dataset_root": str(root), "bucket": bucket,
                "bucket_exists": store.exists(bucket_key(root, bucket)),
                "execution": p.execution.record(), "assessment_id": assessment_id}

    results: Any
    if raster_path is not None:
        try:
            results = [_raster_pass(project, root, bucket, p, raster_path,
                                    {**pass_identity, "execution": p.execution.record()},
                                    require_masks=require_masks, resumed=recorded is not None)]
        except VersionConflictError:
            return {"error": f"a raster pass toward bucket {bucket!r} is already recorded: resume "
                             "it (resume=True), or name a bucket no pass has been recorded "
                             "toward."}
    else:
        results = list(_image_results(p, progress, canceled))
        if not results and canceled is not None and canceled():
            return {"dataset_root": str(root), "bucket": bucket, "image_count": 0}
    try:
        published = publish(project, root, bucket, pass_documents(p, results),
                            producer=p.checkpoint.producer, scope=p.scope, execution=p.execution,
                            raster_path=raster_path, raster_identity=raster_identity,
                            assessment_id=assessment_id, actor=actor)
    except ValueError as exc:
        return {"error": str(exc)}
    if raster_path is not None:
        _clear_raster_pass_progress(project, root, bucket)
    from tcip_mcp.buckets import detection_rows

    rows = detection_rows(published) if p.execution.conf is not None else None
    return {
        "dataset_root": str(published.root), "bucket": published.name,
        "image_count": len(published.documents),
        "total_detections": sum(r["count"] for r in rows) if rows is not None else None,
        "execution": p.execution.record(), "assessment_id": published.assessment_id,
        **published.producer, "date": published.date, "dropped_boxes": published.dropped_boxes,
    }


def _image_results(p: Any, progress: Callable[[int, int], None] | None,
                   canceled: Callable[[], bool] | None):
    """The pass's prediction per image, one image at a time, reporting each and stopping at the
    image boundary a cancel is first seen at."""
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
    bucket: str = "",
    assessment_id: str | None = None,
    stated: Stated | None = None,
    device: str | None = None,
    tile_batch_size: int = DEFAULT_TILE_BATCH_SIZE,
    dry_run: bool = False,
    require_masks: bool = True,
    resume: bool = False,
) -> dict:
    """Run a trained model over images or a raster, and publish the predictions as a bucket.

    Provide exactly one of ``images_dir`` (a capture directory under a dataset's ``images/``
    tree) or ``raster_path`` (a single raster there, potentially too large to decode whole, always
    tiled). The predictions are published under the dataset root those images lie under as the
    bucket named ``bucket``: one prediction document per image (one for the raster, in
    full-raster pixels) and the bucket's record of the checkpoint, class scope, execution record,
    capture and assessment behind them, in one commit. A bucket is published once: a name already
    published under that root refuses once the pass has predicted, and a new run names a new
    bucket.

    With ``assessment_id`` the pass runs exactly that assessment's execution record (its conf,
    cap, tile edge, overlap, merge and threshold) and the bucket names the assessment, which is
    what a delivery reads to call its numbers validated; a stated execution value it records
    differently refuses by name. Without one, every execution value is the stated one, else the
    checkpoint's own recorded geometry, else a documented default, and the bucket is unassessed.

    Args:
        checkpoint_path: A checkpoint registered in this project.
        images_dir: A capture directory, ``<root>/images/<capture>`` (exclusive with
            ``raster_path``); any other directory refuses naming it.
        raster_path: A single raster (exclusive with ``images_dir``).
        bucket: The name to publish the predictions under, e.g. ``<model>/<date>``.
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
        resume: ``raster_path`` only: continue a raster pass toward ``bucket`` whose progress
            the project records, under the execution record that progress recorded; a stated
            value it records differently refuses by name, and so does a different checkpoint,
            raster, assessment, tile batch size or mask choice.
    """
    return infer(project, checkpoint_path=checkpoint_path, images_dir=images_dir,
                 raster_path=raster_path, bucket=bucket, assessment_id=assessment_id,
                 stated=stated, device=device, tile_batch_size=tile_batch_size, dry_run=dry_run,
                 require_masks=require_masks, resume=resume, actor=None)


def _progress_key(project: Path, root: Path, bucket: str, segment: str) -> Key:
    """One raster pass's progress record toward the bucket ``bucket`` under the dataset ``root``,
    kept under ``project``: the identity (``segment="identity"``) or one flushed tile batch
    (``segment=f"batch-{index:06d}"``)."""
    return Key(RASTER_PASS_PROGRESS_STORE, str(project),
               (canonical_path(root), bucket, segment))


def _progress_keys(project: Path, root: Path, bucket: str) -> list[Key]:
    """Every progress record a raster pass toward ``bucket`` under ``root`` left under
    ``project``."""
    return store.keys(RASTER_PASS_PROGRESS_STORE, str(project), (canonical_path(root), bucket))


def _raster_pass(project: Path, root: Path, bucket: str, p: Any, raster_path: str,
                 pass_identity: dict, *, require_masks: bool, resumed: bool) -> dict:
    """The one tiled pass over the raster: its identity recorded first on a fresh pass, each
    flushed tile batch recorded as it lands, the batches already recorded fed back in on a
    resumed one."""
    from tcip_mcp.pipelines.raster_source import open_raster

    prior: dict[str, list] | None = None
    if resumed:
        indexed = sorted((int(key.parts[-1][len("batch-"):]), key)
                         for key in _progress_keys(project, root, bucket)
                         if key.parts[-1].startswith("batch-"))
        prior = {"slices": [], "predictions": []}
        for _index, key in indexed:
            batch = store.read(key)
            prior["slices"].extend(batch["slices"])
            prior["predictions"].extend(batch["predictions"])
    else:
        store.replace(_progress_key(project, root, bucket, "identity"), pass_identity,
                      expect=Version.ABSENT)

    def record(start: int, _end: int, batch: dict) -> None:
        store.replace(_progress_key(project, root, bucket, f"batch-{start:06d}"), batch,
                      expect=Version.ABSENT)

    with open_raster(raster_path, p.predictor.in_chans) as reader:
        return p.predictor.predict_sliced(
            reader, execution=p.execution, tile_batch_size=p.tile_batch_size,
            require_masks=require_masks, source_label=str(raster_path), prior=prior,
            progress=record)


def _clear_raster_pass_progress(project: Path, root: Path, bucket: str) -> None:
    """Delete every progress record a raster pass toward ``bucket`` under ``root`` left, in one
    transaction: a published pass has nothing left to resume."""
    keys = _progress_keys(project, root, bucket)
    if not keys:
        return
    with store.transaction(*keys) as txn:
        for key in keys:
            txn.delete(key)


@tool()
def deliver_per_image_counts(project: Path, dataset_root: str, bucket: str, output_path: str, *,
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
        dataset_root: The dataset root the bucket is published under.
        bucket: The published bucket's name.
        output_path: The CSV to write; a relative path is under the project.
        trait: The trait whose confirmed per-image-count operationalization this rests on.
        acknowledgment_id: A breeder's recorded acknowledgment of this unvalidated result.
    """
    from tcip_mcp.delivery import DeliveryRefusedError
    from tcip_mcp.operationalization import OperationalizationRefusedError
    from tcip_mcp.pipelines.postprocessing.export import deliver_per_image_counts_csv
    from tcip_mcp.traits import TraitUnknownError

    try:
        return deliver_per_image_counts_csv(
            project, dataset_root, bucket, str(Path(project, output_path)),
            trait=trait, acknowledgment_id=acknowledgment_id, door="deliver_per_image_counts",
            actor=None)
    except (DeliveryRefusedError, OperationalizationRefusedError, TraitUnknownError,
            ValueError) as exc:
        return {"error": str(exc)}

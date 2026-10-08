"""Orchestrates a checkpoint evaluation run (tile-level or delivery-grade full-frame) and returns
its scored result; ``evaluation.py`` keeps the metrics computation itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tcip_mcp.pipelines.data.label_queries import Admission
    from tcip_mcp.pipelines.execution import Pass, Stated
    from tcip_mcp.traits import TraitEntry


# Pinned by name and presence only; a regime's own fields travel in the writer's `extra` instead.
_COMMON_EVAL_FIELDS = (
    "model_path", "task", "checkpoint_sha256", "experiment_id",
    "iou_threshold", "execution", "eval_regime",
)


def evaluation_result(common: dict, extra: dict) -> dict:
    """One evaluation's result: ``common``, the identity tuple both regimes share
    (``_COMMON_EVAL_FIELDS``, ``experiment_id`` possibly ``None`` but every key present, or this
    refuses), followed by ``extra``, this regime's own fields (metrics included), unmodified. A key
    present in both refuses.
    """
    missing = [field for field in _COMMON_EVAL_FIELDS if field not in common]
    if missing:
        raise ValueError(
            f"evaluation_result: common is missing required field(s): {missing}")
    collisions = sorted(set(common) & set(extra))
    if collisions:
        raise ValueError(
            f"evaluation_result: extra collides with common on field(s): {collisions}")
    return {**{field: common[field] for field in _COMMON_EVAL_FIELDS}, **extra}


def run_test_evaluation(
    pass_: Pass, loader, device, *, iou_threshold: float = 0.5,
    score_weights: dict | None = None, tiling: dict | None = None,
    trait: TraitEntry | None = None,
) -> dict:
    """Score ``loader`` against the untiled ``pass_``'s model, governed by the pass's execution
    record as inference governs it, and return the result (:func:`evaluation_result`) with that
    record under ``execution``; ``trait`` is the confirmed entry whose criterion governs the
    count, as ``evaluate`` reads it.

    ``tiling`` describes the loader's regime for provenance only: a tile-level run scores
    per-tile predictions against per-tile ground truth, a diagnostic, not the delivery regime.
    """
    from tcip_mcp.pipelines.data.datasets import run_tiling
    from tcip_mcp.pipelines.training.evaluation import evaluate

    checkpoint, execution = pass_.checkpoint, pass_.execution
    task = checkpoint.task
    model = pass_.predictor.governed(execution).to(device)

    metrics = evaluate(model, loader, device, task, dims=pass_.predictor.dims,
                       conf_threshold=execution.conf, iou_threshold=iou_threshold,
                       score_weights=score_weights, trait=trait)
    tiled = run_tiling(task, tiling) is not None
    common = {
        "model_path": checkpoint.path, "task": task, **checkpoint.producer,
        "iou_threshold": iou_threshold,
        "execution": execution.record(),
        "eval_regime": "tile-level" if tiled else "full-frame-single-pass",
    }
    return evaluation_result(common, metrics)


def run_full_frame_evaluation(
    checkpoint, admitted: "Admission", *, stated: Stated,
    iou_threshold: float = 0.5, device: str | None = None, trait: TraitEntry | None = None,
) -> dict:
    """``checkpoint`` evaluated full frame: its tiled pass over ``stated`` (its conf and cap the
    stated ones), merged across tiles at the stated merge threshold or the one the evaluated
    ground truth derives (:func:`~tcip_mcp.pipelines.execution.execution_record`, whose refusals
    propagate), and matched to the full-frame ground truth of the detection loader a run over
    ``admitted`` builds, under ``trait``'s criterion when given. Returns
    :func:`~tcip_mcp.pipelines.training.evaluation.detection_metrics`, by mask for an
    instance-segmentation checkpoint and by box otherwise, beside the reference's object and
    detection counts and the pass's ``execution`` record. An empty admission, and a document
    carrying the subject only in geometry a detector cannot read, refuse by name."""
    from tcip_mcp.pipelines.execution import Reference, prepare
    from tcip_mcp.pipelines.operating_point import cap_saturated_frac
    from tcip_mcp.pipelines.training.evaluation import (
        detection_metrics, gt_objects, gt_records, prediction_record,
    )

    prep = prepare(checkpoint, stated.model_copy(update={"tile": True}), device=device)
    predictor = prep.predictor
    by_mask = predictor.task == "instance_seg"

    # The loader a run over this same ground truth builds, over the samples the producer admits.
    from tcip_mcp.pipelines.data.datasets import DocumentDataset, build_dataset, resolve_sizes
    from tcip_mcp.pipelines.data.label_queries import require_admitted

    require_admitted(admitted)
    # The task's own loader, at the width the predictor reads at: its targets are the objects the
    # task's geometry selects, and the predictor reads each source itself.
    measured_samples = admitted.every_sample()
    measured = build_dataset(
        predictor.task, scope=admitted.scope, samples=measured_samples,
        sizes=resolve_sizes(predictor.task, {"num_channels": predictor.in_chans},
                            measured_samples))
    assert isinstance(measured, DocumentDataset), "a detector's build over samples is one of these"
    gt_by_key = {key: gt_records(measured.det_targets(measured.document(key)))
                 for key in measured.stems}
    pass_ = prep.runnable(Reference(
        boxes_per_image=[[a["bbox"] for a in gt_objects({"gt": gt})] for gt in gt_by_key.values()],
        counted=None, footprint=None))
    execution = pass_.execution
    assert execution.conf is not None and execution.max_dets is not None, "a detector's record"
    per_image = [
        prediction_record(
            predictor.predict_sliced(measured.image_of(key), execution=execution,
                                     tile_batch_size=pass_.tile_batch_size, require_masks=by_mask),
            gt, image_id=measured.sample_of(key).member)
        for key, gt in gt_by_key.items()]

    common = {
        "model_path": checkpoint.path, "task": predictor.task, **checkpoint.producer,
        "iou_threshold": iou_threshold, "execution": execution.record(),
        "eval_regime": "full-frame-tiled-inference",
    }
    # scored_images/tallies: which images this number was computed over and which the
    # admission held out, so a reviewer can reconstruct the denominator.
    metrics = detection_metrics(per_image, trait=trait, conf_threshold=execution.conf,
                                iou_threshold=iou_threshold, by_mask=by_mask)
    del metrics["matchings"]
    return evaluation_result(common, {
        **metrics,
        "n_gt": sum(len(gt_objects(r)) for r in per_image),
        "n_pred": sum(len(r["dt"]) for r in per_image),
        "max_dets_cap_saturated_frac": cap_saturated_frac(per_image),
        "scored_images": len(per_image), "tallies": admitted.tallies,
    })

"""Orchestrates a checkpoint evaluation run (tile-level or delivery-grade full-frame) and returns
its scored result; ``evaluation.py`` keeps the metrics computation itself.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from tcip_mcp.pipelines.execution import Pass, Stated
    from tcip_mcp.traits import TraitEntry


# Pinned by name and presence only; a regime's own fields travel in the writer's `extra` instead.
_COMMON_EVAL_FIELDS = (
    "model_path", "task", "checkpoint_sha256", "experiment_id", "iou_type",
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
    iou_type: str | None = None, score_weights: dict | None = None,
    tiling: dict | None = None, trait: TraitEntry | None = None,
) -> dict:
    """Score ``loader`` against the untiled ``pass_``'s model, governed by the pass's execution
    record as inference governs it, and return the result (:func:`evaluation_result`) with that
    record under ``execution``; ``trait`` is the confirmed entry whose criterion governs the
    count, as ``evaluate`` reads it.

    ``tiling`` describes the loader's regime for provenance only: a tile-level run scores
    per-tile predictions against per-tile ground truth, a diagnostic, not the delivery regime.
    """
    from tcip_mcp.pipelines.model_build import recorded_model_dims
    from tcip_mcp.pipelines.training.evaluation import effective_iou_type, evaluate

    checkpoint, execution = pass_.checkpoint, pass_.execution
    task = checkpoint.task
    model = pass_.predictor.governed(execution).to(device)

    scored = partial(evaluate, model, loader, device, task,
                     dims=recorded_model_dims(checkpoint.payload.get("config") or {}),
                     iou_threshold=iou_threshold, iou_type=iou_type, score_weights=score_weights,
                     trait=trait)
    metrics = scored() if execution.conf is None else scored(
        conf_threshold=execution.conf, max_dets=cast(int, execution.max_dets))
    tiled = bool(tiling and tiling.get("enabled", True) and task == "detection")
    common = {
        "model_path": checkpoint.path, "task": task, **checkpoint.producer,
        "iou_type": effective_iou_type(task, iou_type), "iou_threshold": iou_threshold,
        "execution": execution.record(),
        "eval_regime": "tile-level" if tiled else "full-frame-single-pass",
    }
    return evaluation_result(common, metrics)


def run_full_frame_evaluation(
    checkpoint, images_dir: str, labels_dir: str, *, stated: Stated,
    iou_threshold: float = 0.5, device: str | None = None, trait: TraitEntry | None = None,
) -> dict:
    """Delivery-grade detection eval: tiled inference reconstructed to full frame, matched to
    full-frame GT, under ``trait``'s confirmed entry's criterion when given.

    Exercises the cross-tile merge and scores against un-fragmented GT. Tile-level
    (``run_test_evaluation`` with ``tiling``) is a diagnostic, never the delivery metric; for a
    checkpoint trained without tiling the untiled full-frame path is the delivery gate.

    The pass is :func:`~tcip_mcp.pipelines.execution.prepare_pass`'s tiled pass over ``stated``,
    and refuses as it does; its execution record is returned under ``execution``.

    The measured set is the detection loader a run over ``images_dir``/``labels_dir`` would build,
    over the platform's own admission under the class space the checkpoint records, its map
    included: the capture date whose confirmed negatives count is the one that admission reads,
    and a document carrying the subject only in geometry a detector cannot read refuses by name.

    A box metric (``iou_type="bbox"``): it requests boxes-only tiled inference
    (``predict_sliced(require_masks=False)``), so an instance_seg checkpoint is gated here on its
    boxes/counts, never on its masks.
    """
    from tcip_mcp.pipelines.execution import prepare_pass
    from tcip_mcp.pipelines.operating_point import cap_saturated_frac
    from tcip_mcp.pipelines.training.evaluation import (
        coco_detection_metrics, governing_counts, gt_records, prediction_record,
        resolve_match_criterion,
    )

    pass_ = prepare_pass(checkpoint, stated.model_copy(update={"tile": True}), device=device)
    predictor, execution = pass_.predictor, pass_.execution
    assert execution.conf is not None and execution.max_dets is not None, "a detector's record"

    # The loader a run over this same ground truth builds, over the samples the producer admits.
    from tcip_mcp.pipelines.data.datasets import DetectionDataset, build_dataset, resolve_sizes
    from tcip_mcp.pipelines.data.label_queries import admit, require_admitted
    from tcip_mcp.pipelines.data.selection import ClassScope

    contradicted_negatives: set[str] = set()
    admitted = admit(images_dir, labels_dir, scope=ClassScope.of(checkpoint.data_config),
                     contradicted_out=contradicted_negatives)
    require_admitted(admitted)
    # At the width the predictor reads at, like every other measurement door: this gate reads
    # targets and source paths off the loader, and the predictor reads each source itself.
    measured_samples = admitted.every_sample()
    measured = build_dataset(
        "detection", scope=admitted.scope, samples=measured_samples,
        sizes=resolve_sizes("detection", {"num_channels": predictor.in_chans}, measured_samples))
    assert isinstance(measured, DetectionDataset), "a detection build over samples is one of these"
    per_image = [
        prediction_record(
            predictor.predict_sliced(measured.sample_of(key).source, execution=execution,
                                     tile_batch_size=pass_.tile_batch_size, require_masks=False),
            gt_records(measured.det_targets(key)), image_id=measured.sample_of(key).member)
        for key in measured.stems]

    m = coco_detection_metrics(per_image, iou_threshold=iou_threshold,
                               conf_threshold=execution.conf, max_dets=execution.max_dets)
    keys = ("map", "map50", "map75", "map_at_maxdets", "map50_at_maxdets",
            "precision", "recall", "f1", "tp", "fp", "fn", "n_images", "n_gt", "n_pred")
    # task: the predictor's own real task, never a hardcoded "detection". iou_type stays the
    # literal "bbox": this gate always computes a box-only metric by design, see the docstring.
    common = {
        "model_path": checkpoint.path, "task": predictor.task,
        "iou_type": "bbox", **checkpoint.producer,
        "iou_threshold": iou_threshold, "execution": execution.record(),
        "eval_regime": "full-frame-tiled-inference",
    }
    # scored_images/tallies: which images this number was computed over and which the
    # admission held out, so a reviewer can reconstruct the denominator.
    extra: dict = {
        **{k: m[k] for k in keys},
        "max_dets_cap_saturated_frac": cap_saturated_frac(per_image),
        "scored_images": len(per_image), "tallies": admitted.tallies,
        # Names recorded negative whose label file now holds subject content; scored on that
        # content, not filtered out, but the stale confirmation needs re-review.
        "contradicted_negatives": sorted(contradicted_negatives),
    }
    # For a count trait, the delivery-grade count that gates the phenotype is the derived
    # criterion's tp/fp/fn (center-match, for a trait so configured), not AP@0.5, kept alongside, clearly labeled.
    if trait is not None:
        criterion = resolve_match_criterion(trait, per_image)
        gc = governing_counts(per_image, criterion, conf_threshold=execution.conf)
        extra.update({
            "governing_counts": gc, "governing_criterion": criterion,
            "map50_role": "comparability_only",
            "iou_tp": m["tp"], "iou_fp": m["fp"], "iou_fn": m["fn"],
            "iou_precision": m["precision"], "iou_recall": m["recall"], "iou_f1": m["f1"],
            "tp": gc["tp"], "fp": gc["fp"], "fn": gc["fn"],
            "precision": gc["precision"], "recall": gc["recall"], "f1": gc["f1"],
        })
    return evaluation_result(common, extra)

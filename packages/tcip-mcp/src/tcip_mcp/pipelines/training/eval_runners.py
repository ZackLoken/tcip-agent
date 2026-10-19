"""Checkpoint evaluation runs, tile-level or delivery-grade full-frame, each returning a result."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from tcip_mcp.model_registry import Producer
from tcip_mcp.pipelines.execution import Execution

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader

    from tcip_mcp.model_registry import VerifiedCheckpoint
    from tcip_mcp.pipelines.data.label_queries import Admission
    from tcip_mcp.pipelines.execution import Pass, Stated
    from tcip_mcp.pipelines.schemas import TilingSpec
    from tcip_mcp.traits import TraitEntry


class EvaluationIdentity(Producer):
    """What one evaluation's metrics are of, in either regime.

    The checkpoint at ``model_path`` (its sha256 and run), its ``task``, the ``iou_threshold`` its
    matches were made at, the pass's ``execution`` and the ``eval_regime`` it ran under.
    """

    model_path: str
    task: str
    iou_threshold: float
    execution: Execution
    eval_regime: Literal[
        "tile-level", "full-frame-single-pass", "full-frame-tiled-inference"
    ]


def evaluation_result(identity: EvaluationIdentity, extra: dict) -> dict:
    """Return one evaluation's result, ``identity``'s fields followed by ``extra``.

    ``extra`` is this regime's own fields (metrics included), unmodified, laid beside the
    identity's in one mapping; a key it shares with the identity refuses.
    """
    common = identity.model_dump(mode="json")
    collisions = sorted(set(common) & set(extra))
    if collisions:
        raise ValueError(
            f"evaluation_result: extra collides with the identity on field(s): {collisions}"
        )
    return {**common, **extra}


def run_test_evaluation(
    pass_: Pass,
    loader: DataLoader,
    device: torch.device,
    *,
    iou_threshold: float,
    tiling: TilingSpec | None = None,
    trait: TraitEntry | None,
) -> dict:
    """Score ``loader`` against the untiled ``pass_``'s model and return the evaluation result.

    Its boxes are counted at the pass's conf, each image capped at the pass's density and the
    objective weighted by the checkpoint's recorded ``evaluation.score_weights``
    (``evaluation.evaluate``); the result (:func:`evaluation_result`) carries the pass's record
    under ``execution``. ``trait`` is the confirmed entry whose criterion governs the count, as
    ``evaluate`` reads it.

    ``tiling`` describes the loader's regime for provenance only: a tile-level run scores
    per-tile predictions against per-tile ground truth, a diagnostic, not the delivery regime.
    """
    from tcip_mcp.pipelines.schemas import run_tiling
    from tcip_mcp.pipelines.training.evaluation import evaluate

    checkpoint, execution, predictor = (
        pass_.checkpoint,
        pass_.execution,
        pass_.predictor,
    )
    task = checkpoint.task

    metrics = evaluate(
        predictor.model.to(device),
        loader,
        device,
        task,
        dims=predictor.dims,
        conf_threshold=execution.conf,
        iou_threshold=iou_threshold,
        score_weights=checkpoint.spec.evaluation.score_weights,
        trait=trait,
        density=execution.density,
        mask_cut=execution.mask_cut,
    )
    tiled = run_tiling(task, tiling) is not None
    identity = EvaluationIdentity(
        **dict(checkpoint.producer),
        model_path=checkpoint.path,
        task=task,
        iou_threshold=iou_threshold,
        execution=execution,
        eval_regime="tile-level" if tiled else "full-frame-single-pass",
    )
    return evaluation_result(identity, metrics)


def run_full_frame_evaluation(
    checkpoint: VerifiedCheckpoint,
    admitted: Admission,
    *,
    stated: Stated,
    iou_threshold: float,
    device: str | None = None,
    trait: TraitEntry | None,
) -> dict:
    """Evaluate ``checkpoint`` full frame over ``admitted`` and return the evaluation result.

    The checkpoint runs its tiled pass over ``stated`` (its conf the stated one), its density,
    any tile geometry the checkpoint does not record and its merge threshold unless stated
    derived from the evaluated ground truth
    (:func:`~tcip_mcp.pipelines.execution.execution_record`, whose refusals propagate), matched
    to the full-frame ground truth of the detection loader a run over ``admitted`` builds, under
    ``trait``'s criterion when given. Returns
    :func:`~tcip_mcp.pipelines.training.evaluation.detection_metrics` without its matchings, by
    mask for a pass collecting masks and by box otherwise, beside the reference's object and
    detection counts, the scored image count, the admission's tallies and the pass's
    ``execution`` record. An empty admission, and a document carrying the subject only in
    geometry a detector cannot read, refuse by name.
    """
    from tcip_mcp.pipelines.execution import Reference, prepare
    from tcip_mcp.pipelines.operating_point import cap_saturated_frac
    from tcip_mcp.pipelines.training.evaluation import (
        detection_metrics,
        gt_objects,
        gt_records,
        result_record,
    )

    prep = prepare(
        checkpoint, stated.model_copy(update={"tile": True}), device=device
    )
    predictor = prep.predictor

    from tcip_mcp.pipelines.data.datasets import DocumentDataset
    from tcip_mcp.pipelines.data.label_queries import require_admitted
    from tcip_mcp.pipelines.data.split_construction import predictor_dataset

    require_admitted(admitted)
    # The task's own loader: its targets are the objects the task's geometry selects, and the
    # predictor reads each source itself.
    measured = predictor_dataset(
        predictor.task,
        admitted.every_sample(),
        admitted.scope,
        predictor,
        None,
    )
    assert isinstance(measured, DocumentDataset), (
        "a detector's build over samples is one of these"
    )
    gt_by_key = {
        key: gt_records(measured.det_targets(measured.document(key)))
        for key in measured.stems
    }
    pass_ = prep.runnable(Reference(regions=measured.regions))
    execution = pass_.execution
    assert execution.conf is not None, "a detector's record"
    per_image = [
        result_record(
            predictor.predict_sliced(
                measured.image_of(key),
                execution=execution,
                tile_batch_size=pass_.tile_batch_size,
            ),
            gt,
            image_id=measured.sample_of(key).member,
        )
        for key, gt in gt_by_key.items()
    ]

    identity = EvaluationIdentity(
        **dict(checkpoint.producer),
        model_path=checkpoint.path,
        task=predictor.task,
        iou_threshold=iou_threshold,
        execution=execution,
        eval_regime="full-frame-tiled-inference",
    )
    # scored_images/tallies: which images this number was computed over and which the
    # admission held out, so a reviewer can reconstruct the denominator.
    metrics = detection_metrics(
        per_image,
        trait=trait,
        conf_threshold=execution.conf,
        iou_threshold=iou_threshold,
        by_mask=execution.masks,
    )
    del metrics["matchings"]
    return evaluation_result(
        identity,
        {
            **metrics,
            "n_gt": sum(len(gt_objects(r)) for r in per_image),
            "n_pred": sum(len(r["dt"]) for r in per_image),
            "cap_saturated_frac": cap_saturated_frac(per_image),
            "scored_images": len(per_image),
            "tallies": admitted.tallies,
        },
    )

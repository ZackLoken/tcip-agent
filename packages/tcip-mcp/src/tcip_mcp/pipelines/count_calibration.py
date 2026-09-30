"""Resolve the count operating point over a locked, disjoint cal/holdout split of a labeled
directory, through a prepared pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tcip_mcp.pipelines.calibration import (
    calibration_reference_inputs, collect_calibration_records, pass_resolver_inputs,
)
from tcip_mcp.pipelines.data.splits import DEFAULT_CAL_SEED, DEFAULT_HOLDOUT_RATIO
from tcip_mcp.pipelines.resolution import ResolvedBundle


class CalibrationUsageError(ValueError):
    """A cal/holdout usage refusal, never a data-integrity one."""


@dataclass(frozen=True)
class CountCalibrationBundle:
    """Everything a caller needs from one resolution pass: the bundle itself for inspection, and
    the exact resolver inputs and checkpoint identity a validation record is earned from, without
    re-running the model.
    """

    trait: str
    dataset_hash: str
    bundle: ResolvedBundle
    resolver_inputs: dict[str, Any]
    reference_inputs: dict[str, Any]
    checkpoint_sha256: str
    locked: dict[str, Any]
    labels_dir: str


def resolve_count_operating_point(
    checkpoint_path: str,
    trait: str,
    labels_dir: str,
    images_dir: str,
    dataset_root: str,
    project: Path,
    *,
    group_by: str | None = None,
    group_key_map: dict[str, str] | None = None,
    selection_dir: str | None = None,
    holdout_ratio: float = DEFAULT_HOLDOUT_RATIO,
    seed: int = DEFAULT_CAL_SEED,
    device: str | None = None,
    regime: dict[str, Any] | None = None,
) -> CountCalibrationBundle:
    """One low-threshold pass over a disjoint calibration/holdout split
    (:func:`~tcip_mcp.pipelines.calibration.collect_calibration_records`), resolved into the
    count-unbiased operating point and its held-out count-bias gate.

    The pass is the one ``inference_tools._prepare_pass`` prepares from the checkpoint and
    ``regime``, its ``tile``/``tile_size``/``overlap``/``postprocess``/``cross_tile_nms`` as a
    ``run_inference`` caller states them (the checkpoint's own regime for any left out).
    ``checkpoint_path`` must be named by a registry entry under ``project``; an unregistered
    checkpoint raises :class:`~tcip_mcp.model_registry.UnregisteredCheckpoint`, and a pass that
    cannot be prepared raises :class:`CalibrationUsageError` with its refusal.

    ``labels_dir`` is this calibration's measurement reference, so it clears
    ``require_reference_ground_truth`` before anything else runs.

    The cal/holdout split locks on its first draw for this labels directory's identity
    (``resolve_locked_cal_holdout_split``, scoped under ``dataset_root``); ``val_ratio``/``seed``
    only take effect on that first call. Ground truth is read under the class space the checkpoint
    records. Without ``selection_dir`` every labeled stem the producer admits under it is the
    universe; ``selection_dir`` restricts it to a selection's calibration samples under
    ``labels_dir``, and conflicts with ``group_by``/``group_key_map``.

    Raises :class:`CalibrationUsageError` (a ``ValueError``) for every usage refusal above, for
    fewer than two labeled stems to split, and for whatever
    :func:`~tcip_mcp.pipelines.data.splits.selection_calibration_universe` raises under
    ``selection_dir``; ``require_reference_ground_truth``'s own refusal stays a bare
    ``ValueError``.
    """
    from tcip_mcp.pipelines.data.splits import selection_policy_conflict

    policy_conflict = selection_policy_conflict(selection_dir, group_by, group_key_map)
    if policy_conflict:
        raise CalibrationUsageError(policy_conflict)

    from tcip_annotation.json_io import require_reference_ground_truth
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.data.datasets import build_dataset, resolve_sizes
    from tcip_mcp.pipelines.data.splits import (
        resolve_locked_cal_holdout_split,
    )
    from tcip_mcp.pipelines.operating_point import (
        STAGED_CONF_FLOOR, apply_operating_point, attach_split_policy_provenance,
        derive_max_dets_from_counts, resolve_operating_point,
    )
    from tcip_mcp.pipelines.data.label_queries import (
        admit, foreground_counts, require_admitted,
    )
    from tcip_mcp.pipelines.resolution import (
        DEFAULT_POSTPROCESS, DEFAULT_TILE_BATCH_SIZE, dataset_hash,
    )
    from tcip_mcp.pipelines.training.evaluation import gt_records
    from tcip_mcp.tools.inference_tools import _prepare_pass

    # labels_dir is this function's measurement reference, cleared before any model/dataset work.
    require_reference_ground_truth(labels_dir)

    stated = {"tile": None, "tile_size": None, "overlap": None, "cross_tile_nms": None,
              "postprocess": DEFAULT_POSTPROCESS, **(regime or {})}
    p = _prepare_pass(
        load_registered_checkpoint(checkpoint_path, project=project), images_dir=None,
        conf_threshold=None, device=device, max_dets=None,
        tile_batch_size=DEFAULT_TILE_BATCH_SIZE, **stated)
    if isinstance(p, str):
        raise CalibrationUsageError(p)

    selection_sha256 = None
    # One membership for this pass, whichever named it: the selection's own held-out samples, or
    # the producer's admission over the place this door was pointed at.
    scope = p.scope
    counted: dict[str, Any]
    if selection_dir:
        from tcip_mcp.pipelines.data.selection import read_selection
        from tcip_mcp.pipelines.data.splits import selection_calibration_universe
        from tcip_mcp.pipelines.resolution import selection_digest

        selection = read_selection(selection_dir, project=project)
        selection_sha256 = selection_digest(selection, project)
        try:
            (stems, group_by, group_key_map, _excluded, annotation_counts,
             counted) = selection_calibration_universe(selection, labels_dir, scope)
        except ValueError as exc:
            raise CalibrationUsageError(str(exc)) from exc
    else:
        # Through the producer, so this door and a run over the same data admit one membership.
        admitted = admit(images_dir, labels_dir, scope=scope)
        require_admitted(admitted)
        stems = sorted(record.member for record in admitted.records)
        counted = {s.member: s for s in admitted.every_sample()}
        # The one per-sample counter, over each member's own recorded ground truth.
        annotation_counts = foreground_counts(counted, scope)
    if len(stems) < 2:
        raise CalibrationUsageError(
            f"Need >=2 labeled stems to split cal/holdout; found {len(stems)}.")

    dh = dataset_hash(labels_dir, stems=(stems if selection_dir else None))
    # Density-derived collection cap (the same formula resolve_operating_point uses for the
    # shipped max_dets), so the sweep isn't measured against a constant below a dense scene's need.
    density_cap = derive_max_dets_from_counts(list(annotation_counts.values()))
    # The first call for this labels_dir's GT identity draws and locks the cal/holdout split; a
    # later call over unchanged labels returns that split rather than a fresh, possibly weaker cut.
    locked = resolve_locked_cal_holdout_split(
        stems, identity_hash=dh, scope_root=dataset_root,
        annotation_counts=annotation_counts,
        group_by=group_by, group_key_map=group_key_map,
        holdout_ratio=holdout_ratio, seed=seed,
    )
    cal_stems, hold_stems = locked["calibration"], locked["holdout"]

    applied, _applied_attribute_path = apply_operating_point(
        p.predictor, STAGED_CONF_FLOOR, density_cap)

    # This pass's own recorded samples, each target and source read through the dataset a run
    # over them builds, at the width the predictor reads at.
    samples = [counted[s] for s in cal_stems + hold_stems]
    ds: Any = build_dataset("detection", tiling=None, samples=samples, scope=scope,
                            sizes=resolve_sizes("detection", {"num_channels": p.predictor.in_chans},
                                                samples))
    gt_of = {ds.member_of(k): gt_records(ds.det_targets(k)) for k in ds.stems}
    source_of = {ds.member_of(k): ds.sample_sources[k] for k in ds.stems}
    cal_records, hold_records = collect_calibration_records(
        p, [s for s in cal_stems if s in gt_of], [s for s in hold_stems if s in gt_of],
        source_of, gt_of)

    resolver_inputs: dict[str, Any] = {
        **pass_resolver_inputs(p),
        "dataset_hash": dh,
        "calibration_records": cal_records,
        "holdout_records": hold_records,
        "staged_conf_floor": applied.get("score_thresh"),
        "selection_dir": selection_dir,
        "calibration_labels_dir": labels_dir,
        "selection_sha256": selection_sha256,
    }
    bundle = resolve_operating_point(
        trait, project=project, experiment_id=p.identity["experiment_id"],
        **resolver_inputs)
    attach_split_policy_provenance(bundle, locked)

    return CountCalibrationBundle(
        trait=trait, dataset_hash=dh, bundle=bundle, resolver_inputs=resolver_inputs,
        reference_inputs=calibration_reference_inputs(resolver_inputs, locked, stems),
        checkpoint_sha256=p.identity["sha256"], locked=locked, labels_dir=str(Path(labels_dir)),
    )

"""Resolve the count operating point over a locked, disjoint calibration/holdout split of a
labeled directory: one bundle, read by the offline inspector
(``tcip calibrate-operating-point``, which prints it and writes nothing) and earned into a
validation record by :func:`tcip_mcp.tools.calibration_tools.calibrate_count_operating_point`.

Distinct from :func:`tcip_mcp.pipelines.calibration.calibrate_operating_point`, which resolves the
operating point inline for a live tiled ``run_inference`` pass over the *same* checkpoint's
production predictor; this one is the untiled, held-out-GT calibration a checkpoint is
independently validated against, over a labels directory the caller names directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tcip_mcp.pipelines.resolution import ResolvedBundle


class CalibrationUsageError(ValueError):
    """A cal/holdout draw refusal the offline script and the audited door both exit cleanly for:
    a usage or configuration mistake, never a data-integrity refusal. Distinct from the bare
    ``ValueError`` :func:`~tcip_annotation.json_io.require_reference_ground_truth` raises for an
    agent-authored, unadjudicated reference, which a caller lets propagate instead."""


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
    project_root: str,
    *,
    subject: str | None = None,
    attribute: str | None = None,
    experiment_id: str | None = None,
    group_by: str | None = None,
    group_key_map: dict[str, str] | None = None,
    selection_dir: str | None = None,
    val_ratio: float = 0.5,
    seed: int = 0,
    device: str | None = None,
) -> CountCalibrationBundle:
    """One low-threshold model pass over a disjoint calibration/holdout split, resolved into the
    count-unbiased operating point and its held-out count-bias gate.

    ``labels_dir`` is this calibration's measurement reference, so it clears
    ``require_reference_ground_truth`` before anything else runs. ``checkpoint_path`` must be
    named by a registry entry under ``project_root`` (``register_model``); an unregistered
    checkpoint raises :class:`~tcip_mcp.model_registry.UnregisteredCheckpoint`.

    The cal/holdout split locks on its first draw for this labels directory's identity
    (``resolve_locked_cal_holdout_split``, scoped under ``dataset_root``); ``val_ratio``/``seed``
    only take effect on that first call. ``selection_dir`` restricts the calibration universe to
    a selection's calibration samples under ``labels_dir`` instead of every labeled stem,
    requires ``subject``, and conflicts with ``group_by``/``group_key_map``.

    Raises :class:`CalibrationUsageError` (a ``ValueError``) for every usage refusal above, for
    fewer than two labeled stems to split, and for whatever
    :func:`~tcip_mcp.pipelines.data.splits.selection_calibration_universe` raises under
    ``selection_dir``; a caller (the offline script, the audited door) can catch that one
    type for a clean exit without also swallowing ``require_reference_ground_truth``'s own bare
    ``ValueError``, a data-integrity refusal rather than a usage one.
    """
    if selection_dir and not subject:
        raise CalibrationUsageError("selection_dir requires subject.")
    if selection_dir and (group_by is not None or group_key_map):
        raise CalibrationUsageError(
            "selection_dir conflicts with group_by/group_key_map: the group keys the selection "
            "recorded on its own samples govern the locked draw."
        )

    from torch.utils.data import DataLoader

    from tcip_annotation.json_io import require_reference_ground_truth
    from tcip_mcp.model_registry import load_registered_checkpoint, resolve_model_identity
    from tcip_mcp.pipelines.data.datasets import build_dataset, resolve_sizes
    from tcip_mcp.pipelines.data.splits import resolve_locked_cal_holdout_split
    from tcip_mcp.pipelines.inference.predictor import build_predictor
    from tcip_mcp.pipelines.operating_point import (
        attach_split_policy_provenance, derive_max_dets_from_counts, records_over_loader,
        resolve_operating_point, set_detector_operating_point,
    )
    from tcip_mcp.pipelines.data.label_queries import (
        admit, foreground_counts, require_admitted,
    )
    from tcip_mcp.pipelines.resolution import DEFAULT_MAX_DETS, dataset_hash
    from tcip_mcp.pipelines.training.collation import task_collate

    # labels_dir is this function's measurement reference, cleared before any model/dataset work.
    require_reference_ground_truth(labels_dir)

    checkpoint = load_registered_checkpoint(checkpoint_path, project_path=project_root)

    # Matches the MCP inference path's initial predictor construction; superseded below, once
    # this split's density is known, by set_detector_operating_point.
    predictor = build_predictor(checkpoint, device=device, max_dets=DEFAULT_MAX_DETS)
    tile_size = getattr(predictor, "train_tile_size", None)

    selection_sha256 = None
    # One membership and one class space for this pass, whichever named it: the selection's own
    # held-out samples, or the producer's admission over the place this door was pointed at.
    counted: dict[str, Any]
    if selection_dir:
        from tcip_mcp.pipelines.data.selection import read_selection
        from tcip_mcp.pipelines.data.splits import selection_calibration_universe
        from tcip_mcp.pipelines.resolution import selection_digest

        selection = read_selection(selection_dir)
        selection_sha256 = selection_digest(selection)
        try:
            (stems, group_by, group_key_map, _excluded,
             counted) = selection_calibration_universe(selection, labels_dir)
        except ValueError as exc:
            raise CalibrationUsageError(str(exc)) from exc
        scope = selection.scope
    else:
        # Through the producer, so this door and a run over the same data admit one membership.
        admitted = admit(images_dir, labels_dir, subject=subject, attribute=attribute)
        require_admitted(admitted)
        stems = sorted(record.member for record in admitted.records)
        counted = {s.member_stem: s for s in admitted.every_sample()}
        scope = admitted.scope
    if len(stems) < 2:
        raise CalibrationUsageError(
            f"Need >=2 labeled stems to split cal/holdout; found {len(stems)}.")

    dh = dataset_hash(labels_dir, stems=(stems if selection_dir else None))
    # The one per-sample counter, over each member's own recorded ground truth.
    annotation_counts = foreground_counts(counted, scope)
    # Density-derived collection cap (the same formula resolve_operating_point uses for the
    # shipped max_dets), so the sweep isn't measured against a constant below a dense scene's need.
    density_cap = derive_max_dets_from_counts(list(annotation_counts.values()))
    # The first call for this labels_dir's GT identity draws and locks the cal/holdout split; a
    # later call over unchanged labels returns that split rather than a fresh, possibly weaker cut.
    locked = resolve_locked_cal_holdout_split(
        stems, identity_hash=dh, scope_root=dataset_root,
        annotation_counts=annotation_counts,
        group_by=(group_by or "tile_prefix"), group_key_map=group_key_map,
        holdout_ratio=val_ratio, seed=seed,
        selection_dir=selection_dir,
    )
    cal_stems, hold_stems = locked["calibration"], locked["holdout"]

    # Floor the in-model conf so hesitant detections survive to be swept, and raise the cap to
    # this split's own density; executes after build_predictor's construction-time cap and wins.
    applied, _applied_attribute_path = set_detector_operating_point(
        predictor.model, score_thresh=0.01, detections_per_img=density_cap)

    def _records(sub: list[str]) -> list[dict]:
        # This pass's own recorded samples, at the width the predictor reads at: the measurement
        # reads the pixels its membership named, in the shape the model below takes.
        samples = [counted[s] for s in sub]
        ds = build_dataset("detection", tiling=None, samples=samples, scope=scope,
                           sizes=resolve_sizes("detection",
                                               {"num_channels": predictor.in_chans}, samples))
        loader = DataLoader(ds, batch_size=4, collate_fn=task_collate("detection"))
        return records_over_loader(predictor.model, loader, predictor.device, "detection")

    resolver_inputs: dict[str, Any] = {
        "dataset_hash": dh,
        "calibration_records": _records(cal_stems),
        "holdout_records": _records(hold_stems),
        "tile_size": tile_size,
        # tile_size above is the checkpoint's persisted training geometry when present: say so,
        # or an unclaimed value would wrongly stamp "default" rather than "derived".
        "tile_size_source": ("derived" if tile_size is not None else "default"),
        # This pass (_records, above) is always untiled, never predict_tiled/predict_batch
        # (tile=...), so tiled=False is stated rather than left to the tiled=True default.
        "tiled": False,
        "staged_conf_floor": applied.get("score_thresh"),
        "selection_dir": selection_dir,
        "calibration_labels_dir": labels_dir,
        "selection_sha256": selection_sha256,
    }
    bundle = resolve_operating_point(trait, experiment_id=experiment_id, **resolver_inputs)
    attach_split_policy_provenance(bundle, locked)

    checkpoint_sha256 = resolve_model_identity(checkpoint, experiment_id=experiment_id)["sha256"]

    drawn = (locked.get("redraw_history") or [{}])[-1]
    stated_values = {"split_identity_hash": dh, "split_content_hash": drawn.get("new_content_hash")}
    if selection_dir:
        stated_values["selection_dir"] = selection_dir
        reference_inputs: dict[str, Any] = {
            "label_stems": {"calibration": {"path": labels_dir, "stems": stems}},
            "stated_values": stated_values,
        }
    else:
        reference_inputs = {
            "label_dirs": {"calibration": labels_dir}, "stated_values": stated_values,
        }

    return CountCalibrationBundle(
        trait=trait, dataset_hash=dh, bundle=bundle, resolver_inputs=resolver_inputs,
        reference_inputs=reference_inputs, checkpoint_sha256=checkpoint_sha256, locked=locked,
        labels_dir=str(Path(labels_dir)),
    )

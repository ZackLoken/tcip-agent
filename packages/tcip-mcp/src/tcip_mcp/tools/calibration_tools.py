"""Calibration-administration tools: redrawing a locked cal/holdout split, calibrating a scalar
(ordinal-rank or continuous-value) trait against a disjoint held-out split, and earning a validated
count operating point over an already-published prediction bucket.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from tcip_mcp.server import mcp
from tcip_mcp.pipelines.data.splits import DEFAULT_CAL_SEED, DEFAULT_GROUP_BY, DEFAULT_HOLDOUT_RATIO

logger = logging.getLogger(__name__)


@mcp.tool()
def redraw_calibration_holdout(
    dataset_root: str,
    labels_dir: str | None = None,
    images_dir: str | None = None,
    identity_hash: str | None = None,
    group_by: str | None = None,
    group_key_map: dict[str, str] | None = None,
    seed: int = DEFAULT_CAL_SEED,
    holdout_ratio: float = DEFAULT_HOLDOUT_RATIO,
    reason: str = "",
    selection_dir: str | None = None,
    subject: str | None = None,
    attribute: str | None = None,
) -> dict:
    """Deliberately redraw a locked calibration/holdout split.

    ``reason`` is required and non-empty. Every redraw is appended to the lock's ``redraw_history``
    with its policy, seed, and the old and new split's content hashes; the old and new split
    membership is recorded in the dataset's own audit log alongside the reason (and, when given,
    ``selection_dir``).

    Provide either ``labels_dir`` (the identity is derived as ``dataset_hash(labels_dir)``, and its
    stems are re-scanned) or ``identity_hash`` directly (e.g. a review-reference hash; the existing
    lock's own calibration+holdout stems are then the redraw's stem universe).

    Args:
        dataset_root: The root the lock is stored under, required. With ``labels_dir`` given, it is
            the root that dir's own lock lives under (its dataset root, or the dir itself when the
            layout places it under none), and a root disagreeing with it refuses.
        labels_dir: Labeled dir whose GT identity locked the split (mutually exclusive with
            ``identity_hash``; if both are omitted, or ``identity_hash`` is given with no existing
            lock and no ``labels_dir``, this refuses).
        images_dir: Images for ``labels_dir``. When given, stems are the
            labels-intersect-images-on-disk universe ``run_inference``'s calibration uses. Omitted
            -> every labeled stem is used regardless of whether an image still exists for it.
        identity_hash: The locked split's identity hash directly.
        group_by: New grouping policy, ``"tile_prefix"`` / ``"stem"`` (ignored if ``group_key_map``
            is given). ``None`` (default) resolves to ``"tile_prefix"`` without a selection; a
            value beside ``selection_dir`` refuses, naming both.
        group_key_map: Explicit ``{stem: group_key}`` map covering every stem, overriding
            ``group_by``. Conflicts with ``selection_dir`` the same way ``group_by`` does.
        seed: New split seed.
        holdout_ratio: New calibration/holdout fraction.
        reason: Required, non-empty justification for this redraw, recorded in the dataset's own
            audit log alongside the old and new split membership.
        selection_dir: Restrict the redraw's universe to a selection's ``calibration`` samples
            under ``labels_dir`` (``pipelines.data.selection.read_selection``), the restriction
            ``run_inference`` applies. Requires ``labels_dir``. The scope is the selection's own; a
            selection over per-image label documents that records no subject refuses by name. The
            identity is ``dataset_hash(labels_dir, stems=universe)``.
        subject: The object class this redraw's foreground is counted for, for the whole-directory
            universe. A selection states its own and overrides it.
        attribute: The attribute the foreground count is scoped to, the same way.
    """
    if not reason or not reason.strip():
        return {"error": "reason is required (a non-empty justification) for a force_redraw"}
    if not labels_dir and not identity_hash:
        return {"error": "provide either labels_dir or identity_hash"}
    if selection_dir is not None:
        if not labels_dir:
            return {"error": "selection_dir requires labels_dir: the universe is drawn "
                             "from the selection's held-out samples under that directory."}
        from tcip_mcp.pipelines.data.splits import selection_policy_conflict

        policy_conflict = selection_policy_conflict(selection_dir, group_by, group_key_map)
        if policy_conflict:
            return {"error": policy_conflict}

    from datetime import datetime, timezone

    from tcip_annotation.json_io import UnreadableLabelDocument

    from tcip_mcp.pipelines.data.splits import (
        cal_holdout_scope_root, count_label_lines, label_image_stems,
        resolve_locked_cal_holdout_split,
    )
    from tcip_mcp.pipelines.resolution import dataset_hash

    selection_stems: list[str] | None = None
    if selection_dir is not None:
        from tcip_mcp.pipelines.data.selection import read_selection, unscoped_document_issue
        from tcip_mcp.pipelines.data.splits import selection_calibration_universe

        from tcip_mcp.pipelines.data.label_queries import refuse_inadmissible_samples

        assert labels_dir is not None, "the selection_dir refusal above requires it"
        selection = read_selection(selection_dir)
        # The scope is the selection's own, read off it: a subject scopes a document selection
        # and nothing else, so a mask or table selection redraws without one.
        unscoped = unscoped_document_issue(selection, selection_dir)
        if unscoped:
            return {"error": unscoped}
        try:
            (selection_stems, group_by, group_key_map, _excluded, selection_counts,
             universe_samples) = selection_calibration_universe(selection, labels_dir)
            # The one re-admission over recorded samples, run before a lock is drawn over them:
            # a member the calibration that reads this lock would refuse is not lockable here.
            refuse_inadmissible_samples(
                [universe_samples[stem] for stem in selection_stems], selection.scope)
        except (ValueError, UnreadableLabelDocument) as exc:
            return {"error": str(exc)}

    scope_root = Path(dataset_root).resolve()
    if labels_dir:
        labels_scope = cal_holdout_scope_root(labels_dir)
        if labels_scope != scope_root:
            return {"error": f"labels_dir {labels_dir!r} locks its cal/holdout split under "
                             f"{str(labels_scope)!r}, and dataset_root states {str(scope_root)!r}. "
                             "A redraw under the stated root would replace a lock the calibration "
                             "never reads, so state the root those labels' own lock lives under."}

    if identity_hash is None:
        assert labels_dir is not None, "the earlier refusal above requires one of the two"
        # dataset_hash enumerates through prediction_documents and hashes each file's raw bytes;
        # it never parses one, so it cannot raise the named error the other reads here guard for.
        identity_hash = dataset_hash(labels_dir, stems=selection_stems)

    if selection_stems is not None:
        # Set only inside the selection_dir branch above, which already required labels_dir.
        assert labels_dir is not None, "selection_stems is only set where labels_dir was required"
        stems, annotation_counts = selection_stems, selection_counts
    elif labels_dir:
        # The same labels-intersect-images scan calibrate_operating_point uses, not a second
        # independent glob (images_dir omitted degrades to the labels-only scan).
        from tcip_mcp.dataset_layout import label_filename

        stems, _ = label_image_stems(labels_dir, images_dir)
        try:
            # The one caller here holding a name rather than a record composes its path, here.
            annotation_counts = {
                s: count_label_lines(Path(labels_dir) / label_filename(s),
                                     subject=subject, attribute=attribute)
                for s in stems
            }
        except UnreadableLabelDocument as exc:
            return {"error": str(exc)}
    else:
        # No labels to re-scan: the library takes the existing lock's own members as the universe.
        stems, annotation_counts = None, None

    try:
        new_lock = resolve_locked_cal_holdout_split(
            stems, identity_hash=identity_hash, scope_root=scope_root,
            annotation_counts=annotation_counts,
            group_by=group_by, group_key_map=group_key_map,
            holdout_ratio=holdout_ratio, seed=seed,
            force_redraw=True, timestamp=datetime.now(timezone.utc).isoformat(),
            selection_dir=selection_dir, reason=reason,
        )
    except ValueError as exc:
        return {"error": str(exc)}
    new_membership = {"calibration": new_lock["calibration"], "holdout": new_lock["holdout"]}
    return {"identity_hash": identity_hash, "reason": reason,
            "old_membership": new_lock["old_membership"], "new_membership": new_membership}


def _scalar_predictions(predictor, image_source, stems: list[str], suffix: str) -> dict[str, float]:
    """Run ``predictor`` over ``stems``' images and pull each image's single scalar prediction.

    ``suffix`` is the bespoke model's own output-key convention for this task (``"_ranks"`` for
    ordinal, ``"_values"`` for regression, an ``OrdinalHead``/``RegressionHead`` decode output
    prefixed ``head{i}_``), matched by key suffix. Only the first matching key per prediction is
    used.
    """
    if not stems:
        return {}
    results = predictor.predict_batch([image_source[s] for s in stems])
    out: dict[str, float] = {}
    for stem, pred in zip(stems, results):
        for key, val in pred.items():
            if key.endswith(suffix) and isinstance(val, list) and val:
                out[stem] = float(val[0])
                break
    return out


_ORDINAL_REGRESSION_TASKS = {
    "ordinal": {"true_key": "true_rank", "pred_key": "predicted_rank", "suffix": "_ranks"},
    "regression": {"true_key": "true_value", "pred_key": "predicted_value", "suffix": "_values"},
}


@mcp.tool()
def calibrate_scalar_operating_point(
    trait_name: str,
    task: str,
    checkpoint_path: str,
    images_dir: str,
    csv_path: str,
    criterion: str,
    output_dir: str,
    dataset_root: str,
    experiment_id: str | None = None,
    group_by: str = DEFAULT_GROUP_BY,
    group_key_map: dict[str, str] | None = None,
    seed: int = DEFAULT_CAL_SEED,
    holdout_ratio: float = DEFAULT_HOLDOUT_RATIO,
) -> dict:
    """Calibrate and validate a trait's ordinal-rank or continuous-value prediction against a
    disjoint held-out split.

    Runs live inference over a locked cal/holdout split (``resolve_locked_cal_holdout_split``) of
    the CSV's own stems, then the calibration gate
    (``operating_point.resolve_ordinal_operating_point``/``resolve_regression_operating_point``:
    disjointness, train-disjointness, a derived compensating-error floor on a holdout-only
    criterion score), stamped into ``<output_dir>/ordinal_operating_point.json`` or
    ``regression_operating_point.json`` (see
    ``resolution.read_ordinal_operating_point_sidecar``/``read_regression_operating_point_sidecar``).

    The universe is every row the producer admits from this table.

    A stamp that claims validation names the record it was earned from:
    ``resolution.open_validation`` runs the gate over the evidence, ``seal_validation`` files the
    row and returns the stamp with its pointer merged in, and the stamp is written last. A
    calibration that does not clear its gate stamps unvalidated, with its failures, and earns
    nothing.

    Record and stamp carry ``checkpoint_sha256`` from ``resolve_model_identity`` over the
    checkpoint this door ran.

    Args:
        trait_name: The registered trait whose rank/value prediction is being calibrated.
        task: ``"ordinal"`` or ``"regression"``, dispatches which criterion toolkit, item shape and
            sidecar file apply.
        checkpoint_path: The trained checkpoint to calibrate. Must be registered under this
            process's platform state root (``register_model``, explicit mode for a foreign or
            bespoke checkpoint) or this door refuses before loading it.
        images_dir: Directory holding the CSV's images.
        csv_path: The ``(stem, value)`` CSV ``OrdinalDataset``/``RegressionDataset`` reads.
        criterion: Which registered criterion to calibrate against
            (``operating_point.ORDINAL_CRITERIA``/``REGRESSION_CRITERIA``), required.
        output_dir: Where to write the sidecar.
        dataset_root: The dataset this calibration's claim hangs off: the record's reference
            locations (the CSV, the images directory, the locked split) are written against it, and
            the cal/holdout lock is stored under it. Refuses when the images directory's own layout
            places it under a different root; a loose directory the layout cannot place refuses
            nothing.
        experiment_id: The checkpoint's own training run's record id (``tcip_mcp.experiments``), if
            known, gates train-disjointness. ``None`` (a foreign/unregistered checkpoint) skips
            that check.
        group_by / group_key_map / seed / holdout_ratio: The locked cal/holdout split's grouping
        policy, same semantics as ``run_inference``'s own calibration arguments; only the first
        call for this CSV's identity draws the split.
    """
    if task not in _ORDINAL_REGRESSION_TASKS:
        return {"error": f"task must be one of {sorted(_ORDINAL_REGRESSION_TASKS)}, got {task!r}"}

    from tcip_mcp.tools.phenology_tools import _stated_root_disagreement

    disagreement = _stated_root_disagreement(dataset_root, {"images_dir": images_dir})
    if disagreement:
        return {"error": disagreement}

    from tcip_mcp.model_registry import UnregisteredCheckpoint, load_registered_checkpoint
    from tcip_mcp.pipelines.data.splits import cal_holdout_scope_root, resolve_locked_cal_holdout_split
    from tcip_mcp.pipelines.inference.predictor import build_predictor
    from tcip_mcp.pipelines.operating_point import (
        resolve_ordinal_operating_point,
        resolve_regression_operating_point,
    )
    from tcip_mcp.pipelines.resolution import csv_dataset_hash
    from tcip_mcp.traits import TraitUnknownError, get_trait

    try:
        get_trait(trait_name)
    except TraitUnknownError as e:
        return {"error": str(e)}

    try:
        checkpoint = load_registered_checkpoint(checkpoint_path)
    except UnregisteredCheckpoint as exc:
        return {"error": str(exc)}

    shape = _ORDINAL_REGRESSION_TASKS[task]
    is_ordinal = task == "ordinal"

    # Through the producer, so this door and a run over the same table hold one membership: the
    # rows it admits are the rows this door measures, each with the source it resolved.
    from tcip_mcp.pipelines.data.label_queries import admit, ground_truth_table, require_admitted
    from tcip_mcp.pipelines.image_utils import resolve_source_path

    try:
        admitted = admit(images_dir, csv_path)
        require_admitted(admitted)
    except ValueError as exc:
        return {"error": str(exc)}
    stems = sorted(record.member for record in admitted.records)
    source_of = {record.member: resolve_source_path(record.source)
                 for record in admitted.records}

    cast = int if is_ordinal else float
    try:
        table = ground_truth_table(csv_path)
        true_by_stem: dict[str, float] = {stem: cast(table[stem]) for stem in stems}
    except ValueError as exc:
        return {"error": f"{csv_path!r} does not read as a {task} ground-truth table: {exc}"}

    identity_hash = csv_dataset_hash(csv_path)
    try:
        locked = resolve_locked_cal_holdout_split(
            stems, identity_hash=identity_hash, scope_root=cal_holdout_scope_root(dataset_root),
            group_by=group_by, group_key_map=group_key_map,
            seed=seed, holdout_ratio=holdout_ratio,
        )
    except ValueError as exc:
        return {"error": str(exc)}
    cal_stems, hold_stems = locked["calibration"], locked["holdout"]

    predictor = build_predictor(checkpoint)
    cal_pred = _scalar_predictions(predictor, source_of, cal_stems, shape["suffix"])
    hold_pred = _scalar_predictions(predictor, source_of, hold_stems, shape["suffix"])

    def _items(sub_stems: list[str], preds: dict[str, float]) -> list[dict]:
        return [{"image_id": s, shape["true_key"]: true_by_stem[s], shape["pred_key"]: cast(preds[s])}
                for s in sub_stems if s in preds]

    cal_items = _items(cal_stems, cal_pred)
    hold_items = _items(hold_stems, hold_pred)

    resolver = resolve_ordinal_operating_point if is_ordinal else resolve_regression_operating_point
    # The table is where these rows' ground truth lives, the scope the run's own partition records
    # them under, so the selection check can name a bound run's own validation row.
    resolver_inputs: dict[str, Any] = {
        "criterion": criterion, "calibration_items": cal_items, "holdout_items": hold_items,
        "calibration_labels_dir": str(csv_path)}
    result = resolver(trait_name, experiment_id=experiment_id, **resolver_inputs)

    from tcip_mcp.project_paths import resolve_output_path

    from tcip_mcp.model_registry import resolve_model_identity
    from tcip_mcp.pipelines.resolution import open_validation, seal_validation, write_sidecar

    out = resolve_output_path(output_dir)
    document = f"{task}_operating_point"
    checkpoint_sha256 = resolve_model_identity(
        checkpoint, experiment_id=experiment_id)["sha256"]
    stamp = {
        "operating_point": {task: {"validated_against": result["validated_against"],
                                   "criterion": criterion}},
        "validated": result["passed"],
        "validated_by": None,
        "failures": result["failures"],
        "gate_evidence": result["gate_evidence"],
        "checkpoint_sha256": checkpoint_sha256,
        "experiment_id": experiment_id,
        "trait": trait_name,
    }
    if result["passed"]:
        draft = open_validation(
            document=document,
            # Named off the function this door reported from, so record and report share one gate.
            evidence={"resolver": resolver.__name__, "inputs": resolver_inputs},
            trait=trait_name, checkpoint_sha256=checkpoint_sha256,
            producing_experiment_id=experiment_id,
            reference_inputs={
                "dataset_root": dataset_root,
                "label_csvs": {"reference": csv_path},
                "scope_roots": {"images": images_dir},
                "stated_values": {"split_identity": identity_hash},
            },
        )
        stamp = seal_validation(draft, dataset_root=dataset_root, bucket_dirs=[],
                                   stamp_body=stamp)
    write_sidecar(out, stamp, document)
    return {
        "output_dir": str(out),
        "validated_against": result["validated_against"],
        "passed": result["passed"],
        "failures": result["failures"],
        "validated_by": stamp["validated_by"],
        "n_calibration_items": len(cal_items),
        "n_holdout_items": len(hold_items),
        "criterion": criterion,
    }


@mcp.tool()
def calibrate_count_operating_point(
    checkpoint_path: str,
    trait: str,
    labels_dir: str,
    images_dir: str,
    dataset_root: str,
    pred_dir: str,
    *,
    subject: str | None = None,
    attribute: str | None = None,
    experiment_id: str | None = None,
    group_by: str | None = None,
    group_key_map: dict[str, str] | None = None,
    selection_dir: str | None = None,
    holdout_ratio: float = DEFAULT_HOLDOUT_RATIO,
    seed: int = DEFAULT_CAL_SEED,
    device: str | None = None,
) -> dict:
    """Calibrate the count operating point against held-out GT, and earn a validated claim over
    ``pred_dir`` only when the earned conf is the conf its stored detections were produced at.

    Runs :func:`tcip_mcp.pipelines.count_calibration.resolve_count_operating_point`: one
    low-threshold model pass over a disjoint, locked calibration/holdout split of ``labels_dir``,
    resolved into the count-unbiased conf and its held-out count-bias gate. A validated stamp is
    written only when the earned conf equals the conf ``pred_dir``'s own stamp already records as
    its production conf (``operating_point.conf.value``, read before the pass runs). Any other
    earned conf refuses by name, stating both values, and points at ``run_inference``, whose
    calibrated path re-predicts a bucket at the earned conf; this is decided before
    ``open_validation``/``seal_validation`` run. The same equality is re-decided under the stamp's
    own lock (``resolution.update_sidecar``) against the stamp as stored.

    Decided against the stamp as stored: whether ``pred_dir`` already carries an earned claim,
    through ``resolution.verify_stamp_binding``; the tile-geometry floor
    (``resolution.fold_tile_validation``); ``trait``, written only when this calibration earns the
    claim; and ``shippable_issues``, refreshed to this run's own list. A calibration whose
    count-bias gate does not clear, or whose tile geometry never validated, merges an honest
    ``conf`` with ``validated=false`` and earns nothing.

    ``pred_dir`` must already carry an ``operating_point.json`` stamp naming a checkpoint identity,
    hold at least one prediction document, not name a whole-raster bucket
    (``resolution.stamp_names_raster``), and sit under ``dataset_root``
    (``resolution.bucket_relative_key``). A bucket outside ``dataset_root``, with no stamp at all,
    empty, a whole-raster bucket, whose stamp carries no ``checkpoint_sha256``, whose stamped
    checkpoint disagrees with ``checkpoint_path``, or that already carries a claim
    ``verify_stamp_binding`` answers for, all refuse by name before the calibration pass draws its
    cal/holdout lock. A stamp changing to a different production conf while the pass is running
    refuses after a record was sealed; that leaves a minted calibration experiment and an inert
    validation row behind, which the response names.

    Args:
        checkpoint_path: The trained checkpoint to calibrate; must be registered under the platform
            state root (``register_model``) or this door refuses before loading it.
        trait: The registered trait whose count is being calibrated.
        labels_dir: Labeled dir (per-image JSON), this calibration's measurement reference.
        images_dir: Images for ``labels_dir``.
        dataset_root: The root the cal/holdout split lock is stored under and ``pred_dir`` must sit
            beneath; the labels' dataset root, or ``labels_dir`` itself when the layout places it
            under none.
        pred_dir: The already-published prediction bucket this claim covers.
        subject / attribute: The object class / assessed attribute the labeled reference is scoped
        to; when ``pred_dir``'s stamp already records a scope, an omitted pair takes the bucket's
        own recorded scope and a stated pair must equal it, refusing by name otherwise.
        experiment_id: The checkpoint's own training run's record id (``tcip_mcp.experiments``), if
            known, gates train-disjointness; ``None`` (a foreign/unregistered checkpoint) skips
            that check.
        group_by / group_key_map: The locked cal/holdout split's grouping policy; only the first
        call for this labels_dir's identity draws the split.
        selection_dir: Restrict the calibration universe to a selection's calibration samples under
            the labels directory instead of every labeled stem; conflicts with
            ``group_by``/``group_key_map``.
        holdout_ratio / seed: The locked split's holdout fraction and seed; take effect only on the
        first draw for this labels_dir's identity.
        device: cuda / cpu (auto if omitted).
    """
    from tcip_annotation.json_io import prediction_documents
    from tcip_mcp.model_registry import (
        UnregisteredCheckpoint, load_registered_checkpoint, resolve_model_identity,
    )
    from tcip_mcp.pipelines.calibration import gate_evidence_summary
    from tcip_mcp.pipelines.count_calibration import resolve_count_operating_point
    from tcip_mcp.pipelines.resolution import (
        bucket_relative_key, bucket_scope, claim_payload,
        fold_tile_validation, open_validation, read_operating_point_sidecar, seal_validation,
        stamp_names_raster, update_sidecar, verify_stamp_binding,
    )
    from tcip_mcp.project_paths import platform_state_root
    from tcip_store import StoreError
    from tcip_store.errors import SchemaVersionRefused, StoreBusy

    root = Path(dataset_root).resolve()
    bucket = Path(pred_dir).resolve()
    try:
        bucket_relative_key(bucket, root, document="operating_point")
    except ValueError as exc:
        return {"error": str(exc)}

    existing = read_operating_point_sidecar(bucket)
    if not existing:
        return {"error": f"{bucket} carries no operating_point.json stamp; "
                         "calibrate_count_operating_point earns a claim over an already-"
                         "published inference bucket, never an empty one."}
    try:
        existing_scope = bucket_scope(bucket)
    except StoreError as exc:
        return {"error": str(exc)}
    if subject is None and attribute is None:
        if existing_scope is not None:
            subject, attribute = existing_scope.subject, existing_scope.attribute
    elif existing_scope is not None and (existing_scope.subject, existing_scope.attribute) != (
            subject, attribute):
        return {"error": (
            f"{bucket}'s stamp records scope (subject={existing_scope.subject!r}, "
            f"attribute={existing_scope.attribute!r}), not the (subject={subject!r}, "
            f"attribute={attribute!r}) this calibration states: evidence earned under one scope "
            "is never merged into a bucket stamped for another. State the bucket's own scope, or "
            "calibrate a bucket that matches the scope you intend."
        )}
    if stamp_names_raster(existing):
        return {"error": f"{bucket} is a whole-raster bucket (its stamp records raster_path): "
                         "the count-unbiased calibration reasons over per-image predictions, "
                         "which one mosaic total is not."}
    if not prediction_documents(bucket):
        return {"error": f"{bucket} carries a readable operating_point.json stamp but no "
                         "prediction documents: an empty bucket is not this claim's subject "
                         "either."}
    existing_binding = verify_stamp_binding(existing, bucket, document="operating_point")
    if existing_binding.claimed and existing_binding.ok:
        return {"error": f"{bucket} already carries a validated operating_point.json stamp a "
                         "record answers for (earned at the conf its predictions were produced "
                         "at); calibrate_count_operating_point never overwrites an earned stamp."}

    # Checkpoint identity is derived and refused on before the pass below draws its lock.
    existing_sha = existing.get("checkpoint_sha256")
    try:
        checkpoint = load_registered_checkpoint(
            checkpoint_path, project_path=str(platform_state_root()))
    except UnregisteredCheckpoint as exc:
        return {"error": str(exc)}
    checkpoint_sha256 = resolve_model_identity(checkpoint, experiment_id=experiment_id)["sha256"]
    if not existing_sha:
        return {"error": f"{bucket} carries no checkpoint_sha256 in its operating_point.json "
                         "stamp; a count-calibration claim sealed under a digest the stamp does "
                         "not carry could never bind at delivery."}
    if existing_sha != checkpoint_sha256:
        return {"error": f"{bucket} was produced by checkpoint {existing_sha!r}, not "
                         f"{checkpoint_sha256!r} ({checkpoint_path}); a count-"
                         "calibration claim covers the checkpoint that produced these "
                         "predictions."}

    try:
        resolved = resolve_count_operating_point(
            checkpoint_path=checkpoint_path, trait=trait, labels_dir=labels_dir,
            images_dir=images_dir, dataset_root=dataset_root,
            project_root=str(platform_state_root()), subject=subject, attribute=attribute,
            experiment_id=experiment_id, group_by=group_by, group_key_map=group_key_map,
            selection_dir=selection_dir, holdout_ratio=holdout_ratio, seed=seed, device=device,
        )
    except (ValueError, UnregisteredCheckpoint) as exc:
        return {"error": str(exc)}

    conf = resolved.bundle.get("conf")
    conf_provenance = conf.to_provenance()
    earned_conf_value = conf_provenance["value"]
    gate_summary = gate_evidence_summary(conf)
    issues = resolved.bundle.shippable_issues()

    def _conf_mismatch_error(production_conf_value: object) -> str:
        return (
            f"{bucket}'s operating_point.json stamp records its predictions were produced at "
            f"conf={production_conf_value!r}, and this calibration earned "
            f"conf={earned_conf_value!r}: a validated stamp can only claim the conf its stored "
            "predictions were produced at, since calibrate_count_operating_point never "
            "re-predicts pred_dir. Produce a fresh bucket at the earned conf through "
            "run_inference's calibrated path, then calibrate that bucket."
        )

    # Tentative, against the tile floor and production conf read before the pass; the merge
    # below re-decides both against the stamp as actually stored, for a race only.
    existing_conf_value = ((existing.get("operating_point") or {}).get("conf") or {}).get("value")
    validated_tentative = fold_tile_validation(conf.is_shippable, existing.get("tile_size_validated"))
    if validated_tentative and existing_conf_value != earned_conf_value:
        return {"error": _conf_mismatch_error(existing_conf_value)}
    draft = None
    earned = dict(existing)
    earned["operating_point"] = {**(existing.get("operating_point") or {}), "conf": conf_provenance}
    earned["validated"] = validated_tentative
    earned["gate_evidence_summary"] = gate_summary
    earned["shippable_issues"] = issues
    if validated_tentative:
        earned["trait"] = trait
        draft = open_validation(
            document="operating_point",
            evidence={"resolver": "resolve_operating_point", "inputs": resolved.resolver_inputs},
            trait=trait, checkpoint_sha256=resolved.checkpoint_sha256,
            producing_experiment_id=experiment_id,
            reference_inputs={**resolved.reference_inputs, "dataset_root": str(root)},
        )
        earned = seal_validation(
            draft, dataset_root=str(root), bucket_dirs=[bucket], stamp_body=earned)
    else:
        earned["validated_by"] = None

    refusal: dict[str, object] = {}

    def _merge(stored: dict) -> dict | None:
        """Merge this calibration's earned conf into whatever the producing run left, inside the
        stamp's own lock, deciding the answered claim (``verify_stamp_binding``), ``validated``,
        the tile floor and the conf-equality rule against the stamp as stored.
        """
        binding = verify_stamp_binding(stored, bucket, document="operating_point")
        if binding.claimed and binding.ok:
            return None
        validated = fold_tile_validation(conf.is_shippable, stored["tile_size_validated"])
        stored_conf_value = ((stored["operating_point"] or {}).get("conf") or {}).get("value")
        if validated and stored_conf_value != earned_conf_value:
            refusal["stored_conf_value"] = stored_conf_value
            return None
        merged = dict(stored)
        merged["operating_point"] = {**(stored["operating_point"] or {}),
                                     "conf": conf_provenance}
        merged["gate_evidence_summary"] = gate_summary
        merged["shippable_issues"] = issues
        merged["validated"] = validated
        if not validated:
            merged["validated_by"] = None
            return merged
        if draft is None:
            return None
        if (claim_payload(merged, document="operating_point")
                != claim_payload(earned, document="operating_point")):
            return None
        merged["validated_by"] = earned["validated_by"]
        merged["trait"] = trait
        return merged

    try:
        wrote = update_sidecar(bucket, _merge)
    except (StoreBusy, ValueError, SchemaVersionRefused) as exc:
        return {"error": str(exc)}

    if not wrote:
        orphan = (
            f" This calibration's own validation record {earned.get('validated_by')} was filed "
            "before the merge refused and answers for nothing: no bucket names it."
            if draft is not None else ""
        )
        if "stored_conf_value" in refusal:
            return {"error": _conf_mismatch_error(refusal["stored_conf_value"]) + orphan}
        return {"error": f"{bucket}'s operating_point.json stamp changed while this calibration "
                         f"pass was running; recalibrate against the bucket as it is now.{orphan}"}

    new_stamp = read_operating_point_sidecar(bucket) or {}
    new_binding = verify_stamp_binding(new_stamp, bucket, document="operating_point", trait=trait)
    validated_now = bool(new_binding.claimed and new_binding.ok)
    return {
        "pred_dir": str(bucket),
        "trait": trait,
        "dataset_hash": resolved.dataset_hash,
        "validated": validated_now,
        "validated_against": (
            ((new_stamp["operating_point"] or {}).get("conf") or {}).get("validated_against")
            if validated_now else None
        ),
        "validated_by": new_stamp["validated_by"] if validated_now else None,
        "n_calibration_images": len(resolved.resolver_inputs["calibration_records"]),
        "n_holdout_images": len(resolved.resolver_inputs["holdout_records"]),
        "gate_evidence": new_stamp.get("gate_evidence_summary"),
    }

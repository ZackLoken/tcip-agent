"""Calibration-administration tools: redrawing a locked cal/holdout split, and calibrating a
scalar (ordinal-rank or continuous-value) trait against a disjoint held-out split.

Both moved here from their prior donor modules (``inference_tools.py``,
``phenology_tools.py``): neither has anything specific to detection inference or phenology left
in its body, and grouping them here keeps the calibration-administration surface discoverable in
one place.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from tcip_mcp.audit import audited
from tcip_mcp.server import mcp

logger = logging.getLogger(__name__)


@mcp.tool()
@audited(scope_arg="dataset_root")
def force_redraw_cal_holdout_split(
    dataset_root: str,
    labels_dir: str | None = None,
    images_dir: str | None = None,
    identity_hash: str | None = None,
    group_by: str | None = None,
    group_key_map: dict[str, str] | None = None,
    seed: int = 0,
    holdout_ratio: float = 0.5,
    reason: str = "",
    split_manifest_dir: str | None = None,
    subject: str | None = None,
    attribute: str | None = None,
) -> dict:
    """Deliberately redraw a locked calibration/holdout split.

    A cal/holdout split locks on its first draw (``resolve_locked_cal_holdout_split``) so the
    "held-out validation" gate can never silently pass on a different, weaker holdout drawn
    after the fact. Redrawing one is a real, audited decision, never automatic, never a hidden
    kwarg on a high-traffic tool like ``run_inference``, so it is its own small tool. ``reason``
    is required and non-empty, and every redraw (this one included) is appended to the lock's
    ``redraw_history`` with its policy, seed, and the old and new split's content hashes, so a
    redraw-until-it-passes pattern is visible on review even though nothing here enforces that a
    reason differ from a prior one; the old and new split membership itself is recorded in the
    audit log alongside the reason (and, when given, ``split_manifest_dir``), not in
    ``redraw_history``; the defense is a reviewable audit trail, not an automatic block.

    Provide either ``labels_dir`` (the identity is derived as ``dataset_hash(labels_dir)``, and
    its stems are re-scanned) or ``identity_hash`` directly (e.g. a review-reference hash, in
    that case the existing lock's own calibration+holdout stems are reused as the redraw's stem
    universe, since a review reference has no labels directory to re-scan).

    Args:
        dataset_root: The root the lock is stored under, required, no default: a locked split
            travels with the data it was drawn over, and this tool holds an identity hash rather
            than anything the root can be read off. With ``labels_dir`` given, it is the root that
            dir's own lock lives under (its dataset root, or the dir itself when the layout places
            it under none), and a root disagreeing with it refuses rather than redrawing a lock
            nothing reads.
        labels_dir: Labeled dir whose GT identity locked the split (mutually exclusive with
            ``identity_hash``, if both are omitted, or ``identity_hash`` is given with no
            existing lock and no ``labels_dir``, this refuses).
        images_dir: Images for ``labels_dir``. When given, stems are the same labels-intersect-
            images-on-disk universe ``run_inference``'s calibration uses, a stem
            whose image was deleted/renamed never enters the redraw's stem universe. Omitted ->
            every labeled stem is used regardless of whether an image still exists for it, for a
            caller that has no images directory to check against; required alongside
            ``split_manifest_dir``, whose universe must be the same one a manifest-restricted
            calibration draws.
        identity_hash: The locked split's identity hash directly.
        group_by: New grouping policy, ``"tile_prefix"`` / ``"stem"`` (ignored if
            ``group_key_map`` is given). ``None`` (default) resolves to ``"tile_prefix"`` when
            neither this nor a manifest was given; a value beside ``split_manifest_dir`` conflicts
            with the manifest's own grouping policy and refuses, naming both.
        group_key_map: Explicit ``{stem: group_key}`` map covering every stem, overriding
            ``group_by``. Conflicts with ``split_manifest_dir`` the same way ``group_by`` does.
        seed: New split seed.
        holdout_ratio: New calibration/holdout fraction.
        reason: Required, non-empty justification for this redraw, recorded in the audit log
            alongside the old and new split membership.
        split_manifest_dir: Restrict the redraw's universe to one capture date's ``calibration``
            side of a split manifest (``data_tools.read_split_manifest_dir``), the same
            restriction ``run_inference`` applies, instead of every labelled stem with an image.
            Requires ``labels_dir`` and ``subject``: the manifest's own subject/attribute must equal
            ``subject``/``attribute``, the date ``labels_dir`` is under must be one the manifest
            holds members under, and the manifest's ``images_root`` for that date must be
            ``images_dir``, each refusing by name. The identity is
            ``dataset_hash(labels_dir, stems=universe)`` rather than the whole directory's hash,
            so the redraw addresses the same lock a manifest-restricted calibration locked.
        subject: The object class ``split_manifest_dir``'s admission was drawn for; required
            alongside it.
        attribute: The attribute ``split_manifest_dir``'s admission was scoped to, when it was.
    """
    if not reason or not reason.strip():
        return {"error": "reason is required (a non-empty justification) for a force_redraw"}
    if not labels_dir and not identity_hash:
        return {"error": "provide either labels_dir or identity_hash"}
    if split_manifest_dir is not None:
        if not labels_dir:
            return {"error": "split_manifest_dir requires labels_dir: the universe is drawn "
                             "from the manifest's held-out members under the labels' own date."}
        if not subject:
            return {"error": "split_manifest_dir requires subject: the manifest's own subject "
                             "must be checked against the door's."}
        if not images_dir:
            return {"error": "split_manifest_dir requires images_dir: a labels-only universe "
                             "can include a stem whose image is gone, a lock the redraw would "
                             "address that no manifest-restricted calibration ever draws."}
        if group_by is not None or group_key_map is not None:
            return {"error": f"split_manifest_dir={split_manifest_dir!r} conflicts with "
                             "group_by/group_key_map: the manifest's own grouping policy governs "
                             "the redraw; pass neither beside it."}

    from datetime import datetime, timezone

    from tcip_annotation.json_io import UnreadableLabelDocument
    from tcip_store import DecodeError, store

    from tcip_mcp.audit import dataset_scope_of, record_event
    from tcip_mcp.pipelines.data.splits import (
        cal_holdout_lock_key, cal_holdout_scope_root, count_label_lines, label_image_stems,
        resolve_locked_cal_holdout_split,
    )
    from tcip_mcp.pipelines.resolution import dataset_hash

    manifest_stems: list[str] | None = None
    if split_manifest_dir is not None:
        from tcip_mcp.pipelines.data.splits import (
            label_image_stems, resolve_manifest_calibration_universe,
        )
        from tcip_mcp.tools.data_tools import read_split_manifest_dir

        assert labels_dir is not None, "the split_manifest_dir refusal above requires it"
        manifest = read_split_manifest_dir(split_manifest_dir)
        present, _ = label_image_stems(labels_dir, images_dir)
        try:
            manifest_stems, group_by, group_key_map, _excluded, cal_date = \
                resolve_manifest_calibration_universe(
                    manifest, split_manifest_dir, labels_dir, images_dir, subject, attribute,
                    present)
        except ValueError as exc:
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
        identity_hash = dataset_hash(labels_dir, stems=manifest_stems)

    try:
        old_lock = store.read(cal_holdout_lock_key(identity_hash, scope_root=scope_root),
                              default=None)
    except DecodeError:
        # A redraw is the recovery for a lock whose bytes do not decode, so an unreadable one
        # is redrawn over rather than blocking the call; the entry it replaces is unknowable.
        logger.warning("the locked split for %s does not decode; redrawing over it",
                       identity_hash, exc_info=True)
        old_lock = None
    old_membership = ({"calibration": old_lock.get("calibration", []),
                       "holdout": old_lock.get("holdout", [])} if old_lock else None)

    if manifest_stems is not None:
        # Set only inside the split_manifest_dir branch above, which already required labels_dir.
        assert labels_dir is not None, "manifest_stems is only set where labels_dir was required"
        stems = manifest_stems
        try:
            annotation_counts = {
                s: count_label_lines(labels_dir, s, subject=subject, attribute=attribute)
                for s in stems
            }
        except UnreadableLabelDocument as exc:
            return {"error": str(exc)}
    elif labels_dir:
        # The same labels-intersect-images scan calibrate_operating_point uses, not a second
        # independent glob (images_dir omitted degrades to the prior labels-only scan).
        stems, _ = label_image_stems(labels_dir, images_dir)
        try:
            annotation_counts = {
                s: count_label_lines(labels_dir, s, subject=subject, attribute=attribute)
                for s in stems
            }
        except UnreadableLabelDocument as exc:
            return {"error": str(exc)}
    elif old_lock:
        stems = sorted(set(old_lock.get("calibration", [])) | set(old_lock.get("holdout", [])))
        annotation_counts = None
    else:
        return {"error": f"no existing lock for identity_hash={identity_hash!r}, and no "
                          "labels_dir to derive stems from"}

    new_lock = resolve_locked_cal_holdout_split(
        stems, identity_hash=identity_hash, scope_root=scope_root,
        annotation_counts=annotation_counts,
        group_by=(group_by or "tile_prefix"), group_key_map=group_key_map,
        holdout_ratio=holdout_ratio, seed=seed,
        force_redraw=True, timestamp=datetime.now(timezone.utc).isoformat(),
        split_manifest_dir=split_manifest_dir,
    )
    new_membership = {"calibration": new_lock["calibration"], "holdout": new_lock["holdout"]}

    # A distinct tool name under the same scope: @audited logs the call, this logs what it made.
    record_event(
        "force_redraw_cal_holdout_split_result",
        {"identity_hash": identity_hash, "group_by": group_by, "group_key_map": group_key_map,
         "seed": seed, "holdout_ratio": holdout_ratio, "reason": reason,
         "split_manifest_dir": split_manifest_dir},
        scope=dataset_scope_of(str(scope_root)),
        old_membership=old_membership, new_membership=new_membership,
    )

    return {"identity_hash": identity_hash, "reason": reason,
            "old_membership": old_membership, "new_membership": new_membership}


def _scalar_predictions(predictor, image_source, stems: list[str], suffix: str) -> dict[str, float]:
    """Run ``predictor`` over ``stems``' images and pull each image's single scalar prediction.

    ``suffix`` is the agent-authored bespoke model's own output-key convention for this task
    (``"_ranks"`` for ordinal, ``"_values"`` for regression, an ``OrdinalHead``/``RegressionHead``
    decode output prefixed ``head{i}_`` by the model's own ``forward()``), scanned the same
    key-suffix way ``active_learning.selector._confidence_values`` scans ``*_confidences``, never a
    hardcoded ``head0_`` name: the platform does not fix how many heads a bespoke model carries.
    Only the first matching key per prediction is used; a model with more than one head emitting the
    same suffix has no single scalar this function can disambiguate, that is a bespoke-model design
    question outside this calibration path's scope.
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
@audited
def calibrate_ordinal_regression_operating_point(
    trait_name: str,
    task: str,
    checkpoint_path: str,
    images_dir: str,
    csv_path: str,
    criterion: str,
    output_dir: str,
    dataset_root: str,
    experiment_id: str | None = None,
    group_by: str = "tile_prefix",
    group_key_map: dict[str, str] | None = None,
    seed: int = 0,
    holdout_ratio: float = 0.5,
) -> dict:
    """Calibrate and validate a trait's ordinal-rank or continuous-value prediction against a
    disjoint held-out split.

    Unlike :func:`calibrate_classifier_operating_point` (which reads pre-staged per-image
    prediction JSON via ``_classification_items``), there is no such staging mechanism for a
    CSV-sourced scalar trait (``OrdinalDataset``/``RegressionDataset`` are one CSV row per image
    stem, no bbox/geometry concept applies), so this runs live inference directly, mirroring
    ``pipelines.calibration.calibrate_operating_point``'s pattern instead: a locked cal/holdout split
    (``resolve_locked_cal_holdout_split``, dataset-shape-agnostic, a plain stems list + identity
    hash) of the CSV's own stems, the predictor run live over each side, then the same-rigor
    calibration gate (``operating_point.resolve_ordinal_operating_point``/
    ``resolve_regression_operating_point``: disjointness, train-disjointness, a derived
    compensating-error floor on a holdout-only criterion score), stamped into
    ``<output_dir>/ordinal_operating_point.json`` or ``regression_operating_point.json``, a file
    distinct from every other operating-point sidecar (see
    ``resolution.read_ordinal_operating_point_sidecar``/``read_regression_operating_point_sidecar``).

    This door takes no split manifest: its universe is the CSV's own stems, and no split manifest
    is drawn over a CSV-sourced scalar trait.

    A stamp that claims validation names the record it was earned from, the same two phases the
    classifier door goes through: ``resolution.open_validation`` runs the gate over the evidence,
    ``seal_validation`` files the row and returns the stamp with its pointer merged in, and the
    stamp is written last. A calibration that does not clear its gate stamps unvalidated, with its
    failures, and earns nothing.

    Record and stamp carry ``checkpoint_sha256`` from ``resolve_model_identity`` over the checkpoint
    this door itself ran, rather than the classifier door's copied evidence, which has nothing to copy
    here: running the checkpoint is what makes the hash the identity behind the scored predictions.

    Args:
        trait_name: The registered trait whose rank/value prediction is being calibrated.
        task: ``"ordinal"`` or ``"regression"``, dispatches which criterion toolkit, item shape and
            sidecar file apply.
        checkpoint_path: The trained checkpoint to calibrate. Must be registered under this
            process's platform state root (``register_model``, explicit mode for a foreign or
            bespoke checkpoint) or this door refuses before loading it.
        images_dir: Directory holding the CSV's images.
        csv_path: The ``(stem, value)`` CSV ``OrdinalDataset``/``RegressionDataset`` reads.
        criterion: Which registered criterion to calibrate against (``operating_point.
            ORDINAL_CRITERIA``/``REGRESSION_CRITERIA``), required, no default: which statistic is
            scientifically appropriate for this trait's calibration is a CV-scientist judgment call
            the caller makes explicitly, never a platform-prescribed default.
        output_dir: Where to write the sidecar.
        dataset_root: The dataset this calibration's claim hangs off, stated by the caller: the
            record's reference locations (the CSV, the images directory, the locked split) are
            written against it, it is the root the cal/holdout lock itself is stored under, and it
            is the root a reader resolves them from. Refuses when the
            images directory's own layout places it under a different root; a loose directory the
            layout cannot place refuses nothing, since a CSV over a bespoke image set with a stated
            root is legitimate.
        experiment_id: The checkpoint's own training-run id, if known, gates train-disjointness the
            same way the detector/classifier calibration paths do. ``None`` (a foreign/unregistered
            checkpoint) skips that check rather than failing closed.
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
    from tcip_mcp.pipelines.image_utils import list_logical_images
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

    true_by_stem: dict[str, float] = {}
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                value = int(row[1].strip()) if is_ordinal else float(row[1].strip())
                true_by_stem[row[0].strip()] = value

    logical = list_logical_images(images_dir)
    stems = sorted(s for s in true_by_stem if s in logical)
    if not stems:
        return {"error": f"no stem in {csv_path!r} has a matching image under {images_dir!r}"}

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
    cal_pred = _scalar_predictions(predictor, logical, cal_stems, shape["suffix"])
    hold_pred = _scalar_predictions(predictor, logical, hold_stems, shape["suffix"])

    def _items(sub_stems: list[str], preds: dict[str, float]) -> list[dict]:
        cast = int if is_ordinal else float
        return [{"image_id": s, shape["true_key"]: true_by_stem[s], shape["pred_key"]: cast(preds[s])}
                for s in sub_stems if s in preds]

    cal_items = _items(cal_stems, cal_pred)
    hold_items = _items(hold_stems, hold_pred)

    resolver = resolve_ordinal_operating_point if is_ordinal else resolve_regression_operating_point
    result = resolver(
        trait_name, criterion=criterion, calibration_items=cal_items, holdout_items=hold_items,
        experiment_id=experiment_id,
    )

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
        "sweep_data": result["sweep_data"],
        "checkpoint_sha256": checkpoint_sha256,
        "experiment_id": experiment_id,
        "trait": trait_name,
    }
    if result["passed"]:
        draft = open_validation(
            document=document,
            # Named off the function this door reported from, so record and report share one gate.
            evidence={"resolver": resolver.__name__,
                      "inputs": {"criterion": criterion, "calibration_items": cal_items,
                                 "holdout_items": hold_items}},
            trait=trait_name, checkpoint_sha256=checkpoint_sha256,
            producing_experiment_id=experiment_id,
            reference_inputs={
                "dataset_root": dataset_root,
                "label_csvs": {"reference": csv_path},
                "scope_roots": {"images": images_dir},
                "stated_values": {"split_identity": identity_hash},
            },
        )
        _, stamp = seal_validation(draft, dataset_root=dataset_root, bucket_dirs=[],
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

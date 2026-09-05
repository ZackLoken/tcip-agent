"""Calibration-administration tools: redrawing a locked cal/holdout split, calibrating a scalar
(ordinal-rank or continuous-value) trait against a disjoint held-out split, and earning a
validated count operating point over an already-published prediction bucket.

The first two moved here from their prior donor modules (``inference_tools.py``,
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
def redraw_calibration_holdout(
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
    dataset's own audit log alongside the reason (and, when given, ``split_manifest_dir``), not
    in ``redraw_history``; the defense is a reviewable audit trail, not an automatic block.

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
        reason: Required, non-empty justification for this redraw, recorded in the dataset's own
            audit log alongside the old and new split membership.
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
            manifest_stems, group_by, group_key_map, _excluded, cal_date, subject, attribute = \
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
        "redraw_calibration_holdout_result",
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
        "schema_version": 2,
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


@mcp.tool()
@audited(scope_arg="dataset_root")
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
    split_manifest_dir: str | None = None,
    val_ratio: float = 0.5,
    seed: int = 0,
    device: str | None = None,
) -> dict:
    """Calibrate the count operating point against held-out GT, and earn a validated claim over
    ``pred_dir`` only when the earned conf is the conf its stored detections were produced at.

    Runs :func:`tcip_mcp.pipelines.count_calibration.resolve_count_operating_point` unchanged
    (the same resolution ``scripts/calibrate_operating_point.py`` prints and writes nothing for):
    one low-threshold model pass over a disjoint, locked calibration/holdout split of
    ``labels_dir``, resolved into the count-unbiased conf and its held-out count-bias gate. That
    pass runs at a floor far below any production conf and proves the resolved conf against
    held-out GT; it says nothing about the detections already sitting in ``pred_dir``, which were
    filtered to whatever conf actually produced them. So a validated stamp is written only when
    the earned conf equals the conf ``pred_dir``'s own stamp already records as its production
    conf (``operating_point.conf.value``, read before the pass runs): only then are the
    detections sitting in the bucket the detections the validated conf describes. Any other
    earned conf refuses by name, stating both values, and points at ``run_inference``, whose
    calibrated path re-predicts a bucket at the earned conf; this is decided before
    ``open_validation``/``seal_validation`` ever run, so the ordinary mismatch mints no
    calibration experiment and no validation row. The same equality is re-decided under the
    stamp's own lock (``resolution.update_sidecar``) against the stamp as it is actually stored,
    not the copy read before the pass, for the one case the pre-pass read cannot see: a stamp
    another process overwrote with a different production conf while this (potentially long)
    pass was running.

    Every other decision this merge makes follows the same discipline the review-promotion route
    (``routes/validation.py``'s ``_promotion_of``) holds a stamp to, decided against the stamp as
    it is now stored: whether ``pred_dir`` already carries an earned claim is decided through
    ``resolution.verify_stamp_binding``, never a bare ``validated`` boolean, so a stamp asserting
    validated with no record answering for it is promotable over exactly as the route treats it;
    the tile-geometry floor (``resolution.fold_tile_validation``) is re-applied against the
    stamp's own tile field; ``trait`` is written only when this calibration actually earns the
    claim, so an unvalidated merge never relabels a bucket produced for a different trait; and
    ``shippable_issues`` is refreshed to this run's own list. A calibration whose count-bias gate
    does not clear, or whose tile geometry never validated, merges an honest ``conf`` with
    ``validated=false`` and earns nothing, since it claims nothing about the stored detections.

    ``pred_dir`` must already carry an ``operating_point.json`` stamp naming a checkpoint
    identity, hold at least one prediction document (the bucket is this claim's subject; an
    empty one has nothing to validate), and not name a whole-raster bucket
    (``resolution.stamp_names_raster``, the predicate the per-image delivery door refuses a
    mosaic bucket on too: one mosaic total is not a per-image count this calibration's records
    reason over), and sit under ``dataset_root``, the same requirement ``seal_validation`` holds
    every claimed bucket to (``resolution.bucket_relative_key``). A bucket outside
    ``dataset_root``, with no stamp at all, empty, a whole-raster bucket, whose stamp carries no
    ``checkpoint_sha256`` at all (a claim sealed under a digest the stamp does not carry could
    never bind at delivery), whose stamped checkpoint disagrees with ``checkpoint_path``, or that
    already carries a claim ``verify_stamp_binding`` answers for, all refuse by name before the
    calibration pass ever draws its cal/holdout lock. Only the race (the stamp changing to a
    different production conf while the pass is running) is discovered after a record has
    already been sealed against the pre-pass reading: that refusal can leave a minted calibration
    experiment and an appended, inert validation row behind, since no bucket ever comes to name
    it, but the record exists, and the response names it when this happens.

    Args:
        checkpoint_path: The trained checkpoint to calibrate; must be registered under the
            platform state root (``register_model``) or this door refuses before loading it.
        trait: The registered trait whose count is being calibrated.
        labels_dir: Labeled dir (per-image JSON), this calibration's measurement reference.
        images_dir: Images for ``labels_dir``.
        dataset_root: The root the cal/holdout split lock is stored under and ``pred_dir`` must
            sit beneath; the labels' dataset root, or ``labels_dir`` itself when the layout
            places it under none.
        pred_dir: The already-published prediction bucket this claim covers.
        subject / attribute: The object class / assessed attribute the labeled reference is
            scoped to; when ``pred_dir``'s stamp already records a scope, an omitted pair takes
            the bucket's own recorded scope and a stated pair must equal it, refusing by name
            otherwise, since evidence earned under one scope is never merged into a bucket
            stamped for another.
        experiment_id: The checkpoint's own training-run id, if known, gates train-disjointness;
            ``None`` (a foreign/unregistered checkpoint) skips that check.
        group_by / group_key_map: The locked cal/holdout split's grouping policy; only the first
            call for this labels_dir's identity draws the split.
        split_manifest_dir: Restrict the calibration universe to one capture date's calibration
            side of a split manifest instead of every labeled stem; requires ``subject``, and
            conflicts with ``group_by``/``group_key_map``.
        val_ratio / seed: The locked split's holdout fraction and seed; take effect only on the
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
        StampScopeUnstated, bucket_relative_key, bucket_scope, claim_payload,
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
    except (StampScopeUnstated, StoreError) as exc:
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
            split_manifest_dir=split_manifest_dir, val_ratio=val_ratio, seed=seed, device=device,
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
        _digest, earned = seal_validation(
            draft, dataset_root=str(root), bucket_dirs=[bucket], stamp_body=earned)
    else:
        earned["validated_by"] = None

    refusal: dict[str, object] = {}

    def _merge(stored: dict) -> dict | None:
        """Merge this calibration's earned conf into whatever the producing run left, inside the
        stamp's own lock.

        Distinct from the review-promotion route's own updater (``routes/validation.py``'s
        ``_promotion_of``): that one folds one review verdict across every bucket a review pass
        covered, while this one earns a single bucket's own count-calibration claim from a
        resolved conf. Both hold to deciding whether the bucket already carries an answered
        claim through ``verify_stamp_binding``, and deciding ``validated``, the tile floor and
        the conf-equality rule against the stamp as stored, not the copy read before the lock.
        """
        binding = verify_stamp_binding(stored, bucket, document="operating_point")
        if binding.claimed and binding.ok:
            return None
        validated = fold_tile_validation(conf.is_shippable, stored.get("tile_size_validated"))
        stored_conf_value = ((stored.get("operating_point") or {}).get("conf") or {}).get("value")
        if validated and stored_conf_value != earned_conf_value:
            refusal["stored_conf_value"] = stored_conf_value
            return None
        merged = dict(stored)
        merged["operating_point"] = {**(stored.get("operating_point") or {}),
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
            ((new_stamp.get("operating_point") or {}).get("conf") or {}).get("validated_against")
            if validated_now else None
        ),
        "validated_by": new_stamp.get("validated_by") if validated_now else None,
        "n_calibration_images": len(resolved.resolver_inputs["calibration_records"]),
        "n_holdout_images": len(resolved.resolver_inputs["holdout_records"]),
        "gate_evidence": new_stamp.get("gate_evidence_summary"),
    }

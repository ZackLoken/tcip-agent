"""A prediction bucket cannot vouch for itself: what a validated stamp has to be answered for by.

A stamp claiming validation names a row in an experiment log outside the bucket, and a delivery
recomputes that row's identity, compares the claim the stamp asserts against the claim the row was
earned for, and re-hashes the prediction files a count claim covers. Everything here goes through
the reconcilers, which is how the delivery doors reach this module, rather than through the verifier
alone.

The residual is recorded here too rather than implied away: the storage seam's own append stays
reachable in process, and a row written through it delivers.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from tests._binding_fixtures import write_bound_sidecar, write_prediction

pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")

TRAIT = "bud_opening"
CHECKPOINT_SHA = "a1" * 32
PRODUCING_RUN = "exp-021-currant-bud-det"


def _op(*, conf: float = 0.42, reference: str = "held_out_annotations",
        tile_size: int | None = None) -> dict:
    """The resolved-parameter mapping a stamp carries, in the shape ``to_provenance`` produces."""
    op = {"conf": {"name": "conf", "value": conf, "validated_against": reference,
                   "requires_validation": True, "validation_kind": "annotations"}}
    if tile_size is not None:
        op["tile_size"] = {"name": "tile_size", "value": tile_size,
                           "validated_against": "persisted_training_geometry",
                           "requires_validation": True, "validation_kind": "geometry"}
    return op


def _count_stamp(**overrides) -> dict:
    from tcip_mcp.pipelines.resolution import operating_point_stamp

    fields = dict(
        validated=True, validated_by=None, tile_size_validated=None, shippable_issues=[],
        id_map=None, trait=TRAIT, dataset_hash="h1", checkpoint="best",
        checkpoint_sha256=CHECKPOINT_SHA, experiment_id=PRODUCING_RUN, images_dir=None,
        raster_path=None, produced_at="2026-03-04T12:00:00+00:00",
        subject=TRAIT, attribute=None,
    )
    op = overrides.pop("operating_point", _op())
    fields.update(overrides)
    return operating_point_stamp(op, **fields)


def _bucket(root: Path, *, name: str = "live", date: str = "2026-03-04", stems=("img_a",)) -> Path:
    """A prediction bucket in the canonical layout, holding per-image predictions."""
    d = root / "predictions" / name / date
    for stem in stems:
        write_prediction(d, stem)
    return d


def _write_raw(pred_dir: Path, stamp: dict, document: str = "operating_point") -> Path:
    """A stamp written straight through the seam, bypassing write_sidecar's own claim check: the
    way a forged, hand-authored or edited-after-the-fact stamp reaches the store."""
    import tcip_store

    from tcip_mcp.pipelines.resolution import sidecar_key

    pred_dir.mkdir(parents=True, exist_ok=True)
    key = sidecar_key(pred_dir, document)
    with tcip_store.transaction(key) as txn:
        txn.write(key, stamp)
    return pred_dir / f"{document}.json"


def _count_validity(pred_dir: Path) -> dict:
    from tcip_mcp.pipelines.resolution import reconcile_operating_point_validity

    return reconcile_operating_point_validity([str(pred_dir)], trait=TRAIT)


def _delivers(validity: dict) -> bool:
    from tcip_mcp.pipelines.resolution import check_delivery_gate

    return check_delivery_gate({"operating_point": validity["validated"]}).ok


# --- one test per check the reader performs ------------------------------------------------

def test_unbacked_stamp_does_not_deliver(tmp_path):
    """A stamp claiming validation with no pointer at a record floors, and the gate refuses."""
    root = tmp_path / "ds"
    pred_dir = _bucket(root)
    _write_raw(pred_dir, _count_stamp())

    validity = _count_validity(pred_dir)

    assert validity["validated"] == "false"
    assert validity["unvalidated_buckets"] == [str(pred_dir)]
    note = validity["binding_notes"][str(pred_dir)]
    assert "validated_by" in note and "operating_point.json" in note
    assert not _delivers(validity)


def test_record_absent_from_the_experiment_store_floors(tmp_path):
    """A pointer at an experiment that never ran, and at a row no experiment holds, both floor."""
    from tcip_mcp.experiments import create_experiment, experiments_scope

    root = tmp_path / "ds"
    pred_dir = _bucket(root)
    _write_raw(pred_dir, {**_count_stamp(),
                          "validated_by": {"experiment_id": "exp_that_never_ran",
                                           "record_digest": "0123456789abcdef"}})

    no_experiment = _count_validity(pred_dir)
    assert no_experiment["validated"] == "false"
    note = no_experiment["binding_notes"][str(pred_dir)]
    assert "exp_that_never_ran" in note and experiments_scope() in note

    create_experiment("exp_that_never_ran", {"derived_from": "an experiment with no such row"})
    no_row = _count_validity(pred_dir)
    assert no_row["validated"] == "false"
    assert "0123456789abcdef" in no_row["binding_notes"][str(pred_dir)]
    assert not _delivers(no_row)


def test_edited_threshold_under_a_genuine_pointer_floors(tmp_path):
    """A real record's pointer copied onto an edited threshold answers for a value it never saw."""
    root = tmp_path / "ds"
    pred_dir = _bucket(root)
    bound = write_bound_sidecar(pred_dir, _count_stamp(), dataset_root=root)
    assert _count_validity(pred_dir)["validated"] == "held_out_annotations"

    edited = {**bound, "operating_point": _op(conf=0.05)}
    _write_raw(pred_dir, edited)

    validity = _count_validity(pred_dir)
    assert validity["validated"] == "false"
    assert "disagree" in validity["binding_notes"][str(pred_dir)]
    assert not _delivers(validity)


def test_edited_classifier_value_under_a_genuine_pointer_floors(tmp_path):
    """The classifier document covers no bucket content, so its claim is what binds it."""
    from tcip_mcp.pipelines.resolution import reconcile_classifier_validity

    root = tmp_path / "ds"
    pred_dir = _bucket(root)
    stamp = {"operating_point": {"classifier": {"validated_against": "held_out_annotations",
                                                "value": "shedding"}},
             "validated": True, "trait": TRAIT, "checkpoint_sha256": CHECKPOINT_SHA}
    bound = write_bound_sidecar(pred_dir, stamp, document="classifier_operating_point",
                                dataset_root=root)
    assert reconcile_classifier_validity([str(pred_dir)])["validated"] == "held_out_annotations"

    edited = {**bound, "operating_point": {"classifier": {
        "validated_against": "held_out_annotations", "value": "closed"}}}
    _write_raw(pred_dir, edited, "classifier_operating_point")

    validity = reconcile_classifier_validity([str(pred_dir)])
    assert validity["validated"] == "false"
    assert "disagree" in validity["binding_notes"][str(pred_dir)]


def test_edited_criterion_under_a_genuine_pointer_floors(tmp_path):
    """Which statistic an ordinal claim was judged against is part of the claim, not its provenance."""
    from tcip_mcp.pipelines.resolution import reconcile_ordinal_validity

    root = tmp_path / "ds"
    pred_dir = _bucket(root)
    stamp = {"operating_point": {"ordinal": {"validated_against": "held_out_annotations",
                                             "criterion": "quadratic_weighted_kappa"}},
             "validated": True, "trait": TRAIT, "checkpoint_sha256": None}
    bound = write_bound_sidecar(pred_dir, stamp, document="ordinal_operating_point",
                                dataset_root=root)
    assert reconcile_ordinal_validity(
        [str(pred_dir)], trait=TRAIT)["validated"] == "held_out_annotations"

    edited = {**bound, "operating_point": {"ordinal": {
        "validated_against": "held_out_annotations", "criterion": "spearman"}}}
    _write_raw(pred_dir, edited, "ordinal_operating_point")

    assert reconcile_ordinal_validity([str(pred_dir)], trait=TRAIT)["validated"] == "false"


def test_edited_claim_scope_flag_floors_every_dimension(tmp_path):
    """One stamp file, one verdict: conf, tile geometry and claim scope stand or fall together."""
    from tcip_mcp.pipelines.resolution import (
        reconcile_claim_scope_validity, reconcile_tile_size_validity,
    )

    root = tmp_path / "ds"
    pred_dir = _bucket(root)
    stamp = _count_stamp(operating_point=_op(tile_size=640),
                         tile_size_validated="persisted_training_geometry")
    stamp["claim_scope_validated"] = "same_mosaic_georeferenced_identity"
    bound = write_bound_sidecar(pred_dir, stamp, dataset_root=root)

    dirs = [str(pred_dir)]
    assert _count_validity(pred_dir)["validated"] == "held_out_annotations"
    assert reconcile_tile_size_validity(dirs)["validated"] == "persisted_training_geometry"
    assert reconcile_claim_scope_validity(dirs)["validated"] == "same_mosaic_georeferenced_identity"

    _write_raw(pred_dir, {**bound, "claim_scope_validated": "false"})

    assert _count_validity(pred_dir)["validated"] == "false"
    assert reconcile_tile_size_validity(dirs)["validated"] == "false"
    scope = reconcile_claim_scope_validity(dirs)
    assert scope["validated"] == "false"
    assert "disagree" in scope["binding_notes"][str(pred_dir)]


def test_changed_bucket_content_floors_the_stamp(tmp_path):
    """A count claim cannot outlive the prediction files it was earned over."""
    root = tmp_path / "ds"
    pred_dir = _bucket(root)
    write_bound_sidecar(pred_dir, _count_stamp(), dataset_root=root)
    assert _count_validity(pred_dir)["validated"] == "held_out_annotations"

    write_prediction(pred_dir, "img_a", count=9)

    validity = _count_validity(pred_dir)
    assert validity["validated"] == "false"
    assert "replaced" in validity["binding_notes"][str(pred_dir)]
    assert not _delivers(validity)


def test_added_prediction_file_floors_the_stamp(tmp_path):
    """An added prediction changes the covered content just as a replaced one does."""
    root = tmp_path / "ds"
    pred_dir = _bucket(root)
    write_bound_sidecar(pred_dir, _count_stamp(), dataset_root=root)
    assert _count_validity(pred_dir)["validated"] == "held_out_annotations"

    write_prediction(pred_dir, "img_b")

    assert _count_validity(pred_dir)["validated"] == "false"


def test_same_length_replacement_with_restored_timestamp_floors(tmp_path):
    """The content is re-read every delivery, so a restored size and timestamp hide nothing."""
    root = tmp_path / "ds"
    pred_dir = _bucket(root)
    write_bound_sidecar(pred_dir, _count_stamp(), dataset_root=root)
    assert _count_validity(pred_dir)["validated"] == "held_out_annotations"

    path = pred_dir / "img_a.json"
    before = path.stat()
    original = path.read_text(encoding="utf-8")
    replacement = original.replace('"count": 1', '"count": 7')
    assert len(replacement) == len(original)
    path.write_text(replacement, encoding="utf-8")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size

    assert _count_validity(pred_dir)["validated"] == "false"


def test_a_row_stating_split_manifest_dir_with_no_selection_disjointness_floors(tmp_path):
    """A row a writer other than this platform produced, or one sealed before the field existed,
    states a manifest but carries no selection_disjointness at all: the door floors it by name,
    never reads a missing check as a passing one."""
    root = tmp_path / "ds"
    pred_dir = _bucket(root)
    write_bound_sidecar(
        pred_dir, _count_stamp(), dataset_root=root,
        reference_identity={"stated_values": {"split_manifest_dir": "some/manifest"}},
        selection_disjointness=None,
    )

    validity = _count_validity(pred_dir)
    assert validity["validated"] == "false"
    note = validity["binding_notes"][str(pred_dir)]
    assert "selection_disjointness" in note and "some/manifest" in note
    assert not _delivers(validity)


def test_a_row_stating_split_manifest_dir_with_an_unchecked_selection_disjointness_floors(
    tmp_path,
):
    """A row that states a manifest with a selection_disjointness present but neither
    not-applicable-with-a-reason nor checked (the unresolvable shape) still floors: the field
    has to answer the question, not merely exist."""
    root = tmp_path / "ds"
    pred_dir = _bucket(root)
    write_bound_sidecar(
        pred_dir, _count_stamp(), dataset_root=root,
        reference_identity={"stated_values": {"split_manifest_dir": "some/manifest"}},
        selection_disjointness={"applicable": True, "reason": "no record to check against",
                                "checked": False, "group_check": None},
    )

    validity = _count_validity(pred_dir)
    assert validity["validated"] == "false"
    assert "selection_disjointness" in validity["binding_notes"][str(pred_dir)]


def test_a_row_stating_split_manifest_dir_with_a_checked_no_leak_selection_disjointness_delivers(
    tmp_path,
):
    """The admits-valid-work half: a row whose selection_disjointness is checked with no leak
    delivers normally, the floor above never reaches a legitimate manifest-bound claim."""
    root = tmp_path / "ds"
    pred_dir = _bucket(root)
    write_bound_sidecar(
        pred_dir, _count_stamp(), dataset_root=root,
        reference_identity={"stated_values": {"split_manifest_dir": "some/manifest"}},
        selection_disjointness={"applicable": True, "reason": None, "checked": True,
                                "unresolvable": False, "leaked_groups": [], "leaked_stems": [],
                                "group_check": "performed", "labels_moved_draw_to_run": None,
                                "labels_moved_run_to_now": None, "calibration_labels_moved": None,
                                "manifest_redrawn": None, "calibration_labels_dir": None},
    )

    assert _count_validity(pred_dir)["validated"] == "held_out_annotations"


def test_a_row_stating_split_manifest_dir_with_a_leaking_selection_disjointness_floors(tmp_path):
    """A row that reports checked=True but names a leaked group still floors: "checked with no
    leak" is enforced from the row's own leak fields, not read off the pass/fail booleans alone.
    The five label-movement keys are present (null), the shape a genuinely checked row carries,
    so the leak is the only clause that can floor this row; a missing-keys floor is proven
    separately, by a row with no leak and no movement keys."""
    root = tmp_path / "ds"
    pred_dir = _bucket(root)
    write_bound_sidecar(
        pred_dir, _count_stamp(), dataset_root=root,
        reference_identity={"stated_values": {"split_manifest_dir": "some/manifest"}},
        selection_disjointness={"applicable": True, "reason": None, "checked": True,
                                "unresolvable": False, "leaked_groups": ["g1"],
                                "leaked_stems": [], "group_check": "performed",
                                "labels_moved_draw_to_run": None, "labels_moved_run_to_now": None,
                                "calibration_labels_moved": None, "manifest_redrawn": None,
                                "calibration_labels_dir": None},
    )

    validity = _count_validity(pred_dir)
    assert validity["validated"] == "false"
    assert "selection_disjointness" in validity["binding_notes"][str(pred_dir)]


def test_a_row_stating_split_manifest_dir_missing_the_label_movement_keys_floors(tmp_path):
    """A row that is otherwise checked with no leak still floors when it carries none of the
    five label-movement keys: those keys have to answer the question (present, null admitted),
    not merely be absent, the same rule the unchecked-shape test above proves for checked."""
    root = tmp_path / "ds"
    pred_dir = _bucket(root)
    write_bound_sidecar(
        pred_dir, _count_stamp(), dataset_root=root,
        reference_identity={"stated_values": {"split_manifest_dir": "some/manifest"}},
        selection_disjointness={"applicable": True, "reason": None, "checked": True,
                                "unresolvable": False, "leaked_groups": [],
                                "leaked_stems": [], "group_check": "performed"},
    )

    validity = _count_validity(pred_dir)
    assert validity["validated"] == "false"
    assert "selection_disjointness" in validity["binding_notes"][str(pred_dir)]


def test_classifier_trust_set_ignores_unbacked_ids(tmp_path):
    """The runs a classifier stamp is checked against come from verified records, not from what a
    count bucket declares for itself beside its own predictions."""
    from tcip_mcp.pipelines.resolution import bind_classifier_validity

    root = tmp_path / "ds"
    count_dir = _bucket(root)
    classifier_dir = _bucket(root, name="classifier", stems=())

    # The stamp self-declares one run while its record names the real producer; the record wins.
    write_bound_sidecar(count_dir, _count_stamp(experiment_id="exp-forged"), dataset_root=root,
                        experiment_id="exp-count-record", producing_experiment_id=PRODUCING_RUN)
    classifier_stamp = {
        "operating_point": {"classifier": {"validated_against": "held_out_annotations",
                                           "value": "shedding"}},
        "validated": True, "trait": TRAIT, "checkpoint_sha256": CHECKPOINT_SHA,
    }
    write_bound_sidecar(classifier_dir, classifier_stamp, document="classifier_operating_point",
                        dataset_root=root, experiment_id="exp-classifier-record",
                        producing_experiment_id="exp-forged")

    state, note = bind_classifier_validity(
        "held_out_annotations", [str(classifier_dir)], [str(count_dir)], trait=TRAIT)

    assert state == "false"
    assert "exp-forged" in note and PRODUCING_RUN in note


def test_hand_written_scale_stamp_does_not_deliver(tmp_path):
    """A stamp claiming validated with no well-formed validated_by floors regardless of how it was
    written: calibrate_physical_scale always names a record, so a stamp with none was hand-authored
    or written by a caller that bypassed the tool."""
    from tcip_mcp.pipelines.resolution import reconcile_scale_validity

    root = tmp_path / "ds"
    pred_dir = _bucket(root)
    _write_raw(pred_dir, {"operating_point": {"scale": {
        "validated_against": "physical_measurement", "value": 0.31, "unit": "mm",
        "capture_id": None}},
        "validated": True, "trait": TRAIT}, "resolve_scale")

    validity = reconcile_scale_validity([str(pred_dir)], unit="mm", trait=TRAIT, images_dir="unused")

    assert validity["validated"] == "false"
    assert "validated_by" in validity["binding_notes"][str(pred_dir)]
    assert not _delivers({"validated": validity["validated"]})


def test_claim_scope_reader_floors_an_unbacked_stamp(tmp_path):
    """The claim-scope reader has its own loop, and it reaches the same stamp the others do."""
    from tcip_mcp.pipelines.resolution import reconcile_claim_scope_validity

    root = tmp_path / "ds"
    pred_dir = _bucket(root)
    stamp = _count_stamp()
    stamp["claim_scope_validated"] = "same_mosaic_georeferenced_identity"
    _write_raw(pred_dir, stamp)

    validity = reconcile_claim_scope_validity([str(pred_dir)])

    assert validity["validated"] == "false"
    assert validity["unvalidated_buckets"] == [str(pred_dir)]
    assert "validated_by" in validity["binding_notes"][str(pred_dir)]


def test_validated_stamp_without_a_trait_is_refused_at_write(tmp_path):
    """A claim with no trait names nothing a delivery can be checked against, on either writer."""
    from tcip_mcp.pipelines.resolution import update_sidecar, write_sidecar

    root = tmp_path / "ds"
    pred_dir = _bucket(root)
    # The writer checks the pointer's shape, not the record behind it, so a shaped one is enough here.
    pointer = {"experiment_id": "exp-binding-reference", "record_digest": "0123456789abcdef"}
    untraited = _count_stamp(trait=None, validated_by=pointer)

    with pytest.raises(ValueError, match="no trait"):
        write_sidecar(pred_dir, untraited)
    with pytest.raises(ValueError, match="no trait"):
        update_sidecar(pred_dir, lambda stored: untraited)
    with pytest.raises(ValueError, match="validated_by"):
        write_sidecar(pred_dir, _count_stamp())

    assert write_sidecar(pred_dir, {**untraited, "trait": TRAIT}) is None  # type: ignore[func-returns-value]  # the assert documents the None return this writer contracts to


# --- the residual, recorded rather than implied away ---------------------------------------

def test_a_row_appended_through_the_storage_seam_still_delivers(tmp_path):
    """The storage seam's own append is generic and public, and nothing here closes it.

    Making the validations appender module-private removes the supported way to hand over a verdict;
    it does not remove the way. An in-process caller that appends a row it authored, and stamps a
    bucket naming that row, delivers validated. This is the boundary as it stands, and it fails the
    day someone believes it moved.
    """
    import tcip_store

    from tcip_mcp.experiments import create_experiment, validation_digest, validations_key
    from tcip_mcp.pipelines.resolution import claim_payload, write_sidecar
    from tcip_mcp.prediction_buckets import bucket_content_digest

    root = tmp_path / "ds"
    pred_dir = _bucket(root)
    stamp = _count_stamp()
    create_experiment(PRODUCING_RUN, {"derived_from": "a run whose validations were never earned"})
    body = {
        "document": "operating_point", "trait": TRAIT,
        "claim": claim_payload(stamp, document="operating_point"),
        "validated_against": "held_out_annotations", "checkpoint_sha256": CHECKPOINT_SHA,
        "producing_experiment_id": PRODUCING_RUN,
        "reference_identity": {"stated_values": {"reference": "authored, never presented"}},
        "covered_buckets": {"predictions/live/2026-03-04": bucket_content_digest(pred_dir)},
        "dataset_root": str(root.resolve()), "recorded_at": "2026-03-04T12:00:00+00:00",
    }
    tcip_store.append(validations_key(PRODUCING_RUN), body)

    write_sidecar(pred_dir, {**stamp, "validated_by": {
        "experiment_id": PRODUCING_RUN, "record_digest": validation_digest(body)}})

    validity = _count_validity(pred_dir)
    assert validity["validated"] == "held_out_annotations"
    assert _delivers(validity)


# --- the legitimate calls the refusals have to keep admitting -------------------------------

def _passing_evidence():
    """Calibration and holdout records a real held-out gate passes over."""
    from tests._dense_op_fixtures import dense_records

    common = dict(n_images=20, objects_per_image=80, miss_pattern=[0] * 20,
                  fp_pattern=[1] * 20, score=0.9, fp_score=0.05)
    return (dense_records(id_prefix="c", **common),
            dense_records(id_prefix="h", shift=5.0, **common))


def _earned_bucket(root: Path, *, conf_records=None, stems=("img_a",)):
    """Earn a count claim the way a door does: gate, publish, seal, stamp last."""
    from tcip_mcp.pipelines.resolution import (
        open_validation, operating_point_stamp, seal_validation, write_sidecar,
    )

    cal, hold = conf_records or _passing_evidence()
    labels_dir = root / "annotations" / "2026-03-04"
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / "img_a.json").write_text(json.dumps({"annotations": []}), encoding="utf-8")

    draft = open_validation(
        document="operating_point",
        evidence={"resolver": "resolve_operating_point",
                  "inputs": {"dataset_hash": "h1", "calibration_records": cal,
                             "holdout_records": hold, "staged_conf_floor": 0.01,
                             "tiled": False}},
        trait=TRAIT, checkpoint_sha256=CHECKPOINT_SHA, producing_experiment_id=None,
        reference_inputs={"dataset_root": str(root), "label_dirs": {"calibration": labels_dir},
                          "stated_values": {"split_identity": "d41d8cd98f00b204"}},
    )
    pred_dir = _bucket(root, stems=stems)
    body = operating_point_stamp(
        draft.result.to_provenance()["operating_point"], validated=True, validated_by=None,
        tile_size_validated=None, shippable_issues=draft.result.shippable_issues(), id_map=None,
        trait=TRAIT, dataset_hash="h1", checkpoint="best", checkpoint_sha256=CHECKPOINT_SHA,
        experiment_id=None, images_dir=None, raster_path=None,
        produced_at="2026-03-04T12:00:00+00:00", subject=TRAIT, attribute=None,
    )
    digest, stamped = seal_validation(draft, dataset_root=root, bucket_dirs=[pred_dir],
                                      stamp_body=body)
    write_sidecar(pred_dir, stamped)
    return pred_dir, digest, stamped


def test_a_claim_earned_through_the_two_phases_delivers_validated(tmp_path):
    """The gate runs once over the evidence, the record covers the files that landed, and it
    verifies."""
    pytest.importorskip("torch")
    from tcip_mcp.experiments import read_validations
    from tcip_mcp.prediction_buckets import bucket_content_digest

    root = tmp_path / "ds"
    pred_dir, digest, stamped = _earned_bucket(root)

    experiment_id = stamped["validated_by"]["experiment_id"]
    assert experiment_id.startswith("calibration_")  # no producing run, so the claim hangs off one
    row = next(r for r in read_validations(experiment_id) if r["claim"])
    assert row["producing_experiment_id"] is None
    assert row["covered_buckets"] == {
        "predictions/live/2026-03-04": bucket_content_digest(pred_dir)}
    assert row["claim"]["operating_point"]["conf"]["validated_against"] == "held_out_annotations"

    validity = _count_validity(pred_dir)
    assert validity["validated"] == "held_out_annotations"
    assert validity["binding_notes"] == {}
    assert _delivers(validity)
    assert digest == stamped["validated_by"]["record_digest"]


def test_an_untouched_validated_bucket_reconciles_across_repeated_deliveries(tmp_path):
    """Recomputation is not a one-shot: an untouched bucket keeps reconciling to its record."""
    root = tmp_path / "ds"
    pred_dir = _bucket(root, stems=("img_a", "img_b"))
    write_bound_sidecar(pred_dir, _count_stamp(), dataset_root=root)

    for _ in range(3):
        assert _count_validity(pred_dir)["validated"] == "held_out_annotations"


def test_a_dataset_moved_to_a_new_absolute_path_still_verifies(tmp_path):
    """Covered buckets are keyed inside the dataset, so moving the dataset whole changes nothing.

    Bound to the file backend: the sqlite backend keeps a connection to the pre-move root open
    for the test process's life, and Windows refuses to rename a directory holding an open
    database handle, a mechanical property of that backend rather than of the claim under test,
    which every other case in this module already exercises against both backends.
    """
    import tcip_store
    from tcip_store.file_backend import FileBackend

    tcip_store.bind(FileBackend())
    original = tmp_path / "ds"
    pred_dir = _bucket(original)
    write_bound_sidecar(pred_dir, _count_stamp(), dataset_root=original)
    assert _count_validity(pred_dir)["validated"] == "held_out_annotations"

    moved = tmp_path / "ds_moved"
    shutil.move(str(original), str(moved))

    assert _count_validity(moved / "predictions" / "live" / "2026-03-04")["validated"] == (
        "held_out_annotations")


def test_a_recalibration_at_a_new_threshold_earns_a_new_record(tmp_path):
    """Binding the claim does not freeze it: a genuine re-calibration delivers at the new value."""
    root = tmp_path / "ds"
    pred_dir = _bucket(root)
    first = write_bound_sidecar(pred_dir, _count_stamp(operating_point=_op(conf=0.42)),
                                dataset_root=root, experiment_id="exp-first-calibration")
    assert _count_validity(pred_dir)["conf"] == 0.42

    second = write_bound_sidecar(pred_dir, _count_stamp(operating_point=_op(conf=0.61)),
                                 dataset_root=root, experiment_id="exp-second-calibration")

    assert second["validated_by"]["record_digest"] != first["validated_by"]["record_digest"]
    validity = _count_validity(pred_dir)
    assert validity["validated"] == "held_out_annotations"
    assert validity["conf"] == 0.61
    assert _delivers(validity)


def test_an_unvalidated_bucket_keeps_the_tile_geometry_it_really_persisted(tmp_path):
    """A stamp that claims nothing has nothing to bind, so the dimensions it does not claim are
    read exactly as they were before."""
    from tcip_mcp.pipelines.resolution import reconcile_tile_size_validity

    root = tmp_path / "ds"
    pred_dir = _bucket(root)
    _write_raw(pred_dir, _count_stamp(
        validated=False, operating_point=_op(reference="false", tile_size=640)))

    assert _count_validity(pred_dir)["validated"] == "false"
    tile = reconcile_tile_size_validity([str(pred_dir)])
    assert tile["validated"] == "persisted_training_geometry"
    assert tile["binding_notes"] == {}


def test_an_acknowledgement_still_writes_a_flagged_provisional_path(tmp_path):
    """An honestly-flagged provisional delivery is the escape hatch, and the gate keeps it open
    for a real acknowledgement naming who and why, never a bare boolean."""
    from tcip_mcp.pipelines.resolution import Acknowledgement, check_delivery_gate

    root = tmp_path / "ds"
    pred_dir = _bucket(root)
    _write_raw(pred_dir, _count_stamp())
    validity = _count_validity(pred_dir)

    gate = check_delivery_gate(
        {"operating_point": validity["validated"]},
        acknowledgement=Acknowledgement(acknowledged_by="user:tester", reason="known uncalibrated"))

    assert gate.ok
    assert gate.unvalidated == ("operating_point",)
    assert gate.stamp["operating_point"] == "false"
    assert gate.acknowledged_by == "user:tester"
    assert gate.acknowledgement_reason == "known uncalibrated"

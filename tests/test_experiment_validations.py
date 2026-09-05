"""The experiment record's validation member: what it takes, what it refuses, what reads back.

A validation is a claim earned against evidence and filed on the experiment it was earned on.
The member is append-only, so a re-validation is a second row rather than a rewrite; every
field of a row is required, since a defaulted provenance field is a claim nobody made; and it
is the one member a finished run still accepts, because a validation is a statement made about
a run after it ended.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import tcip_store as ts

REFERENCE_IDENTITY = {
    "calibration_dataset_hash": "9f2c1b0a4d6e8f31",
    "holdout_dataset_hash": "3ab9c7d15e0f2846",
    "split_identity": "d41d8cd98f00b204",
}

TRAINING_CONFIG = {"model_source": {"builder": "my_models:bud_det"}}


def _real_selection_disjointness() -> dict[str, Any]:
    """The full shape resolution.resolver_selection_disjointness returns for a foreign
    checkpoint with no split manifest named, read back through the same
    operating_point._selection_disjointness path a live calibration takes, rather than a
    hand-typed subset the resolver never actually produces."""
    from tcip_mcp.pipelines import resolution
    from tcip_mcp.pipelines.operating_point import _selection_disjointness

    raw = _selection_disjointness(None, set(), set())
    return resolution.resolver_selection_disjointness(
        {"gate_evidence": {"selection_disjointness": raw}}, "operating_point")


def _row(**overrides: Any) -> dict[str, Any]:
    """One complete validation row, with the fields a case varies replaced."""
    body: dict[str, Any] = {
        "schema_version": 2,
        "document": "operating_point",
        "trait": "bud_50per_date",
        "claim": {"operating_point": {"conf": {"value": 0.42,
                                               "validated_against": "held_out_annotations"}}},
        "validated_against": "held_out_annotations",
        "checkpoint_sha256": "0" * 64,
        "producing_experiment_id": "exp-021-currant-bud-det",
        "reference_identity": REFERENCE_IDENTITY,
        "covered_buckets": {"predictions/live/2026-03-04": "7f3a1b9c2d4e5f60"},
        "dataset_root": "/data/currant_valley",
        "recorded_at": "2026-03-04T12:00:00+00:00",
        "train_disjointness": {"checked": True, "group_check": None},
        "selection_disjointness": _real_selection_disjointness(),
    }
    body.update(overrides)
    return body


def _rows_on_disk(root: Path, experiment_id: str) -> list[dict[str, Any]]:
    from tcip_mcp.experiments import validations_key

    return list(ts.read_log(validations_key(experiment_id, root=root)).records)


def test_a_second_validation_of_one_claim_appends_rather_than_replacing(tmp_path):
    from tcip_mcp.experiments import (
        _append_validation, create_experiment, get_experiment, read_validations,
    )

    experiment_id = "exp-021-currant-bud-det"
    create_experiment(experiment_id, TRAINING_CONFIG)

    first = _append_validation(experiment_id, _row())
    second = _append_validation(experiment_id, _row(recorded_at="2026-03-11T09:30:00+00:00"))

    assert "error" not in first and "error" not in second
    assert first["record_digest"] != second["record_digest"]
    rows = read_validations(experiment_id)
    assert [row["recorded_at"] for row in rows] == [
        "2026-03-04T12:00:00+00:00", "2026-03-11T09:30:00+00:00",
    ]
    assert _rows_on_disk(tmp_path, experiment_id) == rows
    assert get_experiment(experiment_id)["validations"] == rows


def test_a_validation_of_an_experiment_that_does_not_exist_is_refused(tmp_path):
    """The record is filed on a run, so there is no filing it against an id nothing named."""
    from tcip_mcp.experiments import _append_validation, create_experiment, read_validations

    experiment_id = "exp-022-chestnut-burr-det"
    create_experiment(experiment_id, TRAINING_CONFIG)

    refused = _append_validation("exp-022-chestnut-burr-det-typo", _row())
    assert "error" in refused
    assert "exp-022-chestnut-burr-det-typo" in refused["error"]
    assert read_validations("exp-022-chestnut-burr-det-typo") == []

    assert "error" not in _append_validation(experiment_id, _row())
    assert len(read_validations(experiment_id)) == 1


def test_a_row_missing_a_required_field_is_refused_and_names_it(tmp_path):
    from tcip_mcp.experiments import _append_validation, create_experiment, read_validations

    experiment_id = "exp-023-currant-cluster-det"
    create_experiment(experiment_id, TRAINING_CONFIG)

    incomplete = _row()
    del incomplete["reference_identity"]
    refused = _append_validation(experiment_id, incomplete)

    assert "error" in refused
    assert "reference_identity" in refused["error"]
    assert read_validations(experiment_id) == []

    assert "error" not in _append_validation(experiment_id, _row())
    assert read_validations(experiment_id) == [_row()]


def test_a_completed_run_takes_a_validation_while_its_metrics_and_state_stay_frozen(tmp_path):
    """The exemption is the member's own meaning: freezing it would make it unwritable in
    every case it exists for, since a run is validated after it finishes."""
    from tcip_mcp.experiments import (
        _append_validation, create_experiment, log_metrics, read_validations, update_status,
    )

    experiment_id = "exp-024-elderberry-umbel-det"
    create_experiment(experiment_id, TRAINING_CONFIG)
    update_status(experiment_id, "running")
    log_metrics(experiment_id, 4, {"val_map50": 0.61})
    update_status(experiment_id, "completed")

    appended = _append_validation(experiment_id, _row())

    assert "error" not in appended
    assert read_validations(experiment_id) == [_row()]

    refused_epoch = log_metrics(experiment_id, 5, {"val_map50": 0.99})
    assert "error" in refused_epoch
    refused_reopen = update_status(experiment_id, "running")
    assert "error" in refused_reopen and refused_reopen["state"] == "completed"


def test_a_row_is_found_by_its_recomputed_identity_and_an_unknown_one_finds_nothing(tmp_path):
    from tcip_mcp.experiments import (
        _append_validation, create_experiment, find_validation, validation_digest,
    )

    experiment_id = "exp-025-persimmon-fruit-det"
    create_experiment(experiment_id, TRAINING_CONFIG)
    _append_validation(experiment_id, _row(trait="fruit_ripe_date"))
    digest = _append_validation(experiment_id, _row())["record_digest"]

    found = find_validation(experiment_id, digest)

    assert found == _row()
    assert validation_digest(found) == digest
    assert find_validation(experiment_id, "0" * 16) is None


def test_the_same_calibration_content_resolves_to_one_experiment(tmp_path):
    from tcip_mcp.experiments import (
        _append_validation, config_key, ensure_calibration_experiment, list_experiments,
    )

    first = ensure_calibration_experiment(
        document="classifier_operating_point", checkpoint_sha256=None,
        reference_identity=REFERENCE_IDENTITY, trait="bud_50per_date",
        config={"notes": "calibrated on the March holdout"},
    )
    again = ensure_calibration_experiment(
        document="classifier_operating_point", checkpoint_sha256=None,
        reference_identity=REFERENCE_IDENTITY, trait="bud_50per_date",
        config={"notes": "a second door, same content"},
    )

    assert again == first
    assert [e["experiment_id"] for e in list_experiments()] == [first]

    config = ts.read(config_key(first))
    assert config["reference_identity"] == REFERENCE_IDENTITY
    assert config["trait"] == "bud_50per_date"
    assert config["notes"] == "calibrated on the March holdout"
    assert "error" not in _append_validation(first, _row(document="classifier_operating_point"))


def test_a_calibration_against_a_different_reference_is_a_different_experiment(tmp_path):
    from tcip_mcp.experiments import ensure_calibration_experiment, list_experiments

    on_march = ensure_calibration_experiment(
        document="classifier_operating_point", checkpoint_sha256=None,
        reference_identity=REFERENCE_IDENTITY, trait="bud_50per_date",
        config={"notes": "calibrated on the March holdout"},
    )
    on_april = ensure_calibration_experiment(
        document="classifier_operating_point", checkpoint_sha256=None,
        reference_identity={**REFERENCE_IDENTITY, "holdout_dataset_hash": "5c0e7a2b48d1f963"},
        trait="bud_50per_date", config={"notes": "calibrated on the April holdout"},
    )

    assert on_april != on_march
    assert sorted(e["experiment_id"] for e in list_experiments()) == sorted([on_march, on_april])


def test_a_calibration_config_restating_an_identity_field_is_refused(tmp_path):
    """The identity fields are written from the content the id came from, so a caller's own
    spelling of one could disagree with the id itself."""
    from tcip_mcp.experiments import config_key, ensure_calibration_experiment, list_experiments

    with pytest.raises(ValueError) as refused:
        ensure_calibration_experiment(
            document="ordinal_operating_point", checkpoint_sha256="0" * 64,
            reference_identity=REFERENCE_IDENTITY, trait="bud_50per_date",
            config={"trait": "something_else", "notes": "free text"},
        )
    assert "trait" in str(refused.value)
    assert list_experiments() == []

    experiment_id = ensure_calibration_experiment(
        document="ordinal_operating_point", checkpoint_sha256="0" * 64,
        reference_identity=REFERENCE_IDENTITY, trait="bud_50per_date",
        config={"notes": "free text", "operator_note": "run from the ordinal door"},
    )
    config = ts.read(config_key(experiment_id))
    assert config["trait"] == "bud_50per_date"
    assert config["operator_note"] == "run from the ordinal door"

def test_the_append_and_the_calibration_creation_each_leave_one_platform_audit_row(tmp_path):
    """The record both mutate is a platform-scoped experiment member, so the rows land in the
    platform log where a reviewer enumerating validations looks first."""
    from tcip_mcp.audit import audit_log_key
    from tcip_mcp.experiments import (
        _append_validation, create_experiment, ensure_calibration_experiment,
    )

    experiment_id = "exp-026-black-locust-raceme-det"
    create_experiment(experiment_id, TRAINING_CONFIG)
    digest = _append_validation(experiment_id, _row())["record_digest"]

    calibration_id = ensure_calibration_experiment(
        document="classifier_operating_point", checkpoint_sha256=None,
        reference_identity=REFERENCE_IDENTITY, trait="bud_50per_date",
        config={"notes": "first calibration"},
    )
    ensure_calibration_experiment(
        document="classifier_operating_point", checkpoint_sha256=None,
        reference_identity=REFERENCE_IDENTITY, trait="bud_50per_date",
        config={"notes": "repeat resolves, creates nothing"},
    )

    rows = ts.read_log(audit_log_key(tmp_path)).records
    appended = [r for r in rows if r.get("tool") == "experiment_validation_recorded"]
    assert [r["arguments"]["record_digest"] for r in appended] == [digest]
    created = [r for r in rows if r.get("tool") == "calibration_experiment_created"]
    assert [r["arguments"]["experiment_id"] for r in created] == [calibration_id]


@pytest.mark.usefixtures("seed_bud_trait_spec")
def test_a_version_2_row_earned_through_the_real_gate_round_trips(tmp_path):
    """The producer-fed round trip: a row earned through open_validation/seal_validation (the
    platform's own two-phase writer, not _append_validation directly) carries schema_version 2 and
    reads back unchanged."""
    pytest.importorskip("torch")
    from tests._dense_op_fixtures import dense_records

    from tcip_mcp.experiments import read_validations, validation_digest
    from tcip_mcp.pipelines.resolution import (
        open_validation, operating_point_stamp, seal_validation,
    )

    # producing_experiment_id=None (foreign checkpoint) skips train-disjointness; seal_validation
    # mints the calibration experiment the row lands on.
    common = dict(n_images=20, objects_per_image=80, miss_pattern=[0] * 20,
                 fp_pattern=[1] * 20, score=0.9, fp_score=0.05)
    cal = dense_records(id_prefix="c", **common)
    hold = dense_records(id_prefix="h", shift=5.0, **common)
    labels_dir = tmp_path / "annotations" / "2026-03-04"
    labels_dir.mkdir(parents=True, exist_ok=True)

    draft = open_validation(
        document="operating_point",
        evidence={"resolver": "resolve_operating_point",
                  "inputs": {"dataset_hash": "H", "calibration_records": cal,
                             "holdout_records": hold, "staged_conf_floor": 0.01, "tiled": False}},
        trait="bud_opening", checkpoint_sha256="0" * 64, producing_experiment_id=None,
        reference_inputs={"dataset_root": str(tmp_path), "label_dirs": {"calibration": labels_dir}},
    )
    stamp = operating_point_stamp(
        draft.result.to_provenance()["operating_point"], validated=True, validated_by=None,
        tile_size_validated=None, shippable_issues=draft.result.shippable_issues(), id_map=None,
        trait="bud_opening", dataset_hash="H", checkpoint="best", checkpoint_sha256="0" * 64,
        experiment_id=None, images_dir=None, raster_path=None,
        produced_at="2026-03-04T12:00:00+00:00", subject="bud", attribute=None,
    )
    digest, stamped = seal_validation(draft, dataset_root=tmp_path, bucket_dirs=(), stamp_body=stamp)
    experiment_id = stamped["validated_by"]["experiment_id"]

    rows = read_validations(experiment_id)
    assert len(rows) == 1
    assert rows[0]["schema_version"] == 2
    assert validation_digest(rows[0]) == digest


def test_read_validations_admits_an_old_row_with_no_schema_version(tmp_path):
    """Lazy absence: a row filed before schema_version existed carries no such key, and the store's
    ceiling (declared schema_version 2 on the experiment_validations store) still reads it, the
    absent key meaning the frozen version 1."""
    from tcip_mcp.experiments import (
        create_experiment, read_validations, validation_digest, validations_key,
    )

    experiment_id = "exp-028-quince-old-vintage-det"
    create_experiment(experiment_id, TRAINING_CONFIG)
    old_row = _row()
    del old_row["schema_version"]
    # Digest taken at write time and the row relocated by it beside a version-2 sibling,
    # so the proof is identification in a mixed-vintage log, not dict self-equality.
    digest_at_write = validation_digest(old_row)
    ts.append(validations_key(experiment_id), old_row)
    ts.append(validations_key(experiment_id), _row())

    rows = read_validations(experiment_id)
    assert old_row in rows and len(rows) == 2
    matched = [r for r in rows if validation_digest(r) == digest_at_write]
    assert matched == [old_row]

"""A run's validations: what the file takes, what it refuses, what reads back.

A validation is a claim earned against evidence and filed in the run it was earned on, its
``validations.jsonl``. The file is append-only, so a re-validation is a second row rather than a
rewrite; every field of a row is required, since a defaulted provenance field is a claim nobody
made; and a finished run still takes one, because a validation is a statement made about a run
after it ended. A calibration of a checkpoint no run of the project produced files its claims in a
run directory it opens for itself.
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


def _real_selection_disjointness(project: Path) -> dict[str, Any]:
    """The full shape resolution.resolver_selection_disjointness returns for a foreign
    checkpoint of ``project`` with no selection named, read back through the same
    operating_point._selection_disjointness path a live calibration takes, rather than a
    hand-typed subset the resolver never actually produces."""
    from tcip_mcp.pipelines import resolution
    from tcip_mcp.pipelines.operating_point import _selection_disjointness

    raw = _selection_disjointness(None, set(), set(), project=project)
    return resolution.resolver_selection_disjointness(
        {"gate_evidence": {"selection_disjointness": raw}}, "operating_point")


def _row(project: Path, **overrides: Any) -> dict[str, Any]:
    """One complete validation row of ``project``, with the fields a case varies replaced."""
    body: dict[str, Any] = {
        "document": "operating_point",
        "trait": "bud_50per_date",
        "claim": {"operating_point": {"conf": {"value": 0.42,
                                               "validated_against": "held_out_annotations"}}},
        "validated_against": "held_out_annotations",
        "checkpoint_sha256": "0" * 64,
        "producing_experiment_id": "exp-021-currant-bud-det",
        "reference_identity": REFERENCE_IDENTITY,
        "covered_buckets": {
            str(project.resolve() / "predictions" / "live" / "2026-03-04"): "7f3a1b9c2d4e5f60"},
        "dataset_root": str(project.resolve() / "currant_valley"),
        "recorded_at": "2026-03-04T12:00:00+00:00",
        "train_disjointness": {"checked": True, "group_check": None},
        "selection_disjointness": _real_selection_disjointness(project),
    }
    body.update(overrides)
    return body


def _run(project: Path, experiment_id: str):
    from tests._verified_checkpoint_fixtures import detection_config, fixture_data_dir, opened_run

    return opened_run(project, detection_config(fixture_data_dir(project, experiment_id)),
                      experiment_id=experiment_id)


def _validations(run_dir) -> list[dict[str, Any]]:
    """The run's validation rows as its reader answers them."""
    from tcip_mcp.experiments import observe, validations

    return [row for _digest, row in validations(observe(run_dir))]


def test_a_second_validation_of_one_claim_appends_rather_than_replacing(tmp_path):
    from tcip_mcp.experiments import append_validation, get_experiment

    run_dir = _run(tmp_path, "exp-021-currant-bud-det")

    first = append_validation(run_dir, _row(tmp_path))
    second = append_validation(run_dir, _row(tmp_path, recorded_at="2026-03-11T09:30:00+00:00"))

    assert first != second
    rows = _validations(run_dir)
    assert [row["recorded_at"] for row in rows] == [
        "2026-03-04T12:00:00+00:00", "2026-03-11T09:30:00+00:00",
    ]
    assert get_experiment(run_dir.name, project=tmp_path)["validations"] == rows


def test_a_validation_filed_in_a_directory_holding_no_run_is_refused(tmp_path):
    """The record is filed in a run, so there is no filing it where no launch record exists."""
    from tcip_mcp.experiments import append_validation, experiment_dir

    run_dir = _run(tmp_path, "exp-022-chestnut-burr-det")
    typo = experiment_dir("exp-022-chestnut-burr-det-typo", project=tmp_path)

    with pytest.raises(ValueError, match="exp-022-chestnut-burr-det-typo"):
        append_validation(typo, _row(tmp_path))
    assert not typo.exists()

    append_validation(run_dir, _row(tmp_path))
    assert len(_validations(run_dir)) == 1


def test_a_row_missing_a_required_field_is_refused_and_names_it(tmp_path):
    from tcip_mcp.experiments import append_validation

    run_dir = _run(tmp_path, "exp-023-currant-cluster-det")

    incomplete = _row(tmp_path)
    del incomplete["reference_identity"]
    with pytest.raises(ValueError, match="reference_identity"):
        append_validation(run_dir, incomplete)
    assert _validations(run_dir) == []

    append_validation(run_dir, _row(tmp_path))
    assert _validations(run_dir) == [_row(tmp_path)]


def test_a_completed_run_takes_a_validation(tmp_path):
    """A run is validated after it finishes, so its final status never closes its validations."""
    from tcip_mcp.experiments import append_validation, observe
    from tests._verified_checkpoint_fixtures import finished_run

    run_dir = finished_run(tmp_path, experiment_id="exp-024-elderberry-umbel-det")
    assert observe(run_dir).state == "completed"

    append_validation(run_dir, _row(tmp_path))

    assert _validations(run_dir) == [_row(tmp_path)]


def test_a_row_is_found_by_its_recomputed_identity_and_an_unknown_one_finds_nothing(tmp_path):
    from tcip_mcp.experiments import (
        VALIDATIONS_FILE, append_validation, find_validation, observe, read_rows,
        validation_digest,
    )

    run_dir = _run(tmp_path, "exp-025-persimmon-fruit-det")
    append_validation(run_dir, _row(tmp_path, trait="fruit_ripe_date"))
    digest = append_validation(run_dir, _row(tmp_path))

    found = find_validation(observe(run_dir), digest)

    assert found == _row(tmp_path)
    stored = read_rows(run_dir / VALIDATIONS_FILE)[0][-1]
    assert stored["dataset_root"] == "currant_valley"
    assert validation_digest(stored) == digest
    assert find_validation(observe(run_dir), "0" * 16) is None


def _calibration(project: Path, reference_identity=REFERENCE_IDENTITY, *,
                 derived_from: str = "first"):
    from tcip_mcp.experiments import open_calibration_run

    return open_calibration_run({
        "document": "classifier_operating_point", "checkpoint_sha256": None,
        "reference_identity": reference_identity, "trait": "bud_50per_date",
        "derived_from": derived_from}, project=project)


def test_every_calibration_opens_a_run_of_its_own_recording_what_it_calibrated(tmp_path):
    from tcip_mcp.experiments import RUN_FILE, append_validation, list_experiments, read_record

    first = _calibration(tmp_path)
    again = _calibration(tmp_path, derived_from="a second door, same content")

    assert again != first
    assert sorted(e["experiment_id"] for e in list_experiments(tmp_path)) == sorted(
        [first.name, again.name])
    calibrated = read_record(first / RUN_FILE)["calibrated"]
    assert calibrated["reference_identity"] == REFERENCE_IDENTITY
    assert calibrated["trait"] == "bud_50per_date"
    assert calibrated["derived_from"] == "first"
    append_validation(first, _row(tmp_path, document="classifier_operating_point"))


def test_the_append_and_the_calibration_creation_each_leave_one_project_audit_row(tmp_path):
    """Both acts land in the project's log where a reviewer enumerating validations looks first,
    one row per act."""
    from tcip_mcp.audit import audit_log_key
    from tcip_mcp.experiments import append_validation

    digest = append_validation(_run(tmp_path, "exp-026-black-locust-raceme-det"), _row(tmp_path))
    calibrations = [_calibration(tmp_path), _calibration(tmp_path, derived_from="a second calibration")]

    rows = ts.read_log(audit_log_key(tmp_path)).records
    appended = [r for r in rows if r.get("tool") == "experiment_validation_recorded"]
    assert [r["arguments"]["record_digest"] for r in appended] == [digest]
    created = [r for r in rows if r.get("tool") == "calibration_experiment_created"]
    assert [r["arguments"]["experiment_id"] for r in created] == [c.name for c in calibrations]


@pytest.mark.usefixtures("seed_bud_trait_spec")
def test_a_row_earned_through_the_real_gate_round_trips(tmp_path):
    """The producer-fed round trip: a row earned through open_validation/seal_validation (the
    platform's own two-phase writer) for a checkpoint no run produced lands in the calibration
    run directory the claim names, which reads completed with that checkpoint's sha256 on its
    launch record, and reads back under the digest the stamp carries."""
    pytest.importorskip("torch")
    from tests._dense_op_fixtures import dense_records

    from tcip_mcp.experiments import experiment_dir, observe, validations
    from tcip_mcp.pipelines.data.selection import ClassScope
    from tcip_mcp.pipelines.resolution import (
        open_validation, operating_point_stamp, seal_validation,
    )

    sha256 = "ab" * 32
    common = dict(n_images=20, objects_per_image=80, miss_pattern=[0] * 20,
                  fp_pattern=[1] * 20, score=0.9, fp_score=0.05)
    cal = dense_records(id_prefix="c", **common)
    hold = dense_records(id_prefix="h", shift=5.0, **common)
    labels_dir = tmp_path / "annotations" / "2026-03-04"
    labels_dir.mkdir(parents=True, exist_ok=True)

    draft = open_validation(
        project=tmp_path, document="operating_point",
        evidence={"resolver": "resolve_operating_point",
                  "inputs": {"dataset_hash": "H", "calibration_records": cal,
                             "holdout_records": hold, "staged_conf_floor": 0.01, "slicing": None}},
        trait="bud_opening", checkpoint_sha256=sha256, producing_experiment_id=None,
        reference_inputs={"dataset_root": str(tmp_path), "label_dirs": {"calibration": labels_dir}},
    )
    stamp = operating_point_stamp(
        draft.result.to_provenance()["operating_point"], slicing=None, validated=True,
        validated_by=None,
        tile_size_validated=None, shippable_issues=draft.result.shippable_issues(),
        trait="bud_opening", dataset_hash="H", checkpoint="best", checkpoint_sha256=sha256,
        experiment_id=None, images_dir=None, raster_path=None,
        produced_at="2026-03-04T12:00:00+00:00", scope=ClassScope(subject="bud"),
    )
    stamped = seal_validation(draft, dataset_root=tmp_path, bucket_dirs=(), stamp_body=stamp)
    run_dir = experiment_dir(stamped["validated_by"]["experiment_id"], project=tmp_path)

    assert run_dir.name.startswith("calibration_")
    calibration = observe(run_dir)
    assert calibration.state == "completed"
    assert calibration.record["calibrated"]["checkpoint_sha256"] == sha256
    ((digest, _row_read),) = validations(calibration)
    assert digest == stamped["validated_by"]["record_digest"]


def test_append_validation_raises_when_its_audit_line_cannot_be_written(tmp_path, monkeypatch):
    """The row is already on disk by the time the audit line is attempted, so a failed append
    must not be swallowed: the caller is told through AuditEntryNotWritten, not a log line."""
    import tcip_mcp.audit as audit_module
    from tcip_mcp.experiments import append_validation

    run_dir = _run(tmp_path, "exp-029-quince-second-vintage-det")

    def _refuse(*args, **kwargs):
        raise RuntimeError("the audit log could not be appended to")

    monkeypatch.setattr(audit_module, "append", _refuse)

    with pytest.raises(audit_module.AuditEntryNotWritten) as caught:
        append_validation(run_dir, _row(tmp_path))

    assert caught.value.tool == "experiment_validation_recorded"
    assert _validations(run_dir) == [_row(tmp_path)]


def test_a_calibration_run_raises_when_its_audit_line_cannot_be_written(tmp_path, monkeypatch):
    """The directory is already created by the time the audit line is attempted, so a failed
    append must not be swallowed the same way."""
    import tcip_mcp.audit as audit_module
    from tcip_mcp.experiments import list_experiments

    def _refuse(*args, **kwargs):
        raise RuntimeError("the audit log could not be appended to")

    monkeypatch.setattr(audit_module, "append", _refuse)

    with pytest.raises(audit_module.AuditEntryNotWritten) as caught:
        _calibration(tmp_path)

    assert caught.value.tool == "calibration_experiment_created"
    assert len(list_experiments(tmp_path)) == 1

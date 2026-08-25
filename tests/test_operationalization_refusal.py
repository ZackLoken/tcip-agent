"""The precondition every delivery door runs: six failure states, in one order, and what admits.

A delivered phenotype without a recorded, breeder-confirmed meaning is a number nobody defined.
These cases pin each refusal separately, pin that the earlier state reports alone when more than
one applies, and pin the calls that must still succeed, because a rail that only rejects is a rail
that has not been shown to admit valid work.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_mcp import operationalization as op
from tcip_mcp.traits import TraitUnknownError
from tcip_web.app import app
from tests import _operationalization_fixtures as fx
from tests._binding_fixtures import PRODUCER_CHECKPOINT_SHA256
from tests.test_tcip_web_results_routes import _expected_validation_record, _phenology_fixture


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return fx.seed_project(tmp_path / "project")


def _confirmed_crossing(project: Path) -> op.ResolvedOperationalization:
    record = fx.state_crossing(project)
    fx.confirm(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES, record)
    return fx.resolve(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES)


def _confirmed_count(project: Path) -> op.ResolvedOperationalization:
    record = fx.state_count(project)
    fx.confirm(project, fx.COUNT_TRAIT, op.PER_IMAGE_COUNT, record)
    return fx.resolve(project, fx.COUNT_TRAIT, op.PER_IMAGE_COUNT)


def _confirmed_aggregate(project: Path, kind: str) -> op.ResolvedOperationalization:
    record = fx.state_aggregate(project, kind)
    fx.confirm(project, fx.COUNT_TRAIT, kind, record)
    return fx.resolve(project, fx.COUNT_TRAIT, kind)


# ── the six failure states, one at a time ────────────────────────────────────


def test_an_unstated_delivery_refuses_and_names_the_statement_primitive(project: Path):
    spec, record, _ = fx.resolve(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES)

    result = op.check_operationalization(spec, record, op.STATE_CROSSING_DATES)

    assert result.state == 1 and not result.ok
    assert "state_trait_operationalization(" in result.message
    assert "bloom_05per_date" in result.message
    assert "Date when 5%" in result.message
    assert result.basis is None


def test_a_stated_but_unconfirmed_delivery_refuses_and_names_who_confirms(project: Path):
    fx.state_crossing(project)
    spec, record, _ = fx.resolve(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES)

    result = op.check_operationalization(spec, record, op.STATE_CROSSING_DATES)

    assert result.state == 2
    assert "not confirmed by the breeder" in result.message
    assert "Results tab" in result.message
    assert op.STATEMENT_SURFACE in result.message


def test_a_relayed_note_is_surfaced_and_does_not_clear_the_refusal(project: Path):
    fx.state_crossing(project, relayed_note="answered on the phone before the visit")
    spec, record, _ = fx.resolve(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES)

    result = op.check_operationalization(spec, record, op.STATE_CROSSING_DATES)

    assert result.state == 2
    assert "answered on the phone before the visit" in result.message
    assert "does not clear this refusal" in result.message


def test_a_moved_constituting_field_refuses_and_names_it_with_both_values(project: Path):
    _confirmed_crossing(project)
    fx.write_spec(project, dataclasses.replace(fx.CROSSING_SPEC, positive_class_name="shed"))
    spec, record, _ = fx.resolve(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES)

    result = op.check_operationalization(spec, record, op.STATE_CROSSING_DATES)

    assert result.state == 3
    assert "positive_class_name" in result.message
    assert "'open'" in result.message and "'shed'" in result.message
    assert result.superseded == (
        {"field": "positive_class_name", "confirmed_value": "open", "current_value": "shed"},
    )


def test_an_empty_constituting_field_refuses_and_names_the_field(project: Path):
    """The spec is emptied and re-confirmed, so the record covers the emptiness rather than a move."""
    record = fx.state_crossing(project)
    fx.write_spec(project, dataclasses.replace(fx.CROSSING_SPEC, milestone_on=""))
    fx.confirm(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES, record)
    spec, stored, _ = fx.resolve(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES)

    result = op.check_operationalization(spec, stored, op.STATE_CROSSING_DATES)

    assert result.state == 4
    assert "milestone_on" in result.message
    assert "update_trait_spec_fields(" in result.message


def test_a_value_key_outside_the_confirmed_set_refuses(project: Path):
    spec, record, _ = _confirmed_aggregate(project, op.PER_PLANT_COUNT_AGGREGATE)

    result = op.check_operationalization(
        spec, record, op.PER_PLANT_COUNT_AGGREGATE, value_keys=["stem_count", "leaf_length"]
    )

    assert result.state == 5
    assert "leaf_length" in result.message
    assert "never confirmed" in result.message


def test_a_row_carrying_no_value_key_refuses_and_counts_them(project: Path):
    spec, record, _ = _confirmed_aggregate(project, op.PER_PLANT_COUNT_AGGREGATE)

    result = op.check_operationalization(
        spec, record, op.PER_PLANT_COUNT_AGGREGATE, value_keys=["stem_count", "", None]
    )

    assert result.state == 5
    assert "2 of these rows carry no value key" in result.message


def test_a_delivered_phenotype_outside_the_confirmed_set_refuses(project: Path):
    spec, record, _ = _confirmed_crossing(project)

    result = op.check_operationalization(
        spec, record, op.STATE_CROSSING_DATES, delivered_phenotype="bloom_95per_date"
    )

    assert result.state == 5
    assert "bloom_95per_date" in result.message


def test_a_measured_subject_absent_from_every_id_map_refuses(project: Path):
    spec, record, _ = _confirmed_count(project)

    result = op.check_operationalization(
        spec, record, op.PER_IMAGE_COUNT, id_maps={"predictions/live/2026-03-04": {"leaf": 0}}
    )

    assert result.state == 5
    assert "stem" in result.message
    assert "predictions/live/2026-03-04" in result.message


def test_a_record_rewritten_mid_delivery_refuses_against_the_basis_the_door_checked(project: Path):
    spec, record, _ = _confirmed_crossing(project)
    basis = op.check_operationalization(spec, record, op.STATE_CROSSING_DATES).basis
    restated = fx.state_crossing(project, statement="a different quantity entirely")
    fx.confirm(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES, restated)
    spec, moved, _ = fx.resolve(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES)

    result = op.check_operationalization(spec, moved, op.STATE_CROSSING_DATES, basis=basis)

    assert result.state == 6
    assert "changed while this delivery was being produced" in result.message
    assert "Nothing was written" in result.message


def test_a_basis_taken_against_other_spec_values_refuses_on_the_spec_half(project: Path):
    """The basis carries the constituting values as well as the record's token, so both are compared."""
    spec, record, _ = _confirmed_crossing(project)
    current = op.check_operationalization(spec, record, op.STATE_CROSSING_DATES).basis
    elsewhere = op.OperationalizationBasis(
        record_version=current.record_version, constituting={"milestone_on": "something else"}
    )

    result = op.check_operationalization(spec, record, op.STATE_CROSSING_DATES, basis=elsewhere)

    assert result.state == 6
    assert "milestone_on" in result.message


# ── ordering ─────────────────────────────────────────────────────────────────


def test_an_unstated_record_reports_state_one_rather_than_a_later_binding(project: Path):
    fx.write_spec(project, dataclasses.replace(fx.CROSSING_SPEC, milestone_on=""))
    spec, record, _ = fx.resolve(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES)

    result = op.check_operationalization(
        spec, record, op.STATE_CROSSING_DATES, delivered_phenotype="not_covered"
    )

    assert result.state == 1
    assert "no operationalization is recorded" in result.message
    assert "not_covered" not in result.message


def test_an_unconfirmed_record_reports_state_two_rather_than_an_empty_field_or_a_binding(
    project: Path,
):
    fx.state_crossing(project)
    fx.write_spec(project, dataclasses.replace(fx.CROSSING_SPEC, milestone_on=""))
    spec, record, _ = fx.resolve(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES)

    result = op.check_operationalization(
        spec, record, op.STATE_CROSSING_DATES, delivered_phenotype="not_covered"
    )

    assert result.state == 2
    assert "milestone_on" not in result.message
    assert "not_covered" not in result.message


def test_a_moved_field_reports_state_three_rather_than_the_binding_that_also_fails(project: Path):
    _confirmed_crossing(project)
    fx.write_spec(project, dataclasses.replace(fx.CROSSING_SPEC, positive_class_name="shed"))
    spec, record, _ = fx.resolve(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES)

    result = op.check_operationalization(
        spec, record, op.STATE_CROSSING_DATES, delivered_phenotype="not_covered"
    )

    assert result.state == 3
    assert "not_covered" not in result.message


# ── what the rail admits ─────────────────────────────────────────────────────


def test_a_confirmed_delivery_passes_and_returns_the_basis_its_door_re_checks_with(project: Path):
    spec, record, _ = _confirmed_crossing(project)

    result = op.check_operationalization(
        spec, record, op.STATE_CROSSING_DATES, delivered_phenotype="bloom_50per_date"
    )

    assert result.ok and result.state is None and result.message == ""
    assert result.basis is not None
    assert result.basis.constituting["milestone_fractions"] == [0.05, 0.50]
    again = op.check_operationalization(
        spec, record, op.STATE_CROSSING_DATES, basis=result.basis
    )
    assert again.ok


def test_a_record_read_back_through_the_store_reports_no_supersession(project: Path):
    """A tuple spec field stored as a JSON array must not read back as a field that moved."""
    record = fx.state_crossing(project)
    fx.confirm(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES, record)

    spec, stored, _ = fx.resolve(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES)

    assert isinstance(stored.value["confirmed_fields"]["milestone_fractions"], list)
    assert spec.milestone_fractions == (0.05, 0.50)
    result = op.check_operationalization(spec, stored, op.STATE_CROSSING_DATES)
    assert result.ok, result.message
    assert op.superseded_confirmations(project, fx.CROSSING_TRAIT) == []


def test_a_per_image_count_record_refuses_a_delivered_phenotype_and_admits_none(project: Path):
    with pytest.raises(ValueError, match="names no phenotype"):
        fx.state_count(project, delivered_phenotypes=["stem_count"])

    record = fx.state_count(project)

    assert record["delivered_phenotypes"] == []
    assert record["delivered_value_keys"] == []


@pytest.mark.parametrize(
    "delivery_kind",
    [op.PER_PLANT_COUNT_AGGREGATE, op.PER_PLANT_ORDINAL_AGGREGATE, op.PER_PLANT_REGRESSION_AGGREGATE],
)
def test_an_aggregate_record_refuses_empty_value_keys_and_admits_named_ones(
    project: Path, delivery_kind: str
):
    with pytest.raises(ValueError, match="delivered_value_keys"):
        fx.state_aggregate(project, delivery_kind, delivered_value_keys=[])

    record = fx.state_aggregate(project, delivery_kind)

    assert record["delivered_value_keys"] == ["stem_count"]
    fx.confirm(project, fx.COUNT_TRAIT, delivery_kind, record)
    spec, stored, _ = fx.resolve(project, fx.COUNT_TRAIT, delivery_kind)
    assert set(stored.value["confirmed_fields"]) == set(op.constituting_fields(delivery_kind))
    assert op.check_operationalization(
        spec, stored, delivery_kind, delivered_phenotype="stem_count", value_keys=["stem_count"]
    ).ok


def test_confirming_one_kind_leaves_another_kinds_record_alone(project: Path):
    count = fx.state_aggregate(project, op.PER_PLANT_COUNT_AGGREGATE)
    ordinal = fx.state_aggregate(project, op.PER_PLANT_ORDINAL_AGGREGATE)
    fx.confirm(project, fx.COUNT_TRAIT, op.PER_PLANT_COUNT_AGGREGATE, count)

    _, confirmed, _ = fx.resolve(project, fx.COUNT_TRAIT, op.PER_PLANT_COUNT_AGGREGATE)
    _, untouched, _ = fx.resolve(project, fx.COUNT_TRAIT, op.PER_PLANT_ORDINAL_AGGREGATE)

    assert confirmed.value["confirmed_by"] == "user:grüne"
    assert untouched.value["confirmed_by"] is None
    assert untouched.value["stated_at"] == ordinal["stated_at"]


# ── the resolver's own refusals ──────────────────────────────────────────────


def test_an_unknown_delivery_kind_refuses_and_names_the_kinds(project: Path):
    with pytest.raises(ValueError, match="unknown delivery kind"):
        op.constituting_fields("per_plant_aggregate")
    with pytest.raises(ValueError, match="unknown delivery kind"):
        fx.resolve(project, fx.CROSSING_TRAIT, "per_plant_aggregate")


def test_a_trait_the_named_project_does_not_register_refuses(project: Path, tmp_path: Path):
    with pytest.raises(TraitUnknownError):
        fx.resolve(project, "not_registered_here", op.STATE_CROSSING_DATES)
    with pytest.raises(TraitUnknownError):
        fx.resolve(tmp_path / "empty_project", fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES)


def test_the_record_is_read_from_the_project_the_caller_names(project: Path, monkeypatch):
    """The pinned platform root is somewhere else entirely, and the record still resolves."""
    record = fx.state_crossing(project)
    fx.confirm(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES, record)
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(project.parent / "unrelated"))

    spec, stored, specs_dir = fx.resolve(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES)

    assert op.check_operationalization(spec, stored, op.STATE_CROSSING_DATES).ok
    assert specs_dir == project / ".tcip" / "state" / "trait_specs"


# ── the crossing delivery doors ──────────────────────────────────────────────


def delivered_golden(body: dict) -> bytes:
    """What a confirmed crossing delivery writes, byte for byte, for the golden inputs below.

    Both doors produce this through the one writer they share (``write_phenology_csv``). A
    precondition in front of a door must leave what the door delivers untouched, so this is
    asserted as bytes rather than as the absence of an error.

    The one cell no constant can hold is ``validation_record``: a record's digest covers the buckets
    at their absolute dataset root, which is this run's own temporary directory. It is read from the
    buckets' own stamps, so the golden still compares the delivered cell against the records the
    stamps name rather than against whatever the delivery put there.
    """
    record = _expected_validation_record(body).encode()
    sha = PRODUCER_CHECKPOINT_SHA256.encode()
    row = (b",2,2,0,0,2026-02-24,2026-02-12,2026-02-18,2026-02-24,interpolated,interpolated,"
           b"interpolated,interpolated,true,0.4,held_out_annotations,held_out_annotations,"
           + sha + b",exp-1," + record + b"\r\n")
    return (
        b"plant_id,accession,n_dates,n_observed_dates,n_dates_unclassified,n_dates_missing_images,"
        b"catkin_elongation_date,catkin_05per_date,catkin_50per_date,catkin_95per_date,"
        b"catkin_elongation_date_bound,catkin_05per_date_bound,catkin_50per_date_bound,"
        b"catkin_95per_date_bound,catkin_elongation_provisional,operating_point_conf,"
        b"operating_point_validated,positive_state_classifier_validated,producer_model_sha256,"
        b"producer_experiment_id,validation_record\r\n"
        + b"PLANT_A,AccA" + row
        + b"PLANT_B,AccB" + row
    )


GOLDEN_INPUTS = {"fractions": (0.0, 1.0), "detections": 2}


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _delivery(tmp_path: Path, **kwargs) -> dict:
    """A registered, stated and confirmed catkin project with buckets the doors can deliver from.

    The pinned platform root is this test's ``tmp_path``, and the body names the same root, so the
    tool door and the web doors read one project rather than two.
    """
    return _phenology_fixture(tmp_path, **kwargs)


def _compute(body: dict, out_csv: Path, **kwargs) -> dict:
    from tcip_mcp.tools.phenology_tools import compute_phenology

    return compute_phenology(
        trait=body["trait"],
        mapping_path=body["mapping_path"],
        predictions_by_date=body["predictions_by_date"],
        output_csv_path=str(out_csv),
        **kwargs,
    )


def _validated_call(body: dict) -> dict:
    """The arguments that clear every evidence dimension for a fully stamped fixture."""
    return {
        "classifier_pred_dirs": list(body["predictions_by_date"].values()),
        "operating_point_conf": 0.4,
        "operating_point_validated": "held_out_annotations",
    }


def _withdraw(project_root: Path, trait: str) -> None:
    """Withdraw the breeder's confirmation, leaving the statement on file."""
    _spec, record, _ = fx.resolve(project_root, trait, op.STATE_CROSSING_DATES)
    op.confirm_trait_operationalization(
        project_root, trait, op.STATE_CROSSING_DATES,
        user="rosalind", record_seen=op.record_seen_hash(record.value),
        identity_from_request=True, confirmed=False,
    )


def _hand_edit_spec(project_root: Path, trait: str, **fields) -> None:
    """Edit the trait's own stored record directly, the way an out-of-band writer does, around
    every writer this platform has."""
    import tcip_store as ts

    from tcip_mcp import traits

    key = traits.trait_spec_key(traits.trait_specs_dir(project_root), trait)
    stored = ts.read_versioned(key)
    data = dict(stored.value)
    data.update(fields)
    ts.replace(key, data, expect=stored.version)


def _web_refusal(client: TestClient, body: dict, route: str) -> dict:
    sent = {**body, "payload": "milestones", "filename": "x.csv"} if route == "export_csv" else body
    resp = client.post(f"/api/results/{route}", json=sent)
    assert resp.status_code == 400, (route, resp.status_code, resp.text)
    return resp.json()["detail"]


def test_unconfirmed_crossing_door_refuses(tmp_path: Path):
    """A statement nobody confirmed is the agent's own definition, so the tool door refuses on it.

    The refusal names who confirms and where, and the delivery writes nothing.
    """
    body = _delivery(tmp_path, validated=True)
    _withdraw(tmp_path, "catkin")
    out_csv = tmp_path / "delivered.csv"

    res = _compute(body, out_csv, **_validated_call(body))

    assert "stated but not confirmed by the breeder" in res["error"]
    assert "open the Results tab" in res["error"]
    assert not out_csv.exists()


def test_superseded_confirmation_names_the_field(tmp_path: Path):
    """A confirmation covers the values it was given, so a moved field refuses naming both of them."""
    body = _delivery(tmp_path, validated=True)
    _hand_edit_spec(tmp_path, "catkin", positive_class_name="dormant")
    out_csv = tmp_path / "delivered.csv"

    res = _compute(body, out_csv, **_validated_call(body))

    assert "positive_class_name has changed since" in res["error"]
    assert "'elongated'" in res["error"] and "'dormant'" in res["error"]
    assert not out_csv.exists()


def test_empty_constituting_field_refuses_before_class_id(tmp_path: Path):
    """A trait resting on a field its spec leaves empty is refused for that, not for a class id.

    The class-id resolution downstream would fail too, and its message names the wrong problem: it
    would send the agent looking through the buckets for a class nothing has named. The
    precondition runs first, so the refusal describes what is actually missing.
    """
    body = _delivery(tmp_path, validated=True)
    _hand_edit_spec(tmp_path, "catkin", positive_class_name="")
    fx.seed_confirmed_crossing(tmp_path, "catkin")
    out_csv = tmp_path / "delivered.csv"

    res = _compute(body, out_csv, **_validated_call(body))

    assert "rests on positive_class_name, which this trait's spec leaves empty" in res["error"]
    assert "id_map" not in res["error"]
    assert not out_csv.exists()


def test_all_three_web_doors_refuse_identically(client: TestClient, tmp_path: Path):
    """One precondition, one refusal body: a curve the breeder sees is never one Download refuses."""
    body = _delivery(tmp_path, validated=True)
    _withdraw(tmp_path, "catkin")

    details = [_web_refusal(client, body, route)
               for route in ("per_plant_curves", "onset_dates", "export_csv")]

    assert all(d == details[0] for d in details), details
    assert details[0]["kind"] == "operationalization"
    assert details[0]["state"] == 2
    assert details[0]["trait"] == "catkin"
    assert details[0]["delivery_kind"] == op.STATE_CROSSING_DATES
    assert not (tmp_path / "results_export").exists()


def test_hand_edited_spec_is_caught_at_read_time(client: TestClient, tmp_path: Path):
    """The spec is an ordinary record, so the comparison that catches an edit happens at every read.

    Nothing here goes through the spec writer or its supersession signal: the record is replaced
    directly, and the door still refuses, because the confirmation records the values it covered.
    """
    body = _delivery(tmp_path, validated=True)
    _hand_edit_spec(tmp_path, "catkin", milestone_fractions=[0.5])

    detail = _web_refusal(client, body, "onset_dates")

    assert detail["state"] == 3
    assert "milestone_fractions has changed since" in detail["message"]


def test_acknowledge_does_not_clear_the_precondition_at_every_door(
    client: TestClient, tmp_path: Path,
):
    """Acknowledging an unvalidated measurement says nothing about whether one was defined.

    An acknowledged provisional number whose meaning is stated ships stamped false. An acknowledged
    number whose meaning is unstated is not a number, so no door takes the acknowledgement for it.
    """
    body = _delivery(tmp_path, validated=True)
    _withdraw(tmp_path, "catkin")
    out_csv = tmp_path / "delivered.csv"

    res = _compute(body, out_csv, acknowledge_unvalidated=True, **_validated_call(body))
    assert "stated but not confirmed by the breeder" in res["error"]
    assert not out_csv.exists()

    acknowledged = {**body, "acknowledge_unvalidated": True}
    for route in ("per_plant_curves", "onset_dates", "export_csv"):
        assert _web_refusal(client, acknowledged, route)["kind"] == "operationalization", route


def test_acknowledge_still_clears_the_gate_dimensions_in_the_same_call(tmp_path: Path):
    """The two rules are separate, and this is the direction that proves it rather than assumes it.

    Same call, same acknowledgement: with the meaning confirmed, the unvalidated evidence ships
    stamped false exactly as it did before the precondition existed.
    """
    body = _delivery(tmp_path, validated=False)
    out_csv = tmp_path / "delivered.csv"

    res = _compute(body, out_csv, acknowledge_unvalidated=True)

    assert "error" not in res, res
    assert res["positive_state_classifier_validated"] == "false"
    assert out_csv.exists()


def test_confirmation_withdrawn_during_delivery_refuses_before_the_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A confirmation withdrawn while the numbers were being computed leaves nothing written.

    The check that admitted this delivery and the check that lets it write are one function called
    twice, so the window between them is closed rather than assumed to be empty. The withdrawal
    happens inside the measurement, which is where it would happen for real.
    """
    from tcip_mcp.pipelines.postprocessing import phenology

    body = _delivery(tmp_path, validated=True)
    real = phenology.per_plant_phenology

    def withdraw_then_measure(*args, **kwargs):
        _withdraw(tmp_path, "catkin")
        return real(*args, **kwargs)

    monkeypatch.setattr(phenology, "per_plant_phenology", withdraw_then_measure)
    out_csv = tmp_path / "delivered.csv"

    res = _compute(body, out_csv, **_validated_call(body))

    assert "not confirmed by the breeder" in res["error"]
    assert not out_csv.exists()


def test_spec_edited_during_delivery_refuses_before_the_write(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A constituting field moved mid-delivery refuses at the export door with no file written.

    Reported as the supersession it is: the confirmation covered the old value, and the states are
    checked in one order so the first that applies reports alone.
    """
    from tcip_mcp.pipelines.postprocessing import phenology

    body = _delivery(tmp_path, validated=True)
    real = phenology.per_plant_phenology

    def edit_then_measure(*args, **kwargs):
        _hand_edit_spec(tmp_path, "catkin", milestone_on="detected_count")
        return real(*args, **kwargs)

    monkeypatch.setattr(phenology, "per_plant_phenology", edit_then_measure)

    detail = _web_refusal(client, body, "export_csv")

    assert detail["state"] == 3
    assert "milestone_on has changed since" in detail["message"]
    assert not (tmp_path / "results_export").exists()


def test_write_phenology_csv_refuses_without_a_basis(tmp_path: Path):
    """The canonical writer is a ninth entry point, and it demands proof its caller ran the check.

    It holds a spec rather than a project, so it cannot read the record itself without becoming a
    second resolver. Its refusal names the primitives that produce a basis.
    """
    from tcip_mcp.pipelines.postprocessing import phenology

    from tests._trait_fixtures import CATKIN

    with pytest.raises(ValueError) as excinfo:
        phenology.write_phenology_csv(
            "test", [], tmp_path / "out.csv", CATKIN, flags={}, acknowledge_unvalidated=False,
            basis=None, operating_point_conf=None, producer={}, bindings={}, pred_dirs=[],
            project_root=tmp_path)

    assert "compute_phenology and export_csv produce one" in str(excinfo.value)
    assert not (tmp_path / "out.csv").exists()


# ── what the precondition must still admit ───────────────────────────────────


def test_a_confirmed_delivery_writes_the_bytes_it_wrote_before_the_precondition(tmp_path: Path):
    """The tool door's delivered CSV, asserted as bytes rather than as the absence of an error."""
    body = _delivery(tmp_path, validated=True, **GOLDEN_INPUTS)
    out_csv = tmp_path / "delivered.csv"

    res = _compute(body, out_csv, **_validated_call(body))

    assert "error" not in res, res
    assert out_csv.read_bytes() == delivered_golden(body)


def test_the_web_export_door_writes_the_bytes_it_wrote_before_the_precondition(
    client: TestClient, tmp_path: Path,
):
    """The other door onto the same writer, byte for byte, reading its own saved file back."""
    body = _delivery(tmp_path, validated=True, **GOLDEN_INPUTS)

    resp = client.post("/api/results/export_csv",
                       json={**body, "payload": "milestones", "filename": "x.csv"})

    assert resp.status_code == 200, resp.text
    assert resp.content == delivered_golden(body)
    assert (tmp_path / "results_export" / "x.csv").read_bytes() == delivered_golden(body)


def test_write_phenology_csv_with_a_basis_writes_the_delivered_schema(tmp_path: Path):
    """The rail admits the call it was built to admit: a basis and cleared flags in hand, the
    writer writes, with the flags coming from a real reconciliation over the fixture's own
    validated buckets rather than a hand-typed dict."""
    from tcip_mcp.pipelines.postprocessing import phenology
    from tcip_mcp.pipelines.resolution import (
        bind_classifier_validity, reconcile_classifier_validity, reconcile_operating_point_validity,
    )

    from tests._trait_fixtures import CATKIN

    body = _delivery(tmp_path, validated=True)
    spec, record, _ = fx.resolve(tmp_path, "catkin", op.STATE_CROSSING_DATES)
    check = op.check_operationalization(spec, record, op.STATE_CROSSING_DATES)
    pred_dirs = list(body["predictions_by_date"].values())
    recon = reconcile_operating_point_validity(pred_dirs, trait="catkin")
    classifier_recon = reconcile_classifier_validity(pred_dirs)
    classifier_state, _note = bind_classifier_validity(
        classifier_recon["validated"], pred_dirs, pred_dirs, trait="catkin")
    flags = {"classifier": classifier_state, "operating_point": recon["validated"]}
    row = {"plant_id": "P1", "accession": "acc-9", "n_dates": 2, "n_observed_dates": 2}

    phenology.write_phenology_csv(
        "test", [row], tmp_path / "out.csv", CATKIN, flags=flags, acknowledge_unvalidated=False,
        basis=check.basis, operating_point_conf=0.4, producer={}, bindings=recon["bindings"],
        pred_dirs=pred_dirs, project_root=tmp_path)

    header = (tmp_path / "out.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == phenology.phenology_csv_columns(CATKIN)


def test_a_provisional_majority_reading_delivers_and_flipping_it_invalidates_nothing(
    tmp_path: Path,
):
    """The majority alias marker qualifies one disclosure column, not the crossing measurement.

    It is deliberately outside what a confirmation covers, so a trait carrying it delivers, and
    flipping it does not supersede the confirmation the way a constituting field would.
    """
    body = _delivery(tmp_path, validated=True)
    out_csv = tmp_path / "delivered.csv"
    assert "error" not in _compute(body, out_csv, **_validated_call(body))

    _hand_edit_spec(tmp_path, "catkin", majority_provisional=False)
    flipped = tmp_path / "flipped.csv"

    res = _compute(body, flipped, **_validated_call(body))

    assert "error" not in res, res
    assert ",false,0.4," in flipped.read_text(encoding="utf-8")


def test_the_screen_doors_still_honor_acknowledge_unvalidated_for_the_evidence_gate(
    client: TestClient, tmp_path: Path,
):
    """A confirmed meaning plus unvalidated evidence still reaches the screen, marked provisional.

    The precondition is about meaning and must not have absorbed the evidence gate's job, which is
    what would strand a breeder who has nothing to look at.
    """
    body = _delivery(tmp_path, validated=False)
    acknowledged = {**body, "acknowledge_unvalidated": True}

    for route in ("per_plant_curves", "onset_dates"):
        resp = client.post(f"/api/results/{route}", json=acknowledged)
        assert resp.status_code == 200, (route, resp.text)
        assert resp.json()["provisional"] is True, route
    assert client.post("/api/results/onset_dates", json=body).status_code == 400


def test_an_unstated_and_gate_unvalidated_delivery_reports_the_precondition_alone(
    client: TestClient, tmp_path: Path,
):
    """Two refusals apply and the earlier one reports by itself.

    A number with no defined meaning has nothing for a reference to validate, so naming the missing
    evidence beside the missing definition would point the agent at the wrong repair.
    """
    body = _delivery(tmp_path, validated=False)
    _withdraw(tmp_path, "catkin")

    detail = _web_refusal(client, body, "onset_dates")

    assert detail["kind"] == "operationalization"
    assert "validated classifier and count operating point" not in detail["message"]


def test_a_confirmed_delivery_with_an_unbound_classifier_stamp_reports_that_refusal(
    client: TestClient, tmp_path: Path,
):
    """With the meaning confirmed, the refusal families behind the precondition report unchanged.

    A classifier stamp earned for another trait does not validate this delivery, and that is what
    the breeder is told, in the words that family already used.
    """
    from tests.test_tcip_web_results_routes import _rewrite_classifier_sidecars

    body = _delivery(tmp_path, validated=True)
    _rewrite_classifier_sidecars(body, trait="chestnut_bur")

    resp = client.post("/api/results/export_csv",
                       json={**body, "payload": "milestones", "filename": "x.csv"})

    assert resp.status_code == 400
    assert "was earned for trait" in resp.json()["detail"]
    assert "chestnut_bur" in resp.json()["detail"]


# ── the count and aggregate delivery doors ───────────────────────────────────


@pytest.fixture
def delivery_root(tmp_path: Path) -> Path:
    """The pinned platform root, carrying every trait the count and aggregate doors deliver under.

    These doors resolve the record from the root this process is pinned to, which is the root the
    tool and the writer both read, so a test of them seeds that one rather than a project of its own.
    """
    return fx.seed_delivery_traits(tmp_path)


def _count_rows() -> list[dict]:
    return [{"image": "a.jpg", "count": 3, "scores": [0.9]}]


def _aggregate_rows(
    value_key: str | None = "count", *, measurement_document: str = "operating_point"
) -> list[dict]:
    row = {"plant_id": "p1", "value": 5, "observations": 2,
          "measurement_document": measurement_document}
    return [row if value_key is None else {**row, "value_key": value_key}]


def _bucket_recording(tmp_path: Path, id_map: dict) -> str:
    """A prediction bucket whose sidecar records which names its labels decoded to."""
    from tcip_mcp.pipelines.resolution import write_sidecar

    bucket = tmp_path / "ds" / "predictions" / "run"
    bucket.mkdir(parents=True)
    write_sidecar(bucket, {"validated": False, "id_map": id_map}, "operating_point")
    return str(bucket)


def test_unstated_count_door_refuses(delivery_root: Path, tmp_path: Path):
    """A per-image count CSV under a trait nobody has defined a count for is not a measurement."""
    from tcip_mcp.pipelines.postprocessing.export import export_detection_csv

    out_csv = tmp_path / "counts.csv"
    with pytest.raises(ValueError) as excinfo:
        export_detection_csv(_count_rows(), str(out_csv), trait=fx.COUNT_TRAIT,
                             acknowledge_unvalidated=True)

    assert "no operationalization is recorded" in str(excinfo.value)
    assert op.PER_IMAGE_COUNT in str(excinfo.value)
    assert not out_csv.exists()


def test_the_count_tool_refuses_before_it_has_any_counts_to_return(
    delivery_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """The refusal happens ahead of the pass, so the response carries no count of its own.

    This tool returns image_count and total_detections on its refusal paths as well as its success
    one, so a check placed after the pass would hand back exactly the numbers it refused to write.
    """
    import tcip_mcp.tools.inference_tools as itools

    def _never_runs(**kwargs):
        raise AssertionError("run_inference must not run: the precondition refuses ahead of it")

    monkeypatch.setattr(itools, "run_inference", _never_runs)

    res = itools.tabulate_counts("m.pt", str(tmp_path), str(tmp_path / "o.csv"),
                                 trait=fx.COUNT_TRAIT)

    assert "no operationalization is recorded" in res["error"]
    assert "image_count" not in res
    assert "total_detections" not in res


def test_measured_subject_absent_from_id_maps_refuses(delivery_root: Path, tmp_path: Path):
    """The counts have to be counts of the subject the breeder confirmed, read from the buckets."""
    from tcip_mcp.pipelines.postprocessing.export import export_detection_csv

    fx.seed_confirmed_count(tmp_path)
    bucket = _bucket_recording(tmp_path, {"leaf": 0})
    out_csv = tmp_path / "counts.csv"

    with pytest.raises(ValueError) as excinfo:
        export_detection_csv(_count_rows(), str(out_csv), trait=fx.COUNT_TRAIT,
                             pred_dirs=[bucket], acknowledge_unvalidated=True)

    assert fx.COUNT_SUBJECT in str(excinfo.value)
    assert "The counts are of something else" in str(excinfo.value)
    assert not out_csv.exists()


def test_a_bucket_recording_the_subject_delivers_and_one_recording_none_is_unchecked(
    delivery_root: Path, tmp_path: Path,
):
    """The rail admits both shapes it must: the subject recorded, and no subject recorded at all."""
    from tcip_mcp.pipelines.postprocessing.export import export_detection_csv

    fx.seed_confirmed_count(tmp_path)
    matching = _bucket_recording(tmp_path, {fx.COUNT_SUBJECT: 0})
    silent = tmp_path / "ds" / "predictions" / "silent"
    silent.mkdir(parents=True)
    (silent / "operating_point.json").write_text(json.dumps({"validated": False}), encoding="utf-8")

    for name, bucket in (("recorded.csv", matching), ("silent.csv", str(silent))):
        out_csv = tmp_path / name
        export_detection_csv(_count_rows(), str(out_csv), trait=fx.COUNT_TRAIT,
                             pred_dirs=[bucket], acknowledge_unvalidated=True)
        assert out_csv.exists(), name


def test_row_without_value_key_refuses(delivery_root: Path, tmp_path: Path):
    """A row naming no quantity has nothing to check against what the breeder confirmed."""
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    fx.confirm_aggregate(tmp_path, fx.COUNT_TRAIT, op.PER_PLANT_COUNT_AGGREGATE,
                         delivered_phenotype="stem_count", value_keys=["count"])
    out_csv = tmp_path / "agg.csv"

    with pytest.raises(ValueError) as excinfo:
        export_aggregated_csv(_aggregate_rows(None), str(out_csv), trait_name="stem_count",
                              acknowledge_unvalidated=True)

    assert "1 of these rows carry no value key" in str(excinfo.value)
    assert not out_csv.exists()


def test_value_key_outside_confirmed_set_refuses(delivery_root: Path, tmp_path: Path):
    """A quantity nobody confirmed cannot ship under a trait's name, even a plausible one."""
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    fx.confirm_aggregate(tmp_path, fx.COUNT_TRAIT, op.PER_PLANT_COUNT_AGGREGATE,
                         delivered_phenotype="stem_count", value_keys=["count"])
    out_csv = tmp_path / "agg.csv"

    with pytest.raises(ValueError) as excinfo:
        export_aggregated_csv(_aggregate_rows("leaf_length"), str(out_csv),
                              trait_name="stem_count", acknowledge_unvalidated=True)

    assert "leaf_length" in str(excinfo.value)
    assert "never confirmed" in str(excinfo.value)
    assert not out_csv.exists()


def test_a_phenotype_no_registered_trait_delivers_refuses_and_names_the_authoring_step(
    delivery_root: Path, tmp_path: Path,
):
    """The CSV column is a crop-vocabulary name; the record behind it is keyed by a registry trait.

    A phenotype no spec claims has no record to rest on, and the refusal says which step supplies
    one rather than letting the delivery ship under a name nothing answers for.
    """
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    out_csv = tmp_path / "agg.csv"
    with pytest.raises(ValueError) as excinfo:
        export_aggregated_csv(_aggregate_rows(), str(out_csv), trait_name="cluster_nut_count",
                              acknowledge_unvalidated=True)

    assert "no trait registered for this project delivers" in str(excinfo.value)
    assert not out_csv.exists()


def test_a_phenotype_two_registered_traits_deliver_refuses_as_ambiguous(
    delivery_root: Path, tmp_path: Path,
):
    """Two specs claiming one phenotype means two records could say two different things."""
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    fx.write_spec(tmp_path, dataclasses.replace(fx.COUNT_SPEC, name="second_deliverer"))
    out_csv = tmp_path / "agg.csv"

    with pytest.raises(ValueError) as excinfo:
        export_aggregated_csv(_aggregate_rows(), str(out_csv), trait_name="stem_count",
                              acknowledge_unvalidated=True)

    assert "each deliver" in str(excinfo.value)
    assert "second_deliverer" in str(excinfo.value)
    assert not out_csv.exists()


def test_ordinal_and_count_aggregates_need_their_own_records(delivery_root: Path, tmp_path: Path):
    """One trait, two aggregate shapes, two confirmations, each covering only its own delivery.

    A count aggregate rests on the trait's count objective and an ordinal aggregate on its agreement
    floor, so a breeder who confirmed one has said nothing about the other.
    """
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    fx.confirm_aggregate(tmp_path, "astringency", op.PER_PLANT_ORDINAL_AGGREGATE,
                         delivered_phenotype="astringency", value_keys=["astringency"])

    ordinal_rows = _aggregate_rows("astringency", measurement_document="ordinal_operating_point")
    ordinal_csv = tmp_path / "ordinal.csv"
    export_aggregated_csv(ordinal_rows, str(ordinal_csv), trait_name="astringency",
                          acknowledge_unvalidated=True)
    assert ordinal_csv.exists()

    count_rows = _aggregate_rows("astringency")
    count_csv = tmp_path / "count.csv"
    with pytest.raises(ValueError) as excinfo:
        export_aggregated_csv(count_rows, str(count_csv), trait_name="astringency",
                              acknowledge_unvalidated=True)
    assert op.PER_PLANT_COUNT_AGGREGATE in str(excinfo.value)
    assert not count_csv.exists()

    fx.confirm_aggregate(tmp_path, "astringency", op.PER_PLANT_COUNT_AGGREGATE,
                         delivered_phenotype="astringency", value_keys=["astringency"])
    export_aggregated_csv(count_rows, str(count_csv), trait_name="astringency",
                          acknowledge_unvalidated=True)

    assert count_csv.exists()
    _spec, ordinal_record, _dir = fx.resolve(
        tmp_path, "astringency", op.PER_PLANT_ORDINAL_AGGREGATE)
    assert ordinal_record.value["confirmed_by"] == "user:grüne"


def test_a_count_csv_no_longer_ships_under_no_trait_at_all(delivery_root: Path, tmp_path: Path):
    """The permissive delivery this door used to allow, named directly: a count under no trait.

    A confirmed meaning is keyed by the trait, so a call naming none has nothing to check against
    and wrote the file regardless. The argument is required now and nothing is written without it.
    """
    from tcip_mcp.pipelines.postprocessing.export import export_detection_csv

    out_csv = tmp_path / "counts.csv"
    with pytest.raises(TypeError, match="'trait'"):
        export_detection_csv(_count_rows(), str(out_csv), acknowledge_unvalidated=True)  # type: ignore[call-arg]  # the omission is the subject; the raises pins it to trait

    assert not out_csv.exists()


def test_the_count_tool_no_longer_tabulates_under_no_trait_at_all(
    delivery_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """The same permissive delivery at the tool, which used to hand back the counts as well."""
    import tcip_mcp.tools.inference_tools as itools

    monkeypatch.setattr(itools, "run_inference", lambda **kwargs: {
        "results": [{"image": "a.png", "count": 3}], "image_count": 1, "total_detections": 3,
        "operating_point": {"conf": {"value": 0.5}}, "validated": False, "conf_source": "default"})
    out_csv = tmp_path / "o.csv"

    with pytest.raises(TypeError):
        itools.tabulate_counts("m.pt", str(tmp_path), str(out_csv), acknowledge_unvalidated=True)

    assert not out_csv.exists()


def test_a_per_plant_csv_no_longer_ships_under_the_writers_own_default_name(
    delivery_root: Path, tmp_path: Path,
):
    """The other permissive delivery: a per-plant CSV whose trait_name came from a default.

    That default shipped a trait_name column holding a word the crop vocabulary does not carry, so
    no record could be keyed by it. The argument is required now and nothing is written without it.
    """
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    out_csv = tmp_path / "agg.csv"
    with pytest.raises(TypeError, match="'trait_name'"):
        export_aggregated_csv(_aggregate_rows(), str(out_csv), acknowledge_unvalidated=True)  # type: ignore[call-arg]  # the omission is the subject; the raises pins it to trait_name

    assert not out_csv.exists()

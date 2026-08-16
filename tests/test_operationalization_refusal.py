"""The precondition every delivery door runs: six failure states, in one order, and what admits.

A delivered phenotype without a recorded, breeder-confirmed meaning is a number nobody defined.
These cases pin each refusal separately, pin that the earlier state reports alone when more than
one applies, and pin the calls that must still succeed, because a rail that only rejects is a rail
that has not been shown to admit valid work.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from tcip_mcp import operationalization as op
from tcip_mcp.traits import TraitUnknownError
from tests import _operationalization_fixtures as fx


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

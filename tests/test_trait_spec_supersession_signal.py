"""Editing a trait spec tells the agent which confirmations that edit just moved out from under.

A confirmation covers the field values it was confirmed against, and the delivery precondition is
what enforces that. This is the convenience beside it: the spec writer names what it superseded at
edit time, so the agent learns there rather than at the next delivery refusal. It reports, and
refuses nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_mcp import operationalization as op
from tcip_mcp.tools.phenology_tools import update_trait_spec_fields
from tests import _operationalization_fixtures as fx


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return fx.seed_project(tmp_path / "project")


def _confirmed_crossing(project: Path) -> dict:
    record = fx.state_crossing(project)
    return fx.confirm(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES, record)


def test_moving_a_constituting_field_reports_the_superseded_confirmation(project: Path):
    _confirmed_crossing(project)

    result = update_trait_spec_fields(
        str(project), fx.CROSSING_TRAIT, {"positive_class_name": "shed"},
        ["positive_class_name: domain_expert_correction"],
    )

    assert result["positive_class_name"] == "shed"
    assert result["superseded"] == [
        {"delivery_kind": op.STATE_CROSSING_DATES, "field": "positive_class_name",
         "confirmed_value": "open", "current_value": "shed"},
    ]


def test_moving_a_field_no_confirmation_rests_on_reports_nothing(project: Path):
    _confirmed_crossing(project)

    result = update_trait_spec_fields(
        str(project), fx.CROSSING_TRAIT, {"notes": "the breeder walked the row again"},
        ["notes: domain_expert_confirmed"],
    )

    assert result["superseded"] == []


def test_an_unconfirmed_statement_is_not_reported_as_superseded(project: Path):
    fx.state_crossing(project)

    result = update_trait_spec_fields(
        str(project), fx.CROSSING_TRAIT, {"positive_class_name": "shed"},
        ["positive_class_name: domain_expert_correction"],
    )

    assert result["superseded"] == []


def test_only_the_kinds_resting_on_the_moved_field_are_reported(project: Path):
    count = fx.state_aggregate(project, op.PER_PLANT_COUNT_AGGREGATE)
    fx.confirm(project, fx.COUNT_TRAIT, op.PER_PLANT_COUNT_AGGREGATE, count)
    ordinal = fx.state_aggregate(project, op.PER_PLANT_ORDINAL_AGGREGATE)
    fx.confirm(project, fx.COUNT_TRAIT, op.PER_PLANT_ORDINAL_AGGREGATE, ordinal)

    result = update_trait_spec_fields(
        str(project), fx.COUNT_TRAIT, {"ordinal_agreement_floor": 0.8},
        ["ordinal_agreement_floor: domain_expert_correction"],
    )

    assert result["superseded"] == [
        {"delivery_kind": op.PER_PLANT_ORDINAL_AGGREGATE, "field": "ordinal_agreement_floor",
         "confirmed_value": 0.6, "current_value": 0.8},
    ]


def test_the_spec_is_written_to_the_project_the_call_names(project: Path, tmp_path: Path):
    """The pinned platform root is another project, and the edit still lands in the named one."""
    other = fx.seed_project(tmp_path / "other_project")

    update_trait_spec_fields(
        str(project), fx.CROSSING_TRAIT, {"positive_class_name": "shed"},
        ["positive_class_name: domain_expert_correction"],
    )

    assert fx.resolve(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES).spec.positive_class_name == "shed"
    assert fx.resolve(other, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES).spec.positive_class_name == "open"


def test_the_reported_supersession_is_what_the_delivery_precondition_then_refuses_on(project: Path):
    _confirmed_crossing(project)

    reported = update_trait_spec_fields(
        str(project), fx.CROSSING_TRAIT, {"positive_class_name": "shed"},
        ["positive_class_name: domain_expert_correction"],
    )["superseded"]
    spec, record, _ = fx.resolve(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES)
    refusal = op.check_operationalization(spec, record, op.STATE_CROSSING_DATES)

    assert refusal.state == 3
    assert [{"delivery_kind": op.STATE_CROSSING_DATES, **moved} for moved in refusal.superseded] == reported

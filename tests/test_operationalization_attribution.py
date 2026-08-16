"""Who wrote what: the statement writer, the confirmation writer, and the seal between them.

The confirmation is the breeder's act, so a statement write cannot reach a confirmation field, a
restatement cannot keep a confirmation it changed the meaning of, and a click cannot land on
content the surface never rendered. The last of those is a hash over every field a statement owns,
which is what stops a rewrite of the measured subject or the value keys from riding along under an
untouched statement sentence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_mcp import operationalization as op
from tests import _operationalization_fixtures as fx


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return fx.seed_project(tmp_path / "project")


# ── the compare-and-set covers every statement-owned field ───────────────────


def test_changing_measured_subject_invalidates_the_seen_hash(project: Path):
    seen = fx.state_crossing(project)
    fx.state_crossing(project, measured_subject="bush")

    with pytest.raises(op.RecordMoved) as raised:
        fx.confirm(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES, seen)

    assert raised.value.record["measured_subject"] == "bush"
    assert raised.value.record_seen != op.record_seen_hash(seen)
    _, stored, _ = fx.resolve(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES)
    assert stored.value["confirmed_by"] is None


def test_changing_delivered_value_keys_invalidates_the_seen_hash(project: Path):
    seen = fx.state_aggregate(project, op.PER_PLANT_COUNT_AGGREGATE)
    fx.state_aggregate(
        project, op.PER_PLANT_COUNT_AGGREGATE, delivered_value_keys=["stem_count", "leaf_length"]
    )

    with pytest.raises(op.RecordMoved) as raised:
        fx.confirm(project, fx.COUNT_TRAIT, op.PER_PLANT_COUNT_AGGREGATE, seen)

    assert raised.value.record["delivered_value_keys"] == ["stem_count", "leaf_length"]
    assert raised.value.record_seen == op.record_seen_hash(raised.value.record)


def test_the_seen_hash_covers_every_statement_field_and_no_confirmation_field(project: Path):
    """A field the click authorizes and the hash misses is the same defect in a new place."""
    record = fx.state_crossing(project)
    baseline = op.record_seen_hash(record)

    for field in op.STATEMENT_FIELDS:
        moved = dict(record)
        moved[field] = ["a different value"] if isinstance(record[field], list) else "changed"
        assert op.record_seen_hash(moved) != baseline, field
    for field in op.CONFIRMATION_FIELDS:
        stamped = {**record, field: "anything"}
        assert op.record_seen_hash(stamped) == baseline, field


def test_the_seen_hash_does_not_vary_with_sequence_type(project: Path):
    record = fx.state_crossing(project)
    as_tuples = {**record, "delivered_phenotypes": tuple(record["delivered_phenotypes"])}

    assert op.record_seen_hash(as_tuples) == op.record_seen_hash(record)


# ── the statement writer's own refusals ──────────────────────────────────────


def test_a_statement_write_refuses_a_confirmation_field(project: Path):
    for field in op.CONFIRMATION_FIELDS:
        with pytest.raises(ValueError, match="cannot carry the confirmation field"):
            fx.state_crossing(project, **{field: "user:someone"})

    _, stored, _ = fx.resolve(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES)
    assert stored.value is None


def test_a_statement_stamps_the_producing_surface_rather_than_taking_one(project: Path):
    with pytest.raises(ValueError, match="unknown statement field"):
        fx.state_crossing(project, stated_by="a name the caller chose")

    record = fx.state_crossing(project)

    assert record["stated_by"] == op.STATEMENT_SURFACE
    assert record["stated_at"].endswith("+00:00")


def test_a_statement_refuses_an_empty_meaning_or_mechanism_or_subject(project: Path):
    for field in ("statement", "mechanism", "measured_subject"):
        with pytest.raises(ValueError, match=f"{field} is required"):
            fx.state_crossing(project, **{field: "   "})


def test_a_statement_refuses_a_phenotype_the_spec_does_not_deliver(project: Path):
    with pytest.raises(ValueError, match="not in trait 'bloom'"):
        fx.state_crossing(project, delivered_phenotypes=["bloom_95per_date"])


def test_restating_clears_the_confirmation(project: Path):
    record = fx.state_crossing(project)
    fx.confirm(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES, record)

    restated = fx.state_crossing(project, statement="what the breeder actually meant")

    assert all(restated[field] is None for field in op.CONFIRMATION_FIELDS)
    spec, stored, _ = fx.resolve(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES)
    assert op.check_operationalization(spec, stored, op.STATE_CROSSING_DATES).state == 2


# ── the confirmation writer ──────────────────────────────────────────────────


def test_confirming_refuses_when_nothing_is_stated(project: Path):
    with pytest.raises(op.NothingStated, match="state_trait_operationalization"):
        op.confirm_trait_operationalization(
            project,
            fx.CROSSING_TRAIT,
            op.STATE_CROSSING_DATES,
            user="grüne",
            record_seen="whatever the surface would have hashed",
            identity_from_request=True,
        )


def test_confirming_stamps_the_identity_the_clock_and_the_fields_it_covered(project: Path):
    record = fx.state_crossing(project)

    confirmed = fx.confirm(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES, record)

    assert confirmed["confirmed_by"] == "user:grüne"
    assert confirmed["confirmed_at"].endswith("+00:00")
    assert confirmed["identity_from_request"] is True
    assert confirmed["confirmed_fields"] == {
        "positive_class_name": "open",
        "milestone_on": "positive_fraction",
        "milestone_fractions": [0.05, 0.50],
    }
    assert all(confirmed[field] == record[field] for field in op.STATEMENT_FIELDS)


def test_identity_from_request_records_what_the_caller_observed(project: Path):
    record = fx.state_crossing(project)

    confirmed = fx.confirm(
        project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES, record, identity_from_request=False
    )

    assert confirmed["identity_from_request"] is False
    assert confirmed["confirmed_by"] == "user:grüne"


def test_confirming_refuses_a_nameless_actor_rather_than_recording_one(project: Path):
    record = fx.state_crossing(project)

    with pytest.raises(ValueError, match="confirming name is required"):
        fx.confirm(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES, record, user="  ")


def test_withdrawing_clears_exactly_the_four_confirmation_fields(project: Path):
    record = fx.state_crossing(project)
    confirmed = fx.confirm(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES, record)

    withdrawn = op.confirm_trait_operationalization(
        project,
        fx.CROSSING_TRAIT,
        op.STATE_CROSSING_DATES,
        user="grüne",
        record_seen=op.record_seen_hash(confirmed),
        identity_from_request=True,
        confirmed=False,
    )

    assert set(withdrawn) == set(confirmed)
    assert [field for field in withdrawn if withdrawn[field] != confirmed[field]] == list(
        op.CONFIRMATION_FIELDS
    )
    assert all(withdrawn[field] is None for field in op.CONFIRMATION_FIELDS)


def test_no_confirmation_survives_a_hash_taken_before_a_restatement(project: Path):
    record = fx.state_crossing(project)
    fx.confirm(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES, record)
    restated = fx.state_crossing(project, mechanism="a different call entirely")

    with pytest.raises(op.RecordMoved):
        fx.confirm(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES, record)

    assert fx.confirm(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES, restated)[
        "confirmed_by"
    ] == "user:grüne"

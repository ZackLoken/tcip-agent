"""Records the platform's producers write, checked against a golden of their shape or against
the shape their readers require."""

from __future__ import annotations

from typing import Any

TRAIT_UNDER_TEST = "trait_under_test"
DELIVERY_KIND_UNDER_TEST = "state_crossing_dates"

TRAIT_GOLDEN: dict[str, Any] = {"revisions": [{
    "number": 1,
    "entry": {
        "name": TRAIT_UNDER_TEST, "delivers": ["measure_one"],
        "positive_state": {"attribute": "stage", "value": "büsch"},
        "milestone_fractions": [0.5], "milestone_on": "positive_fraction",
        "majority_milestone": "",
        "phenology_prefix": "", "majority_label": "", "count_objective": "",
        "localization": "", "localization_tolerance": "half_class_avg_size",
        "localization_tolerance_frac": 0.5, "count_bias_tolerance_frac": None,
        "count_error_tolerance": None, "classifier_agreement_floor": None,
        "ordinal_agreement_floor": None, "regression_skill_floor": None,
        "regression_criterion": "",
        "scale_tolerance_frac": None, "holdout_match_quality_floor": None, "notes": "ü",
        "operationalizations": {DELIVERY_KIND_UNDER_TEST: {
            "statement": "the date each büsch reached the measured state",
            "mechanism": "the calibrated state classifier over isolated buds",
            "measured_subject": "bud", "delivered_phenotypes": ["measure_one"],
            "delivered_value_keys": []}}},
    "entry_sha256": "7f3a1b9c2d4e5f60",
    "rationale": "the breeder described the state directly", "relayed_note": "",
    "proposed_at": "2026-03-04T12:00:00+00:00",
    "confirmed_by": "user:ü", "confirmed_at": "2026-03-04T12:30:00+00:00",
    "withdrawn_by": None, "withdrawn_at": None}]}
"""One trait's record as the proposing and confirming producers leave it."""

DELIVERY_EVENT_GOLDEN: dict[str, Any] = {
    "event_id": "a1b2c3d4e5f60718", "trait": TRAIT_UNDER_TEST, "trait_revision": 1,
    "trait_revision_sha256": "7f3a1b9c2d4e5f60",
    "delivery_kind": DELIVERY_KIND_UNDER_TEST, "door": "deliver_phenology_milestones",
    "output_path": "büsch_phenology.csv", "output_sha256": "0" * 64,
    "producer": {"checkpoint_sha256": "0" * 64, "experiment_id": "exp_042"},
    "buckets": [{"dataset_root": "dü", "bucket": "live/2026-03-04", "date": "2026-03-04",
                 "assessment_id": "assessment-ü", "validated": True, "reason": None}],
    "scale_assessment_id": None, "validated": True, "acknowledgment": None,
    "population": ["plot_ü"], "require_all_dates_complete": True,
    "plant_mapping": {
        "name": "valley", "dataset_id": "ds-1",
        "dataset_root": "dü", "built_at": "2026-03-04T12:00:00+00:00",
        "record_sha256": "0" * 64, "nn_tolerance_m": {"value": 3.0, "source": "stated"},
        "capture_identity": {"2026-03-04": "0" * 16}, "captures_unverified": [],
        "plant_csvs_unverified": [], "dates_delivered": ["2026-03-04"],
        "images_unattributed": 0, "images_unattributed_scope": "delivered_dates",
        "plant_attribution": "image"},
    "produced_at": "2026-03-04T12:00:00+00:00"}
"""One completed delivery, with the gate's finding for the bucket it shipped."""


def test_the_delivery_events_golden_validates_against_its_declared_shape():
    """The delivery-event golden validates as a ``DeliveryEventRecord``."""
    from tcip_mcp.pipelines.delivery_events_schema import DeliveryEventRecord

    DeliveryEventRecord.model_validate(DELIVERY_EVENT_GOLDEN)


def test_a_selection_record_carries_each_sample_s_own_source_label_group_and_side(tmp_path):
    """A selection's record is its sample list: each entry names its own source and label rather
    than a shared root, with no per-date members block or bare id list."""
    import tcip_store as ts

    from tcip_mcp.dataset_layout import label_key
    from tcip_mcp.pipelines.data import selection
    from tcip_mcp.pipelines.data.label_queries import registry_scope

    selection.write_selection(
        tmp_path / "splits",
        selection.Selection(
            samples=(
                selection.Sample(member="a_1", source=str(tmp_path / "images/2026-03-04/a_1.jpg"),
                                 ground_truth=label_key(tmp_path, "2026-03-04", "a_1"),
                                 group="a", side="train",
                                 ground_truth_digest="7f3a1b9c2d4e5f60"),
            ),
            scope=registry_scope(tmp_path, "bud"), seed=42,
            group_by="stem", dataset_fingerprint="7ac1",
        ),
        project=tmp_path,
    )
    fresh = ts.read(selection.selection_key(tmp_path / "splits"))

    assert "members" not in fresh and "splits" not in fresh and "date" not in fresh
    assert set(fresh["samples"][0]) >= {"member", "source", "ground_truth", "group", "side"}


def test_the_delivery_events_golden_carries_every_key_a_delivery_records(tmp_path):
    """A real delivery through the chain's own producers leaves an event whose keys, and whose
    bucket finding's keys, are exactly the golden's."""
    import pytest

    pytest.importorskip("torch")
    from tcip_mcp.delivery import read_delivery_events
    from tcip_mcp.tools.inference_tools import deliver_per_image_counts

    from tests import _trait_fixtures as fx
    from tests._chain_fixtures import run_the_chain

    chain = run_the_chain(tmp_path, experiment_id="exp-golden-event")
    delivered = deliver_per_image_counts(tmp_path, str(chain.root), chain.bucket,
                                         output_path=str(tmp_path / "out.csv"),
                                         trait=fx.COUNT_TRAIT)
    assert "error" not in delivered, delivered
    (event,) = read_delivery_events(tmp_path)
    fresh = event.model_dump(mode="json")
    golden = DELIVERY_EVENT_GOLDEN
    assert set(golden) == set(fresh)
    assert set(golden["buckets"][0]) == set(fresh["buckets"][0])
    assert set(golden["producer"]) == set(fresh["producer"])


def test_the_traits_golden_carries_every_field_the_proposing_and_confirming_producers_write(
    tmp_path,
):
    """The record ``propose_trait`` and ``confirm_revision`` leave on the store carries exactly
    the golden's keys, at the record, the revision, the entry and the operationalization, so a
    field added to the schema is caught here."""
    import tcip_store as ts
    from tcip_mcp import traits

    from tests import _trait_fixtures as fx

    fx.seed_confirmed_count(tmp_path)
    fresh = ts.read(traits.trait_key(tmp_path, fx.COUNT_TRAIT))

    (golden_revision,), fresh_revision = TRAIT_GOLDEN["revisions"], fresh["revisions"][-1]
    assert set(TRAIT_GOLDEN) == set(fresh)
    assert set(golden_revision) == set(fresh_revision)
    assert set(golden_revision["entry"]) == set(fresh_revision["entry"])
    (golden_op,), (fresh_op,) = (golden_revision["entry"]["operationalizations"].values(),
                                 fresh_revision["entry"]["operationalizations"].values())
    assert set(golden_op) == set(fresh_op)

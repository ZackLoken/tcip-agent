"""Per-plant aggregation over a cohort of several plants, through to the delivery CSV.

A delivery is normally many plants with genuinely different statistics, so these fixtures keep the
per-plant groups distinguishable from each other and from the cohort as a whole: every plant's
summary, observation count and identity-provenance must describe that plant's own records, and a
continuous trait's mean and standard deviation must describe the same estimator of the same values.
The gate is exercised here only where the kind an assessment measured could be mistaken for the
kind a delivery ships.
"""

from __future__ import annotations

import pytest

from tcip_mcp.pipelines.postprocessing.aggregation import aggregate_per_plant
from tcip_mcp.traits import PER_IMAGE_COUNT, PER_PLANT_COUNT_AGGREGATE
from tests import _trait_fixtures as fx, csv_rows


@pytest.fixture(autouse=True)
def _recorded_meaning(tmp_path):
    """Every delivery below ships under a trait whose delivered number has a confirmed meaning."""
    fx.seed_delivery_traits(tmp_path)
    # The count measures what the chain's unassessed bucket detects.
    fx.seed_confirmed_aggregate(tmp_path, "stem_count", value_keys=["count"],
                                measured_subject="bud")


def _by_plant(rows: list[dict]) -> dict[str, dict]:
    """Index summary dicts (or CSV rows) by plant_id, asserting the cohort is neither empty nor
    collapsed: a per-plant assertion means nothing if the rows it quantifies over do not exist."""
    indexed = {r["plant_id"]: r for r in rows}
    assert len(indexed) == len(rows) > 1
    return indexed


# -- each plant's summary comes from that plant's own records ----------------


def test_each_plants_summary_describes_only_that_plants_records():
    """Three plants with deliberately different count distributions, none of which equals the
    cohort-wide statistic: a summary attributed to a plant must be computed over that plant's group,
    not over the whole delivery, which would ship one cohort number under every plant_id."""
    results = [
        {"image": "a1", "plant_id": "PLANT_A", "count": 2},
        {"image": "a2", "plant_id": "PLANT_A", "count": 4},
        {"image": "a3", "plant_id": "PLANT_A", "count": 9},
        {"image": "b1", "plant_id": "PLANT_B", "count": 10},
        {"image": "b2", "plant_id": "PLANT_B", "count": 20},
        {"image": "c1", "plant_id": "PLANT_C", "count": 7},
    ]
    out = _by_plant(aggregate_per_plant(results, strategy="count", value_key="count"))

    assert out["PLANT_A"]["value"] == 4
    assert out["PLANT_B"]["value"] == 15.0
    assert out["PLANT_C"]["value"] == 7
    assert (out["PLANT_A"]["min_count"], out["PLANT_A"]["max_count"]) == (2, 9)
    assert (out["PLANT_B"]["min_count"], out["PLANT_B"]["max_count"]) == (10, 20)
    assert (out["PLANT_C"]["min_count"], out["PLANT_C"]["max_count"]) == (7, 7)
    assert [out[p]["observations"] for p in ("PLANT_A", "PLANT_B", "PLANT_C")] == [3, 2, 1]


def test_identity_provenance_is_summarized_per_plant_not_across_the_cohort():
    """One plant resolved from a single source, another from two, a third with no provenance at all.
    Mixing these across the cohort would report every plant as 'mixed' and give a well-resolved plant
    another plant's worst assignment distance."""
    results = [
        {"image": "a1", "plant_id": "PLANT_A", "count": 2,
         "source": "gnss_sequence", "distance_m": 0.4},
        {"image": "a2", "plant_id": "PLANT_A", "count": 4,
         "source": "gnss_sequence", "distance_m": 1.9},
        {"image": "b1", "plant_id": "PLANT_B", "count": 10,
         "source": "gnss_sequence", "distance_m": 6.5},
        {"image": "b2", "plant_id": "PLANT_B", "count": 20, "source": "qr_code"},
        {"image": "c1", "plant_id": "PLANT_C", "count": 7},
    ]
    out = _by_plant(aggregate_per_plant(results, strategy="count", value_key="count"))

    assert out["PLANT_A"]["plant_id_source"] == "gnss_sequence"
    assert out["PLANT_A"]["plant_id_distance_m_max"] == pytest.approx(1.9)
    assert out["PLANT_B"]["plant_id_source"] == "mixed"
    assert out["PLANT_B"]["plant_id_distance_m_max"] == pytest.approx(6.5)
    assert "plant_id_source" not in out["PLANT_C"]
    assert "plant_id_distance_m_max" not in out["PLANT_C"]


def test_delivery_csv_carries_each_plants_own_value_and_image_count(tmp_path):
    """The whole path a mosaic delivery takes, aggregate then export: every CSV row's value,
    n_images and identity columns belong to the plant named in that row. A cohort-wide number in
    n_images tells the breeder a single-image plant was measured from six."""
    pytest.importorskip("torch")
    from tests._chain_fixtures import deliver_acknowledged

    results = [
        {"image": "a1", "plant_id": "PLANT_A", "count": 2, "plant_attribution": "image",
         "source": "gnss_sequence", "distance_m": 0.4},
        {"image": "a2", "plant_id": "PLANT_A", "count": 4, "plant_attribution": "image",
         "source": "gnss_sequence", "distance_m": 1.9},
        {"image": "a3", "plant_id": "PLANT_A", "count": 9, "plant_attribution": "image",
         "source": "gnss_sequence"},
        {"image": "b1", "plant_id": "PLANT_B", "count": 10, "plant_attribution": "image",
         "source": "qr_code"},
        {"image": "b2", "plant_id": "PLANT_B", "count": 20, "plant_attribution": "image",
         "source": "qr_code"},
        {"image": "c1", "plant_id": "PLANT_C", "count": 7, "plant_attribution": "image"},
    ]
    summaries = aggregate_per_plant(results, strategy="count", value_key="count")

    out_path = tmp_path / "per_plant.csv"
    deliver_acknowledged(tmp_path, summaries, out_path, "stem_count",
                         delivery_kind=PER_PLANT_COUNT_AGGREGATE, crop="currant")
    rows = _by_plant(csv_rows(out_path))

    assert [int(rows[p]["n_images"]) for p in ("PLANT_A", "PLANT_B", "PLANT_C")] == [3, 2, 1]
    assert float(rows["PLANT_A"]["value"]) == 4
    assert float(rows["PLANT_B"]["value"]) == 15.0
    assert float(rows["PLANT_C"]["value"]) == 7
    assert rows["PLANT_A"]["plant_id_distance_m_max"] == "1.9"
    assert rows["PLANT_B"]["plant_id_source"] == "qr_code"
    assert rows["PLANT_B"]["plant_id_distance_m_max"] == ""
    assert rows["PLANT_C"]["plant_id_source"] == ""


# -- a continuous trait's two summary statistics describe one estimator ------


def test_continuous_summary_reports_the_mean_beside_its_own_standard_deviation():
    """A row labeled as a mean carries the arithmetic mean, and the deviation beside it is the
    sample standard deviation of the same values. Skewed samples, where the mean and the median are
    several units apart, are the ordinary case for a count-derived continuous trait, and a median
    reported under a mean's label travels with a deviation that describes a different estimator."""
    results = [
        {"image": "a1", "plant_id": "PLANT_A", "value": 1.0},
        {"image": "a2", "plant_id": "PLANT_A", "value": 2.0},
        {"image": "a3", "plant_id": "PLANT_A", "value": 9.0},
        {"image": "b1", "plant_id": "PLANT_B", "value": 10.0},
        {"image": "b2", "plant_id": "PLANT_B", "value": 11.0},
        {"image": "b3", "plant_id": "PLANT_B", "value": 30.0},
    ]
    out = _by_plant(aggregate_per_plant(results, strategy="mean", value_key="value"))

    assert out["PLANT_A"]["value"] == pytest.approx(4.0)
    assert out["PLANT_A"]["std"] == pytest.approx(4.3589, abs=1e-9)
    assert out["PLANT_A"]["n_observations_with_value"] == 3
    assert out["PLANT_B"]["value"] == pytest.approx(17.0)
    assert out["PLANT_B"]["std"] == pytest.approx(11.2694, abs=1e-9)
    assert out["PLANT_B"]["n_observations_with_value"] == 3


def test_summed_areas_stay_within_their_own_plant():
    """The sum strategy over plants with different numbers of contributing images: a cohort-wide sum
    would give every plant the delivery's total area."""
    results = [
        {"image": "a1", "plant_id": "PLANT_A", "area_mm2": 100.0},
        {"image": "a2", "plant_id": "PLANT_A", "area_mm2": 250.0},
        {"image": "b1", "plant_id": "PLANT_B", "area_mm2": 40.0},
    ]
    out = _by_plant(aggregate_per_plant(results, strategy="sum", value_key="area_mm2"))

    assert out["PLANT_A"]["value"] == pytest.approx(350.0)
    assert out["PLANT_A"]["n_observations_with_value"] == 2
    assert out["PLANT_B"]["value"] == pytest.approx(40.0)
    assert out["PLANT_B"]["n_observations_with_value"] == 1


# -- an assessment answers for the kind it measured, and no other -------


def _assessed(project, kind: str, experiment_id: str):
    """A bucket published under an assessment of ``kind``, the count trait confirmed first with both
    its per-image and its per-plant count operationalizations; the bucket as recorded."""
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.tools.calibration_tools import assess_checkpoint
    from tcip_mcp.tools.inference_tools import run_inference
    from tests._chain_fixtures import (
        DATE, SUBJECT, draw_reference_selection, synthetic_capture, train_on,
    )

    entry = fx.with_operationalization(fx.COUNT_SPEC, PER_IMAGE_COUNT, measured_subject=SUBJECT)
    fx.propose_and_confirm(project, fx.with_operationalization(
        entry, PER_PLANT_COUNT_AGGREGATE, measured_subject=SUBJECT,
        delivered_phenotypes=("stem_count",), delivered_value_keys=("count",)))
    root = project / "ds"
    images_dir = synthetic_capture(root)
    draw_reference_selection(project, root, project / "selection")
    checkpoint = train_on(project / "selection", project, experiment_id)
    assessment = assess_checkpoint(project, checkpoint_path=checkpoint, trait=fx.COUNT_TRAIT,
                                   delivery_kind=kind, selection_dir=str(project / "selection"))
    assert assessment.get("passed") is True, assessment
    bucket = f"{kind}/{DATE}"
    published = run_inference(project, checkpoint_path=checkpoint, images_dir=str(images_dir),
                              bucket=bucket, assessment_id=assessment["assessment_id"])
    assert "error" not in published, published
    return read_bucket(root, bucket)


_ROWS = [{"plant_id": "PLANT_A", "value": 4, "observations": 3, "value_key": "count",
          "plant_attribution": "image"}]


def test_a_per_plant_count_assessment_validates_a_per_plant_count_delivery(tmp_path):
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.postprocessing.aggregation import deliver_per_plant_aggregate

    bucket = _assessed(tmp_path, PER_PLANT_COUNT_AGGREGATE, "exp-kind-matched")
    delivered = deliver_per_plant_aggregate(
        tmp_path, _ROWS, str(tmp_path / "matched.csv"), delivered_phenotype="stem_count",
        delivery_kind=PER_PLANT_COUNT_AGGREGATE, buckets=[bucket], plants=["PLANT_A"],
        door="test", actor=None)

    assert delivered["validated"] is True


def test_a_per_image_count_assessment_never_answers_for_a_per_plant_delivery(tmp_path):
    """The per-image count the assessment measured is not the per-plant number the delivery ships:
    the gate refuses, naming both kinds, rather than lend one measurement to another."""
    pytest.importorskip("torch")
    from tcip_mcp.delivery import DeliveryRefused
    from tcip_mcp.pipelines.postprocessing.aggregation import deliver_per_plant_aggregate

    bucket = _assessed(tmp_path, PER_IMAGE_COUNT, "exp-kind-crossed")
    with pytest.raises(DeliveryRefused, match=f"{PER_IMAGE_COUNT} delivery, not a "
                                              f"{PER_PLANT_COUNT_AGGREGATE} one"):
        deliver_per_plant_aggregate(
            tmp_path, _ROWS, str(tmp_path / "crossed.csv"), delivered_phenotype="stem_count",
            delivery_kind=PER_PLANT_COUNT_AGGREGATE, buckets=[bucket], plants=["PLANT_A"],
            door="test", actor=None)
    assert not (tmp_path / "crossed.csv").exists()

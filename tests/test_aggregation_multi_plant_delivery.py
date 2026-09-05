"""Per-plant aggregation over a cohort of several plants, through to the delivery CSV.

A delivery is normally many plants with genuinely different statistics, so these fixtures keep the
per-plant groups distinguishable from each other and from the cohort as a whole: every plant's
summary, observation count and identity-provenance must describe that plant's own records, and a
continuous trait's mean and standard deviation must describe the same estimator of the same values.
The delivery gate's validity dimensions are exercised here only where the count dimension and the
ordinal dimension could be mistaken for one another.
"""

from __future__ import annotations

import csv
import json

import pytest

from tcip_mcp.pipelines.postprocessing.aggregation import (
    aggregate_per_plant,
    export_aggregated_csv,
)
from tcip_mcp.pipelines.resolution import DeliveryRefused, VALIDATED_FALSE, VALIDATED_HELD_OUT
from tests import _operationalization_fixtures as fx
from tests._binding_fixtures import write_bound_sidecar, write_prediction


@pytest.fixture(autouse=True)
def _recorded_meaning(tmp_path):
    """Every delivery below ships under a trait whose delivered number has a confirmed meaning."""
    fx.seed_delivery_traits(tmp_path)
    fx.seed_confirmed_aggregate(tmp_path, "stem_count", value_keys=["count"])
    fx.seed_confirmed_aggregate(tmp_path, "astringency", value_keys=["astringency"],
                                measurement_document="ordinal_operating_point")
    # A delivery stating operating_point rests on the same trait's count aggregate, its own record.
    fx.seed_confirmed_aggregate(tmp_path, "astringency", value_keys=["astringency"])


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
         "plant_id_source": "gnss_sequence", "plant_id_distance_m": 0.4},
        {"image": "a2", "plant_id": "PLANT_A", "count": 4,
         "plant_id_source": "gnss_sequence", "plant_id_distance_m": 1.9},
        {"image": "b1", "plant_id": "PLANT_B", "count": 10,
         "plant_id_source": "gnss_sequence", "plant_id_distance_m": 6.5},
        {"image": "b2", "plant_id": "PLANT_B", "count": 20, "plant_id_source": "qr_code"},
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
    results = [
        {"image": "a1", "plant_id": "PLANT_A", "count": 2, "plant_attribution": "image", "measurement_document": "operating_point",
         "plant_id_source": "gnss_sequence", "plant_id_distance_m": 0.4},
        {"image": "a2", "plant_id": "PLANT_A", "count": 4, "plant_attribution": "image", "measurement_document": "operating_point",
         "plant_id_source": "gnss_sequence", "plant_id_distance_m": 1.9},
        {"image": "a3", "plant_id": "PLANT_A", "count": 9, "plant_attribution": "image", "measurement_document": "operating_point",
         "plant_id_source": "gnss_sequence"},
        {"image": "b1", "plant_id": "PLANT_B", "count": 10, "plant_attribution": "image", "measurement_document": "operating_point",
         "plant_id_source": "qr_code"},
        {"image": "b2", "plant_id": "PLANT_B", "count": 20, "plant_attribution": "image", "measurement_document": "operating_point",
         "plant_id_source": "qr_code"},
        {"image": "c1", "plant_id": "PLANT_C", "count": 7, "plant_attribution": "image", "measurement_document": "operating_point"},
    ]
    summaries = aggregate_per_plant(results, strategy="count", value_key="count")

    bucket = _count_bucket(tmp_path, "count_preds")
    out_path = tmp_path / "per_plant.csv"
    export_aggregated_csv(summaries, str(out_path), delivered_phenotype="stem_count", crop="currant",
                          pred_dirs=[bucket])
    with open(out_path, newline="") as f:
        rows = _by_plant(list(csv.DictReader(f)))

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
    """A row labelled as a mean carries the arithmetic mean, and the deviation beside it is the
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


# -- the count and ordinal validity dimensions are not interchangeable -------


def _count_bucket(tmp_path, name, *, validated=True):
    root = tmp_path / "ds"
    d = root / "predictions" / name
    write_prediction(d, "img_a")
    stamp = {
        "validated": validated, "trait": fx.COUNT_TRAIT,
        "operating_point": {"conf": {
            "value": 0.55,
            "validated_against": VALIDATED_HELD_OUT if validated else VALIDATED_FALSE,
        }},
        "subject": fx.COUNT_SUBJECT, "attribute": None,
    }
    if validated:
        write_bound_sidecar(d, stamp, dataset_root=root, experiment_id=f"exp-{name}")
    else:
        (d / "operating_point.json").write_text(json.dumps(stamp), encoding="utf-8")
    return str(d)


def _ordinal_bucket(tmp_path, name, *, validated=True):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    stamp = {
        "validated": validated, "trait": "astringency",
        "operating_point": {"ordinal": {
            "validated_against": VALIDATED_HELD_OUT if validated else VALIDATED_FALSE,
            "criterion": "quadratic_weighted_kappa",
        }},
    }
    if validated:
        write_bound_sidecar(d, stamp, document="ordinal_operating_point", dataset_root=tmp_path,
                            experiment_id=f"exp-{name}-ordinal")
    else:
        (d / "ordinal_operating_point.json").write_text(json.dumps(stamp), encoding="utf-8")
    return str(d)


def test_an_ordinal_delivery_never_clears_the_gate_on_the_count_dimension(tmp_path):
    """Which sidecar dimension a delivery reconciles against comes from the records' own stated
    measurement_document, so a bucket holding only operating_point.json cannot answer for records
    naming ordinal_operating_point, and must refuse rather than ship stamped with a dimension
    nothing validated. The two matching pairings still ship, so the rail admits the legitimate
    deliveries it exists to protect."""
    ordinal_only = _ordinal_bucket(tmp_path, "ordinal_preds")
    count_only = _count_bucket(tmp_path, "count_preds")

    with pytest.raises(DeliveryRefused, match="unvalidated dimension"):
        export_aggregated_csv(
            [{"plant_id": "PLANT_A", "value": 2, "observations": 3, "value_key": "astringency",
             "plant_attribution": "image", "measurement_document": "ordinal_operating_point", "scale_document": None}],
            str(tmp_path / "wrong_dimension.csv"), delivered_phenotype="astringency",
            operating_point_validated=VALIDATED_HELD_OUT, pred_dirs=[count_only])

    matched_ordinal = tmp_path / "matched_ordinal.csv"
    export_aggregated_csv(
        [{"plant_id": "PLANT_A", "value": 2, "observations": 3, "value_key": "astringency",
         "plant_attribution": "image", "measurement_document": "ordinal_operating_point", "scale_document": None}],
        str(matched_ordinal), delivered_phenotype="astringency",
        operating_point_validated=VALIDATED_HELD_OUT, pred_dirs=[ordinal_only])
    matched_count = tmp_path / "matched_count.csv"
    export_aggregated_csv(
        [{"plant_id": "PLANT_A", "value": 4, "observations": 3, "value_key": "count",
         "plant_attribution": "image", "measurement_document": "operating_point", "scale_document": None}],
        str(matched_count), delivered_phenotype="stem_count",
        operating_point_validated=VALIDATED_HELD_OUT, pred_dirs=[count_only])

    for path in (matched_ordinal, matched_count):
        with open(path, newline="") as f:
            assert next(csv.DictReader(f))["operating_point_validated"] == VALIDATED_HELD_OUT


def test_a_count_stamp_earned_for_one_trait_floors_a_delivery_of_another(tmp_path):
    """A count stamp validated for one trait must not answer for a delivery under a different
    trait: the refusal names the sidecar and both traits."""
    bucket = _count_bucket(tmp_path, "count_preds")  # stamped trait=fx.COUNT_TRAIT ("stem")

    with pytest.raises(DeliveryRefused) as exc:
        export_aggregated_csv(
            [{"plant_id": "PLANT_A", "value": 4, "observations": 3, "value_key": "astringency",
             "plant_attribution": "image", "measurement_document": "operating_point", "scale_document": None}],
            str(tmp_path / "mismatched_trait.csv"), delivered_phenotype="astringency",
            operating_point_validated=VALIDATED_HELD_OUT, pred_dirs=[bucket])
    message = str(exc.value)
    assert bucket in message
    assert fx.COUNT_TRAIT in message and "astringency" in message


def test_a_delivery_naming_no_measurement_document_refuses(tmp_path):
    """A record set that states nothing about which sidecar document its value rests on refuses
    naming the field, rather than falling through to any particular reconciler (the statement rail
    that replaces the old task-omission gap, count-delivery-door design section 5, P4-20)."""
    ordinal_only = _ordinal_bucket(tmp_path, "ordinal_preds")
    rows = [{"plant_id": "PLANT_A", "value": 2, "observations": 3, "value_key": "astringency"}]

    with pytest.raises(ValueError, match="measurement_document"):
        export_aggregated_csv(rows, str(tmp_path / "unstated.csv"), delivered_phenotype="astringency",
                              operating_point_validated=VALIDATED_HELD_OUT, pred_dirs=[ordinal_only])


def test_a_caller_downgrade_floors_a_validated_ordinal_sidecar(tmp_path):
    """The caller's own assertion may only lower what the sidecar says, and it must keep doing so on
    the ordinal dimension: a caller who knows the measurement is not validated saying so must refuse
    the delivery, not be overruled by the on-disk stamp."""
    bucket = _ordinal_bucket(tmp_path, "ordinal_preds")
    rows = [{"plant_id": "PLANT_A", "value": 2, "observations": 3, "value_key": "astringency",
             "plant_attribution": "image", "measurement_document": "ordinal_operating_point", "scale_document": None}]

    with pytest.raises(DeliveryRefused, match="unvalidated dimension"):
        export_aggregated_csv(rows, str(tmp_path / "downgraded.csv"),
                              delivered_phenotype="astringency",
                              operating_point_validated=VALIDATED_FALSE,
                              pred_dirs=[bucket])

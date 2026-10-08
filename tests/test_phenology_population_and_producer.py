"""A phenology delivery is for the plants its caller names, by one producer.

The CSV carries one row per plant in the population and no other, a plant the mapping never
covers included; dates whose buckets were produced by different checkpoints refuse rather than
splice two models into one series.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests import csv_rows

pytest.importorskip("torch")


def test_the_csv_row_count_equals_the_population(tmp_path: Path) -> None:
    """The mapping names PLANT_A and PLANT_B; a population of PLANT_A and a plant the mapping
    never covers delivers exactly those two rows, in that order, and never PLANT_B; the uncovered
    plant is imaged on none of the delivered dates, so it is incomplete on every one."""
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones
    from tests._chain_fixtures import attributed_series

    body = attributed_series(tmp_path).body()
    out_csv = tmp_path / "out" / "bud.csv"

    res = deliver_phenology_milestones(
        tmp_path, trait=body["trait"], mapping_name=body["mapping_name"],
        dataset_root=body["dataset_root"], buckets=body["buckets"], output_csv_path=str(out_csv),
        plants=["PLANT_A", "PLANT_UNMAPPED"])

    assert "error" not in res, res
    rows = csv_rows(out_csv)
    assert [r["plant_id"] for r in rows] == ["PLANT_A", "PLANT_UNMAPPED"]
    assert res["n_plants"] == 2
    by_plant = {r["plant_id"]: r for r in rows}
    assert by_plant["PLANT_A"]["n_dates"] == "4"
    assert by_plant["PLANT_UNMAPPED"]["n_dates"] == "4"
    assert by_plant["PLANT_UNMAPPED"]["n_dates_missing_images"] == "4"
    assert by_plant["PLANT_UNMAPPED"]["complete"] == "False"
    assert by_plant["PLANT_UNMAPPED"]["bud_50per_date"] == ""


def test_an_empty_population_refuses_naming_the_argument(tmp_path: Path) -> None:
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones
    from tests._chain_fixtures import attributed_series

    body = attributed_series(tmp_path, fractions=(0.0, 1.0), assessed=False).body()

    res = deliver_phenology_milestones(
        tmp_path, trait=body["trait"], mapping_name=body["mapping_name"],
        dataset_root=body["dataset_root"], buckets=body["buckets"],
        output_csv_path=str(tmp_path / "out" / "bud.csv"), plants=[])

    assert "plants=[...]" in res["error"]
    assert not (tmp_path / "out" / "bud.csv").exists()


def test_the_web_route_delivers_the_population_it_was_given(tmp_path: Path, client) -> None:
    from tests._chain_fixtures import attributed_series

    body = attributed_series(tmp_path).body(plants=["PLANT_B"])

    resp = client.post("/api/results/phenology_measurement", json=body)

    assert resp.status_code == 200, resp.text
    assert [r["plant_id"] for r in resp.json()["milestones"]["rows"]] == ["PLANT_B"]
    assert resp.json()["curves"]["n_plants"] == 1


def test_differing_checkpoints_across_dates_refuse(tmp_path: Path) -> None:
    """Two dates published by two checkpoints are two producers: the tool refuses naming each
    bucket's producer and writes nothing, while the same dates from one producer deliver."""
    from tcip_mcp.experiments import observe
    from tcip_mcp.tools.inference_tools import run_inference
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones
    from tests._chain_fixtures import BLOB_BUILDER, attributed_series, chain_pass, run_config
    from tests._verified_checkpoint_fixtures import worker_run

    series = attributed_series(tmp_path, fractions=(0.0, 1.0))
    body = series.body()
    first, second = sorted(series.buckets)
    one = tmp_path / "out" / "one_producer.csv"

    res = deliver_phenology_milestones(
        tmp_path, trait=body["trait"], mapping_name=body["mapping_name"],
        dataset_root=body["dataset_root"], buckets=body["buckets"], output_csv_path=str(one),
        plants=["PLANT_A"])
    assert "error" not in res, res

    config = run_config(tmp_path / "selection", BLOB_BUILDER)
    other = observe(worker_run(tmp_path, config, experiment_id="exp-other")).checkpoint
    assert other is not None
    other_bucket = f"other/{second}"
    published = run_inference(tmp_path, checkpoint_path=other["path"],
                              images_dir=str(series.root / "images" / second),
                              bucket=other_bucket, stated=chain_pass())
    assert "error" not in published, published

    out_csv = tmp_path / "out" / "two_producers.csv"
    res = deliver_phenology_milestones(
        tmp_path, trait=body["trait"], mapping_name=body["mapping_name"],
        dataset_root=body["dataset_root"], buckets=[series.buckets[first], other_bucket],
        output_csv_path=str(out_csv), plants=["PLANT_A"])

    assert "more than one checkpoint or run" in res["error"], res
    assert other_bucket in res["error"]
    assert not out_csv.exists()

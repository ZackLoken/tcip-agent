"""A phenology delivery is for the plants its caller names, by one producer.

The CSV carries one row per plant in the population and no other, a plant the mapping never
covers included; dates whose buckets were produced by different checkpoints refuse rather than
splice two models into one series.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tests.test_tcip_web_results_routes import _phenology_fixture


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_the_csv_row_count_equals_the_population(tmp_path: Path) -> None:
    """The mapping names PLANT_A and PLANT_B; a population of PLANT_A and a plant the mapping
    never covers delivers exactly those two rows, in that order, and never PLANT_B."""
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones

    body = _phenology_fixture(tmp_path, validated=True)
    out_csv = tmp_path / "out" / "bud.csv"

    res = deliver_phenology_milestones(
        trait=body["trait"], mapping_name=body["mapping_name"],
        predictions_by_date=body["predictions_by_date"], output_csv_path=str(out_csv),
        plants=["PLANT_A", "PLANT_UNMAPPED"],
        classifier_pred_dirs=list(body["predictions_by_date"].values()))

    assert "error" not in res, res
    rows = _rows(out_csv)
    assert [r["plant_id"] for r in rows] == ["PLANT_A", "PLANT_UNMAPPED"]
    assert res["n_plants"] == 2
    by_plant = {r["plant_id"]: r for r in rows}
    assert by_plant["PLANT_A"]["n_dates"] == "4"
    assert by_plant["PLANT_UNMAPPED"]["n_dates"] == "0"
    assert by_plant["PLANT_UNMAPPED"]["bud_50per_date"] == ""


def test_an_empty_population_refuses_naming_the_argument(tmp_path: Path) -> None:
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones

    body = _phenology_fixture(tmp_path, validated=True)

    res = deliver_phenology_milestones(
        trait=body["trait"], mapping_name=body["mapping_name"],
        predictions_by_date=body["predictions_by_date"],
        output_csv_path=str(tmp_path / "out" / "bud.csv"), plants=[],
        classifier_pred_dirs=list(body["predictions_by_date"].values()))

    assert "plants=[...]" in res["error"]
    assert not (tmp_path / "out" / "bud.csv").exists()


def test_the_web_route_delivers_the_population_it_was_given(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from tcip_web.app import app

    client = TestClient(app, base_url="http://127.0.0.1")
    body = {**_phenology_fixture(tmp_path, validated=True), "plants": ["PLANT_B"]}

    resp = client.post("/api/results/phenology_measurement", json=body)

    assert resp.status_code == 200, resp.text
    assert [r["plant_id"] for r in resp.json()["milestones"]["rows"]] == ["PLANT_B"]
    assert resp.json()["curves"]["n_plants"] == 1


def test_differing_checkpoints_across_dates_refuse(tmp_path: Path) -> None:
    """Two validated dates whose sidecars name different checkpoints are two producers; the
    tool refuses naming each date's producer and writes nothing, never a placeholder cell."""
    from tcip_mcp.pipelines.resolution import (
        ProducerDiffers,
        read_operating_point_sidecar,
        stamped_producer,
    )
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones

    body = _phenology_fixture(tmp_path, validated=True, fractions=(0.0, 1.0))
    dates = list(body["predictions_by_date"])
    first = read_operating_point_sidecar(body["predictions_by_date"][dates[0]])
    second = read_operating_point_sidecar(body["predictions_by_date"][dates[1]])
    assert first["checkpoint_sha256"] == second["checkpoint_sha256"]
    assert "error" not in deliver_phenology_milestones(
        trait=body["trait"], mapping_name=body["mapping_name"],
        predictions_by_date=body["predictions_by_date"],
        output_csv_path=str(tmp_path / "out" / "one_producer.csv"), plants=["PLANT_A"],
        classifier_pred_dirs=list(body["predictions_by_date"].values()))

    from tests._binding_fixtures import record_producing_run, write_bound_sidecar

    other_sha = record_producing_run(tmp_path / "other_run", "exp-other")
    second_stamp = {k: v for k, v in second.items() if k != "validated_by"}
    second_stamp.update({"checkpoint_sha256": other_sha, "experiment_id": "exp-other"})
    write_bound_sidecar(body["predictions_by_date"][dates[1]], second_stamp,
                        dataset_root=tmp_path / "ds", experiment_id="exp-op-other",
                        producing_experiment_id="exp-other", trait="bud_opening")

    with pytest.raises(ProducerDiffers, match=dates[1]):
        stamped_producer(body["predictions_by_date"])
    out_csv = tmp_path / "out" / "two_producers.csv"
    res = deliver_phenology_milestones(
        trait=body["trait"], mapping_name=body["mapping_name"],
        predictions_by_date=body["predictions_by_date"], output_csv_path=str(out_csv),
        plants=["PLANT_A"], classifier_pred_dirs=list(body["predictions_by_date"].values()))

    assert "error" in res
    assert "more than one" in res["error"]
    assert not out_csv.exists()

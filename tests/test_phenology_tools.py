"""Tests for the phenology MCP tools (build_plant_mapping + deliver_phenology_milestones).

The tools are the agent-facing surface for the per-plant phenology pipeline. These tests pin:
(1) build_plant_mapping wraps build + persist and reports a compact summary + error paths;
(2) deliver_phenology_milestones writes the canonical column schema from buckets whose scope
declares the positive state's attribute and a persisted plant mapping, validated when every
bucket was published under an assessment that answers for the delivery and refused otherwise (the
tool takes no acknowledgment); and (3) its measurement-integrity guard refuses to deliver a CSV
when no bucket's scope declares the positive state's attribute.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

import tcip_store as ts
from tcip_mcp.pipelines.postprocessing.plant_mapping import (
    NEAREST_MATCH_FACTOR,
    plant_mapping_key,
)
from tcip_mcp.tools.phenology_tools import build_plant_mapping
from tests._chain_fixtures import deliver_milestones


def _plant_csv(path: Path) -> None:
    path.write_text(
        "plot_name,accession_name,WGS84_centroid_x,WGS84_centroid_y\n"
        "P1,acc-9,-90.058,43.197\n",
        encoding="utf-8",
    )


def test_build_plant_mapping_wraps_build_and_persists(tmp_path: Path) -> None:
    from datetime import datetime

    from tcip_mcp.tools.project_tools import initialize_project, register_dataset
    from tcip_mcp.traits import registered_crops
    from tests._image_fixtures import write_geo_image
    from tests._mapping_fixtures import register_plant_registry_for

    assert "error" not in initialize_project(str(tmp_path), "Orchard", site="orchard block")
    images_root = tmp_path / "images"
    write_geo_image(
        images_root / "2026-02-11" / "img1.jpg", 43.19670, -90.058000,
        datetime(2026, 2, 11, 9, 30))
    register_dataset(tmp_path, str(tmp_path), crop=sorted(registered_crops())[0])
    csv_path = tmp_path / "plants.csv"
    _plant_csv(csv_path)
    name = "valley"

    registry = register_plant_registry_for(tmp_path, [csv_path])
    res = build_plant_mapping(
        tmp_path, name=name,
        images_root=str(images_root),
        plant_registry=registry,
        nn_tolerance_m=10.0,
    )

    assert "error" not in res, res
    assert res["name"] == name
    totals = res["summary"]["totals"]
    assert totals["n_dates"] == 1
    assert totals["n_images"] == 1
    assert totals["n_mapped"] + totals["n_unattributed"] == 1
    assert "2026-02-11" in res["summary"]["per_date"]
    assert res["nn_tolerance_m"] == {"value": 10.0, "source": "stated"}
    assert res["max_match_distance_m"] == pytest.approx(10.0 * NEAREST_MATCH_FACTOR)
    persisted = ts.read(plant_mapping_key(tmp_path, name))
    assert list(persisted["assignments"].keys()) == ["2026-02-11"]
    assert persisted["assignments"]["2026-02-11"][0]["stem"] == "img1"
    assert "confidence" not in persisted["assignments"]["2026-02-11"][0]


def test_build_plant_mapping_missing_images_root(tmp_path: Path) -> None:
    from tcip_mcp.tools.project_tools import initialize_project

    assert "error" not in initialize_project(str(tmp_path), "Orchard", site="orchard block")
    res = build_plant_mapping(
        tmp_path, images_root=str(tmp_path / "nope"),
        plant_registry="unregistered",
        name="m",
    )
    assert "error" in res
    assert "is not a dataset's own images/ root" in res["error"]


def test_build_plant_mapping_missing_registry(tmp_path: Path) -> None:
    from tcip_mcp.tools.project_tools import initialize_project, register_dataset
    from tcip_mcp.traits import registered_crops

    assert "error" not in initialize_project(str(tmp_path), "Orchard", site="orchard block")
    images_root = tmp_path / "images"
    (images_root / "2026-02-11").mkdir(parents=True)
    register_dataset(tmp_path, str(tmp_path), crop=sorted(registered_crops())[0])
    res = build_plant_mapping(
        tmp_path, images_root=str(images_root),
        plant_registry="does-not-exist",
        name="m",
    )
    assert "error" in res
    assert "plant registry not found" in res["error"]
    assert "register_plant_registry" in res["error"]


# ── deliver_phenology_milestones over a published, assessed series ─────────────────────────


def _rows(out_csv: Path) -> list[dict]:
    with out_csv.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_an_assessed_series_delivers_validated_milestones_under_its_revision(tmp_path: Path):
    pytest.importorskip("torch")
    from tcip_mcp.delivery import read_delivery_events
    from tcip_mcp.operationalization import latest_confirmed
    from tcip_mcp.pipelines.postprocessing.phenology import phenology_csv_columns
    from tests._chain_fixtures import attributed_series

    series = attributed_series(tmp_path)
    out_csv = tmp_path / "out" / "bud_phenology.csv"

    res = deliver_milestones(tmp_path, series.body(), out_csv)

    assert "error" not in res, res
    revision = latest_confirmed("bud_opening", tmp_path)
    assert res["validated"] is True
    assert res["trait_revision"] == revision.number
    assert res["columns"] == phenology_csv_columns(revision.entry)
    assert res["n_images_unattributed"] > 0  # the reference frames stand at no plant
    rows = _rows(out_csv)
    assert [r["plant_id"] for r in rows] == ["PLANT_A", "PLANT_B"]
    assert {r["validated"] for r in rows} == {"True"}
    assert {r["delivery_event_id"] for r in rows} == {res["delivery_event_id"]}
    (event,) = read_delivery_events(tmp_path)
    assert event.door == "deliver_phenology_milestones"
    assert event.require_all_dates_complete is True
    assert event.producer.model_dump() == series.assessment["producer"]
    assert {b.assessment_id for b in event.buckets} == {series.assessment["assessment_id"]}


def test_the_delivery_line_lands_in_the_log_of_the_dataset_its_buckets_sit_in(tmp_path: Path):
    """The delivery's one audit line, in the log that travels with the data, names the event."""
    pytest.importorskip("torch")
    from tcip_mcp.audit import audit_log_key
    from tcip_mcp.delivery import read_delivery_events
    from tests._chain_fixtures import attributed_series

    series = attributed_series(tmp_path, fractions=(0.0, 1.0))
    res = deliver_milestones(tmp_path, series.body(), tmp_path / "out" / "bud.csv")
    assert "error" not in res, res

    (event,) = read_delivery_events(tmp_path)
    lines = [e for e in ts.read_log(audit_log_key(series.root)).records
             if e["tool"] == "delivery_event"]
    assert [e["arguments"] for e in lines] == [{"event_id": event.event_id}]
    assert not [e for e in ts.read_log(audit_log_key(tmp_path)).records
                if e["tool"] == "delivery_event"]


def test_an_unassessed_series_refuses_at_the_door_that_takes_no_acknowledgment(tmp_path: Path):
    pytest.importorskip("torch")
    from tests._chain_fixtures import attributed_series

    body = attributed_series(tmp_path, fractions=(0.0, 1.0), assessed=False).body()
    out_csv = tmp_path / "out" / "bud.csv"

    res = deliver_milestones(tmp_path, body, out_csv)

    assert "no assessment answers" in res["error"], res
    assert not out_csv.exists()


def test_an_unreadable_prediction_document_is_reported_by_name(tmp_path: Path):
    """A present, unreadable document is an error naming the file, never a raise through the tool
    boundary and never read as this plant's date contributing nothing."""
    pytest.importorskip("torch")
    from tests._chain_fixtures import attributed_series

    series = attributed_series(tmp_path, fractions=(0.0, 1.0))
    date = sorted(series.buckets)[1]
    bad = Path(series.buckets[date]) / f"PLANT_A_{date}_0.json"
    bad.write_text("not json {][", encoding="utf-8")

    res = deliver_milestones(tmp_path, series.body(), tmp_path / "out" / "bud.csv")

    assert str(bad) in res["error"], res


def test_predictions_that_never_classified_the_positive_state_refuse(tmp_path: Path):
    """Detector buckets over the same captures carry no opening axis at all: the delivery refuses
    rather than reporting full coverage."""
    pytest.importorskip("torch")
    from tests._chain_fixtures import attributed_series, predicted, published

    series = attributed_series(tmp_path, fractions=(0.0, 1.0), assessed=False)
    bare = []
    for date in series.buckets:
        images = sorted((series.root / "images" / date).iterdir())
        results = [{**predicted(p.stem, ["bud", "bud"]), "image": str(p)} for p in images]
        bare.append(str(published(tmp_path, series.root / "predictions" / "bare" / date,
                                  results, scope={"subject": "bud"}).path))
    out_csv = tmp_path / "out" / "bud.csv"

    res = deliver_milestones(tmp_path, series.body(buckets=bare), out_csv)

    assert "classify no opening='open'" in res["error"], res
    assert not out_csv.exists()


def test_each_bucket_stands_for_the_date_its_record_states_and_two_on_one_date_refuse(
    tmp_path: Path,
):
    """The date a bucket stands for is the capture its own record states, whatever its directory
    is named: a bucket published under a directory naming another date delivers as its recorded
    date, and two buckets recording one date refuse naming both."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.inference_tools import run_inference
    from tests._chain_fixtures import attributed_series

    series = attributed_series(tmp_path, fractions=(0.0, 1.0))
    first, second = sorted(series.buckets)
    misnamed = series.root / "predictions" / "misnamed" / "2099-12-31"
    published = run_inference(tmp_path, checkpoint_path=series.checkpoint_path,
                              images_dir=str(series.root / "images" / second),
                              output_dir=str(misnamed),
                              assessment_id=series.assessment["assessment_id"])
    assert "error" not in published, published
    assert published["date"] == second

    shipped = deliver_milestones(
        tmp_path, series.body(buckets=[series.buckets[first], str(misnamed)]),
        tmp_path / "out" / "a.csv")
    both = deliver_milestones(
        tmp_path, series.body(buckets=[*series.buckets.values(), str(misnamed)]),
        tmp_path / "out" / "b.csv")

    assert "error" not in shipped, shipped
    assert _rows(tmp_path / "out" / "a.csv")[0]["n_dates"] == "2"
    assert f"both record capture date {second}" in both["error"]
    assert not (tmp_path / "out" / "b.csv").exists()


def test_a_missing_mapping_and_an_unknown_trait_each_refuse(tmp_path: Path):
    pytest.importorskip("torch")
    from tests._chain_fixtures import attributed_series

    body = attributed_series(tmp_path, fractions=(0.0, 1.0)).body()

    missing = deliver_milestones(tmp_path, {**body, "mapping_name": "nope"}, tmp_path / "a.csv")
    unknown = deliver_milestones(tmp_path, {**body, "trait": "not-a-real-trait"},
                                 tmp_path / "b.csv")

    assert "not found" in missing["error"]
    assert "error" in unknown
    assert not (tmp_path / "a.csv").exists() and not (tmp_path / "b.csv").exists()

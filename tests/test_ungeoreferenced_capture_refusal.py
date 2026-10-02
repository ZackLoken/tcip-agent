"""No positioned capture refuses a plant-mapping build or delivery by name; a partly positioned
mapping keeps delivering with its unattributed count disclosed. Reuses the platform's own
producers (``initialize_project``, ``register_dataset``, ``build_plant_mapping``, publication and
the phenology delivery) and the binding family's own geolocated-scene writer, a second registered
trait rather than the pilot's, so nothing here generalizes from one trait's own vocabulary.
"""

from __future__ import annotations

import asyncio
import csv
from datetime import datetime
from pathlib import Path

import pytest
from PIL import Image

import tcip_store as ts
from tcip_mcp.pipelines.postprocessing import plant_mapping
from tcip_mcp.pipelines.postprocessing.plant_mapping import Assignment, MappingBuild
from tcip_mcp.tools.phenology_tools import build_plant_mapping, deliver_phenology_milestones

from tests._image_fixtures import write_geo_image as _write_geo_image
from tests._mapping_fixtures import register_plant_registry_for
from tests.test_plant_mapping_binding import (
    PLANTS, _dataset, _deliver, _events, _init, _publish, _write_scene,
)
from tests.test_second_trait_acceptance import _seed_currant_bloom_trait

DATE = "2026-02-11"
PLANT_IDS = [p["plot"] for p in PLANTS]


def _write_ungeoreferenced_image(path: Path) -> None:
    """A JPEG carrying no EXIF at all: readable, but with no timestamp and no GPS block."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8)).save(path)


def _write_corrupt_image(path: Path) -> None:
    """A few bytes no decoder can open: unreadable, never a stamp with a position to miss."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a jpeg")


def _write_plant_csv(path: Path, plants: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["plot_name", "accession_name", "WGS84_centroid_x", "WGS84_centroid_y"])
        for p in plants:
            w.writerow([p["plot"], p["accession"], p["lon"], p["lat"]])


# ── build: no positioned capture refuses, at both doors ────────────────────


def test_build_plant_mapping_refuses_when_every_capture_carries_no_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A build over captures none of which carry a position this door reads refuses rather than
    persisting a mapping with ``n_mapped == 0``."""
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root = dataset_root / "images"
    _write_ungeoreferenced_image(images_root / DATE / "P1_a.jpg")
    plant_csv = dataset_root.parent / f"{dataset_root.name}_plants.csv"
    _write_plant_csv(plant_csv, PLANTS)
    registry = register_plant_registry_for(tmp_path, [plant_csv])

    res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root), plant_registry=registry)
    assert "error" in res
    assert "plant-tag mechanism" in res["error"]
    assert not ts.exists(plant_mapping.plant_mapping_key(tmp_path, "valley"))


def test_build_plant_mapping_names_the_unreadable_capture_before_the_position_clause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A capture PIL cannot open at all is named as unreadable, not folded into the no-position
    sentence as if it had been opened and found blank."""
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root = dataset_root / "images"
    _write_corrupt_image(images_root / DATE / "P1_corrupt.jpg")
    plant_csv = dataset_root.parent / f"{dataset_root.name}_plants.csv"
    _write_plant_csv(plant_csv, PLANTS)
    registry = register_plant_registry_for(tmp_path, [plant_csv])

    res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root), plant_registry=registry)
    assert "error" in res
    assert "P1_corrupt.jpg could not be opened" in res["error"]
    assert "plant-tag mechanism" in res["error"]
    assert not ts.exists(plant_mapping.plant_mapping_key(tmp_path, "valley"))


def test_build_route_refuses_when_every_capture_carries_no_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi.testclient import TestClient
    from tcip_web.app import app
    from tcip_web.state import store

    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root = dataset_root / "images"
    _write_ungeoreferenced_image(images_root / DATE / "P1_a.jpg")
    plant_csv = dataset_root.parent / f"{dataset_root.name}_plants.csv"
    _write_plant_csv(plant_csv, PLANTS)
    registry = register_plant_registry_for(tmp_path, [plant_csv])
    asyncio.run(store.open_project(tmp_path.resolve()))

    client = TestClient(app, base_url="http://127.0.0.1")
    resp = client.post("/api/results/plant_mapping/build", json={
        "name": "valley", "images_root": str(images_root), "plant_registry": registry,
    })
    assert resp.status_code == 400
    assert "plant-tag mechanism" in resp.json()["detail"]
    assert not (tmp_path / ".tcip" / "state" / "plant_mappings" / "valley.json").exists()


def test_build_route_refuses_a_selected_date_with_no_captures_never_persisting_an_empty_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A selected date with no captures at all refuses in the shared builder, so both the build
    door and this route refuse it the same way rather than the route persisting an empty
    mapping."""
    from fastapi.testclient import TestClient
    from tcip_web.app import app
    from tcip_web.state import store

    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, _ = _write_scene(dataset_root, dates=[DATE])
    (images_root / "2099-01-01").mkdir()
    registry = register_plant_registry_for(tmp_path, [plant_csv])
    asyncio.run(store.open_project(tmp_path.resolve()))

    client = TestClient(app, base_url="http://127.0.0.1")
    resp = client.post("/api/results/plant_mapping/build", json={
        "name": "valley", "images_root": str(images_root), "plant_registry": registry,
        "dates": ["2099-01-01"],
    })
    assert resp.status_code == 400
    assert "no capture under" in resp.json()["detail"]
    assert not (tmp_path / ".tcip" / "state" / "plant_mappings" / "valley.json").exists()


# ── delivery: three sentences, by the record's own evidence ────────────────


def _persist_synthetic_mapping(
    project_root: Path, dataset_root: Path, name: str, *, plant_csvs: list[dict],
    assignments: dict[str, list[Assignment]],
) -> MappingBuild:
    """A hand-composed ``MappingBuild``, persisted through the platform's own ``persist_mapping``,
    for a delivery scenario ``build_mapping`` itself would refuse to construct (no positioned
    capture, or every capture beyond the match distance): the delivery-time refusal is what these
    tests pin, not the build-time one, so the record is built directly."""
    from tcip_mcp.dataset_layout import require_dataset_identity
    from tcip_mcp.registry_paths import stored_path

    dataset_id = require_dataset_identity(dataset_root)["id"]
    registry_name, registry_digest = f"{name}-registry", "0" * 64
    ts.replace(
        plant_mapping.plant_registry_key(project_root, registry_name),
        {
            "name": registry_name, "crop": "currant", "site": "test", "csvs": plant_csvs,
            "n_plants": sum(e["n_plants"] for e in plant_csvs), "digest": registry_digest,
            "registered_at": "2026-02-11T00:00:00+00:00",
        },
        expect=ts.Version.ABSENT,
    )
    build = MappingBuild(
        name=name, dataset_root=stored_path(dataset_root, project_root),
        dataset_id=dataset_id,
        built_at="2026-02-11T00:00:00+00:00", dates_requested=None,
        dates=sorted(assignments), nn_tolerance_m={"value": 10.0, "source": "stated"},
        plant_registry={"name": registry_name, "digest": registry_digest},
        capture_identity={d: "0" * 16 for d in assignments},
        capture_digests={d: {} for d in assignments}, unreadable={d: [] for d in assignments},
        assignments=assignments,
    )
    plant_mapping.persist_mapping(build, project_root)
    return build


def _unmapped_row(stem: str, distance_m: float | None) -> Assignment:
    return Assignment(
        image=f"{stem}.jpg", stem=stem, date_folder=DATE, plot_name=None,
        accession_name=None, source="unmapped", distance_m=distance_m)


def _delivery_scene(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, str]]:
    """A registered dataset with one real prediction bucket, and the trait this module's
    deliveries run under; returns ``(dataset_root, {date: bucket})``.
    """
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    _, _, preds_by_date = _write_scene(dataset_root, dates=[DATE])
    _seed_currant_bloom_trait(tmp_path)
    return dataset_root, preds_by_date


def _assert_all_doors_refuse(
    tmp_path: Path, preds_by_date: dict[str, str], mapping_name: str, expected_fragment: str,
) -> None:
    from fastapi.testclient import TestClient
    from tcip_web.app import app
    from tcip_web.state import store

    out_csv = tmp_path / "out.csv"
    res = deliver_phenology_milestones(
        tmp_path, trait="currant_bloom", mapping_name=mapping_name, plants=["P1"],
        buckets=list(preds_by_date.values()), output_csv_path=str(out_csv))
    assert "error" in res
    assert expected_fragment in res["error"]
    assert not out_csv.exists()

    asyncio.run(store.open_project(tmp_path.resolve()))
    client = TestClient(app, base_url="http://127.0.0.1")
    payload = {
        "mapping_name": mapping_name,
        "buckets": list(preds_by_date.values()), "trait": "currant_bloom",
        "plants": ["P1"],
    }
    resp = client.post("/api/results/export_csv",
                       json={**payload, "payload": "milestones", "filename": "x.csv"})
    assert resp.status_code == 400, resp.text
    assert expected_fragment in resp.json()["detail"]

    resp = client.post("/api/results/phenology_measurement", json=payload)
    assert resp.status_code == 400, resp.text
    assert expected_fragment in resp.json()["detail"]


def test_delivery_refuses_naming_a_date_recorded_with_no_capture_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root, preds_by_date = _delivery_scene(tmp_path, monkeypatch)
    plant_csv = tmp_path / "plants.csv"
    _write_plant_csv(plant_csv, PLANTS)
    plant_csvs = [{"path": str(plant_csv), "sha256": "0" * 64, "n_plants": len(PLANTS)}]
    _persist_synthetic_mapping(
        tmp_path, dataset_root, "valley", plant_csvs=plant_csvs, assignments={DATE: []})

    _assert_all_doors_refuse(tmp_path, preds_by_date, "valley", "recorded no capture at all")


def test_delivery_refuses_naming_the_plant_csvs_when_none_parsed_a_plant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root, preds_by_date = _delivery_scene(tmp_path, monkeypatch)
    _persist_synthetic_mapping(
        tmp_path, dataset_root, "valley", plant_csvs=[],
        assignments={DATE: [_unmapped_row("P1_20260211", None)]})

    _assert_all_doors_refuse(tmp_path, preds_by_date, "valley", "parsed no plant")


def test_delivery_refuses_with_the_ungeoreferenced_sentence_when_every_distance_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root, preds_by_date = _delivery_scene(tmp_path, monkeypatch)
    plant_csv = tmp_path / "plants.csv"
    _write_plant_csv(plant_csv, PLANTS)
    plant_csvs = [{"path": str(plant_csv), "sha256": "0" * 64, "n_plants": len(PLANTS)}]
    _persist_synthetic_mapping(
        tmp_path, dataset_root, "valley", plant_csvs=plant_csvs,
        assignments={DATE: [_unmapped_row("P1_20260211", None)]})

    _assert_all_doors_refuse(tmp_path, preds_by_date, "valley", "plant-tag mechanism")


def test_delivery_refuses_naming_the_match_distance_when_every_position_is_too_far(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root, preds_by_date = _delivery_scene(tmp_path, monkeypatch)
    plant_csv = tmp_path / "plants.csv"
    _write_plant_csv(plant_csv, PLANTS)
    plant_csvs = [{"path": str(plant_csv), "sha256": "0" * 64, "n_plants": len(PLANTS)}]
    _persist_synthetic_mapping(
        tmp_path, dataset_root, "valley", plant_csvs=plant_csvs,
        assignments={DATE: [_unmapped_row("P1_20260211", 5_000.0)]})

    _assert_all_doors_refuse(tmp_path, preds_by_date, "valley", "beyond the accepted match")


# ── the predicate: a blank plant name is unattributed, everywhere it is decided ─────────


def test_a_blank_plant_name_is_unattributed_by_the_one_predicate(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.postprocessing import phenology
    from tcip_mcp.pipelines.postprocessing.plant_mapping import assignment_is_attributed

    blank = Assignment(image="a.jpg", stem="a", date_folder=DATE, plot_name="",
                       accession_name=None, source="unmapped", distance_m=None)
    named = Assignment(image="b.jpg", stem="b", date_folder=DATE, plot_name="P1",
                       accession_name="acc-9", source="sequence", distance_m=1.0)
    assert assignment_is_attributed(blank) is False
    assert assignment_is_attributed(named) is True

    build = MappingBuild(
        name="m", dataset_root="ds", dataset_id="ds-1",
        built_at="2026-02-11T00:00:00+00:00", dates_requested=None,
        dates=[DATE], nn_tolerance_m={"value": 10.0, "source": "stated"},
        plant_registry={"name": "unregistered", "digest": "0" * 64},
        capture_identity={DATE: "0" * 16}, capture_digests={DATE: {}}, unreadable={DATE: []},
        assignments={DATE: [blank, named]},
    )
    assert build.unattributed() == 1

    per_plant = phenology.per_plant_series(
        {DATE: [blank, named]}, {}, positive_value="open", plants=["P1"])
    assert list(per_plant) == ["P1"]


def _republished(dataset_root: Path, date: str) -> dict[str, str]:
    """Every image now under ``date`` published as a fresh bucket of that date, for a scene whose
    capture set grew after :func:`_write_scene` published it."""
    images = sorted((dataset_root / "images" / date).glob("*.jpg"))
    bucket = dataset_root / "predictions" / "all" / date
    return {date: _publish(dataset_root.parent, bucket, images)}


def _delivered_disclosure(project: Path, preds_by_date: dict[str, str]) -> dict:
    """Deliver ``preds_by_date`` through ``valley`` for both plants; the event's plant-mapping
    disclosure."""
    res = _deliver(project, trait="currant_bloom", mapping_name="valley", plants=PLANT_IDS,
                   buckets=preds_by_date.values(), output_csv_path=str(project / "out.csv"))
    assert "error" not in res, res
    return _events(project)[-1]["plant_mapping"]


# ── admits valid work: a partly positioned dataset keeps delivering, disclosed ──────────


def test_a_partly_positioned_scene_builds_and_delivers_with_the_count_disclosed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two images positioned, one not: both doors build, ``summary()`` reports one unattributed
    per date and in total, the load route returns the same summary, and the delivery event
    discloses one unattributed image over the delivered date."""
    from fastapi.testclient import TestClient
    from tcip_web.app import app
    from tcip_web.state import store

    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, _ = _write_scene(dataset_root, dates=[DATE])
    _write_ungeoreferenced_image(images_root / DATE / "P3_extra.jpg")
    preds_by_date = _republished(dataset_root, DATE)

    registry = register_plant_registry_for(tmp_path, [plant_csv])
    build_res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root), plant_registry=registry)
    assert "error" not in build_res, build_res
    assert build_res["summary"]["per_date"][DATE]["n_unattributed"] == 1
    assert build_res["summary"]["totals"]["n_unattributed"] == 1

    asyncio.run(store.open_project(tmp_path.resolve()))
    client = TestClient(app, base_url="http://127.0.0.1")
    load_resp = client.post("/api/results/plant_mapping/load", json={"name": "valley"})
    assert load_resp.status_code == 200, load_resp.text
    loaded_summary = load_resp.json()["summary"]
    assert loaded_summary["totals"]["n_unattributed"] == 1

    _seed_currant_bloom_trait(tmp_path)
    pm = _delivered_disclosure(tmp_path, preds_by_date)
    assert pm["images_unattributed"] == 1
    assert pm["dates_delivered"] == [DATE]
    assert pm["images_unattributed_scope"] == "delivered_dates"


def test_a_delivery_naming_one_of_two_mapping_dates_carries_the_delivered_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mapping covering two dates, one delivered: the disclosed count is scoped to the
    delivered date alone, never the mapping's full span."""
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    dates = ["2026-02-11", "2026-02-25"]
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root, dates=dates)
    _write_ungeoreferenced_image(images_root / dates[1] / "P3_extra.jpg")

    registry = register_plant_registry_for(tmp_path, [plant_csv])
    build_res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root), plant_registry=registry)
    assert "error" not in build_res, build_res
    assert build_res["summary"]["totals"]["n_unattributed"] == 1

    _seed_currant_bloom_trait(tmp_path)
    pm = _delivered_disclosure(tmp_path, {dates[0]: preds_by_date[dates[0]]})
    assert pm["images_unattributed"] == 0
    assert pm["dates_delivered"] == [dates[0]]


def test_a_date_recorded_with_no_capture_still_delivers_beside_an_attributed_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The no-capture-at-all refusal fires only when nothing across the delivered dates
    attributes: a mapping recording one date with no capture at all beside a fully attributed
    one still delivers the attributed date's bucket. A bucket of no documents states no capture
    date, so no delivery names the empty date itself."""
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root, dates=[DATE])
    empty_date = "2026-02-25"
    (images_root / empty_date).mkdir()

    registry = register_plant_registry_for(tmp_path, [plant_csv])
    build_res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root), plant_registry=registry,
        dates=[DATE, empty_date])
    assert "error" not in build_res, build_res
    assert build_res["summary"]["per_date"][empty_date]["n_images"] == 0

    _seed_currant_bloom_trait(tmp_path)
    pm = _delivered_disclosure(tmp_path, preds_by_date)
    assert pm["dates_delivered"] == [DATE]


def test_a_fully_positioned_scene_keeps_delivering_with_zero_unattributed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root, dates=[DATE])
    registry = register_plant_registry_for(tmp_path, [plant_csv])
    build_res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root), plant_registry=registry)
    assert "error" not in build_res, build_res
    assert build_res["summary"]["totals"]["n_unattributed"] == 0

    _seed_currant_bloom_trait(tmp_path)
    assert _delivered_disclosure(tmp_path, preds_by_date)["images_unattributed"] == 0


def test_a_raster_beside_positioned_photographs_still_delivers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root, dates=[DATE])
    (images_root / DATE / "orthomosaic_block.tif").write_bytes(b"")

    registry = register_plant_registry_for(tmp_path, [plant_csv])
    build_res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root), plant_registry=registry)
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    _delivered_disclosure(tmp_path, preds_by_date)
    assert (tmp_path / "out.csv").exists()


def test_a_capture_at_the_origin_is_admitted_as_positioned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``(0.0, 0.0)`` is a real GPS position (off the coast of west Africa), never treated as
    the absence of one: the build's own position check is ``is not None``, not truthiness."""
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    images_root = dataset_root / "images"
    _write_geo_image(images_root / DATE / "P1_a.jpg", 0.0, 0.0, datetime(2026, 2, 11, 9, 30))
    plant_csv = tmp_path / "plants.csv"
    _write_plant_csv(plant_csv, [{"plot": "P1", "accession": "acc-A", "lat": 0.0, "lon": 0.0}])

    registry = register_plant_registry_for(tmp_path, [plant_csv])
    res = build_plant_mapping(
        tmp_path, name="valley", images_root=str(images_root), plant_registry=registry, nn_tolerance_m=10.0)
    assert "error" not in res, res
    assert res["summary"]["totals"]["n_mapped"] == 1

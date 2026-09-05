"""No positioned capture refuses a plant-mapping build or delivery by name; a partly positioned
mapping keeps delivering with its unattributed count disclosed. Reuses the platform's own
producers (``initialize_project``, ``register_dataset``, ``build_plant_mapping``, ``deliver_phenology_milestones``)
and the binding family's own geolocated-scene writer, a second registered trait rather than the
pilot's, so nothing here generalizes from one trait's own vocabulary.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import pytest
from PIL import Image

import tcip_store as ts
from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox
from tcip_mcp.pipelines.postprocessing import plant_mapping
from tcip_mcp.pipelines.postprocessing.plant_mapping import Assignment, MappingBuild
from tcip_mcp.tools.phenology_tools import build_plant_mapping, deliver_phenology_milestones

from tests._binding_fixtures import register_plant_registry_for
from tests.test_plant_mapping_binding import PLANTS, _dataset, _init, _write_geo_image, _write_scene
from tests.test_second_trait_acceptance import _seed_currant_bloom_trait

DATE = "2026-02-11"


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
    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    images_root = dataset_root / "images"
    _write_ungeoreferenced_image(images_root / DATE / "P1_a.jpg")
    plant_csv = dataset_root.parent / f"{dataset_root.name}_plants.csv"
    _write_plant_csv(plant_csv, PLANTS)
    registry = register_plant_registry_for([plant_csv])

    res = build_plant_mapping(
        name="valley", images_root=str(images_root), plant_registry=registry)
    assert "error" in res
    assert "plant-tag mechanism" in res["error"]
    assert not ts.exists(plant_mapping.plant_mapping_key(tmp_path, "valley"))


def test_build_plant_mapping_names_the_unreadable_capture_before_the_position_clause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A capture PIL cannot open at all is named as unreadable, not folded into the no-position
    sentence as if it had been opened and found blank."""
    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    images_root = dataset_root / "images"
    _write_corrupt_image(images_root / DATE / "P1_corrupt.jpg")
    plant_csv = dataset_root.parent / f"{dataset_root.name}_plants.csv"
    _write_plant_csv(plant_csv, PLANTS)
    registry = register_plant_registry_for([plant_csv])

    res = build_plant_mapping(
        name="valley", images_root=str(images_root), plant_registry=registry)
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

    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    images_root = dataset_root / "images"
    _write_ungeoreferenced_image(images_root / DATE / "P1_a.jpg")
    plant_csv = dataset_root.parent / f"{dataset_root.name}_plants.csv"
    _write_plant_csv(plant_csv, PLANTS)
    registry = register_plant_registry_for([plant_csv])
    store.open_project(tmp_path.resolve())

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

    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, _ = _write_scene(dataset_root, dates=[DATE])
    (images_root / "2099-01-01").mkdir()
    registry = register_plant_registry_for([plant_csv])
    store.open_project(tmp_path.resolve())

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

    dataset_id = require_dataset_identity(dataset_root)["id"]
    registry_name, registry_digest = f"{name}-registry", "0" * 64
    ts.replace(
        plant_mapping.plant_registry_key(project_root, registry_name),
        {
            "name": registry_name, "crop": "currant", "site": "test", "csvs": plant_csvs,
            "n_plants": sum(e["n_plants"] for e in plant_csvs), "digest": registry_digest,
            "registered_by": "agent:test", "registered_at": "2026-02-11T00:00:00+00:00",
        },
        expect=ts.Version.ABSENT,
    )
    build = MappingBuild(
        name=name, project_root=str(project_root), dataset_root=str(dataset_root),
        dataset_id=dataset_id, built_by="build_plant_mapping",
        built_at="2026-02-11T00:00:00+00:00", dates_requested=None,
        dates=sorted(assignments), nn_tolerance_m={"value": 10.0, "source": "fallback"},
        plant_registry={"name": registry_name, "digest": registry_digest},
        capture_identity={d: "0" * 16 for d in assignments},
        capture_digests={d: {} for d in assignments}, unreadable={d: [] for d in assignments},
        assignments=assignments,
    )
    plant_mapping.persist_mapping(build, project_root, name)
    return build


def _unmapped_row(stem: str, distance_m: float | None) -> Assignment:
    return Assignment(
        image_path=f"{stem}.jpg", stem=stem, date_folder=DATE, plot_name=None,
        accession_name=None, source="unmapped", distance_m=distance_m)


def _delivery_scene(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, str]]:
    """A registered dataset with one real prediction bucket, and the trait this module's
    deliveries run under; returns ``(dataset_root, predictions_by_date)``.

    ``_init`` points ``TCIP_WORKSPACE`` away from ``tmp_path`` so ``initialize_project`` does not hold an
    arbitrary test directory to the workspace naming scheme; a web-route delivery then guards
    ``project_root`` against the allowed roots, so ``TCIP_IMAGE_ROOTS`` names ``tmp_path`` as a
    legitimate root the same way an operator would for a project outside any workspace.
    """
    _init(tmp_path, monkeypatch)
    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(tmp_path))
    dataset_root = _dataset(tmp_path)
    _, _, preds_by_date = _write_scene(dataset_root, dates=[DATE])
    _seed_currant_bloom_trait(tmp_path)

    from tcip_mcp.class_registry import copy_registry
    from tcip_mcp.dataset_layout import classes_path

    # The project's own registry (seeded by _seed_currant_bloom_trait) is copied to the delivered
    # dataset root, since the web door resolves its registry from there, not the project.
    copy_registry(classes_path(tmp_path), classes_path(dataset_root))
    return dataset_root, preds_by_date


def _assert_all_doors_refuse(
    tmp_path: Path, preds_by_date: dict[str, str], mapping_name: str, expected_fragment: str,
) -> None:
    from fastapi.testclient import TestClient
    from tcip_web.app import app
    from tcip_web.state import store

    out_csv = tmp_path / "out.csv"
    res = deliver_phenology_milestones(
        trait="currant_bloom", mapping_name=mapping_name, predictions_by_date=preds_by_date,
        output_csv_path=str(out_csv))
    assert "error" in res
    assert expected_fragment in res["error"]
    assert not out_csv.exists()

    store.open_project(tmp_path.resolve())
    client = TestClient(app, base_url="http://127.0.0.1")
    payload = {
        "project_root": str(tmp_path), "mapping_name": mapping_name,
        "predictions_by_date": preds_by_date, "trait": "currant_bloom",
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

    blank = Assignment(image_path="a.jpg", stem="a", date_folder=DATE, plot_name="",
                       accession_name=None, source="unmapped", distance_m=None)
    named = Assignment(image_path="b.jpg", stem="b", date_folder=DATE, plot_name="P1",
                       accession_name="acc-9", source="sequence", distance_m=1.0)
    assert assignment_is_attributed(blank) is False
    assert assignment_is_attributed(named) is True

    build = MappingBuild(
        name="m", project_root="/p", dataset_root="/p/ds", dataset_id="ds-1",
        built_by="test", built_at="2026-02-11T00:00:00+00:00", dates_requested=None,
        dates=[DATE], nn_tolerance_m={"value": 10.0, "source": "fallback"},
        plant_registry={"name": "unregistered", "digest": "0" * 64},
        capture_identity={DATE: "0" * 16}, capture_digests={DATE: {}}, unreadable={DATE: []},
        assignments={DATE: [blank, named]},
    )
    assert build.unattributed() == 1

    per_plant = phenology.per_plant_series(
        {DATE: [blank, named]}, {}, positive_class_name="open")
    assert list(per_plant) == ["P1"]


def _validate_delivery_buckets(
    preds_by_date: dict[str, str], dataset_root: Path,
) -> list[str]:
    """Bind a genuinely validated operating_point and classifier_operating_point sidecar onto
    every bucket a delivery names, all naming one shared producing run, so a delivery earns its
    result the way the door requires rather than through the acknowledgement it no longer takes.
    Mirrors ``test_second_trait_acceptance._currant_bloom_fixture``'s own validated branch, over
    buckets this module's own ``_write_scene`` already wrote. Returns the bucket paths, so a
    caller can pass them straight through as ``classifier_pred_dirs``.
    """
    from tests._binding_fixtures import write_bound_sidecar
    from tests.test_second_trait_acceptance import _ID_MAP

    producing = "exp-currant-run"
    classifier_dirs = []
    for date, bucket in preds_by_date.items():
        sidecar = {
            "id_map": _ID_MAP, "validated": True, "trait": "currant_bloom",
            "subject": "flower", "attribute": "bloom_state",
            "operating_point": {"conf": {"value": 0.4, "validated_against": "held_out_annotations"}},
            "experiment_id": producing, "checkpoint_sha256": "abc123",
        }
        write_bound_sidecar(bucket, sidecar, dataset_root=dataset_root,
                            experiment_id=f"exp-op-{date}", producing_experiment_id=producing,
                            trait="currant_bloom")
        classifier_stamp = {
            "validated": True, "trait": "currant_bloom", "experiment_id": producing,
            "operating_point": {"classifier": {"value": "open",
                                               "validated_against": "held_out_annotations"}},
        }
        write_bound_sidecar(bucket, classifier_stamp, document="classifier_operating_point",
                            dataset_root=dataset_root, experiment_id=f"exp-cls-{date}",
                            producing_experiment_id=producing, trait="currant_bloom")
        classifier_dirs.append(bucket)
    return classifier_dirs


# ── admits valid work: a partly positioned dataset keeps delivering, disclosed ──────────


def test_a_partly_positioned_scene_builds_and_delivers_with_the_count_disclosed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two images positioned, one not: both doors build, ``summary()`` reports one unattributed
    per date and in total, the load route returns the same summary, and ``deliver_phenology_milestones``
    delivers with ``images_unattributed == 1`` and ``dates_delivered`` in the tool's return, the
    CSV's own row and the delivery event's block."""
    from fastapi.testclient import TestClient
    from tcip_web.app import app
    from tcip_web.state import store

    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root, dates=[DATE])
    _write_ungeoreferenced_image(images_root / DATE / "P3_extra.jpg")
    json_io.write_annotations(
        Path(preds_by_date[DATE]) / "P3_extra.json",
        [Annotation(subject="flower", geometry=BBox(1.0, 1.0, 3.0, 3.0), score=0.9,
                   attributes={"bloom_state": "open"})], 8, 8)

    registry = register_plant_registry_for([plant_csv])
    build_res = build_plant_mapping(
        name="valley", images_root=str(images_root), plant_registry=registry)
    assert "error" not in build_res, build_res
    assert build_res["per_date"][DATE]["n_unattributed"] == 1
    assert build_res["n_unattributed"] == 1

    store.open_project(tmp_path.resolve())
    client = TestClient(app, base_url="http://127.0.0.1")
    load_resp = client.post("/api/results/plant_mapping/load", json={"name": "valley"})
    assert load_resp.status_code == 200, load_resp.text
    loaded_summary = load_resp.json()["summary"]
    assert loaded_summary["totals"]["n_unattributed"] == 1

    _seed_currant_bloom_trait(tmp_path)
    classifier_dirs = _validate_delivery_buckets(preds_by_date, dataset_root)
    out_csv = tmp_path / "out.csv"
    res = deliver_phenology_milestones(
        trait="currant_bloom", mapping_name="valley", predictions_by_date=preds_by_date,
        output_csv_path=str(out_csv), classifier_pred_dirs=classifier_dirs)
    assert "error" not in res, res
    assert res["n_images_unattributed"] == 1
    assert res["dates_delivered"] == [DATE]

    rows = list(csv.DictReader(out_csv.open(newline="", encoding="utf-8")))
    assert rows[0]["images_unattributed"] == "1"
    assert rows[0]["dates_delivered"] == DATE

    from tcip_mcp.pipelines import resolution

    scope = resolution.delivery_events_scope(tmp_path)
    keys = ts.keys(resolution.DELIVERY_EVENTS_STORE, str(scope))
    events = [ts.read(k) for k in keys if ts.read(k)["door"] == "deliver_phenology_milestones"]
    pm = events[-1]["plant_mapping"]
    assert pm["images_unattributed"] == 1
    assert pm["dates_delivered"] == [DATE]
    assert pm["images_unattributed_scope"] == "delivered_dates"


def test_a_delivery_naming_one_of_two_mapping_dates_carries_the_delivered_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mapping covering two dates, one delivered: the disclosed count is scoped to the
    delivered date alone, never the mapping's full span."""
    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    dates = ["2026-02-11", "2026-02-25"]
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root, dates=dates)
    _write_ungeoreferenced_image(images_root / dates[1] / "P3_extra.jpg")
    json_io.write_annotations(
        Path(preds_by_date[dates[1]]) / "P3_extra.json",
        [Annotation(subject="flower", geometry=BBox(1.0, 1.0, 3.0, 3.0), score=0.9,
                   attributes={"bloom_state": "open"})], 8, 8)

    registry = register_plant_registry_for([plant_csv])
    build_res = build_plant_mapping(
        name="valley", images_root=str(images_root), plant_registry=registry)
    assert "error" not in build_res, build_res
    assert build_res["n_unattributed"] == 1

    _seed_currant_bloom_trait(tmp_path)
    delivered = {dates[0]: preds_by_date[dates[0]]}
    classifier_dirs = _validate_delivery_buckets(delivered, dataset_root)
    out_csv = tmp_path / "out.csv"
    res = deliver_phenology_milestones(
        trait="currant_bloom", mapping_name="valley", predictions_by_date=delivered,
        output_csv_path=str(out_csv), classifier_pred_dirs=classifier_dirs)
    assert "error" not in res, res
    assert res["n_images_unattributed"] == 0
    assert res["dates_delivered"] == [dates[0]]


def test_a_date_recorded_with_no_capture_still_delivers_beside_an_attributed_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The no-capture-at-all refusal fires only when nothing across the delivered dates
    attributes: a delivery naming a fully attributed date beside a date recorded with no
    capture at all still ships."""
    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root, dates=[DATE])
    empty_date = "2026-02-25"
    (images_root / empty_date).mkdir()
    empty_bucket = dataset_root / "predictions" / "live" / empty_date
    empty_bucket.mkdir(parents=True)

    registry = register_plant_registry_for([plant_csv])
    build_res = build_plant_mapping(
        name="valley", images_root=str(images_root), plant_registry=registry,
        dates=[DATE, empty_date])
    assert "error" not in build_res, build_res
    assert build_res["per_date"][empty_date]["n_images"] == 0

    _seed_currant_bloom_trait(tmp_path)
    delivered = {**preds_by_date, empty_date: str(empty_bucket)}
    classifier_dirs = _validate_delivery_buckets(delivered, dataset_root)
    out_csv = tmp_path / "out.csv"
    res = deliver_phenology_milestones(
        trait="currant_bloom", mapping_name="valley", predictions_by_date=delivered,
        output_csv_path=str(out_csv), classifier_pred_dirs=classifier_dirs)
    assert "error" not in res, res
    assert sorted(res["dates_delivered"]) == sorted([DATE, empty_date])


def test_a_fully_positioned_scene_keeps_delivering_with_zero_unattributed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root, preds_by_date = _delivery_scene(tmp_path, monkeypatch)
    images_root, plant_csv, _ = _write_scene(dataset_root, dates=[DATE])
    registry = register_plant_registry_for([plant_csv])
    build_res = build_plant_mapping(
        name="valley", images_root=str(images_root), plant_registry=registry)
    assert "error" not in build_res, build_res
    assert build_res["n_unattributed"] == 0

    classifier_dirs = _validate_delivery_buckets(preds_by_date, dataset_root)
    out_csv = tmp_path / "out.csv"
    res = deliver_phenology_milestones(
        trait="currant_bloom", mapping_name="valley", predictions_by_date=preds_by_date,
        output_csv_path=str(out_csv), classifier_pred_dirs=classifier_dirs)
    assert "error" not in res, res
    assert res["n_images_unattributed"] == 0


def test_a_raster_beside_positioned_photographs_still_delivers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root, dates=[DATE])
    (images_root / DATE / "orthomosaic_block.tif").write_bytes(b"")

    registry = register_plant_registry_for([plant_csv])
    build_res = build_plant_mapping(
        name="valley", images_root=str(images_root), plant_registry=registry)
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    classifier_dirs = _validate_delivery_buckets(preds_by_date, dataset_root)
    out_csv = tmp_path / "out.csv"
    res = deliver_phenology_milestones(
        trait="currant_bloom", mapping_name="valley", predictions_by_date=preds_by_date,
        output_csv_path=str(out_csv), classifier_pred_dirs=classifier_dirs)
    assert "error" not in res, res
    assert out_csv.exists()


def test_a_capture_at_the_origin_is_admitted_as_positioned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``(0.0, 0.0)`` is a real GPS position (off the coast of west Africa), never treated as
    the absence of one: the build's own position check is ``is not None``, not truthiness."""
    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    images_root = dataset_root / "images"
    _write_geo_image(images_root / DATE / "P1_a.jpg", 0.0, 0.0, datetime(2026, 2, 11, 9, 30))
    plant_csv = tmp_path / "plants.csv"
    _write_plant_csv(plant_csv, [{"plot": "P1", "accession": "acc-A", "lat": 0.0, "lon": 0.0}])

    registry = register_plant_registry_for([plant_csv])
    res = build_plant_mapping(
        name="valley", images_root=str(images_root), plant_registry=registry)
    assert "error" not in res, res
    assert res["n_mapped"] == 1

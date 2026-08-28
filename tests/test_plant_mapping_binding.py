"""Rails for the plant-mapping binding family: the record is bound to its inputs, to a
registered dataset by minted id, and to the receipt that proves it was written by the
platform's own producers, never a hand-filed file. Every scenario is built through the
platform's own producers (init_project, register_dataset, build_plant_mapping,
compute_phenology), a second registered trait rather than the pilot's, so nothing here
generalizes from one trait's own vocabulary.
"""

from __future__ import annotations

import csv
import os
import shutil
import threading
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from PIL import Image

import tcip_store as ts
from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox
from tcip_mcp.pipelines.postprocessing import plant_mapping
from tcip_mcp.tools.phenology_tools import build_plant_mapping, compute_phenology
from tcip_mcp.tools.project_tools import init_project, register_dataset
from tcip_mcp.traits import registered_crops

from tests.test_second_trait_acceptance import _ID_MAP, _seed_currant_bloom_trait

PLANTS = [
    {"plot": "P1", "accession": "acc-A", "lat": 43.19670, "lon": -90.058000},
    {"plot": "P2", "accession": "acc-B", "lat": 43.19670, "lon": -90.058037},
]
DATES = ["2026-02-11", "2026-02-25"]


def _init(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # tmp_path sits directly under this test's workspace; point the workspace elsewhere so
    # init_project's naming rail (which only holds under the workspace) doesn't apply here.
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    result = init_project(str(tmp_path), site="orchard block")
    assert "error" not in result, result


def _dataset(tmp_path: Path, name: str = "ds") -> Path:
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    result = register_dataset(str(root), crop=sorted(registered_crops())[0], project_root=str(tmp_path))
    assert "error" not in result, result
    return root


def _deg_to_dms(value: float) -> tuple[float, float, float]:
    v = abs(value)
    d = int(v)
    m_full = (v - d) * 60
    m = int(m_full)
    s = round((m_full - m) * 60, 4)
    return (float(d), float(m), s)


def _write_geo_image(path: Path, lat: float, lon: float, when: datetime) -> None:
    """A tiny JPEG carrying EXIF DateTimeOriginal + GPS lat/lon (as plant_mapping reads)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    exif[0x8769] = {0x9003: when.strftime("%Y:%m:%d %H:%M:%S")}
    exif[0x8825] = {
        0x0001: "N" if lat >= 0 else "S", 0x0002: _deg_to_dms(lat),
        0x0003: "E" if lon >= 0 else "W", 0x0004: _deg_to_dms(lon),
    }
    Image.new("RGB", (8, 8)).save(path, exif=exif)


def _write_scene(dataset_root: Path, *, dates: list[str] = DATES) -> tuple[Path, Path, dict[str, str]]:
    """Real geolocated images for two plants across ``dates``, plus matching classified
    prediction buckets (id_map only, unvalidated: these rails are about the mapping's own
    binding, not the measurement-validity gate). Returns (images_root, plant_csv, preds_by_date).
    """
    from tcip_mcp.pipelines.resolution import write_sidecar

    images_root = dataset_root / "images"
    preds_root = dataset_root / "predictions" / "live"
    preds_by_date: dict[str, str] = {}
    for date in dates:
        base_time = datetime.strptime(date, "%Y-%m-%d").replace(hour=9, minute=30)
        bucket = preds_root / date
        bucket.mkdir(parents=True, exist_ok=True)
        for j, plant in enumerate(PLANTS):
            stem = f"{plant['plot']}_{date.replace('-', '')}"
            _write_geo_image(
                images_root / date / f"{stem}.jpg", plant["lat"], plant["lon"],
                base_time + timedelta(minutes=j))
            json_io.write_annotations(
                bucket / f"{stem}.json",
                [Annotation(subject="open", geometry=BBox(1.0, 1.0, 3.0, 3.0), score=0.9)], 8, 8)
        write_sidecar(bucket, {"id_map": _ID_MAP}, "operating_point")
        preds_by_date[date] = str(bucket)

    plant_csv = dataset_root.parent / f"{dataset_root.name}_plants.csv"
    with plant_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["plot_name", "accession_name", "WGS84_centroid_x", "WGS84_centroid_y"])
        for p in PLANTS:
            w.writerow([p["plot"], p["accession"], p["lon"], p["lat"]])
    return images_root, plant_csv, preds_by_date


# ── rail 6: no project record ────────────────────────────────────────────


def test_build_plant_mapping_under_no_project_record_names_init_project(tmp_path: Path) -> None:
    images_root = tmp_path / "images"
    (images_root / DATES[0]).mkdir(parents=True)
    res = build_plant_mapping(name="valley", images_root=str(images_root), plant_csv_paths=[])
    assert "error" in res
    assert "init_project" in res["error"]


# ── rail 4: dataset identity, NAME_SEGMENT, variously-spelled roots, dataset mismatch ────


def test_build_plant_mapping_refuses_a_name_outside_name_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path, monkeypatch)
    res = build_plant_mapping(name="Not Legal!", images_root=str(tmp_path), plant_csv_paths=[])
    assert "error" in res
    assert "lowercase" in res["error"]


def test_build_plant_mapping_over_an_unregistered_images_dir_names_register_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path, monkeypatch)
    images_root = tmp_path / "ds" / "images"
    (images_root / DATES[0]).mkdir(parents=True)
    res = build_plant_mapping(name="valley", images_root=str(images_root), plant_csv_paths=[])
    assert "error" in res
    assert "register_dataset" in res["error"]
    assert not (tmp_path / ".tcip" / "state" / "plant_mappings").exists()


def test_build_plant_mapping_admits_the_dataset_images_root_spelled_variously(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, _ = _write_scene(dataset_root)

    trailing = str(images_root) + os.sep
    res = build_plant_mapping(name="trailing-sep", images_root=trailing, plant_csv_paths=[str(plant_csv)])
    assert "error" not in res, res

    forward = str(images_root).replace("\\", "/")
    res = build_plant_mapping(name="forward-slash", images_root=forward, plant_csv_paths=[str(plant_csv)])
    assert "error" not in res, res

    monkeypatch.chdir(dataset_root)
    res = build_plant_mapping(name="relative", images_root="images", plant_csv_paths=[str(plant_csv)])
    assert "error" not in res, res


def test_compute_phenology_refuses_predictions_from_a_different_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, _ = _write_scene(dataset_root)
    build_res = build_plant_mapping(
        name="valley", images_root=str(images_root), plant_csv_paths=[str(plant_csv)])
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    other_root = _dataset(tmp_path, name="ds2")
    _, _, other_preds = _write_scene(other_root)

    res = compute_phenology(
        trait="currant_bloom", mapping_name="valley", predictions_by_date=other_preds,
        output_csv_path=str(tmp_path / "out.csv"), acknowledge_unvalidated=True)
    assert "error" in res
    assert "different dataset" in res["error"]


# ── rail 5: an extra predictions_by_date date the mapping does not name ─────────────────


def test_compute_phenology_refuses_a_date_the_mapping_does_not_cover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root, dates=[DATES[0]])
    build_res = build_plant_mapping(
        name="valley", images_root=str(images_root), plant_csv_paths=[str(plant_csv)])
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    extra_date = "2026-03-01"
    extra = dataset_root / "predictions" / "live" / extra_date
    extra.mkdir(parents=True)
    json_io.write_annotations(
        extra / "P1_20260301.json",
        [Annotation(subject="open", geometry=BBox(1.0, 1.0, 3.0, 3.0), score=0.9)], 8, 8)
    from tcip_mcp.pipelines.resolution import write_sidecar

    write_sidecar(extra, {"id_map": _ID_MAP}, "operating_point")
    preds_by_date[extra_date] = str(extra)

    res = compute_phenology(
        trait="currant_bloom", mapping_name="valley", predictions_by_date=preds_by_date,
        output_csv_path=str(tmp_path / "out.csv"), acknowledge_unvalidated=True)
    assert "error" in res
    assert extra_date in res["error"]


# ── rail 1: a hand-written record refuses, missing provenance or missing receipt ────────


def test_compute_phenology_refuses_a_hand_written_record_missing_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    _, _, preds_by_date = _write_scene(dataset_root)
    _seed_currant_bloom_trait(tmp_path)

    ts.replace(plant_mapping.plant_mapping_key(tmp_path, "forged"), {
        "assignments": {d: [] for d in DATES},
    })
    out_csv = tmp_path / "out.csv"
    res = compute_phenology(
        trait="currant_bloom", mapping_name="forged", predictions_by_date=preds_by_date,
        output_csv_path=str(out_csv), acknowledge_unvalidated=True)
    assert "error" in res
    assert "is missing" in res["error"]
    assert not out_csv.exists()


def test_compute_phenology_refuses_a_record_with_provenance_and_no_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    _, _, preds_by_date = _write_scene(dataset_root)
    _seed_currant_bloom_trait(tmp_path)

    build = plant_mapping.MappingBuild(
        name="forged", project_root=str(tmp_path), dataset_root=str(dataset_root),
        dataset_id="whatever-id", built_by="build_plant_mapping",
        built_at="2026-02-11T00:00:00+00:00", dates_requested=None, dates=list(DATES),
        nn_tolerance_m={"value": 10.0, "source": "fallback"}, plant_csvs=[],
        capture_identity={d: "0" * 16 for d in DATES}, unreadable={d: [] for d in DATES},
        assignments={d: [] for d in DATES},
    )
    ts.replace(plant_mapping.plant_mapping_key(tmp_path, "forged"), build.to_record())
    out_csv = tmp_path / "out.csv"
    res = compute_phenology(
        trait="currant_bloom", mapping_name="forged", predictions_by_date=preds_by_date,
        output_csv_path=str(out_csv), acknowledge_unvalidated=True)
    assert "error" in res
    assert "receipt" in res["error"]
    assert not out_csv.exists()


# ── rail 3: a plant CSV rewritten in place refuses, naming the file ─────────────────────


def test_compute_phenology_refuses_a_plant_csv_rewritten_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root)
    build_res = build_plant_mapping(
        name="valley", images_root=str(images_root), plant_csv_paths=[str(plant_csv)])
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    plant_csv.write_text(
        "plot_name,accession_name,WGS84_centroid_x,WGS84_centroid_y\n"
        "P1,acc-Z,-90.058000,43.19670\n", encoding="utf-8")

    out_csv = tmp_path / "out.csv"
    res = compute_phenology(
        trait="currant_bloom", mapping_name="valley", predictions_by_date=preds_by_date,
        output_csv_path=str(out_csv), acknowledge_unvalidated=True)
    assert "error" in res
    assert str(plant_csv) in res["error"]
    assert not out_csv.exists()


# ── rail 7: a readability flip in either direction refuses, naming the file ─────────────


def test_a_capture_unreadable_at_build_and_readable_at_verify_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root, dates=[DATES[0]])
    bad = images_root / DATES[0] / "bad.jpg"
    bad.write_bytes(b"not a real jpeg")

    build_res = build_plant_mapping(
        name="valley", images_root=str(images_root), plant_csv_paths=[str(plant_csv)])
    assert "error" not in build_res, build_res
    assert "bad.jpg" in build_res["unreadable"][DATES[0]]
    _seed_currant_bloom_trait(tmp_path)

    _write_geo_image(bad, 43.1968, -90.0581, datetime(2026, 2, 11, 9, 5))
    out_csv = tmp_path / "out.csv"
    res = compute_phenology(
        trait="currant_bloom", mapping_name="valley", predictions_by_date=preds_by_date,
        output_csv_path=str(out_csv), acknowledge_unvalidated=True)
    assert "error" in res
    assert "bad.jpg" in res["error"]
    assert not out_csv.exists()


def test_a_capture_readable_at_build_and_unreadable_at_verify_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root, dates=[DATES[0]])
    build_res = build_plant_mapping(
        name="valley", images_root=str(images_root), plant_csv_paths=[str(plant_csv)])
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    stem = f"{PLANTS[0]['plot']}_{DATES[0].replace('-', '')}"
    target = images_root / DATES[0] / f"{stem}.jpg"
    target.write_bytes(b"corrupted after the build")

    out_csv = tmp_path / "out.csv"
    res = compute_phenology(
        trait="currant_bloom", mapping_name="valley", predictions_by_date=preds_by_date,
        output_csv_path=str(out_csv), acknowledge_unvalidated=True)
    assert "error" in res
    assert target.name in res["error"]
    assert not out_csv.exists()


# ── rail 8: a receipt that cannot be written fails loudly and the record stays refused ──


def test_a_receipt_that_cannot_be_written_fails_persist_mapping_and_the_record_stays_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At the pipeline level: ``persist_mapping``'s own contract, not the MCP tool's ``@audited``
    wrapper (which writes its own, separate audit line for the call and would otherwise also
    contend for the same lock this test holds for the whole body)."""
    from tcip_mcp.audit import AuditEntryNotWritten
    from tcip_store.file_backend import FileBackend, path_lock

    # Bound before anything is written, so this root's state is file-backed throughout: the
    # lock below has to guard the exact file the receipt append writes to.
    ts.bind(FileBackend(lock_timeout_s=0.2))
    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, _ = _write_scene(dataset_root)
    build = plant_mapping.build_mapping(
        images_root, [plant_csv], name="valley", dataset_root=dataset_root,
        dataset_id="whatever-id", project_root=tmp_path, built_by="build_plant_mapping")

    audit_path = tmp_path / ".tcip" / "audit.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    holding, release = threading.Event(), threading.Event()

    def hold() -> None:
        with path_lock(audit_path, timeout_s=30):
            holding.set()
            release.wait(30)

    holder = threading.Thread(target=hold)
    holder.start()
    try:
        assert holding.wait(30)
        with pytest.raises(AuditEntryNotWritten, match="plant_mapping_built"):
            plant_mapping.persist_mapping(build, tmp_path, "valley")
    finally:
        release.set()
        holder.join(30)

    with pytest.raises(ValueError, match="receipt"):
        plant_mapping.load_mapping(tmp_path, "valley")


def test_the_web_build_route_answers_409_when_the_receipt_cannot_be_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi.testclient import TestClient
    from tcip_store.file_backend import FileBackend, path_lock
    from tcip_web.app import app
    from tcip_web.state import store

    ts.bind(FileBackend(lock_timeout_s=0.2))
    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, _ = _write_scene(dataset_root)
    store.open_project(tmp_path.resolve())

    audit_path = tmp_path / ".tcip" / "audit.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    holding, release = threading.Event(), threading.Event()

    def hold() -> None:
        with path_lock(audit_path, timeout_s=30):
            holding.set()
            release.wait(30)

    holder = threading.Thread(target=hold)
    holder.start()
    try:
        assert holding.wait(30)
        client = TestClient(app, base_url="http://127.0.0.1")
        resp = client.post("/api/results/plant_mapping/build", json={
            "name": "valley", "images_root": str(images_root), "plant_csv_paths": [str(plant_csv)],
        })
        assert resp.status_code == 409, resp.text
        assert "plant_mapping_built" in resp.json()["detail"]
    finally:
        release.set()
        holder.join(30)


# ── rails 9, 10: the full round trip through the platform's own producers ───────────────


def test_full_round_trip_delivers_and_a_rebuild_reads_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root)

    build_res = build_plant_mapping(
        name="valley", images_root=str(images_root), plant_csv_paths=[str(plant_csv)])
    assert "error" not in build_res, build_res
    assert build_res["n_dates"] == len(DATES)

    build = plant_mapping.load_mapping(tmp_path, "valley")
    assert build is not None
    assert build.name == "valley"
    assert set(build.dates) == set(DATES)
    for date in DATES:
        assert len(build.assignments[date]) == len(PLANTS)
        assert {a.plot_name for a in build.assignments[date]} == {p["plot"] for p in PLANTS}

    _seed_currant_bloom_trait(tmp_path)
    out_csv = tmp_path / "out" / "bloom_phenology.csv"
    res = compute_phenology(
        trait="currant_bloom", mapping_name="valley", predictions_by_date=preds_by_date,
        output_csv_path=str(out_csv), acknowledge_unvalidated=True)
    assert "error" not in res, res
    assert out_csv.exists()

    from tcip_mcp.pipelines import resolution

    scope = resolution.delivery_events_scope(tmp_path)
    keys = ts.keys(resolution.DELIVERY_EVENTS_STORE, str(scope))
    events = [ts.read(k) for k in keys if ts.read(k)["door"] == "compute_phenology"]
    assert len(events) == 1, events
    pm = events[0]["plant_mapping"]
    assert pm["name"] == "valley"
    assert pm["project_root"] == str(tmp_path)
    assert pm["record_sha256"] == build.record_sha256
    assert set(pm["capture_identity"].keys()) == set(DATES)

    # A rebuild under the same name reads back the rebuild and delivers.
    build_res2 = build_plant_mapping(
        name="valley", images_root=str(images_root), plant_csv_paths=[str(plant_csv)])
    assert "error" not in build_res2, build_res2
    out_csv2 = tmp_path / "out2" / "bloom_phenology.csv"
    res2 = compute_phenology(
        trait="currant_bloom", mapping_name="valley", predictions_by_date=preds_by_date,
        output_csv_path=str(out_csv2), acknowledge_unvalidated=True)
    assert "error" not in res2, res2


# ── rail 12: a moved plant CSV and an archived date deliver, disclosed rather than refused


def test_a_moved_plant_csv_and_an_archived_date_deliver_with_disclosures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root)
    build_res = build_plant_mapping(
        name="valley", images_root=str(images_root), plant_csv_paths=[str(plant_csv)])
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    moved = tmp_path / "plants_moved.csv"
    plant_csv.rename(moved)
    shutil.rmtree(images_root / DATES[0])

    out_csv = tmp_path / "out.csv"
    res = compute_phenology(
        trait="currant_bloom", mapping_name="valley", predictions_by_date=preds_by_date,
        output_csv_path=str(out_csv), acknowledge_unvalidated=True)
    assert "error" not in res, res
    assert out_csv.exists()

    from tcip_mcp.pipelines import resolution

    scope = resolution.delivery_events_scope(tmp_path)
    keys = ts.keys(resolution.DELIVERY_EVENTS_STORE, str(scope))
    events = [ts.read(k) for k in keys if ts.read(k)["door"] == "compute_phenology"]
    pm = events[-1]["plant_mapping"]
    assert str(plant_csv) in pm["plant_csvs_unverified"]
    assert DATES[0] in pm["captures_unverified"]


# ── rail 15: the build route's happy path, and a moved+re-registered dataset still binds ─


def test_a_moved_and_re_registered_dataset_still_delivers_through_the_earlier_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root)
    build_res = build_plant_mapping(
        name="valley", images_root=str(images_root), plant_csv_paths=[str(plant_csv)])
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    # Copied, not moved: a live sqlite handle under the tree can hold a Windows file lock a
    # real rename would trip over; id preservation only needs dataset.json at the new root.
    moved_root = tmp_path / "ds_moved"
    shutil.copytree(str(dataset_root), str(moved_root))
    reg = register_dataset(str(moved_root), crop=sorted(registered_crops())[0], project_root=str(tmp_path))
    assert "error" not in reg, reg
    original = plant_mapping.load_mapping(tmp_path, "valley")
    assert original is not None
    assert reg["id"] == original.dataset_id, "register_dataset must preserve the id across the move"

    moved_preds = {d: str(moved_root / "predictions" / "live" / d) for d in preds_by_date}
    out_csv = tmp_path / "out.csv"
    res = compute_phenology(
        trait="currant_bloom", mapping_name="valley", predictions_by_date=moved_preds,
        output_csv_path=str(out_csv), acknowledge_unvalidated=True)
    assert "error" not in res, res
    assert out_csv.exists()


# ── rail 13 (listing only): legal names are listed, an illegally-named stray is not ─────


def test_plant_mapping_names_lists_legal_names_and_omits_a_stray_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, _ = _write_scene(dataset_root)
    res_a = build_plant_mapping(
        name="valley-a", images_root=str(images_root), plant_csv_paths=[str(plant_csv)])
    assert "error" not in res_a, res_a
    res_b = build_plant_mapping(
        name="valley-b", images_root=str(images_root), plant_csv_paths=[str(plant_csv)])
    assert "error" not in res_b, res_b

    # A record written straight through the store, bypassing the door's NAME_SEGMENT check,
    # stands in for a stray file under either backend.
    ts.replace(plant_mapping.plant_mapping_key(tmp_path, "Not A Legal Name"), {})

    assert plant_mapping.plant_mapping_names(tmp_path) == ["valley-a", "valley-b"]


# ── rail 2: a capture added under a mapped date, through the platform's own writers ─────


def test_an_image_ingested_under_a_mapped_date_refuses_the_delivery_naming_the_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tcip_mcp.tools.ingest_tools import ingest_images

    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root, dates=[DATES[0]])
    build_res = build_plant_mapping(
        name="valley", images_root=str(images_root), plant_csv_paths=[str(plant_csv)])
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    extra_source = tmp_path / "extra_source"
    _write_geo_image(extra_source / "P3_extra.jpg", 43.1968, -90.0581, datetime(2026, 2, 11, 9, 40))
    res = ingest_images(
        source=str(extra_source), name="ds", site="orchard block",
        project_path=str(dataset_root), date_from=DATES[0])
    assert "error" not in res, res
    assert res["buckets"].get(DATES[0]) == 1

    out_csv = tmp_path / "out.csv"
    res = compute_phenology(
        trait="currant_bloom", mapping_name="valley", predictions_by_date=preds_by_date,
        output_csv_path=str(out_csv), acknowledge_unvalidated=True)
    assert "error" in res
    assert DATES[0] in res["error"]
    assert not out_csv.exists()


def test_a_band_group_written_under_a_mapped_date_refuses_the_delivery_the_same_way(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No band group existed at build time; ``detect_and_write_band_groups`` forms one from two
    previously-standalone files, so the date's capture set changes shape (two stems collapse
    into one) the same way an added image does: refused, naming the date, through the same
    whole-date identity check, not the manifest-specific wording (which needs the old build's
    own recorded manifest digest and members, not just its name; that disambiguation is not
    implemented here, see the report).
    """
    from tcip_mcp.pipelines.data.band_groups import detect_and_write_band_groups

    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root, dates=[DATES[0]])
    date_dir = images_root / DATES[0]
    _write_geo_image(date_dir / "aux_b1.jpg", 43.1968, -90.0581, datetime(2026, 2, 11, 9, 41))
    _write_geo_image(date_dir / "aux_b2.jpg", 43.1968, -90.0581, datetime(2026, 2, 11, 9, 42))

    build_res = build_plant_mapping(
        name="valley", images_root=str(images_root), plant_csv_paths=[str(plant_csv)])
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)

    grouped = detect_and_write_band_groups(
        date_dir, explicit_groups={"aux": {"b1": "aux_b1.jpg", "b2": "aux_b2.jpg"}})
    assert grouped["formed"], grouped

    out_csv = tmp_path / "out.csv"
    res = compute_phenology(
        trait="currant_bloom", mapping_name="valley", predictions_by_date=preds_by_date,
        output_csv_path=str(out_csv), acknowledge_unvalidated=True)
    assert "error" in res
    assert DATES[0] in res["error"]
    assert not out_csv.exists()


# ── rail 11: an unreadable image, a raster and a band group all build and deliver ───────


def test_a_date_with_an_unreadable_image_a_raster_and_a_band_group_builds_and_delivers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tcip_mcp.pipelines.data.band_groups import detect_and_write_band_groups
    from tcip_mcp.pipelines.image_utils import list_logical_images
    from tcip_mcp.pipelines.postprocessing.plant_mapping import _read_date_stamps

    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, preds_by_date = _write_scene(dataset_root, dates=[DATES[0]])
    date_dir = images_root / DATES[0]

    bad = date_dir / "bad.jpg"
    bad.write_bytes(b"not a real jpeg")
    (date_dir / "raster.npy").write_bytes(b"not really a numpy array, list_logical_images "
                                           b"never decodes it")
    _write_geo_image(date_dir / "aux_b1.jpg", 43.1968, -90.0581, datetime(2026, 2, 11, 9, 41))
    _write_geo_image(date_dir / "aux_b2.jpg", 43.1968, -90.0581, datetime(2026, 2, 11, 9, 42))
    grouped = detect_and_write_band_groups(
        date_dir, explicit_groups={"aux": {"b1": "aux_b1.jpg", "b2": "aux_b2.jpg"}})
    assert grouped["formed"], grouped

    # White-box: the raster and band-group stamps carry readable=None (no EXIF to fail reading),
    # the unreadable image carries readable=False, before build_mapping ever touches a store.
    logical = list_logical_images(date_dir)
    stamps_by_stem = {s.stem: s for s in _read_date_stamps(logical, DATES[0])}
    assert stamps_by_stem["bad"].kind == "image" and stamps_by_stem["bad"].readable is False
    assert stamps_by_stem["raster"].kind == "raster" and stamps_by_stem["raster"].readable is None
    assert stamps_by_stem["aux"].kind == "band_group" and stamps_by_stem["aux"].readable is None

    build_res = build_plant_mapping(
        name="valley", images_root=str(images_root), plant_csv_paths=[str(plant_csv)])
    assert "error" not in build_res, build_res
    assert build_res["unreadable"][DATES[0]] == ["bad.jpg"]
    _seed_currant_bloom_trait(tmp_path)

    build = plant_mapping.load_mapping(tmp_path, "valley")
    assert build is not None
    by_stem = {a.stem: a for a in build.assignments[DATES[0]]}
    assert by_stem["bad"].source == "unmapped"
    assert by_stem["raster"].source == "unmapped"
    assert by_stem["aux"].source == "unmapped"

    out_csv = tmp_path / "out.csv"
    res = compute_phenology(
        trait="currant_bloom", mapping_name="valley", predictions_by_date=preds_by_date,
        output_csv_path=str(out_csv), acknowledge_unvalidated=True)
    assert "error" not in res, res
    assert out_csv.exists()


# ── rail 14 (cursor memo): the second receipt scan in one process reads only what was ───
# ── appended since the first, never the whole log again ─────────────────────────────────


def test_second_receipt_scan_in_one_process_reads_only_what_was_appended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path, monkeypatch)
    dataset_root = _dataset(tmp_path)
    images_root, plant_csv, _ = _write_scene(dataset_root)

    calls: list[str | None] = []
    real_read_log = ts.read_log

    def spy(key, after=None):
        calls.append(after)
        return real_read_log(key, after=after)

    monkeypatch.setattr(ts, "read_log", spy)

    res_a = build_plant_mapping(
        name="valley-a", images_root=str(images_root), plant_csv_paths=[str(plant_csv)])
    assert "error" not in res_a, res_a
    build_a = plant_mapping.load_mapping(tmp_path, "valley-a")
    assert build_a is not None
    assert calls == [None]

    res_b = build_plant_mapping(
        name="valley-b", images_root=str(images_root), plant_csv_paths=[str(plant_csv)])
    assert "error" not in res_b, res_b
    build_b = plant_mapping.load_mapping(tmp_path, "valley-b")
    assert build_b is not None
    assert len(calls) == 2
    assert calls[1] is not None

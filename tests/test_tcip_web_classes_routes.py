"""Tests for the class / per-image-status routes."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_annotation import json_io
from tcip_annotation.state import BBox
from tcip_web.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_load_empty_registry(client: TestClient, tmp_path: Path) -> None:
    resp = client.get("/api/classes/load", params={"project_root": str(tmp_path)})
    assert resp.status_code == 200
    assert resp.json() == {"classes": []}


def test_save_then_load_round_trip(client: TestClient, tmp_path: Path) -> None:
    save = client.post(
        "/api/classes/save",
        json={
            "project_root": str(tmp_path),
            "classes": [
                {"id": 0, "name": "catkin", "color": "#FF0000"},
                {"id": 1, "name": "bud", "color": "#00FFFF"},
            ],
        },
    )
    assert save.status_code == 200
    assert save.json()["n_classes"] == 2

    load = client.get("/api/classes/load", params={"project_root": str(tmp_path)}).json()
    assert len(load["classes"]) == 2
    assert load["classes"][0] == {"id": 0, "name": "catkin", "color": "#FF0000"}
    # File on disk matches
    on_disk = json.loads((tmp_path / ".tcip" / "state" / "classes.json").read_text())
    assert on_disk["0"]["name"] == "catkin"


def test_per_trait_maps_are_scoped(client: TestClient, tmp_path: Path) -> None:
    # catkin and bush each use class 0 for their own object — trait-scoped maps keep them apart.
    for trait, name in (("catkin", "catkin"), ("bush", "bush")):
        r = client.post(
            "/api/classes/save",
            json={
                "project_root": str(tmp_path),
                "trait": trait,
                "classes": [{"id": 0, "name": name, "color": "#FF0000"}],
            },
        )
        assert r.status_code == 200

    cat = client.get(
        "/api/classes/load", params={"project_root": str(tmp_path), "trait": "catkin"}
    ).json()
    bush = client.get(
        "/api/classes/load", params={"project_root": str(tmp_path), "trait": "bush"}
    ).json()
    assert cat["classes"][0]["name"] == "catkin"
    assert bush["classes"][0]["name"] == "bush"
    # each map lands under its own per-trait path
    assert (tmp_path / ".tcip" / "state" / "classes" / "catkin.json").is_file()
    assert (tmp_path / ".tcip" / "state" / "classes" / "bush.json").is_file()


def test_load_derives_from_labels_when_map_absent(client: TestClient, tmp_path: Path) -> None:
    # No saved map for this trait, but its labels exist → derive a provisional registry so the
    # canvas never loads empty (names default to class_<id>).
    det = tmp_path / "annotations" / "catkin" / "d" / "detect"
    det.mkdir(parents=True)
    json_io.write_detect(
        str(det / "IMG_A.json"),
        [BBox(50.0, 50.0, 60.0, 60.0, 0), BBox(20.0, 20.0, 30.0, 30.0, 2)],
        100,
        100,
    )
    load = client.get(
        "/api/classes/load",
        params={"project_root": str(tmp_path), "trait": "catkin", "annotations_detect_dir": str(det)},
    ).json()
    assert [c["id"] for c in load["classes"]] == [0, 2]
    assert load["classes"][0]["name"] == "class_0"


def test_load_rejects_invalid_trait(client: TestClient, tmp_path: Path) -> None:
    resp = client.get(
        "/api/classes/load", params={"project_root": str(tmp_path), "trait": "../evil"}
    )
    assert resp.status_code == 400


def test_auto_color_is_deterministic(client: TestClient) -> None:
    c0 = client.get("/api/classes/auto_color/0").json()
    c10 = client.get("/api/classes/auto_color/10").json()
    assert c0["color"] == c10["color"]  # wraps every 10


def test_image_status_round_trip(client: TestClient, tmp_path: Path) -> None:
    client.post(
        "/api/classes/image_status",
        json={"project_root": str(tmp_path), "image_name": "IMG_0001.JPG", "status": "complete"},
    )
    resp = client.get("/api/classes/image_status", params={"project_root": str(tmp_path)})
    body = resp.json()
    assert body["statuses"]["IMG_0001.JPG"] == "complete"


def test_image_status_rejects_invalid(client: TestClient, tmp_path: Path) -> None:
    resp = client.post(
        "/api/classes/image_status",
        json={"project_root": str(tmp_path), "image_name": "x", "status": "bogus"},
    )
    assert resp.status_code == 400


def test_derive_statuses_negatives_are_intentional(client: TestClient, tmp_path: Path) -> None:
    # A negative is intentional (completed + empty), never inferred from an empty file alone —
    # so an accidental empty file, or flipping through an image, doesn't become a training negative.
    det = tmp_path / "detect"
    det.mkdir()
    json_io.write_detect(str(det / "IMG_A.json"), [BBox(50.0, 50.0, 60.0, 60.0, 0)], 100, 100)  # has objects, not completed
    json_io.write_detect(str(det / "IMG_B.json"), [], 100, 100, keep_empty=True)  # present {"objects": []}, not completed
    json_io.write_detect(str(det / "IMG_E.json"), [BBox(50.0, 50.0, 60.0, 60.0, 0)], 100, 100)  # has objects, completed
    # IMG_C.json missing; IMG_D completed but has no file (empty)

    resp = client.post(
        "/api/classes/image_status/derive",
        json={
            "project_root": str(tmp_path),
            "annotations_detect_dir": str(det),
            "annotations_segment_dir": None,
            "image_list": ["IMG_A.JPG", "IMG_B.JPG", "IMG_C.JPG", "IMG_D.JPG", "IMG_E.JPG"],
            "complete_override": ["IMG_D.JPG", "IMG_E.JPG"],
        },
    )
    assert resp.json()["statuses"] == {
        "IMG_A.JPG": "partial",  # has objects, not completed
        "IMG_B.JPG": "unannotated",  # empty file alone is not a negative — needs review
        "IMG_C.JPG": "unannotated",  # no file
        "IMG_D.JPG": "negative",  # completed + empty → confirmed negative (intentional)
        "IMG_E.JPG": "complete",  # completed + has objects
    }


def test_derive_statuses_cache_invalidates_on_label_write(client: TestClient, tmp_path: Path) -> None:
    # The per-file label-JSON memo is keyed on mtime_ns — a write after a prior derive() call
    # must not serve the stale (pre-write) parse of the same path.
    det = tmp_path / "detect"
    det.mkdir()
    label = det / "IMG_A.json"
    json_io.write_detect(str(label), [], 100, 100, keep_empty=True)
    os.utime(label, (1_000_000, 1_000_000))

    req = {
        "project_root": str(tmp_path),
        "annotations_detect_dir": str(det),
        "annotations_segment_dir": None,
        "image_list": ["IMG_A.JPG"],
        "complete_override": [],
    }
    first = client.post("/api/classes/image_status/derive", json=req).json()
    assert first["statuses"]["IMG_A.JPG"] == "unannotated"

    json_io.write_detect(str(label), [BBox(50.0, 50.0, 60.0, 60.0, 0)], 100, 100)
    os.utime(label, (2_000_000, 2_000_000))  # force a distinct mtime_ns from the first write

    second = client.post("/api/classes/image_status/derive", json=req).json()
    assert second["statuses"]["IMG_A.JPG"] == "partial"


def test_load_derived_registry_cache_invalidates_on_label_write(
    client: TestClient, tmp_path: Path
) -> None:
    # Same memo, exercised through load_classes' label-derived registry path.
    det = tmp_path / "annotations" / "catkin" / "d" / "detect"
    det.mkdir(parents=True)
    label = det / "IMG_A.json"
    json_io.write_detect(str(label), [BBox(50.0, 50.0, 60.0, 60.0, 0)], 100, 100)
    os.utime(label, (1_000_000, 1_000_000))

    params = {"project_root": str(tmp_path), "trait": "catkin", "annotations_detect_dir": str(det)}
    first = client.get("/api/classes/load", params=params).json()
    assert [c["id"] for c in first["classes"]] == [0]

    json_io.write_detect(
        str(label), [BBox(50.0, 50.0, 60.0, 60.0, 0), BBox(20.0, 20.0, 30.0, 30.0, 2)], 100, 100
    )
    os.utime(label, (2_000_000, 2_000_000))

    second = client.get("/api/classes/load", params=params).json()
    assert [c["id"] for c in second["classes"]] == [0, 2]


def test_image_status_bulk(client: TestClient, tmp_path: Path) -> None:
    resp = client.post(
        "/api/classes/image_status/bulk",
        json={
            "project_root": str(tmp_path),
            "statuses": {
                "A.JPG": "complete",
                "B.JPG": "partial",
                "C.JPG": "invalid_ignored",
            },
        },
    )
    assert resp.json()["n"] == 3
    loaded = client.get(
        "/api/classes/image_status", params={"project_root": str(tmp_path)}
    ).json()
    assert loaded["statuses"]["A.JPG"] == "complete"
    assert loaded["statuses"]["B.JPG"] == "partial"
    assert "C.JPG" not in loaded["statuses"]  # invalid skipped

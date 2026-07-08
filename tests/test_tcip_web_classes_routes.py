"""Tests for the class / per-image-status routes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

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


def test_derive_statuses_from_label_files(client: TestClient, tmp_path: Path) -> None:
    det = tmp_path / "detect"
    det.mkdir()
    (det / "IMG_A.txt").write_text("0 0.5 0.5 0.1 0.1\n")  # partial
    (det / "IMG_B.txt").write_text("")  # empty file exists = confirmed negative
    # IMG_C.txt missing = unannotated (never looked at)

    resp = client.post(
        "/api/classes/image_status/derive",
        json={
            "project_root": str(tmp_path),
            "annotations_detect_dir": str(det),
            "annotations_segment_dir": None,
            "image_list": ["IMG_A.JPG", "IMG_B.JPG", "IMG_C.JPG", "IMG_D.JPG"],
            "complete_override": ["IMG_D.JPG"],
        },
    )
    body = resp.json()
    assert body["statuses"] == {
        "IMG_A.JPG": "partial",
        "IMG_B.JPG": "negative",
        "IMG_C.JPG": "unannotated",
        "IMG_D.JPG": "complete",
    }


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

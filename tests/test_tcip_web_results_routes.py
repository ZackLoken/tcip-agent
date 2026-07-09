"""Slice 3 tests: plant mapping + per-plant curves + onset dates + CSV export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_web.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_plant_mapping_build_with_empty_images(client: TestClient, tmp_path: Path) -> None:
    resp = client.post(
        "/api/results/plant_mapping/build",
        json={
            "images_root": str(tmp_path / "nope"),
            "plant_csv_paths": [],
            "persist_path": str(tmp_path / "mapping.json"),
        },
    )
    body = resp.json()
    assert body["mapping"] == {}
    assert body["summary"] == {}


def test_plant_mapping_load_missing_returns_empty(client: TestClient, tmp_path: Path) -> None:
    resp = client.post(
        "/api/results/plant_mapping/load",
        json={"persist_path": str(tmp_path / "missing.json")},
    )
    body = resp.json()
    assert body["mapping"] == {}


def test_per_plant_curves_uses_mapping_and_counts(client: TestClient, tmp_path: Path) -> None:
    # Fabricate a mapping + predictions for two dates, two plants
    mapping_path = tmp_path / "mapping.json"
    mapping_data = {
        "2026-02-11": [
            {
                "image_path": "/x/IMG_A.JPG",
                "stem": "IMG_A",
                "date_folder": "2026-02-11",
                "plot_name": "PLANT_A",
                "accession_name": "AccA",
                "confidence": 0.9,
                "source": "sequence",
                "distance_m": 1.0,
            },
            {
                "image_path": "/x/IMG_B.JPG",
                "stem": "IMG_B",
                "date_folder": "2026-02-11",
                "plot_name": "PLANT_B",
                "accession_name": "AccB",
                "confidence": 0.9,
                "source": "sequence",
                "distance_m": 1.0,
            },
        ],
        "2026-03-24": [
            {
                "image_path": "/x/IMG_A2.JPG",
                "stem": "IMG_A2",
                "date_folder": "2026-03-24",
                "plot_name": "PLANT_A",
                "accession_name": "AccA",
                "confidence": 0.9,
                "source": "sequence",
                "distance_m": 1.0,
            },
        ],
    }
    mapping_path.write_text(json.dumps(mapping_data), encoding="utf-8")

    preds_211 = tmp_path / "preds_2-11-26"
    preds_324 = tmp_path / "preds_3-24-26"
    preds_211.mkdir()
    preds_324.mkdir()
    # PLANT_A on 2-11 → 4 detections, none elongated (h < 0.02)
    (preds_211 / "IMG_A.txt").write_text(
        "\n".join("0 0.9 0.5 0.5 0.01 0.01" for _ in range(4)), encoding="utf-8"
    )
    # PLANT_B on 2-11 → 2 detections, both elongated
    (preds_211 / "IMG_B.txt").write_text(
        "0 0.9 0.5 0.5 0.01 0.05\n0 0.9 0.4 0.4 0.01 0.05\n", encoding="utf-8"
    )
    # PLANT_A on 3-24 → 3 detections, all elongated
    (preds_324 / "IMG_A2.txt").write_text(
        "0 0.9 0.5 0.5 0.01 0.05\n0 0.9 0.4 0.4 0.01 0.05\n0 0.9 0.3 0.3 0.01 0.05\n",
        encoding="utf-8",
    )

    resp = client.post(
        "/api/results/per_plant_curves",
        json={
            "project_root": str(tmp_path),
            "mapping_path": str(mapping_path),
            "predictions_by_date": {
                "2026-02-11": str(preds_211),
                "2026-03-24": str(preds_324),
            },
            "elongation_height": 0.02,
        },
    )
    body = resp.json()
    assert body["n_plants"] == 2

    by_key = {(r["plant_id"], r["date"]): r for r in body["rows"]}
    # PLANT_A on 2-11: 4 total, 0 elongated → ratio 0
    assert by_key[("PLANT_A", "2026-02-11")]["ratio"] == 0.0
    assert by_key[("PLANT_A", "2026-02-11")]["n_total"] == 4
    # PLANT_B on 2-11: 2 total, 2 elongated → ratio 1.0
    assert by_key[("PLANT_B", "2026-02-11")]["ratio"] == 1.0
    # PLANT_A on 3-24: 3 total, 3 elongated → ratio 1.0
    assert by_key[("PLANT_A", "2026-03-24")]["ratio"] == 1.0


def test_onset_dates_finds_crossings(client: TestClient) -> None:
    curves = [
        {
            "plant_id": "PLANT_A",
            "accession": "A",
            "date": "2026-02-11",
            "n_images": 1,
            "n_total": 100,
            "n_elongated": 0,
            "ratio": 0.0,
        },
        {
            "plant_id": "PLANT_A",
            "accession": "A",
            "date": "2026-03-02",
            "n_images": 1,
            "n_total": 100,
            "n_elongated": 10,
            "ratio": 0.10,
        },
        {
            "plant_id": "PLANT_A",
            "accession": "A",
            "date": "2026-03-09",
            "n_images": 1,
            "n_total": 100,
            "n_elongated": 60,
            "ratio": 0.60,
        },
        {
            "plant_id": "PLANT_A",
            "accession": "A",
            "date": "2026-03-18",
            "n_images": 1,
            "n_total": 100,
            "n_elongated": 100,
            "ratio": 1.0,
        },
    ]
    resp = client.post("/api/results/onset_dates", json={"curves": curves})
    rows = resp.json()["rows"]
    assert len(rows) == 1
    onset = rows[0]
    assert onset["catkin_05per_date"] is not None  # reached between 2-11 and 3-2
    assert onset["catkin_50per_date"] is not None  # reached between 3-2 and 3-9
    assert onset["catkin_95per_date"] is not None  # reached between 3-9 and 3-18


def test_export_csv(client: TestClient) -> None:
    rows = [
        {"plant_id": "PLANT_A", "catkin_05per_date": "2026-02-15"},
        {"plant_id": "PLANT_B", "catkin_05per_date": "2026-03-05"},
    ]
    resp = client.post(
        "/api/results/export_csv",
        json={"rows": rows, "filename": "test.csv"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    body = resp.text
    lines = body.strip().split("\n")
    assert len(lines) == 3
    assert "plant_id" in lines[0]
    assert "PLANT_A" in lines[1]
    assert "PLANT_B" in lines[2]


def test_export_csv_empty_rejected(client: TestClient) -> None:
    resp = client.post("/api/results/export_csv", json={"rows": [], "filename": "x.csv"})
    assert resp.status_code == 400


def test_inference_launch_missing_checkpoint(client: TestClient, tmp_path: Path) -> None:
    resp = client.post(
        "/api/inference/launch",
        json={
            "checkpoint_path": str(tmp_path / "no.pt"),
            "images_dir": str(tmp_path),
            "output_dir": str(tmp_path / "out"),
        },
    )
    assert resp.status_code == 404


def test_inference_list_jobs_endpoint(client: TestClient) -> None:
    resp = client.get("/api/inference/jobs")
    assert resp.status_code == 200
    assert "jobs" in resp.json()

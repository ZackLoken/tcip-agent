"""Slice 3 tests: plant mapping + per-plant curves + onset dates + CSV export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tcip_annotation.json_io import write_annotations
from tcip_annotation.state import Annotation, BBox

from tcip_web.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _write_preds(path: Path, n_detections: int) -> None:
    """Per-image JSON prediction file with ``n_detections`` name-based catkin detections.

    Elongation is now a name-based *attribute*, not an integer class, and the attribute→fraction
    bridge is deferred to K4/K5 — so a prediction here is just a detected ``catkin`` (no elongation
    split), which is exactly what the degenerate count reports until that bridge is wired.
    """
    anns = [Annotation(subject="catkin", geometry=BBox(1.0, 1.0, 3.0, 3.0), score=0.9)
            for _ in range(n_detections)]
    write_annotations(str(path), anns, 8, 8)


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
    # Detection totals per (plant, date). Elongation is a name-based attribute whose fraction bridge
    # is deferred (K4/K5), so the endpoint counts detections and reports no elongation split rather
    # than fabricating a bloom fraction from unwired predictions.
    _write_preds(preds_211 / "IMG_A.json", 4)   # PLANT_A on 2-11 → 4 detections
    _write_preds(preds_211 / "IMG_B.json", 2)   # PLANT_B on 2-11 → 2 detections
    _write_preds(preds_324 / "IMG_A2.json", 3)  # PLANT_A on 3-24 → 3 detections

    resp = client.post(
        "/api/results/per_plant_curves",
        json={
            "project_root": str(tmp_path),
            "mapping_path": str(mapping_path),
            "predictions_by_date": {
                "2026-02-11": str(preds_211),
                "2026-03-24": str(preds_324),
            },
        },
    )
    body = resp.json()
    assert body["n_plants"] == 2
    # No elongation split is wired yet — the endpoint must not pass raw detections off as a bloom
    # measurement (measurement-integrity: no fabricated fraction).
    assert body["elongation_classified"] is False

    by_key = {(r["plant_id"], r["date"]): r for r in body["rows"]}
    # Detection totals are reported truthfully; the elongated fraction stays 0 (unwired).
    assert by_key[("PLANT_A", "2026-02-11")]["n_total"] == 4
    assert by_key[("PLANT_A", "2026-02-11")]["ratio"] == 0.0
    assert by_key[("PLANT_B", "2026-02-11")]["n_total"] == 2
    assert by_key[("PLANT_A", "2026-03-24")]["n_total"] == 3


def test_per_plant_curves_flags_unclassified_predictions(client: TestClient, tmp_path: Path) -> None:
    # Single-class detector output (no elongation classification) must not be passed off as
    # a bloom measurement — the endpoint reports elongation_classified=False.
    mapping_path = tmp_path / "m.json"
    mapping_path.write_text(
        json.dumps(
            {
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
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    preds = tmp_path / "preds"
    preds.mkdir()
    # Raw catkin detections, no elongation attribute wired.
    _write_preds(preds / "IMG_A.json", 2)

    body = client.post(
        "/api/results/per_plant_curves",
        json={
            "project_root": str(tmp_path),
            "mapping_path": str(mapping_path),
            "predictions_by_date": {"2026-02-11": str(preds)},
        },
    ).json()
    assert body["elongation_classified"] is False
    assert body["classes_seen"] == []  # no elongation class ids surface until the K4/K5 bridge lands


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
    # catkin_elongation_date = "most catkins elongated" (crops.yml) = the 95% majority crossing,
    # synonymous with catkin_95per_date.
    assert onset["catkin_elongation_date"] == onset["catkin_95per_date"]


def test_onset_dates_ignores_undated_bucket(client: TestClient) -> None:
    # The ingest 'undated/' bucket (and any non-ISO folder) sorts to the (0,0,0) sentinel.
    # It must not crash interpolation (date(0,0,0)) or leak '0000-00-00' into the CSV.
    curves = [
        {"plant_id": "P", "accession": "A", "date": "undated", "ratio": 0.9},
        {"plant_id": "P", "accession": "A", "date": "2026-02-11", "ratio": 0.0},
        {"plant_id": "P", "accession": "A", "date": "2026-03-09", "ratio": 1.0},
    ]
    resp = client.post("/api/results/onset_dates", json={"curves": curves})
    assert resp.status_code == 200
    onset = resp.json()["rows"][0]
    for key in ("catkin_05per_date", "catkin_50per_date", "catkin_95per_date"):
        assert onset[key] != "0000-00-00"
        if onset[key] is not None:
            assert onset[key].startswith("2026-")


def test_onset_dates_rejects_row_missing_required_field(client: TestClient) -> None:
    # A malformed row (missing plant_id) must be a structured 422, not an unhandled
    # KeyError/500 — onset_dates dereferences plant_id/date/ratio directly.
    curves = [{"accession": "A", "date": "2026-02-11", "ratio": 0.0}]
    resp = client.post("/api/results/onset_dates", json={"curves": curves})
    assert resp.status_code == 422


def test_registered_models_confines_project_path_to_allowed_roots(
    client: TestClient, tmp_path: Path, monkeypatch
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(allowed))
    outside = tmp_path / "outside"
    outside.mkdir()
    resp = client.get("/api/results/models/registered", params={"project_path": str(outside)})
    assert resp.status_code == 403


def test_registered_models_unconfined_when_no_image_roots(
    client: TestClient, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("TCIP_IMAGE_ROOTS", raising=False)
    resp = client.get("/api/results/models/registered", params={"project_path": str(tmp_path)})
    assert resp.status_code == 200
    assert resp.json()["models"] == []


def test_export_csv(client: TestClient) -> None:
    # A phenology delivery (milestone columns) must carry the validation stamp to be exported.
    stamp = {"elongation_classifier_validated": "validated_held_out",
             "operating_point_validated": "validated_held_out"}
    rows = [
        {"plant_id": "PLANT_A", "catkin_05per_date": "2026-02-15", **stamp},
        {"plant_id": "PLANT_B", "catkin_05per_date": "2026-03-05", **stamp},
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


def test_export_csv_refuses_unvalidated_phenology(client: TestClient) -> None:
    # Milestone rows without the held-out-validated stamp must not ship (the second-door hole).
    rows = [{"plant_id": "PLANT_A", "catkin_05per_date": "2026-02-15"}]
    resp = client.post("/api/results/export_csv", json={"rows": rows, "filename": "x.csv"})
    assert resp.status_code == 400


def test_export_csv_non_phenology_unaffected(client: TestClient) -> None:
    # A diagnostic / inventory export (no trait_name + value) is not a phenotype delivery — not gated.
    rows = [{"plant_id": "PLANT_A", "some_metric": 42}]
    resp = client.post("/api/results/export_csv", json={"rows": rows, "filename": "x.csv"})
    assert resp.status_code == 200


def test_export_csv_refuses_unvalidated_per_plant_phenotype(client: TestClient) -> None:
    # A per-plant phenotype row (trait_name + value) with no validation stamp must not ship (R4:
    # the non-phenology delivery door is gated too).
    rows = [{"plant_id": "PLANT_A", "trait_name": "catkin_count", "value": 7}]
    resp = client.post("/api/results/export_csv", json={"rows": rows, "filename": "x.csv"})
    assert resp.status_code == 400


def test_export_csv_ships_validated_per_plant_phenotype(client: TestClient) -> None:
    # The same per-plant phenotype row ships once it carries a shippable measurement reference.
    rows = [{"plant_id": "PLANT_A", "trait_name": "catkin_count", "value": 7,
             "measurement_validated": "validated_held_out"}]
    resp = client.post("/api/results/export_csv", json={"rows": rows, "filename": "x.csv"})
    assert resp.status_code == 200


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

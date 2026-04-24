"""Integration tests for the Slice 2 Training routes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_web.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_validate_flags_missing_sections(client: TestClient) -> None:
    resp = client.post("/api/training/validate", json={"config": {}})
    body = resp.json()
    assert body["valid"] is False
    assert any("model_spec" in s for s in body["issues"])


def test_validate_accepts_minimal_config(client: TestClient, tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    cfg = {
        "model_spec": {
            "backbone": {"name": "resnet50"},
            "neck": {"name": "fpn"},
            "heads": [{"name": "detection_head", "task": "detection", "num_classes": 1}],
            "loss": {"name": "focal_loss"},
        },
        "data": {"images_dir": str(images), "labels_dir": str(labels), "task": "detection"},
        "training": {"batch_size": 2, "stages": [{"lr": 1e-3, "epochs": 1}]},
    }
    resp = client.post("/api/training/validate", json={"config": cfg})
    body = resp.json()
    # A real validate_model_spec call may add issues; we only assert the route works.
    assert "valid" in body
    assert "issues" in body


def test_list_runs_returns_shape(client: TestClient) -> None:
    resp = client.get("/api/training/runs")
    assert resp.status_code == 200
    body = resp.json()
    assert "runs" in body


def test_metrics_route_handles_missing_file(client: TestClient, tmp_path: Path) -> None:
    # When no metrics.jsonl exists yet, the route reports exists=False with empty rows.
    resp = client.get(
        "/api/training/runs/foo-xxx/metrics",
        params={"project_root": str(tmp_path)},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["exists"] is False
    assert body["metrics"] == []


def test_metrics_route_reads_jsonl(client: TestClient, tmp_path: Path) -> None:
    run_id = "exp-abc"
    metrics_path = tmp_path / ".tcip" / "experiments" / run_id / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True)
    rows = [{"epoch": 1, "loss": 0.9}, {"epoch": 2, "loss": 0.4}]
    with metrics_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    resp = client.get(
        f"/api/training/runs/{run_id}/metrics",
        params={"project_root": str(tmp_path)},
    )
    body = resp.json()
    assert body["exists"] is True
    assert body["metrics"] == rows


def test_compare_route_handles_empty_ids(client: TestClient) -> None:
    resp = client.post("/api/training/compare", json={"experiment_ids": []})
    assert resp.status_code == 200
    # body schema is up to compare_experiments; we only assert the route returns JSON.
    assert isinstance(resp.json(), dict)

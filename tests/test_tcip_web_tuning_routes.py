"""Tests for the Slice 4 Tuning routes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tcip_web.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_list_sweeps_empty(client: TestClient) -> None:
    resp = client.get("/api/tuning/sweeps")
    assert resp.status_code == 200
    body = resp.json()
    assert "sweeps" in body


def test_get_sweep_404(client: TestClient) -> None:
    resp = client.get("/api/tuning/sweeps/nope")
    assert resp.status_code == 404


def test_launch_creates_sweep_then_listed(client: TestClient) -> None:
    resp = client.post(
        "/api/tuning/launch",
        json={
            "base_config": {"model_source": {"builder": "x:y"}, "data": {}, "training": {}},
            "param_space": {"training.batch_size": [2, 4]},
            "n_trials": 1,
            "output_dir": "",
            "search_alg": "random",
            "scheduler": "asha",
        },
    )
    assert resp.status_code == 200
    sweep_id = resp.json()["sweep_id"]
    assert sweep_id.startswith("hpo-")
    # The background thread will fail fast because config is bogus; we only
    # assert the registry got the entry.
    listing = client.get("/api/tuning/sweeps").json()
    assert any(s["sweep_id"] == sweep_id for s in listing["sweeps"])

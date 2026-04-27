"""Tests for session-tracking routes (annotation_stats.json equivalent)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_web.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_start_inserts_open_session(client: TestClient, tmp_path: Path) -> None:
    resp = client.post(
        "/api/sessions/start",
        json={"project_root": str(tmp_path), "user": "alice"},
    )
    assert resp.status_code == 200
    data = client.get(
        "/api/sessions/load", params={"project_root": str(tmp_path)}
    ).json()
    assert len(data["sessions"]) == 1
    s = data["sessions"][0]
    assert s["user"] == "alice"
    assert s["ended"] == ""


def test_start_is_idempotent(client: TestClient, tmp_path: Path) -> None:
    client.post("/api/sessions/start", json={"project_root": str(tmp_path), "user": "alice"})
    client.post("/api/sessions/start", json={"project_root": str(tmp_path), "user": "alice"})
    data = client.get("/api/sessions/load", params={"project_root": str(tmp_path)}).json()
    assert len(data["sessions"]) == 1  # second start was a no-op


def test_image_event_aggregates(client: TestClient, tmp_path: Path) -> None:
    pr = str(tmp_path)
    client.post("/api/sessions/start", json={"project_root": pr, "user": "alice"})
    # Spend 12s on IMG_A and add 3 boxes (final count: 5)
    client.post(
        "/api/sessions/image_event",
        json={
            "project_root": pr,
            "image_name": "IMG_A",
            "session_seconds_delta": 12.0,
            "annotations_added_delta": 3,
            "final_annotation_count": 5,
            "loaded_annotation_count": 2,
        },
    )
    # Another 6s, +2 boxes
    client.post(
        "/api/sessions/image_event",
        json={
            "project_root": pr,
            "image_name": "IMG_A",
            "session_seconds_delta": 6.0,
            "annotations_added_delta": 2,
            "final_annotation_count": 7,
        },
    )
    data = client.get("/api/sessions/load", params={"project_root": pr}).json()
    s = data["sessions"][0]
    img = s["images"]["IMG_A"]
    assert img["session_seconds"] == 18.0
    assert img["annotations_added"] == 5
    assert img["final_annotation_count"] == 7
    assert img["loaded_annotation_count"] == 2
    assert img["avg_seconds_per_annotation"] == round(18.0 / 5, 2)
    # Aggregates roll up
    assert s["total_annotations"] == 5
    assert s["total_time_seconds"] == 18.0


def test_end_marks_ended(client: TestClient, tmp_path: Path) -> None:
    pr = str(tmp_path)
    client.post("/api/sessions/start", json={"project_root": pr, "user": "alice"})
    client.post(
        "/api/sessions/image_event",
        json={
            "project_root": pr,
            "image_name": "IMG_A",
            "session_seconds_delta": 5.0,
            "annotations_added_delta": 2,
            "final_annotation_count": 2,
        },
    )
    end = client.post("/api/sessions/end", json={"project_root": pr}).json()
    assert end["session"]["ended"] != ""
    on_disk = json.loads(
        (tmp_path / ".tcip" / "state" / "annotation_stats.json").read_text()
    )
    assert on_disk["sessions"][0]["ended"] != ""


def test_image_event_drops_empty_entry(client: TestClient, tmp_path: Path) -> None:
    pr = str(tmp_path)
    client.post("/api/sessions/start", json={"project_root": pr, "user": "alice"})
    client.post(
        "/api/sessions/image_event",
        json={
            "project_root": pr,
            "image_name": "IMG_X",
            "session_seconds_delta": 0.0,
            "annotations_added_delta": 0,
            "final_annotation_count": 0,
        },
    )
    data = client.get("/api/sessions/load", params={"project_root": pr}).json()
    assert "IMG_X" not in data["sessions"][0]["images"]


def test_load_missing_returns_empty_shape(client: TestClient, tmp_path: Path) -> None:
    data = client.get(
        "/api/sessions/load", params={"project_root": str(tmp_path)}
    ).json()
    assert data == {"sessions": [], "image_status": {}}

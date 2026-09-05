"""Tests for session-tracking routes (annotation_stats.json equivalent)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import tcip_store
from tcip_mcp.dataset_layout import record_image_statuses, status_bucket
from tcip_web.app import app
from tcip_web.routes.sessions import annotation_stats_key


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


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
    assert img["avg_seconds_per_annotation"] == round(18.0 / 5, 2)
    # Aggregates roll up
    assert s["total_annotations"] == 5
    assert s["total_time_seconds"] == 18.0


def test_negative_confirmation_time_counts_toward_the_session_total(
    client: TestClient, tmp_path: Path
) -> None:
    """Real time spent confirming a negative or reviewing existing annotations, with zero new
    annotations added, must not vanish from total_time_seconds the way it used to."""
    pr = str(tmp_path)
    client.post("/api/sessions/start", json={"project_root": pr, "user": "alice"})
    # 10s reviewed IMG_A, added nothing new (a negative confirmation or pure review).
    client.post(
        "/api/sessions/image_event",
        json={
            "project_root": pr,
            "image_name": "IMG_A",
            "session_seconds_delta": 10.0,
            "annotations_added_delta": 0,
            "final_annotation_count": 3,
        },
    )
    # 20s on IMG_B, added 4 new annotations.
    client.post(
        "/api/sessions/image_event",
        json={
            "project_root": pr,
            "image_name": "IMG_B",
            "session_seconds_delta": 20.0,
            "annotations_added_delta": 4,
            "final_annotation_count": 4,
        },
    )
    s = client.get("/api/sessions/load", params={"project_root": pr}).json()["sessions"][0]
    # The full 30s counts, not just the 20s spent on the image that gained new annotations.
    assert s["total_time_seconds"] == 30.0
    # images_annotated and total_annotations stay scoped to real new-annotation activity.
    assert s["images_annotated"] == 1
    assert s["total_annotations"] == 4
    # The per-annotation average is still a pure figure: 20s / 4 annotations, not 30s / 4.
    assert s["avg_seconds_per_annotation"] == round(20.0 / 4, 2)


def test_load_splits_time_into_new_annotation_review_and_negative_confirmation(
    client: TestClient, tmp_path: Path
) -> None:
    """Read-time classification against image_status.json's current state, not a write-time
    snapshot: negative_confirmation_seconds + review_seconds + new_annotation_seconds sums back
    to total_time_seconds."""
    project_root = tmp_path / "proj"
    dataset_root = tmp_path / "data"
    record_image_statuses(dataset_root, status_bucket("bud", "2026-02-11"),
                          {"IMG_NEG": "negative"}, recorded_by="user:breeder")

    pr = str(project_root)
    common = {"project_root": pr, "dataset_root": str(dataset_root), "subject": "bud",
             "date": "2026-02-11"}
    client.post("/api/sessions/start", json={"project_root": pr, "user": "alice"})
    # IMG_NEG: confirmed negative, no new annotations.
    client.post("/api/sessions/image_event", json={
        **common, "image_name": "IMG_NEG", "session_seconds_delta": 5.0,
        "annotations_added_delta": 0, "final_annotation_count": 0,
    })
    # IMG_REVIEW: has no status recorded, no new annotations (pure review, not a confirmed negative).
    client.post("/api/sessions/image_event", json={
        **common, "image_name": "IMG_REVIEW", "session_seconds_delta": 7.0,
        "annotations_added_delta": 0, "final_annotation_count": 2,
    })
    # IMG_NEW: gained new annotations.
    client.post("/api/sessions/image_event", json={
        **common, "image_name": "IMG_NEW", "session_seconds_delta": 20.0,
        "annotations_added_delta": 4, "final_annotation_count": 4,
    })

    s = client.get("/api/sessions/load", params={"project_root": pr}).json()["sessions"][0]
    assert s["negative_confirmation_seconds"] == 5.0
    assert s["review_seconds"] == 7.0
    assert s["new_annotation_seconds"] == 20.0
    assert s["total_time_seconds"] == 32.0
    assert (
        s["negative_confirmation_seconds"] + s["review_seconds"] + s["new_annotation_seconds"]
        == s["total_time_seconds"]
    )


def test_a_status_store_that_will_not_decode_reports_its_time_as_review(
    client: TestClient, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An unreadable confirmation store must not read as a confirmed negative.

    The route only displays these numbers, so it keeps answering, but it names the store in the
    log and counts the time as review: the reading that claims the least.

    Bound to the file backend: the claim needs bytes on disk no codec decodes, which only the
    file backend ever holds raw.
    """
    from tcip_store.file_backend import FileBackend

    tcip_store.bind(FileBackend())
    project_root = tmp_path / "proj"
    dataset_root = tmp_path / "data"
    (dataset_root / ".tcip" / "state").mkdir(parents=True)
    (dataset_root / ".tcip" / "state" / "image_status.json").write_bytes(b"{not a status store")

    pr = str(project_root)
    client.post("/api/sessions/start", json={"project_root": pr, "user": "alice"})
    client.post("/api/sessions/image_event", json={
        "project_root": pr, "dataset_root": str(dataset_root), "subject": "bud",
        "date": "2026-02-11", "image_name": "IMG_UNREADABLE", "session_seconds_delta": 4.0,
        "annotations_added_delta": 0, "final_annotation_count": 0,
    })

    with caplog.at_level(logging.WARNING):
        s = client.get("/api/sessions/load", params={"project_root": pr}).json()["sessions"][0]
    assert s["review_seconds"] == 4.0
    assert s["negative_confirmation_seconds"] == 0.0
    assert str(dataset_root) in caplog.text


def test_load_reflects_a_negative_confirmed_after_the_session_that_spent_time_ended(
    client: TestClient, tmp_path: Path
) -> None:
    """The split is read fresh, not frozen when image_event fired: confirming a negative later
    still reclassifies that image's already-recorded time on the next load."""
    project_root = tmp_path / "proj"
    dataset_root = tmp_path / "data"
    (dataset_root / ".tcip" / "state").mkdir(parents=True)

    pr = str(project_root)
    client.post("/api/sessions/start", json={"project_root": pr, "user": "alice"})
    client.post("/api/sessions/image_event", json={
        "project_root": pr, "dataset_root": str(dataset_root), "subject": "bud",
        "date": "2026-02-11", "image_name": "IMG_LATE", "session_seconds_delta": 9.0,
        "annotations_added_delta": 0, "final_annotation_count": 0,
    })

    before = client.get("/api/sessions/load", params={"project_root": pr}).json()["sessions"][0]
    assert before["review_seconds"] == 9.0
    assert before["negative_confirmation_seconds"] == 0.0

    record_image_statuses(dataset_root, status_bucket("bud", "2026-02-11"),
                          {"IMG_LATE": "negative"}, recorded_by="user:breeder")

    after = client.get("/api/sessions/load", params={"project_root": pr}).json()["sessions"][0]
    assert after["review_seconds"] == 0.0
    assert after["negative_confirmation_seconds"] == 9.0


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
    stored = tcip_store.read(annotation_stats_key(pr))
    assert stored["sessions"][0]["ended"] != ""


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
    assert data == {"sessions": []}


def test_start_then_image_event_stores_only_a_sessions_key(
    client: TestClient, tmp_path: Path
) -> None:
    """The stored document carries no dead ``image_status`` key: every writer here puts only
    ``sessions`` on disk."""
    pr = str(tmp_path)
    client.post("/api/sessions/start", json={"project_root": pr, "user": "alice"})
    client.post(
        "/api/sessions/image_event",
        json={
            "project_root": pr,
            "image_name": "IMG_A",
            "session_seconds_delta": 5.0,
            "annotations_added_delta": 1,
            "final_annotation_count": 1,
        },
    )
    stored = tcip_store.read(annotation_stats_key(pr))
    assert list(stored.keys()) == ["sessions"]

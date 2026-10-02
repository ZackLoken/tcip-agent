"""Tests for session-tracking routes (annotation_stats.json equivalent), each serving the project
open in the backend."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import tcip_store
from tcip_mcp.web_client import annotation_stats_key
from tcip_web.app import app


@pytest.fixture
def client(opened_project: Path) -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _mark_negative(dataset_root: Path, date: str, stem: str) -> None:
    """``stem``'s image on ``date``, its label document empty and marked complete for ``bud``
    through the editor's own save."""
    from PIL import Image

    from tcip_annotation import json_io
    from tcip_mcp.dataset_layout import annotation_path, image_dir
    from tests._producer_fixtures import mark_complete

    image = image_dir(dataset_root, date) / f"{stem}.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32)).save(image)
    label = annotation_path(dataset_root, date, stem)
    label.parent.mkdir(parents=True, exist_ok=True)
    json_io.write_annotations(label, [], 32, 32, keep_empty=True)
    mark_complete(image, label, "bud", project=dataset_root)


def _load(client: TestClient) -> dict:
    return client.get("/api/sessions/load").json()


def test_start_inserts_open_session(client: TestClient) -> None:
    resp = client.post("/api/sessions/start", json={"user": "alice"})
    assert resp.status_code == 200
    data = _load(client)
    assert len(data["sessions"]) == 1
    s = data["sessions"][0]
    assert s["user"] == "alice"
    assert s["ended"] == ""


def test_start_is_idempotent(client: TestClient) -> None:
    client.post("/api/sessions/start", json={"user": "alice"})
    client.post("/api/sessions/start", json={"user": "alice"})
    assert len(_load(client)["sessions"]) == 1  # second start was a no-op


def test_image_event_aggregates(client: TestClient) -> None:
    client.post("/api/sessions/start", json={"user": "alice"})
    # Spend 12s on IMG_A and add 3 boxes (final count: 5)
    client.post(
        "/api/sessions/image_event",
        json={
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
            "image_name": "IMG_A",
            "session_seconds_delta": 6.0,
            "annotations_added_delta": 2,
            "final_annotation_count": 7,
        },
    )
    s = _load(client)["sessions"][0]
    img = s["images"]["IMG_A"]
    assert img["session_seconds"] == 18.0
    assert img["annotations_added"] == 5
    assert img["final_annotation_count"] == 7
    assert img["avg_seconds_per_annotation"] == round(18.0 / 5, 2)
    # Aggregates roll up
    assert s["total_annotations"] == 5
    assert s["total_time_seconds"] == 18.0


def test_negative_confirmation_time_counts_toward_the_session_total(client: TestClient) -> None:
    """Real time spent confirming a negative or reviewing existing annotations, with zero new
    annotations added, counts toward total_time_seconds rather than vanishing from it."""
    client.post("/api/sessions/start", json={"user": "alice"})
    # 10s reviewed IMG_A, added nothing new (a negative confirmation or pure review).
    client.post(
        "/api/sessions/image_event",
        json={
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
            "image_name": "IMG_B",
            "session_seconds_delta": 20.0,
            "annotations_added_delta": 4,
            "final_annotation_count": 4,
        },
    )
    s = _load(client)["sessions"][0]
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
    """Read-time classification against each label document's current marks, not a write-time
    snapshot: negative_confirmation_seconds + review_seconds + new_annotation_seconds sums back
    to total_time_seconds."""
    dataset_root = tmp_path / "data"
    _mark_negative(dataset_root, "2026-02-11", "IMG_NEG")

    common = {"dataset_root": str(dataset_root), "subject": "bud", "date": "2026-02-11"}
    client.post("/api/sessions/start", json={"user": "alice"})
    # IMG_NEG: confirmed negative, no new annotations.
    client.post("/api/sessions/image_event", json={
        **common, "image_name": "IMG_NEG", "session_seconds_delta": 5.0,
        "annotations_added_delta": 0, "final_annotation_count": 0,
    })
    # IMG_REVIEW: has no label document, no new annotations (pure review, not a confirmed negative).
    client.post("/api/sessions/image_event", json={
        **common, "image_name": "IMG_REVIEW", "session_seconds_delta": 7.0,
        "annotations_added_delta": 0, "final_annotation_count": 2,
    })
    # IMG_NEW: gained new annotations.
    client.post("/api/sessions/image_event", json={
        **common, "image_name": "IMG_NEW", "session_seconds_delta": 20.0,
        "annotations_added_delta": 4, "final_annotation_count": 4,
    })

    s = _load(client)["sessions"][0]
    assert s["negative_confirmation_seconds"] == 5.0
    assert s["review_seconds"] == 7.0
    assert s["new_annotation_seconds"] == 20.0
    assert s["total_time_seconds"] == 32.0
    assert (
        s["negative_confirmation_seconds"] + s["review_seconds"] + s["new_annotation_seconds"]
        == s["total_time_seconds"]
    )


def test_a_label_document_that_will_not_read_fails_the_load(
    client: TestClient, tmp_path: Path
) -> None:
    """A label document is read as written: one that will not decode raises out of the load
    rather than reading as a document holding no mark."""
    from tcip_annotation.json_io import UnreadableLabelDocument
    from tcip_mcp.dataset_layout import annotation_path

    dataset_root = tmp_path / "data"
    label = annotation_path(dataset_root, "2026-02-11", "IMG_UNREADABLE")
    label.parent.mkdir(parents=True)
    label.write_bytes(b"{not a label document")

    client.post("/api/sessions/start", json={"user": "alice"})
    client.post("/api/sessions/image_event", json={
        "dataset_root": str(dataset_root), "subject": "bud",
        "date": "2026-02-11", "image_name": "IMG_UNREADABLE", "session_seconds_delta": 4.0,
        "annotations_added_delta": 0, "final_annotation_count": 0,
    })

    with pytest.raises(UnreadableLabelDocument, match="IMG_UNREADABLE"):
        client.get("/api/sessions/load")


def test_a_recorded_dataset_root_the_server_may_no_longer_read_fails_the_load_naming_it(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Time a session recorded on images under a root the allow-set no longer admits is refused
    with a 403 naming the root, never classified as review time read off no confirmations."""
    import tcip_web.paths as paths

    dataset_root = tmp_path / "data"
    dataset_root.mkdir()
    client.post("/api/sessions/start", json={"user": "alice"})
    assert client.post("/api/sessions/image_event", json={
        "dataset_root": str(dataset_root), "subject": "bud",
        "date": "2026-02-11", "image_name": "IMG_1", "session_seconds_delta": 9.0,
        "annotations_added_delta": 0, "final_annotation_count": 0,
    }).status_code == 200

    real_assert = paths.assert_path_allowed

    def _no_longer_admitted(path, *args, **kwargs):
        if Path(path).resolve() == dataset_root.resolve():
            raise ValueError(f"{path} is outside the allowed roots")
        return real_assert(path, *args, **kwargs)

    monkeypatch.setattr(paths, "assert_path_allowed", _no_longer_admitted)
    resp = client.get("/api/sessions/load")

    assert resp.status_code == 403
    assert str(dataset_root.resolve()) in resp.json()["detail"]


def test_load_reflects_a_negative_confirmed_after_the_session_that_spent_time_ended(
    client: TestClient, tmp_path: Path
) -> None:
    """The split is read fresh, not frozen when image_event fired: confirming a negative later
    still reclassifies that image's already-recorded time on the next load."""
    dataset_root = tmp_path / "data"
    dataset_root.mkdir()

    client.post("/api/sessions/start", json={"user": "alice"})
    client.post("/api/sessions/image_event", json={
        "dataset_root": str(dataset_root), "subject": "bud",
        "date": "2026-02-11", "image_name": "IMG_LATE", "session_seconds_delta": 9.0,
        "annotations_added_delta": 0, "final_annotation_count": 0,
    })

    before = _load(client)["sessions"][0]
    assert before["review_seconds"] == 9.0
    assert before["negative_confirmation_seconds"] == 0.0

    _mark_negative(dataset_root, "2026-02-11", "IMG_LATE")

    after = _load(client)["sessions"][0]
    assert after["review_seconds"] == 0.0
    assert after["negative_confirmation_seconds"] == 9.0


def test_end_marks_ended(client: TestClient, opened_project: Path) -> None:
    client.post("/api/sessions/start", json={"user": "alice"})
    client.post(
        "/api/sessions/image_event",
        json={
            "image_name": "IMG_A",
            "session_seconds_delta": 5.0,
            "annotations_added_delta": 2,
            "final_annotation_count": 2,
        },
    )
    end = client.post("/api/sessions/end", json={}).json()
    assert end["session"]["ended"] != ""
    stored = tcip_store.read(annotation_stats_key(str(opened_project)))
    assert stored["sessions"][0]["ended"] != ""


def test_image_event_drops_empty_entry(client: TestClient) -> None:
    client.post("/api/sessions/start", json={"user": "alice"})
    client.post(
        "/api/sessions/image_event",
        json={
            "image_name": "IMG_X",
            "session_seconds_delta": 0.0,
            "annotations_added_delta": 0,
            "final_annotation_count": 0,
        },
    )
    assert "IMG_X" not in _load(client)["sessions"][0]["images"]


def test_load_missing_returns_empty_shape(client: TestClient) -> None:
    assert _load(client) == {"sessions": []}


def test_start_then_image_event_stores_only_a_sessions_key(
    client: TestClient, opened_project: Path
) -> None:
    """The stored document carries only ``sessions``: every writer here puts nothing else on
    disk."""
    client.post("/api/sessions/start", json={"user": "alice"})
    client.post(
        "/api/sessions/image_event",
        json={
            "image_name": "IMG_A",
            "session_seconds_delta": 5.0,
            "annotations_added_delta": 1,
            "final_annotation_count": 1,
        },
    )
    stored = tcip_store.read(annotation_stats_key(str(opened_project)))
    assert list(stored.keys()) == ["sessions"]


def test_the_session_routes_refuse_while_no_project_is_open() -> None:
    client = TestClient(app, base_url="http://127.0.0.1")
    assert client.post("/api/sessions/start", json={"user": "alice"}).status_code == 409
    assert client.get("/api/sessions/load").status_code == 409

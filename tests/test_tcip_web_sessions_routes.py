"""Tests for the annotation session routes, each serving the project open in the backend."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient

import tcip_store
from tcip_mcp.web_client import annotation_stats_key
from tests import REFUSED_NAMES
from tests._audit_fixtures import audit_rows


def _event(client: TestClient, image: str, seconds: float, added: int, activity: str,
           user: str = "alice", project_id: str | None = None):
    """Post one contribution, to the open project unless ``project_id`` names another."""
    from tcip_web.state import store

    return client.post("/api/sessions/image_event", json={
        "image_name": image, "seconds": seconds, "annotations_added": added,
        "activity": activity, "user": user,
        "project_id": project_id or store.project_id or "no-project-open"})


def _load(client: TestClient) -> dict:
    return client.get("/api/sessions/load").json()


def test_throughput_is_derived_from_the_stored_entries(
    client: TestClient, opened_project: Path
) -> None:
    """The derived figures are the images with additions, the additions, every second, the
    seconds by activity, and the seconds of the images with additions per addition; the record
    stores the entries alone, each naming its activity."""
    _event(client, "IMG_A", 12.0, 3, "new_annotation")
    _event(client, "IMG_B", 10.0, 0, "review")
    _event(client, "IMG_C", 20.0, 4, "new_annotation")
    _event(client, "IMG_D", 5.0, 0, "negative_confirmation")

    s = _load(client)["sessions"][0]
    assert s["images_annotated"] == 2
    assert s["total_annotations"] == 7
    assert s["total_time_seconds"] == 47.0
    assert s["avg_seconds_per_annotation"] == round(32.0 / 7, 2)
    assert s["seconds_by_activity"] == {"new_annotation": 32.0, "review": 10.0,
                                        "negative_confirmation": 5.0}

    (stored,) = tcip_store.read(annotation_stats_key(str(opened_project)))["sessions"]
    assert set(stored) == {"user", "started", "ended", "entries"}
    assert [e["activity"] for e in stored["entries"]] == [
        "new_annotation", "review", "new_annotation", "negative_confirmation"]


def test_a_session_is_its_persons_and_records_them_in_the_actor_spelling(
    opened_client: TestClient
) -> None:
    _event(opened_client, "IMG_A", 3.0, 1, "new_annotation")
    _event(opened_client, "IMG_A", 2.0, 0, "review", user="bob")

    sessions = _load(opened_client)["sessions"]
    assert [s["user"] for s in sessions] == ["user:bob", "user:alice"]
    assert sessions[1]["ended"] is None


def test_a_contribution_naming_no_one_opens_no_session(
    client: TestClient, opened_project: Path
) -> None:
    for name in REFUSED_NAMES:
        resp = _event(client, "IMG_A", 3.0, 1, "new_annotation", user=name)
        assert resp.status_code == 400, (name, resp.text)
    assert tcip_store.read(annotation_stats_key(str(opened_project)), default=None) is None


def test_end_marks_the_open_session_ended_at_a_utc_instant(
    client: TestClient, opened_project: Path
) -> None:
    _event(client, "IMG_A", 5.0, 2, "new_annotation")
    assert client.post("/api/sessions/end", json={}).json() == {"status": "ok"}

    (stored,) = tcip_store.read(annotation_stats_key(str(opened_project)))["sessions"]
    for stamp in (stored["started"], stored["ended"]):
        assert datetime.fromisoformat(stamp).utcoffset() is not None
    assert client.post("/api/sessions/end", json={}).json() == {"status": "noop"}

    _event(client, "IMG_B", 1.0, 0, "review")
    assert len(_load(client)["sessions"]) == 2


def test_a_contribution_with_no_time_and_no_additions_records_nothing(
    client: TestClient, opened_project: Path
) -> None:
    assert _event(client, "IMG_X", 0.0, 0, "review").json() == {"status": "noop"}
    assert _load(client) == {"sessions": []}
    assert audit_rows(opened_project, "image_event") == []


def test_each_session_write_leaves_one_line_by_the_sessions_person(
    client: TestClient, opened_project: Path
) -> None:
    _event(client, "IMG_A", 4.0, 1, "new_annotation")
    client.post("/api/sessions/end", json={})

    (event,) = audit_rows(opened_project, "image_event")
    (ended,) = audit_rows(opened_project, "session_ended")
    assert (event["actor"], ended["actor"]) == ("user:alice", "user:alice")
    assert event["arguments"] == {"image_name": "IMG_A", "seconds": 4.0,
                                  "annotations_added": 1, "activity": "new_annotation"}


def test_a_contribution_naming_another_project_is_refused_and_records_nothing(
    client: TestClient, opened_project: Path
) -> None:
    resp = _event(client, "IMG_A", 4.0, 1, "new_annotation", project_id="another-project")

    assert resp.status_code == 409
    assert tcip_store.read(annotation_stats_key(str(opened_project)), default=None) is None


def test_load_missing_returns_empty_shape(opened_client: TestClient) -> None:
    assert _load(opened_client) == {"sessions": []}


def test_the_session_routes_refuse_while_no_project_is_open(client: TestClient) -> None:
    assert _event(client, "IMG_A", 1.0, 0, "review").status_code == 409
    assert client.get("/api/sessions/load").status_code == 409

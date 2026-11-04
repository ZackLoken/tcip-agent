"""Tests for the annotation-stats record read through its model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

import tcip_store
from tcip_mcp.web_client import annotation_stats_key

if TYPE_CHECKING:
    from tcip_web.state import OpenProject


def _contribute(
    client: TestClient, project_id: str, user: str, contribution_id: str
) -> dict:
    """Post one contribution by ``user`` to ``project_id``; return the posted body."""
    body = {
        "contribution_id": contribution_id,
        "image_name": "IMG_A",
        "seconds": 4.5,
        "annotations_added": 2,
        "activity": "new_annotation",
        "user": user,
        "project_id": project_id,
        "started": None,
    }
    resp = client.post("/api/sessions/image_event", json=body)
    assert resp.status_code == 200, resp.text
    return {**body, "started": resp.json()["started"]}


def test_the_record_the_session_routes_write_reads_back_through_its_model_unchanged(
    client: TestClient, opened: OpenProject
) -> None:
    from tcip_web.routes.sessions import ImageEntry, stored_stats

    alice = _contribute(client, opened.id, "alice", "c-alice")
    bob = _contribute(client, opened.id, "bob", "c-bob")
    ended = client.post(
        "/api/sessions/end",
        json={"project_id": opened.id, "started": alice["started"]},
    )
    assert ended.status_code == 200, ended.text

    key = annotation_stats_key(str(opened.root))
    raw = tcip_store.read(key)
    stats = stored_stats(tcip_store.read, key)

    stored = stats.model_dump(mode="json")
    assert stored == raw
    assert [(s["user"], s["started"]) for s in stored["sessions"]] == [
        ("user:bob", bob["started"]),
        ("user:alice", alice["started"]),
    ]
    assert [s.state.kind for s in stats.sessions] == ["open", "ended"]
    assert [s.entries for s in stats.sessions] == [
        [ImageEntry.model_validate(bob)],
        [ImageEntry.model_validate(alice)],
    ]


def test_a_stored_session_missing_its_person_is_refused_naming_the_field(
    client: TestClient, opened: OpenProject
) -> None:
    _contribute(client, opened.id, "alice", "c-alice")
    key = annotation_stats_key(str(opened.root))
    raw = tcip_store.read(key)
    assert isinstance(raw, dict)
    sessions = raw["sessions"]
    assert isinstance(sessions, list)
    (session,) = sessions
    assert isinstance(session, dict)
    user = session.pop("user")
    tcip_store.replace(key, raw)
    try:
        with pytest.raises(tcip_store.ReadRefusedError) as refused:
            client.get("/api/sessions/load")
    finally:
        # The backend's teardown ends the open session, which reads the record whole.
        session["user"] = user
        tcip_store.replace(key, raw)

    refusal = refused.value.refusal
    assert isinstance(refusal, tcip_store.Invalid)
    assert [issue.loc for issue in refusal.issues] == [["sessions", 0, "user"]]
    assert str(opened.root) in str(refused.value)
    assert "errors.pydantic.dev" not in str(refused.value)

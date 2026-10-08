"""Tests for the annotation session routes."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient

import tcip_store
from tcip_mcp.web_client import annotation_stats_key
from tests import REFUSED_NAMES
from tests._audit_fixtures import audit_rows


def _event(client: TestClient, project_id: str, image: str, seconds: float, added: int,
           activity: str, user: str = "alice", started: str | None = None,
           contribution_id: str | None = None):
    """Post one contribution to the project ``project_id`` names, in the session ``started``
    names or, with ``None``, in the person's latest session of the open project; a fresh
    contribution unless ``contribution_id`` names one."""
    return client.post("/api/sessions/image_event", json={
        "contribution_id": contribution_id or uuid.uuid4().hex,
        "image_name": image, "seconds": seconds, "annotations_added": added,
        "activity": activity, "user": user, "project_id": project_id, "started": started})


def _end(client: TestClient, project_id: str, started: str):
    """End the session ``started`` names in the project ``project_id`` names."""
    return client.post("/api/sessions/end", json={"project_id": project_id, "started": started})


def _stamp(resp) -> str:
    """The start stamp of the session a contribution's answer says it is recorded in."""
    return resp.json()["session"]["started"]


def _open_session(client: TestClient, project_id: str) -> str:
    """One contribution to ``project_id`` by alice; the start stamp of the session it opened."""
    resp = _event(client, project_id, "IMG_A", 5.0, 2, "new_annotation")
    assert resp.status_code == 200, resp.text
    return _stamp(resp)


def _load(client: TestClient) -> dict:
    return client.get("/api/sessions/load").json()


def test_throughput_is_derived_from_the_stored_entries(client: TestClient, opened) -> None:
    """The derived figures are the images with additions, the additions, every second, the
    seconds by activity, and the seconds of the images with additions per addition; the record
    stores the entries alone, each naming its activity."""
    _event(client, opened.id, "IMG_A", 12.0, 3, "new_annotation")
    _event(client, opened.id, "IMG_B", 10.0, 0, "review")
    _event(client, opened.id, "IMG_C", 20.0, 4, "new_annotation")
    _event(client, opened.id, "IMG_D", 5.0, 0, "negative_confirmation")

    s = _load(client)["sessions"][0]
    assert s["images_annotated"] == 2
    assert s["total_annotations"] == 7
    assert s["total_time_seconds"] == 47.0
    assert s["avg_seconds_per_annotation"] == round(32.0 / 7, 2)
    assert s["seconds_by_activity"] == {"new_annotation": 32.0, "review": 10.0,
                                        "negative_confirmation": 5.0}

    (stored,) = tcip_store.read(annotation_stats_key(str(opened.root)))["sessions"]
    assert set(stored) == {"user", "started", "ended", "entries"}
    assert [e["activity"] for e in stored["entries"]] == [
        "new_annotation", "review", "new_annotation", "negative_confirmation"]


def test_a_session_is_its_persons_and_records_them_in_the_actor_spelling(
    opened_client: TestClient, opened,
) -> None:
    _event(opened_client, opened.id, "IMG_A", 3.0, 1, "new_annotation")
    _event(opened_client, opened.id, "IMG_A", 2.0, 0, "review", user="bob")

    sessions = _load(opened_client)["sessions"]
    assert [s["user"] for s in sessions] == ["user:bob", "user:alice"]
    assert sessions[1]["ended"] is None


def test_a_contribution_naming_no_one_opens_no_session(client: TestClient, opened) -> None:
    for name in REFUSED_NAMES:
        resp = _event(client, opened.id, "IMG_A", 3.0, 1, "new_annotation", user=name)
        assert resp.status_code == 400, (name, resp.text)
    assert tcip_store.read(annotation_stats_key(str(opened.root)), default=None) is None


def test_end_marks_the_session_the_contribution_opened_ended_at_a_utc_instant(
    client: TestClient, opened,
) -> None:
    """A contribution answers with the start stamp of the session it landed in, and that stamp
    with the project's id is what ends the session; a second end of the same session is a
    no-op, and the next contribution opens a new one."""
    started = _open_session(client, opened.id)
    assert _end(client, opened.id, started).json() == {"status": "ok", "session": None}

    (stored,) = tcip_store.read(annotation_stats_key(str(opened.root)))["sessions"]
    assert stored["started"] == started
    for stamp in (stored["started"], stored["ended"]):
        assert datetime.fromisoformat(stamp).utcoffset() is not None
    assert _end(client, opened.id, started).json() == {"status": "noop", "session": None}

    _event(client, opened.id, "IMG_B", 1.0, 0, "review")
    assert len(_load(client)["sessions"]) == 2


def test_an_end_naming_another_projects_session_is_refused_and_ends_nothing(
    tmp_path: Path, client: TestClient
) -> None:
    """A page that recorded a session in one project ends that session and no other: once the
    backend has another project open, its end is refused and the session open there stays
    open. The departed project's session was ended by the departure itself."""
    from tests._web_fixtures import named_project

    ws = tmp_path.parent
    valley = named_project(ws / "valley_block", "Valley block")
    hill = named_project(ws / "hill_block", "Hill block")
    for project in (valley, hill):
        opened = client.post("/api/projects/open", json={"id": project.id, "user": "grower"})
        assert opened.status_code == 200, opened.text
        if project is valley:
            valley_started = _open_session(client, valley.id)
    hill_started = _open_session(client, hill.id)

    resp = _end(client, valley.id, valley_started)

    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"]["open_project_id"] == hill.id
    (hill_session,) = tcip_store.read(annotation_stats_key(str(hill.root)))["sessions"]
    assert (hill_session["started"], hill_session["ended"]) == (hill_started, None)
    (valley_session,) = tcip_store.read(annotation_stats_key(str(valley.root)))["sessions"]
    assert valley_session["ended"] is not None
    assert [line["arguments"] for line in audit_rows(valley.root, "session_ended")] == [
        {"started": valley_started}]
    assert audit_rows(hill.root, "session_ended") == []


def _two_open_sessions(client: TestClient, project_id: str) -> list[str]:
    """Alice's then Bob's contribution to ``project_id``, leaving both their sessions unended;
    their start stamps."""
    alice = _open_session(client, project_id)
    bob = _event(client, project_id, "IMG_B", 2.0, 1, "new_annotation", user="bob")
    assert bob.status_code == 200, bob.text
    return [alice, _stamp(bob)]


def _assert_departed(project, started: list[str]) -> None:
    """Every session ``started`` names is ended in ``project``, each with one line by its
    person."""
    sessions = tcip_store.read(annotation_stats_key(str(project.root)))["sessions"]
    assert {s["started"] for s in sessions if s["ended"] is not None} == set(started)
    lines = audit_rows(project.root, "session_ended")
    assert sorted(line["arguments"]["started"] for line in lines) == sorted(started)
    assert sorted(line["actor"] for line in lines) == ["user:alice", "user:bob"]


def test_opening_another_project_ends_every_unended_session_of_the_one_it_replaces(
    tmp_path: Path, client: TestClient,
) -> None:
    from tests._web_fixtures import named_project

    ws = tmp_path.parent
    valley = named_project(ws / "valley_block", "Valley block")
    hill = named_project(ws / "hill_block", "Hill block")
    assert client.post("/api/projects/open", json={"id": valley.id, "user": "g"}).status_code == 200
    started = _two_open_sessions(client, valley.id)

    assert client.post("/api/projects/open", json={"id": hill.id, "user": "g"}).status_code == 200

    _assert_departed(valley, started)


def test_closing_the_open_project_ends_every_unended_session_of_it(
    client: TestClient, opened,
) -> None:
    import asyncio

    from tcip_web.state import store

    started = _two_open_sessions(client, opened.id)

    asyncio.run(store.close_project(opened.id))

    _assert_departed(opened, started)


def test_requests_during_a_departure_never_act_on_the_next_project_with_the_departed_state(
    tmp_path: Path, monkeypatch,
) -> None:
    """While opening Hill holds Valley's departure transaction open, a navigation request that
    has read Valley and its selection, and a contribution naming no session, both wait for the
    switch: neither pairs Hill with Valley's state nor opens a session in Valley after its
    sessions were ended."""
    import contextlib
    import threading

    from tcip_mcp.web_client import gui_snapshot_key
    from tcip_web.app import app
    from tcip_web.state import store
    from tests._web_fixtures import named_project

    ws = tmp_path.parent
    valley = named_project(ws / "valley_block", "Valley block")
    hill = named_project(ws / "hill_block", "Hill block")
    valley_key = annotation_stats_key(str(valley.root))
    real = tcip_store.transaction
    inside, release, read = threading.Event(), threading.Event(), threading.Event()
    held_state = store.held_state

    def reading():
        held = held_state()
        read.set()
        return held

    @contextlib.contextmanager
    def paused(*keys, **kwargs):
        with real(*keys, **kwargs) as txn:
            yield txn
            if keys == (valley_key,) and not inside.is_set():
                inside.set()
                release.wait(10)

    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.post("/api/projects/open",
                           json={"id": valley.id, "user": "g"}).status_code == 200
        assert client.post("/api/state/tab", json={"active_tab": "results"}).status_code == 200
        monkeypatch.setattr(tcip_store, "transaction", paused)
        monkeypatch.setattr(store, "held_state", reading)
        answers: dict[str, int] = {}

        def call(name: str, url: str, body: dict) -> threading.Thread:
            thread = threading.Thread(
                target=lambda: answers.__setitem__(name, client.post(url, json=body).status_code))
            thread.start()
            return thread

        opening = call("open", "/api/projects/open", {"id": hill.id, "user": "g"})
        assert inside.wait(10), "the departure never reached Valley's transaction"
        waiting = [call("nav", "/api/dataset/nav", {"current_image_index": 0}),
                   call("event", "/api/sessions/image_event", {
                       "contribution_id": "c-departing", "image_name": "IMG_A",
                       "seconds": 3.0, "annotations_added": 1,
                       "activity": "new_annotation", "user": "alice",
                       "project_id": valley.id, "started": None})]
        assert read.wait(10), "navigation never read the open project during the departure"
        release.set()
        for thread in (opening, *waiting):
            thread.join(10)

    assert answers == {"open": 200, "nav": 409, "event": 409}
    assert not tcip_store.exists(gui_snapshot_key(hill.root))
    assert tcip_store.read(valley_key, default=None) is None


def test_a_contribution_naming_an_unended_session_lands_whatever_project_is_open(
    tmp_path: Path, client: TestClient, monkeypatch,
) -> None:
    """A contribution names its session, so one for Valley lands in Valley's unended session
    while a backend started afresh has Hill open."""
    import asyncio

    from tcip_web import state
    from tests._web_fixtures import named_project

    ws = tmp_path.parent
    valley = named_project(ws / "valley_block", "Valley block")
    hill = named_project(ws / "hill_block", "Hill block")
    assert client.post("/api/projects/open", json={"id": valley.id, "user": "g"}).status_code == 200
    started = _open_session(client, valley.id)

    restarted = state.StateStore()
    restarted.configure(state.store.workspace, ())
    asyncio.run(restarted.open_project(hill))
    monkeypatch.setattr(state, "store", restarted)
    resp = _event(client, valley.id, "IMG_B", 2.0, 1, "new_annotation", started=started)

    assert resp.status_code == 200, resp.text
    (session,) = tcip_store.read(annotation_stats_key(str(valley.root)))["sessions"]
    assert (session["started"], session["ended"], len(session["entries"])) == (started, None, 2)


def test_a_contribution_naming_a_session_its_departure_ended_is_refused_by_name(
    tmp_path: Path, client: TestClient,
) -> None:
    from tests._web_fixtures import named_project

    ws = tmp_path.parent
    valley = named_project(ws / "valley_block", "Valley block")
    hill = named_project(ws / "hill_block", "Hill block")
    assert client.post("/api/projects/open", json={"id": valley.id, "user": "g"}).status_code == 200
    started = _open_session(client, valley.id)
    assert client.post("/api/projects/open", json={"id": hill.id, "user": "g"}).status_code == 200

    resp = _event(client, valley.id, "IMG_B", 2.0, 1, "new_annotation", started=started)

    assert resp.status_code == 409, resp.text
    assert f"ended that session: {started}" in resp.json()["detail"]
    (session,) = tcip_store.read(annotation_stats_key(str(valley.root)))["sessions"]
    assert len(session["entries"]) == 1


def test_a_contribution_committed_without_its_line_answers_the_session_it_opened(
    client: TestClient, opened, monkeypatch,
) -> None:
    """When the contribution's line cannot be appended, the 409 carries the answer the write
    earned, the session's start stamp included, and that stamp ends the session."""
    from tcip_mcp import audit

    def refuse(*args, **kwargs):
        raise audit.AuditEntryNotWrittenError("image_event", RuntimeError("log unwritable"))

    monkeypatch.setattr(audit, "record_event_or_raise", refuse)
    resp = _event(client, opened.id, "IMG_A", 5.0, 2, "new_annotation")
    monkeypatch.undo()

    assert resp.status_code == 409, resp.text
    (stored,) = tcip_store.read(annotation_stats_key(str(opened.root)))["sessions"]
    assert resp.json()["detail"]["committed"] == {
        "status": "ok", "session": {"started": stored["started"], "ended": False}}
    assert _end(client, opened.id, stored["started"]).json() == {"status": "ok", "session": None}


def test_two_sessions_of_one_project_never_share_a_start_stamp(
    client: TestClient, opened, monkeypatch,
) -> None:
    """Two sessions opening in one clock instant get distinct stamps, the second one microsecond
    past the first, and each stamp ends its own session."""
    from tcip_web.routes import sessions

    frozen = "2026-10-07T14:00:00+00:00"
    monkeypatch.setattr(sessions, "now_iso", lambda: frozen)
    alice = _stamp(_event(client, opened.id, "IMG_A", 5.0, 2, "new_annotation"))
    bob = _stamp(_event(client, opened.id, "IMG_B", 3.0, 1, "new_annotation", user="bob"))
    monkeypatch.undo()

    assert alice == frozen
    assert bob == "2026-10-07T14:00:00.000001+00:00"
    assert _end(client, opened.id, alice).json() == {"status": "ok", "session": None}
    bob_session, alice_session = tcip_store.read(
        annotation_stats_key(str(opened.root)))["sessions"]
    assert (bob_session["started"], bob_session["ended"]) == (bob, None)
    assert alice_session["ended"] is not None
    assert _end(client, opened.id, bob).json() == {"status": "ok", "session": None}
    bob_session, alice_session = tcip_store.read(
        annotation_stats_key(str(opened.root)))["sessions"]
    assert bob_session["ended"] is not None and alice_session["ended"] is not None
    assert bob_session["ended"] >= alice_session["ended"]


def test_an_end_naming_a_start_no_session_carries_is_refused_and_ends_nothing(
    client: TestClient, opened,
) -> None:
    started = _open_session(client, opened.id)

    resp = _end(client, opened.id, "2000-01-01T00:00:00+00:00")

    assert resp.status_code == 409, resp.text
    assert "2000-01-01T00:00:00+00:00" in resp.json()["detail"]
    (stored,) = tcip_store.read(annotation_stats_key(str(opened.root)))["sessions"]
    assert (stored["started"], stored["ended"]) == (started, None)
    assert audit_rows(opened.root, "session_ended") == []


def test_a_contribution_with_no_time_and_no_additions_records_nothing(
    client: TestClient, opened,
) -> None:
    assert _event(client, opened.id, "IMG_X", 0.0, 0, "review").json() == {
        "status": "noop", "session": None}
    assert _load(client) == {"sessions": []}
    assert audit_rows(opened.root, "image_event") == []


def test_a_confirmed_negative_with_no_time_and_no_additions_is_recorded(
    client: TestClient, opened,
) -> None:
    resp = _event(client, opened.id, "IMG_X", 0.0, 0, "negative_confirmation")

    assert resp.json()["status"] == "ok", resp.text
    (session,) = _load(client)["sessions"]
    assert [e["activity"] for e in session["entries"]] == ["negative_confirmation"]


def test_a_contribution_sent_again_after_its_answer_was_lost_is_recorded_once(
    client: TestClient, opened,
) -> None:
    first = _event(client, opened.id, "IMG_A", 10.0, 2, "new_annotation", contribution_id="c1")
    again = _event(client, opened.id, "IMG_A", 10.0, 2, "new_annotation", contribution_id="c1")

    assert again.json() == {"status": "noop", "session": {"started": _stamp(first), "ended": False}}
    (session,) = _load(client)["sessions"]
    assert (session["total_annotations"], session["total_time_seconds"]) == (2, 10.0)
    assert len(audit_rows(opened.root, "image_event")) == 1


def test_a_repeat_of_an_unnamed_contribution_answers_the_session_it_landed_in_and_its_end(
    client: TestClient, opened,
) -> None:
    """Two contributions naming no session land in one session; the first's answer is lost and
    the session ended before it is sent again. The repeat answers that session and that it
    ended, which a page that learned the session from the second can act on, and is recorded
    once."""
    _event(client, opened.id, "IMG_A", 10.0, 2, "new_annotation", contribution_id="a")
    learned = _stamp(_event(client, opened.id, "IMG_B", 4.0, 1, "new_annotation"))
    _end(client, opened.id, learned)

    again = _event(client, opened.id, "IMG_A", 10.0, 2, "new_annotation", contribution_id="a")

    assert again.json() == {"status": "noop", "session": {"started": learned, "ended": True}}
    (session,) = _load(client)["sessions"]
    assert (session["started"], len(session["entries"])) == (learned, 2)


def test_each_session_write_leaves_one_line_by_the_sessions_person(
    client: TestClient, opened,
) -> None:
    started = _stamp(_event(client, opened.id, "IMG_A", 4.0, 1, "new_annotation",
                            contribution_id="c1"))
    _end(client, opened.id, started)

    (event,) = audit_rows(opened.root, "image_event")
    (ended,) = audit_rows(opened.root, "session_ended")
    assert (event["actor"], ended["actor"]) == ("user:alice", "user:alice")
    assert event["arguments"] == {"contribution_id": "c1", "image_name": "IMG_A",
                                  "seconds": 4.0, "annotations_added": 1,
                                  "activity": "new_annotation"}
    assert ended["arguments"] == {"started": started}


def test_an_end_names_its_own_session_and_not_the_latest(client: TestClient, opened) -> None:
    """A page ends the session its stamp names wherever it sits in the project's list; a newer
    session another person opened since stays open."""
    alice_started = _stamp(_event(client, opened.id, "IMG_A", 4.0, 1, "new_annotation"))
    bob_started = _stamp(_event(client, opened.id, "IMG_B", 2.0, 0, "review", user="bob"))
    assert alice_started != bob_started

    assert _end(client, opened.id, alice_started).json() == {"status": "ok", "session": None}

    bob, alice = tcip_store.read(annotation_stats_key(str(opened.root)))["sessions"]
    assert (alice["started"], alice["ended"] is not None) == (alice_started, True)
    assert (bob["started"], bob["ended"]) == (bob_started, None)


def test_a_contribution_naming_another_project_is_refused_and_records_nothing(
    client: TestClient, opened, tmp_path: Path,
) -> None:
    from tests._web_fixtures import named_project

    other = named_project(tmp_path.parent / "other", "Other")
    resp = _event(client, other.id, "IMG_A", 4.0, 1, "new_annotation")

    assert resp.status_code == 409
    assert resp.json()["detail"]["open_project_id"] == opened.id
    assert tcip_store.read(annotation_stats_key(str(opened.root)), default=None) is None


def test_load_missing_returns_empty_shape(opened_client: TestClient) -> None:
    assert _load(opened_client) == {"sessions": []}


def test_the_session_routes_refuse_while_no_project_is_open(client: TestClient, made) -> None:
    """A contribution naming a project while none is open, and a load, each answer the
    backend's none-open refusal."""
    for resp in (_event(client, made.id, "IMG_A", 1.0, 0, "review"),
                 client.get("/api/sessions/load")):
        assert resp.status_code == 409
        assert "no project is open" in resp.json()["detail"]

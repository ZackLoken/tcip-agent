"""Removing a project: ``tcip_mcp.workspace.remove_project`` and ``POST /api/projects/remove``.

One call archives the project, models included, into the workspace's ``.removed/``, moves its
directory there, and records one ``project_removed`` line in the moved project's own log.
Projects are made through the platform's own creation door under the workspace
``tmp_path.parent`` (``conftest.py`` lays ``tmp_path`` out as ``<workspace>/project``).
"""

from __future__ import annotations

import os
import threading
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from PIL import Image

from tcip_mcp import workspace
from tests import REFUSED_NAMES
from tests._audit_fixtures import audit_rows
from tests._web_fixtures import named_project

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from tcip_web.state import OpenProject


def _project(ws: Path, directory: str, display_name: str) -> OpenProject:
    """Make a project through the creation door (``named_project``), with one image."""
    made = named_project(ws / directory, display_name)
    images = made.root / "images" / "2026-03-04"
    images.mkdir(parents=True)
    Image.new("RGB", (8, 8), (0, 0, 0)).save(images / "img.jpg")
    return made


def _deny_moves(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every rename of ``project`` raise ``PermissionError``, with no retry budget."""
    real_rename = os.rename

    def _denied(src: Path, dst: Path) -> None:
        if Path(src) == project:
            raise PermissionError(13, "held open", str(src))
        return real_rename(src, dst)

    monkeypatch.setattr(workspace, "RENAME_BUDGET_S", 0.0)
    monkeypatch.setattr(os, "rename", _denied)


class _ObservedTurn:
    """A lock that delegates to ``lock``, recording each thread's attempt, acquisition and release.

    ``events`` holds ``(event, thread id)`` pairs in the order they happened, and ``changed`` is
    notified at each one.
    """

    def __init__(self, lock: threading.Lock) -> None:
        self.lock = lock
        self.events: list[tuple[str, int]] = []
        self.changed = threading.Condition()

    def _record(self, event: str) -> None:
        with self.changed:
            self.events.append((event, threading.get_ident()))
            self.changed.notify_all()

    def __enter__(self) -> None:
        self._record("attempted")
        self.lock.acquire()
        self._record("acquired")

    def __exit__(self, *exc: object) -> None:
        self._record("released")
        self.lock.release()


def test_removal_archives_moves_and_records_one_line(
    client: TestClient, tmp_path: Path
) -> None:
    """A removal archives, moves, and leaves one ``project_removed`` line as its whole trace.

    The archive the removal writes records nothing of its own.
    """
    ws = tmp_path.parent
    made = _project(ws, "valley_block", "Valley block")
    before = len(audit_rows(made.root))

    resp = client.post(
        "/api/projects/remove",
        json={"id": made.id, "confirm_name": "Valley block", "user": "tester"},
    )

    assert resp.status_code == 200, resp.text
    archive, moved_to = (
        Path(resp.json()["archive_path"]),
        Path(resp.json()["moved_to"]),
    )
    assert (
        archive.parent == ws / workspace.REMOVED_DIRNAME and archive.is_file()
    )
    assert (
        moved_to.parent == ws / workspace.REMOVED_DIRNAME and moved_to.is_dir()
    )
    assert not made.root.exists()
    with zipfile.ZipFile(archive) as zf:
        assert ".tcip/store.db" in zf.namelist()
        assert "images/2026-03-04/img.jpg" in zf.namelist()
    (line,) = audit_rows(moved_to)[before:]
    assert line["tool"] == "project_removed"
    assert line["arguments"]["archive_path"] == str(archive)
    assert line["actor"] == "user:tester"
    assert made.id not in {
        p["id"] for p in client.get("/api/projects").json()["projects"]
    }


def test_a_removal_naming_no_one_leaves_the_project(
    client: TestClient, tmp_path: Path
) -> None:
    made = _project(tmp_path.parent, "valley_block", "Valley block")

    for name in REFUSED_NAMES:
        resp = client.post(
            "/api/projects/remove",
            json={"id": made.id, "confirm_name": "Valley block", "user": name},
        )
        assert resp.status_code == 400, (name, resp.text)
    assert made.root.is_dir()
    assert audit_rows(made.root, "project_removed") == []


def test_an_archive_that_refuses_leaves_the_project_where_it_was_and_still_open(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused archive leaves the project where it was and still open.

    The removal answers 409 carrying the archive's refusal; the project stays at its directory and
    open, nothing is moved into the removed directory, and the log is unchanged.
    """
    from tcip_mcp.tools import project_tools

    ws = tmp_path.parent
    made = _project(ws, "valley_block", "Valley block")
    opened = client.post(
        "/api/projects/open", json={"id": made.id, "user": "grower"}
    )
    assert opened.status_code == 200
    before = audit_rows(made.root)
    monkeypatch.setattr(
        project_tools,
        "write_archive",
        lambda *a, **k: {"error": "the store under the project is unreadable"},
    )

    resp = client.post(
        "/api/projects/remove",
        json={"id": made.id, "confirm_name": "Valley block", "user": "tester"},
    )

    assert resp.status_code == 409
    assert "unreadable" in resp.json()["detail"]
    assert (
        workspace.project_by_id(workspace.project_records(ws), made.id)[0]
        == made.root
    )
    assert [
        p for p in (ws / workspace.REMOVED_DIRNAME).iterdir() if p.is_dir()
    ] == []
    assert client.get("/api/projects").json()["open_id"] == made.id
    assert audit_rows(made.root) == before


def test_a_live_run_refuses_the_removal_and_writes_nothing_then_a_finished_one_admits(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tcip_mcp import experiments
    from tests._verified_checkpoint_fixtures import (
        detection_config,
        opened_run,
    )

    ws = tmp_path.parent
    made = _project(ws, "valley_block", "Valley block")
    opened_run(
        made.root,
        detection_config(ws / "run-data", project=made.root),
        experiment_id="exp-live",
    )
    body = {"id": made.id, "confirm_name": "Valley block", "user": "tester"}

    refused = client.post("/api/projects/remove", json=body)

    assert refused.status_code == 409
    assert "exp-live" in refused.json()["detail"]
    assert made.root.is_dir()
    assert not (ws / workspace.REMOVED_DIRNAME).exists()
    assert audit_rows(made.root, "project_removed") == []

    monkeypatch.setattr(experiments, "HEARTBEAT_STALE_SECONDS", -1.0)
    assert client.post("/api/projects/remove", json=body).status_code == 200


def test_an_unfinished_inference_job_refuses_the_removal(
    client: TestClient, tmp_path: Path
) -> None:
    from tcip_mcp.pipelines.execution import Stated
    from tcip_web.routes import inference

    made = _project(tmp_path.parent, "valley_block", "Valley block")
    job = inference.InferenceJob(
        job_id="job-live",
        actor="user:tester",
        project=str(made.root),
        checkpoint_path="m.pt",
        dataset_root="ds",
        images_dir="ds/images",
        bucket="out/2026-01-01",
        stated=Stated(),
        status="running",
    )
    inference._registry.register(job)
    try:
        resp = client.post(
            "/api/projects/remove",
            json={
                "id": made.id,
                "confirm_name": "Valley block",
                "user": "tester",
            },
        )
    finally:
        job.status = "completed"

    assert resp.status_code == 409
    assert "job-live" in resp.json()["detail"]
    assert made.root.is_dir()


def test_a_confirm_name_other_than_the_display_name_refuses_then_the_display_name_admits(
    client: TestClient,
    tmp_path: Path,
) -> None:
    made = _project(tmp_path.parent, "valley_block", "Valley block")

    wrong = client.post(
        "/api/projects/remove",
        json={"id": made.id, "confirm_name": "valley_block", "user": "tester"},
    )

    assert wrong.status_code == 400
    assert "Valley block" in wrong.json()["detail"]
    assert made.root.is_dir()
    assert (
        client.post(
            "/api/projects/remove",
            json={
                "id": made.id,
                "confirm_name": "Valley block",
                "user": "tester",
            },
        ).status_code
        == 200
    )


def test_an_id_no_project_holds_answers_404(
    client: TestClient, tmp_path: Path
) -> None:
    _project(tmp_path.parent, "valley_block", "Valley block")

    resp = client.post(
        "/api/projects/remove",
        json={
            "id": "0" * 12,
            "confirm_name": "Valley block",
            "user": "tester",
        },
    )

    assert resp.status_code == 404


def test_removing_the_open_project_closes_it_first(
    client: TestClient, tmp_path: Path
) -> None:
    made = _project(tmp_path.parent, "valley_block", "Valley block")
    opened = client.post(
        "/api/projects/open", json={"id": made.id, "user": "grower"}
    )
    assert opened.status_code == 200

    resp = client.post(
        "/api/projects/remove",
        json={"id": made.id, "confirm_name": "Valley block", "user": "tester"},
    )

    assert resp.status_code == 200, resp.text
    assert client.get("/api/projects").json()["open_id"] is None


def test_the_archive_restores_through_import_project_with_its_identity(
    client: TestClient, tmp_path: Path
) -> None:
    from tcip_mcp.project_record import read_record
    from tcip_mcp.tools.project_tools import import_project

    ws = tmp_path.parent
    made = _project(ws, "valley_block", "Valley block")
    resp = client.post(
        "/api/projects/remove",
        json={"id": made.id, "confirm_name": "Valley block", "user": "tester"},
    )
    assert resp.status_code == 200, resp.text

    restored = ws / "restored_block"
    imported = import_project(resp.json()["archive_path"], str(restored))

    assert "error" not in imported, imported
    assert read_record(restored).id == made.id


def test_a_listing_waits_for_a_removal_whose_move_is_denied_at_first(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A listing asked for while the move is retried waits on the turn the removal holds.

    A listing in between would reopen the project's store and hold the folder the move needs. The
    first denied move waits, within the move's retry budget, for the listing to attempt the routes'
    workspace turn; at the move the removal still holds the turn and the listing has attempted it
    without acquiring it, and the listing then answers without the project.
    """
    from concurrent.futures import Future, ThreadPoolExecutor

    from tcip_web.routes import projects

    made = _project(tmp_path.parent, "valley_block", "Valley block")
    turn = _ObservedTurn(projects._WORKSPACE_TURN)
    monkeypatch.setattr(projects, "_WORKSPACE_TURN", turn)
    real_rename = os.rename
    listing: list[tuple[int, Future[dict], bool]] = []
    at_move: list[tuple[int, list[tuple[str, int]]]] = []

    with ThreadPoolExecutor(max_workers=1) as pool:

        def _denied_once(src: Path, dst: Path) -> None:
            if Path(src) == made.root and not listing:
                since = len(turn.events)
                answer = pool.submit(
                    lambda: client.get("/api/projects").json()
                )
                with turn.changed:
                    attempted = turn.changed.wait_for(
                        lambda: len(turn.events) > since,
                        workspace.RENAME_BUDGET_S,
                    )
                listing.append((since, answer, attempted))
                raise PermissionError(13, "held for a moment", str(src))
            if Path(src) == made.root:
                with turn.changed:
                    at_move.append((threading.get_ident(), list(turn.events)))
            real_rename(src, dst)

        monkeypatch.setattr(os, "rename", _denied_once)
        resp = client.post(
            "/api/projects/remove",
            json={
                "id": made.id,
                "confirm_name": "Valley block",
                "user": "tester",
            },
        )
        ((since, answer, attempted),) = listing
        assert attempted, "the listing never attempted the workspace turn"
        assert resp.status_code == 200, resp.text
        mover, events = at_move[0]
        assert [event for event, thread in events if thread == mover][-1] == (
            "acquired"
        ), "the removal let go of the workspace turn before its move"
        assert [event for event, _ in events[since:]] == ["attempted"]
        listed = answer.result(timeout=workspace.RENAME_BUDGET_S)

    assert made.id not in {p["id"] for p in listed["projects"]}


def test_a_move_denied_past_the_budget_answers_409_and_leaves_the_project(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made = _project(tmp_path.parent, "valley_block", "Valley block")
    _deny_moves(made.root, monkeypatch)

    resp = client.post(
        "/api/projects/remove",
        json={"id": made.id, "confirm_name": "Valley block", "user": "tester"},
    )

    assert resp.status_code == 409
    assert "another process" in resp.json()["detail"]
    assert made.root.is_dir()


def test_a_move_the_filesystem_denies_removes_the_archive_and_leaves_the_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = tmp_path.parent
    project = _project(ws, "valley_block", "Valley block").root
    _deny_moves(project, monkeypatch)

    with pytest.raises(OSError):
        workspace.remove_project(
            ws, project, actor="user:tester", release=lambda: None
        )

    assert project.is_dir()
    assert list((ws / workspace.REMOVED_DIRNAME).glob("*.zip")) == []

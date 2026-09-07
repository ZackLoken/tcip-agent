"""Tests for project removal: ``tcip_mcp.project_removal`` and the ``/api/projects`` routes.

Every project a test needs is seeded through the platform's own producers
(``initialize_project``, ``ingest_images``, ``activate_project``, ``register_dataset``), through
the app's test client with ``TCIP_WORKSPACE``/``TCIP_STATE_ROOT`` under ``tmp_path``. ``tmp_path``
is overridden in ``conftest.py`` to ``<workspace>/project``; these tests use ``tmp_path.parent``
as the workspace and name their own projects under it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import tcip_store as ts
from tcip_mcp import workspace
from tcip_mcp.tools.project_tools import initialize_project
from tcip_web.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _init(ws: Path, name: str, site: str = "a site") -> Path:
    result = initialize_project(str(ws / name), site=site)
    assert "error" not in result, result
    return ws / name


def _add_image(project: Path, date: str = "2026-03-04") -> None:
    d = project / "images" / date
    d.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (0, 0, 0)).save(d / "img.jpg")


def _seed(ws: Path) -> tuple[Path, Path]:
    """An open project (activated) and a removable target project, both under ``ws``."""
    open_project = _init(ws, "sample_plot_open")
    target = _init(ws, "sample_plot_target")
    _add_image(target)
    workspace.activate_project("sample_plot_open")
    return open_project, target


def _audit_lines(root: Path) -> list[dict]:
    from tcip_mcp import audit

    return list(ts.read_log(audit.audit_log_key(root)).records)


# ── the request ──────────────────────────────────────────────────────────────


def test_request_archives_marks_and_hides_the_project(client, tmp_path):
    ws = tmp_path.parent
    open_project, target = _seed(ws)

    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "tester"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert Path(body["archive_path"]).is_file()
    assert target.is_dir()  # nothing moved yet: phase one only marks

    pending = workspace.pending_removal_record(target)
    assert pending is not None
    assert pending["archive_path"] == body["archive_path"]

    with pytest.raises(workspace.ProjectPendingRemoval):
        workspace.adoptable_project_root("sample_plot_target")
    assert workspace.workspace_project_name(target) == "sample_plot_target"

    listing = client.get("/api/projects").json()
    names = {p["name"] for p in listing["projects"]}
    assert "sample_plot_target" not in names
    assert listing["pending_removal"][0]["name"] == "sample_plot_target"

    # The removed project's own log ends with the request line; the open project's log carries
    # the archive door's line and this route's own line, read before anything else writes to it.
    target_lines = _audit_lines(target)
    assert target_lines[-1]["tool"] == "project_removal_requested"
    open_lines = _audit_lines(open_project)
    tools = [line["tool"] for line in open_lines]
    assert "archive_project" in tools
    assert tools[-1] == "gui_project_removal_requested"
    assert tools.index("archive_project") < tools.index("gui_project_removal_requested")

    # An unconformed sibling (never touched by this request) still adopts as today.
    sibling = _init(ws, "sample_plot_sibling")
    assert workspace.adoptable_project_root("sample_plot_sibling") == sibling


def test_removal_preview_matches_the_doors_own_refusal(client, tmp_path):
    ws = tmp_path.parent
    _seed(ws)

    preview = client.get("/api/projects/sample_plot_target/removal-preview").json()
    assert preview["refusal"] is None
    assert preview["external_roots"] == []
    assert preview["dependent_projects"] == []


# ── refusals, each with its admitting case ───────────────────────────────────


def test_refuses_an_unsafe_name(client, tmp_path):
    ws = tmp_path.parent
    _seed(ws)
    resp = client.post(
        "/api/projects/remove",
        json={"name": "../escape", "confirm_name": "../escape", "user": "tester"},
    )
    assert resp.status_code == 400


def test_refuses_a_padded_name(client, tmp_path):
    ws = tmp_path.parent
    _seed(ws)
    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target ", "confirm_name": "sample_plot_target ", "user": "t"},
    )
    assert resp.status_code == 400


def test_refuses_the_holding_directory_name(client, tmp_path):
    ws = tmp_path.parent
    _seed(ws)
    resp = client.post(
        "/api/projects/remove",
        json={"name": ".removed", "confirm_name": ".removed", "user": "tester"},
    )
    assert resp.status_code == 400


def test_refuses_a_mismatched_confirmation(client, tmp_path):
    ws = tmp_path.parent
    _seed(ws)
    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_targe", "user": "t"},
    )
    assert resp.status_code == 400
    assert workspace.pending_removal_record(ws / "sample_plot_target") is None


def test_refuses_an_unknown_project_then_admits_once_it_exists(client, tmp_path):
    ws = tmp_path.parent
    _seed(ws)
    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_absent", "confirm_name": "sample_plot_absent", "user": "t"},
    )
    assert resp.status_code == 404

    _init(ws, "sample_plot_absent")
    resp2 = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_absent", "confirm_name": "sample_plot_absent", "user": "t"},
    )
    assert resp2.status_code == 200


def test_refuses_a_second_request_over_an_existing_marker(client, tmp_path):
    ws = tmp_path.parent
    _seed(ws)
    first = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert first.status_code == 200
    second = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert second.status_code == 409
    assert "already has a pending-removal marker" in second.json()["detail"]


def test_refuses_when_no_project_is_open_then_admits_once_one_is(client, tmp_path, tmp_path_factory, monkeypatch):
    """``tmp_path`` (``<workspace>/project``) is the autouse ``_pin_platform_root`` fixture's own
    inherited platform root, and any bare-``@audited`` write under it (``initialize_project``'s
    own) would otherwise leave a ``.tcip`` there that reads as an accidental project; pin the
    root somewhere outside the workspace instead so "no project is open" is genuinely true."""
    ws = tmp_path.parent
    ws.mkdir(exist_ok=True)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path_factory.mktemp("unrelated_root")))
    target = _init(ws, "sample_plot_target")
    _add_image(target)

    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 409
    assert "no project is open" in resp.json()["detail"]

    _init(ws, "sample_plot_open")
    workspace.activate_project("sample_plot_open")
    resp2 = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp2.status_code == 200


def test_refuses_the_markers_own_project_then_admits_a_different_one(client, tmp_path):
    ws = tmp_path.parent
    _seed(ws)
    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_open", "confirm_name": "sample_plot_open", "user": "t"},
    )
    assert resp.status_code == 409
    assert "marker" in resp.json()["detail"]

    resp2 = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp2.status_code == 200


def test_refuses_the_platform_root_by_identity(client, tmp_path):
    """The platform root and the marker name the same project after activate_project, so this
    exercises the same identity comparison the marker case does."""
    ws = tmp_path.parent
    _seed(ws)
    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_open", "confirm_name": "sample_plot_open", "user": "t"},
    )
    assert resp.status_code == 409


def test_refuses_the_canvas_open_binding_then_admits_a_different_project(client, tmp_path):
    ws = tmp_path.parent
    _seed(ws)
    third = _init(ws, "sample_plot_third")
    _add_image(third)

    sel = client.post(
        "/api/dataset/select", json={"project_root": str(third), "dataset_root": str(third)},
    )
    assert sel.status_code == 200

    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_third", "confirm_name": "sample_plot_third", "user": "t"},
    )
    assert resp.status_code == 409
    assert "the GUI has open" in resp.json()["detail"]

    resp2 = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp2.status_code == 200


def test_a_canvas_binding_with_no_project_name_is_read_as_no_binding(client, tmp_path):
    """A canvas binding carrying neither ``project_name`` nor ``root`` reads as no binding,
    rather than raising, in both the preview and the door."""
    from tcip_mcp.web_client import canvas_open_binding_key

    ws = tmp_path.parent
    _seed(ws)
    ts.replace(
        canvas_open_binding_key(),
        {"generation": 1, "issued_at": "2026-03-04T12:00:00+00:00"},
        expect=ts.Version.ABSENT,
    )

    preview = client.get("/api/projects/sample_plot_target/removal-preview").json()
    assert preview["refusal"] is None

    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 200


def test_refuses_a_live_run_then_admits_once_it_is_stale(client, tmp_path):
    from tcip_mcp import experiments

    ws = tmp_path.parent
    open_project, target = _seed(ws)
    exp_id = "exp1"
    ts.replace(
        experiments.status_key(exp_id, root=target),
        {"state": "training", "heartbeat": "2099-01-01T00:00:00+00:00"},
        expect=ts.Version.ABSENT,
    )

    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 409
    assert exp_id in resp.json()["detail"]

    current = ts.read_versioned(experiments.status_key(exp_id, root=target))
    ts.replace(
        experiments.status_key(exp_id, root=target),
        {"state": "completed", "heartbeat": "2026-01-01T00:00:00+00:00"},
        expect=current.version,
    )
    resp2 = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp2.status_code == 200


def test_refuses_a_non_terminal_inference_job_then_admits_once_terminal(client, tmp_path):
    from tcip_web.routes.inference import InferenceJob, _registry

    ws = tmp_path.parent
    open_project, target = _seed(ws)
    job = InferenceJob(
        job_id="j1", checkpoint_path="model.pt", images_dir="images/2026-03-04",
        output_dir="predictions/live/2026-03-04", conf=0.5, iou=0.5,
        slice_hw=(512, 512), overlap=0.2, status="running", platform_root=str(target),
    )
    _registry.register(job.job_id, job, job_root=job.platform_root)
    try:
        resp = client.post(
            "/api/projects/remove",
            json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
        )
        assert resp.status_code == 409
        assert "j1" in resp.json()["detail"]

        job.status = "completed"
        resp2 = client.post(
            "/api/projects/remove",
            json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
        )
        assert resp2.status_code == 200
    finally:
        with _registry.lock:
            _registry.jobs.pop("j1", None)


def test_a_linked_project_is_refused_before_any_write(client, tmp_path):
    ws = tmp_path.parent
    open_project, real = _seed(ws)
    link = ws / "sample_plot_link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks not available on this machine: {exc}")

    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_link", "confirm_name": "sample_plot_link", "user": "t"},
    )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert "junction" in detail or "symbolic link" in detail
    assert workspace.pending_removal_record(real) is None


def test_the_archive_refuses_inside_the_door_and_leaves_the_target_untouched(client, tmp_path):
    """``archive_project``'s own export refuses when the target's database holds rows of a
    store nothing has registered (``tcip_store.export.py``'s own ``UnknownStore`` refusal): the
    reproducible export-refusal path, rather than a damaged record's bytes, which
    ``_export_stores`` copies through unread and which ``archive_project``'s own bundle
    accounting (``derive_roots``/``account_for``) never reads the dataset registry to build."""
    import sqlite3

    if os.environ.get("TCIP_STORE_BACKEND", "sqlite") != "sqlite":
        pytest.skip("the export refusal needs a database under the target: nothing to damage "
                     "on the file backend")

    ws = tmp_path.parent
    open_project, target = _seed(ws)

    from tcip_store.sqlite_backend import database_path

    conn = sqlite3.connect(str(database_path(str(target))), isolation_level=None)
    conn.execute(
        "insert into store_counters (store, change_counter, exported_counter) values (?, ?, ?)",
        ("an_unregistered_store", 1, None),
    )
    conn.close()

    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 409
    assert "an_unregistered_store" in resp.json()["detail"]
    assert workspace.pending_removal_record(target) is None
    assert _audit_lines(target) == []

    # archive_project is bare-@audited: its own refused line lands in the open project's log,
    # the calling process's platform root, never the target's.
    open_lines = _audit_lines(open_project)
    assert open_lines[-1]["tool"] == "archive_project"
    assert open_lines[-1]["status"] == "error"


def test_an_unwritten_route_line_answers_409_naming_the_marker_already_written(
    client, tmp_path, monkeypatch,
):
    """The route's own line (gui_project_removal_requested) failing to append does not undo the
    marker or the removed project's own request line already written: the 409 names the marker,
    and the removal still completes at the next backend start."""
    import tcip_store
    from tcip_mcp import audit

    ws = tmp_path.parent
    open_project, target = _seed(ws)
    real_append = tcip_store.append

    def _flaky_append(key, record):
        if record.get("tool") == "gui_project_removal_requested":
            raise RuntimeError("simulated append failure")
        return real_append(key, record)

    monkeypatch.setattr(audit, "append", _flaky_append)

    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 409
    assert "sample_plot_target" in resp.json()["detail"]

    pending = workspace.pending_removal_record(target)
    assert pending is not None
    target_lines = _audit_lines(target)
    assert target_lines[-1]["tool"] == "project_removal_requested"

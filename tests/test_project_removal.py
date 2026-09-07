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
from tcip_mcp import project_removal, workspace
from tcip_mcp.tools.project_tools import archive_project, import_project, initialize_project
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
    assert "opens by default" in resp.json()["detail"]

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


def test_external_roots_present_flag_is_false_for_a_moved_root(client, tmp_path):
    from tcip_mcp.tools.project_tools import register_dataset

    ws = tmp_path.parent
    open_project, target = _seed(ws)
    external = _init(ws, "sample_plot_external")
    _add_image(external)

    reg = register_dataset(str(external), crop="black locust", project_root=str(target))
    assert "error" not in reg, reg

    preview_before = client.get("/api/projects/sample_plot_target/removal-preview").json()
    entry_before = next(r for r in preview_before["external_roots"] if r["path"] == str(external))
    assert entry_before["present"] is True

    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_external", "confirm_name": "sample_plot_external", "user": "t"},
    )
    assert resp.status_code == 200
    project_removal.complete_pending_removals(ws)
    assert not external.exists()

    preview_after = client.get("/api/projects/sample_plot_target/removal-preview").json()
    entry_after = next(r for r in preview_after["external_roots"] if r["path"] == str(external))
    assert entry_after["present"] is False


def test_external_roots_unreadable_is_named_and_the_request_still_admitted(client, tmp_path):
    from tests._record_damage_fixtures import damage_record

    from tcip_mcp.tools.project_tools import dataset_registry_key, register_dataset

    ws = tmp_path.parent
    open_project, target = _seed(ws)
    external = _init(ws, "sample_plot_external")
    _add_image(external)
    reg = register_dataset(str(external), crop="black locust", project_root=str(target))
    assert "error" not in reg, reg

    damage_record(dataset_registry_key(target), b"not json")

    preview = client.get("/api/projects/sample_plot_target/removal-preview").json()
    assert preview["external_roots"] == []
    assert preview.get("external_roots_unreadable")

    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 200


def test_two_concurrent_requests_produce_one_archive_and_one_marker(client, tmp_path):
    import threading

    ws = tmp_path.parent
    open_project, target = _seed(ws)

    responses: list = []

    def _post() -> None:
        responses.append(client.post(
            "/api/projects/remove",
            json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
        ))

    threads = [threading.Thread(target=_post) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    statuses = sorted(r.status_code for r in responses)
    assert statuses == [200, 409]
    zips = list((ws / ".removed").glob("sample_plot_target-*.zip"))
    assert len(zips) == 1


def test_dependent_project_is_listed_and_never_refused(client, tmp_path):
    from tcip_mcp.tools.project_tools import register_dataset

    ws = tmp_path.parent
    open_project, target = _seed(ws)
    dependent = _init(ws, "sample_plot_dependent")

    reg = register_dataset(str(target), crop="black locust", project_root=str(dependent))
    assert "error" not in reg, reg

    preview = client.get("/api/projects/sample_plot_target/removal-preview").json()
    assert preview["refusal"] is None
    names = {d["project"] for d in preview["dependent_projects"]}
    assert "sample_plot_dependent" in names

    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 200
    names2 = {d["project"] for d in resp.json()["dependent_projects"]}
    assert "sample_plot_dependent" in names2


def test_an_unreadable_sibling_registry_is_listed_as_such(client, tmp_path):
    from tests._record_damage_fixtures import damage_record

    from tcip_mcp.tools.project_tools import dataset_registry_key, register_dataset

    ws = tmp_path.parent
    open_project, target = _seed(ws)
    sibling = _init(ws, "sample_plot_damaged")
    reg = register_dataset(str(sibling), crop="black locust", project_root=str(sibling))
    assert "error" not in reg, reg

    damage_record(dataset_registry_key(sibling), b"not json")

    preview = client.get("/api/projects/sample_plot_target/removal-preview").json()
    entry = next(d for d in preview["dependent_projects"] if d["project"] == "sample_plot_damaged")
    assert entry.get("unreadable") is not None


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


def test_refuses_a_non_terminal_tuning_job_then_admits_once_terminal(client, tmp_path):
    from tcip_web.routes.tuning import HPOJob, _registry

    ws = tmp_path.parent
    open_project, target = _seed(ws)
    job = HPOJob(sweep_id="sw1", status="running", platform_root=str(target))
    _registry.register(job.sweep_id, job, job_root=job.platform_root)
    try:
        resp = client.post(
            "/api/projects/remove",
            json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
        )
        assert resp.status_code == 409
        assert "sw1" in resp.json()["detail"]

        job.status = "completed"
        resp2 = client.post(
            "/api/projects/remove",
            json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
        )
        assert resp2.status_code == 200
    finally:
        with _registry.lock:
            _registry.jobs.pop("sw1", None)


def test_refuses_a_non_terminal_review_priority_queue_job_then_admits_once_terminal(client, tmp_path):
    from tcip_web.routes.review import PriorityQueueJob, _pq_registry

    ws = tmp_path.parent
    open_project, target = _seed(ws)
    job = PriorityQueueJob(
        job_id="pq1", checkpoint_path="model.pt", images_dir="images/2026-03-04",
        dataset_root=str(target), status="running", platform_root=str(target),
    )
    _pq_registry.register(job.job_id, job, job_root=job.platform_root)
    try:
        resp = client.post(
            "/api/projects/remove",
            json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
        )
        assert resp.status_code == 409
        assert "pq1" in resp.json()["detail"]

        job.status = "completed"
        resp2 = client.post(
            "/api/projects/remove",
            json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
        )
        assert resp2.status_code == 200
    finally:
        with _pq_registry.lock:
            _pq_registry.jobs.pop("pq1", None)


def test_job_conflict_skips_a_rehydrated_job_with_an_empty_platform_root(client, tmp_path):
    """A rehydrated ``HPOJob`` can carry an empty ``platform_root``; the scan must never compose
    a sweep directory from it (``training_tools.sweep_dir`` on an empty root)."""
    from tcip_web.routes.tuning import HPOJob, _registry

    ws = tmp_path.parent
    open_project, target = _seed(ws)
    job = HPOJob(sweep_id="sw-empty", status="running", platform_root="")
    with _registry.lock:
        _registry.jobs["sw-empty"] = job
    try:
        resp = client.post(
            "/api/projects/remove",
            json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
        )
        assert resp.status_code == 200
    finally:
        with _registry.lock:
            _registry.jobs.pop("sw-empty", None)


def test_refuses_a_case_variant_spelling_of_the_open_project(client, tmp_path):
    """A case-insensitive filesystem resolves the marker's own project under a case-variant
    spelling to the same directory; the door's samefile comparison catches it as the marker's
    own project rather than a naive string compare admitting it."""
    ws = tmp_path.parent
    _seed(ws)
    try:
        same = os.path.samefile(ws / "SAMPLE_PLOT_OPEN", ws / "sample_plot_open")
    except OSError:
        pytest.skip("this filesystem does not fold case, nothing to prove here")
    if not same:
        pytest.skip("this filesystem does not fold case, nothing to prove here")

    resp = client.post(
        "/api/projects/remove",
        json={"name": "SAMPLE_PLOT_OPEN", "confirm_name": "SAMPLE_PLOT_OPEN", "user": "t"},
    )
    assert resp.status_code == 409


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


def test_select_dataset_refuses_a_pending_root_with_generation_unchanged(client, tmp_path):
    from tcip_mcp.web_client import read_canvas_binding

    ws = tmp_path.parent
    open_project, target = _seed(ws)

    sel = client.post(
        "/api/dataset/select",
        json={"project_root": str(open_project), "dataset_root": str(open_project)},
    )
    assert sel.status_code == 200
    before = read_canvas_binding()

    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 200

    sel2 = client.post(
        "/api/dataset/select", json={"project_root": str(target), "dataset_root": str(target)},
    )
    assert sel2.status_code == 403
    assert "pending removal" in sel2.json()["detail"]

    after = read_canvas_binding()
    assert after["generation"] == before["generation"]


# ── the completion ────────────────────────────────────────────────────────────


def test_completion_moves_the_tree_and_deletes_the_marker(client, tmp_path):
    ws = tmp_path.parent
    open_project, target = _seed(ws)
    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 200
    holding_dir = Path(resp.json()["holding_dir"])

    outcomes = project_removal.complete_pending_removals(ws)
    assert outcomes == [{"name": "sample_plot_target", "moved_to": str(holding_dir),
                          "archive_path": resp.json()["archive_path"]}]
    assert not target.exists()
    assert holding_dir.is_dir()
    assert workspace.pending_removal_record(holding_dir) is None

    moved_lines = _audit_lines(holding_dir)
    assert moved_lines[-1]["tool"] == "project_removal_completed"

    listing = client.get("/api/projects").json()
    assert listing["removal_startup_outcomes"] == outcomes
    assert "sample_plot_target" not in {p["name"] for p in listing["projects"]}
    assert listing["pending_removal"] == []


def test_the_archive_round_trips_through_import_project(client, tmp_path):
    ws = tmp_path.parent
    open_project, target = _seed(ws)
    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
    )
    archive_path = resp.json()["archive_path"]

    project_removal.complete_pending_removals(ws)

    dest = ws / "sample_plot_restored"
    imported = import_project(str(archive_path), str(dest))
    assert "error" not in imported, imported
    assert (dest / "images" / "2026-03-04" / "img.jpg").is_file()


def test_a_tree_moved_back_by_hand_is_listed_and_adoptable(client, tmp_path):
    import os

    ws = tmp_path.parent
    open_project, target = _seed(ws)
    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
    )
    holding_dir = Path(resp.json()["holding_dir"])
    project_removal.complete_pending_removals(ws)

    os.rename(str(holding_dir), str(target))

    assert workspace.adoptable_project_root("sample_plot_target") == target
    listing = client.get("/api/projects").json()
    assert "sample_plot_target" in {p["name"] for p in listing["projects"]}


def test_denied_completion_reports_blocked_by_and_admits_once_released(tmp_path):
    import os
    import sqlite3

    if os.environ.get("TCIP_STORE_BACKEND", "sqlite") != "sqlite":
        pytest.skip("the denied branch needs an open database handle, which only the sqlite "
                     "backend can hold: no handle to hold on the file backend")

    ws = tmp_path.parent
    open_project, target = _seed(ws)

    with TestClient(app, base_url="http://127.0.0.1") as client:
        resp = client.post(
            "/api/projects/remove",
            json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
        )
        assert resp.status_code == 200

        from tcip_store.sqlite_backend import database_path

        held = sqlite3.connect(str(database_path(str(target))), isolation_level=None)
        try:
            held.execute("select 1")
            outcomes = project_removal.complete_pending_removals(ws)
            assert outcomes[0]["name"] == "sample_plot_target"
            assert "blocked_by" in outcomes[0]
            assert target.is_dir()
        finally:
            held.close()

        outcomes2 = project_removal.complete_pending_removals(ws)
        assert outcomes2[0].get("moved_to") is not None
        assert not target.exists()


def test_the_per_project_fold_never_stops_the_walk(tmp_path):
    """A sibling whose state is still loose files (never adopted) is skipped by name, and a
    pending project elsewhere in the workspace still moves in the same walk.

    Bound to the sqlite backend on purpose: the fold is the seam's own refusal to read a loose
    layout through a database-bound process, which needs this process on that backend and only
    the spawned import under the file backend, the one producer of that mismatch."""
    import subprocess
    import sys

    if os.environ.get("TCIP_STORE_BACKEND", "sqlite") != "sqlite":
        pytest.skip("the fold is a database-bound process meeting a loose sibling layout; both "
                     "sides are the same layout on the file backend, so nothing to fold")

    from tcip_store import Version, replace

    ws = tmp_path.parent
    open_project, target = _seed(ws)
    resp_target = archive_project(
        project_path=str(target), output_path=str(ws / "seed.zip"), include_models=False,
    )
    assert "error" not in resp_target

    holding_dir = ws / ".removed" / "sample_plot_target-20260304T120000Z"
    replace(workspace.pending_removal_key(target), {
        "requested_at": "20260304T120000Z", "requested_by": "user:tester",
        "archive_path": str(ws / "seed.zip"), "holding_dir": str(holding_dir),
        "external_roots": [], "dependent_projects": [],
    }, expect=Version.ABSENT)

    sibling = ws / "sample_plot_sibling"
    env = dict(os.environ)
    env["TCIP_STORE_BACKEND"] = "file"
    proc = subprocess.run(
        [sys.executable, "-m", "tcip_web.cli", "import-project", str(ws / "seed.zip"), str(sibling)],
        env=env, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    outcomes = project_removal.complete_pending_removals(ws)
    by_name = {o["name"]: o for o in outcomes}
    assert by_name["sample_plot_target"].get("moved_to") is not None
    assert by_name["sample_plot_sibling"].get("skipped") is not None
    assert "adopt-store" in by_name["sample_plot_sibling"]["skipped"]


def test_the_complete_removals_command_moves_a_pending_project(client, tmp_path):
    import subprocess
    import sys

    from tcip_store import Version, replace

    ws = tmp_path.parent
    open_project, target = _seed(ws)
    resp = archive_project(
        project_path=str(target), output_path=str(ws / ".removed" / "seed.zip"),
        include_models=False,
    )
    assert "error" not in resp
    holding_dir = ws / ".removed" / "sample_plot_target-20260304T120000Z"
    replace(workspace.pending_removal_key(target), {
        "requested_at": "20260304T120000Z", "requested_by": "user:tester",
        "archive_path": resp["output_path"], "holding_dir": str(holding_dir),
        "external_roots": [], "dependent_projects": [],
    }, expect=Version.ABSENT)

    ts.close_connections()  # this process's own cached handle would otherwise deny the rename
    proc = subprocess.run(
        [sys.executable, "-m", "tcip_web.cli", "complete-removals", "--workspace", str(ws)],
        env=dict(os.environ), capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "sample_plot_target" in proc.stdout
    assert not target.exists()
    assert holding_dir.is_dir()


def test_bind_startup_root_moves_a_pending_project_before_serving_a_request(tmp_path):
    """``bind_startup_root`` runs phase two, once, before the marker pin: entering the served
    app's lifespan on a workspace with one pending project moves it before the first response.
    """
    from tcip_mcp import project_paths
    from tcip_store import Version, replace

    ws = tmp_path.parent
    _init(ws, "sample_plot_open")
    target = _init(ws, "sample_plot_target")
    workspace.activate_project("sample_plot_open")
    removed_dir = ws / ".removed"
    removed_dir.mkdir()
    resp = archive_project(
        project_path=str(target), output_path=str(removed_dir / "seed.zip"), include_models=False,
    )
    assert "error" not in resp
    holding_dir = removed_dir / "sample_plot_target-20260304T120000Z"
    replace(workspace.pending_removal_key(target), {
        "requested_at": "20260304T120000Z", "requested_by": "user:tester",
        "archive_path": resp["output_path"], "holding_dir": str(holding_dir),
        "external_roots": [], "dependent_projects": [],
    }, expect=Version.ABSENT)

    # Simulate a fresh process's first bind: no root pinned yet, no connection held.
    project_paths.restore_binding(None)
    ts.close_connections()

    with TestClient(app, base_url="http://127.0.0.1") as fresh_client:
        health = fresh_client.get("/health")
        assert health.status_code == 200
        assert not target.exists()
        assert holding_dir.is_dir()
        listing = fresh_client.get("/api/projects").json()
        outcome = next(o for o in listing["removal_startup_outcomes"] if o["name"] == "sample_plot_target")
        assert outcome["moved_to"] == str(holding_dir)

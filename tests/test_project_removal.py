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
    # coverage: the response carries recorded_in_open_project and audit_note.
    assert body["recorded_in_open_project"] is True
    assert "sample_plot_open" in body["audit_note"]
    assert "''s" not in body["audit_note"]

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


def test_listing_carries_removal_refusal_matching_the_doors_own_answer(client, tmp_path):
    """GET /api/projects carries removal_refusal per project: null for a removable one, and for
    the marker's own project the exact string the door answers a removal request for it with."""
    ws = tmp_path.parent
    _seed(ws)

    listing = client.get("/api/projects").json()
    by_name = {p["name"]: p for p in listing["projects"]}
    assert by_name["sample_plot_target"]["removal_refusal"] is None

    door = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_open", "confirm_name": "sample_plot_open", "user": "t"},
    )
    assert door.status_code == 409

    listing2 = client.get("/api/projects").json()
    by_name2 = {p["name"]: p for p in listing2["projects"]}
    assert by_name2["sample_plot_open"]["removal_refusal"] == door.json()["detail"]


def test_listing_carries_no_reason_naming_no_project_open_with_nothing_bound(
    client, tmp_path, tmp_path_factory, monkeypatch,
):
    """coverage of the landing's own behavior: with nothing bound and no marker set, every card's
    removal_refusal is null and removal_releasable is false, the home of the fact the deleted
    no-project-open vitest once pinned."""
    ws = tmp_path.parent
    ws.mkdir(exist_ok=True)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path_factory.mktemp("unrelated_root")))
    _init(ws, "sample_plot_a")
    _init(ws, "sample_plot_b")

    listing = client.get("/api/projects").json()
    assert listing["projects"]
    for project in listing["projects"]:
        assert project["removal_refusal"] is None
        assert project["removal_releasable"] is False


def test_listing_carries_dependency_warnings_and_problem_through_the_route(client, tmp_path):
    """coverage: dependency_warnings/dependency_problem, converted through DependencyWarning,
    get their assertion through GET /api/projects here, covering the pending, moved and damaged
    cases."""
    from tests._record_damage_fixtures import damage_record
    from tcip_mcp.tools.project_tools import dataset_registry_key, register_dataset

    ws = tmp_path.parent
    open_project, target = _seed(ws)
    dependent = _init(ws, "sample_plot_route-pending")
    reg = register_dataset(str(target), crop="black locust", project_root=str(dependent))
    assert "error" not in reg, reg

    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 200, resp.text

    listing = client.get("/api/projects").json()
    by_name = {p["name"]: p for p in listing["projects"]}
    warning = by_name["sample_plot_route-pending"]["dependency_warnings"][0]
    assert warning["present"] is True
    assert warning["target"] == "sample_plot_target"
    assert warning["archive_path"] == resp.json()["archive_path"]

    project_removal.complete_pending_removals(ws)
    listing2 = client.get("/api/projects").json()
    by_name2 = {p["name"]: p for p in listing2["projects"]}
    warning2 = by_name2["sample_plot_route-pending"]["dependency_warnings"][0]
    assert warning2["present"] is False
    assert warning2["archive_path"] is None

    damaged = _init(ws, "sample_plot_route-damaged")
    reg2 = register_dataset(str(damaged), crop="black locust", project_root=str(damaged))
    assert "error" not in reg2, reg2
    damage_record(dataset_registry_key(damaged), b"not json")

    listing3 = client.get("/api/projects").json()
    by_name3 = {p["name"]: p for p in listing3["projects"]}
    damaged_entry = by_name3["sample_plot_route-damaged"]
    assert damaged_entry["dependency_warnings"] == []
    assert damaged_entry["dependency_problem"] is not None


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


def test_a_malformed_name_with_a_mismatched_confirmation_answers_the_invalid_name_text(client, tmp_path):
    """The name-shape checks run ahead of the confirm-name mismatch check: a malformed name
    answers with its own text, whatever the mismatched confirm_name carries."""
    ws = tmp_path.parent
    _seed(ws)
    resp = client.post(
        "/api/projects/remove",
        json={"name": "../escape", "confirm_name": "something else entirely", "user": "t"},
    )
    assert resp.status_code == 400
    assert "invalid project name" in resp.json()["detail"]


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


def test_the_pending_marker_refusal_names_the_plain_project_with_no_doubled_apostrophe(
    client, tmp_path,
):
    """coverage: the message already names the plain project rather than a quoted repr; this
    pins that a damaged pending-removal marker still reads that way."""
    from tests._record_damage_fixtures import damage_record

    ws = tmp_path.parent
    _seed(ws)
    first = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert first.status_code == 200
    target = ws / "sample_plot_target"
    damage_record(workspace.pending_removal_key(target), b"not json")

    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert "sample_plot_target's pending-removal marker could not be read" in detail
    assert "''" not in detail


def test_admits_the_request_with_no_project_bound_and_records_both_lines_under_the_target(
    client, tmp_path, tmp_path_factory, monkeypatch,
):
    """``tmp_path`` (``<workspace>/project``) is the autouse ``_pin_platform_root`` fixture's own
    inherited platform root, and any bare-``@audited`` write under it (``initialize_project``'s
    own) would otherwise leave a ``.tcip`` there that reads as an accidental project; pin the
    root somewhere outside the workspace instead so nothing is genuinely bound. Admitted rather
    than refused: the target's own log carries both of the request's own lines, and the archive
    door's own line stays under the platform root the pinned variable names, unmoved by which
    project (if any) is bound."""
    from tcip_mcp import audit

    ws = tmp_path.parent
    ws.mkdir(exist_ok=True)
    platform_root = tmp_path_factory.mktemp("unrelated_root")
    monkeypatch.setenv("TCIP_STATE_ROOT", str(platform_root))
    target = _init(ws, "sample_plot_target")
    _add_image(target)

    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["recorded_in_open_project"] is False
    assert body["audit_scope"] == str(target)
    assert "no project open" in body["audit_note"]
    assert platform_root.name in body["audit_note"]

    target_lines = _audit_lines(target)
    assert [line["tool"] for line in target_lines] == [
        "project_removal_requested", "gui_project_removal_requested",
    ]

    # coverage: the archive door's own line stays under the root the pinned variable names, and
    # no request line rides behind it there.
    platform_tools = [line["tool"] for line in ts.read_log(audit.audit_log_key()).records]
    assert platform_tools[-1] == "archive_project"
    assert "project_removal_requested" not in platform_tools
    assert "gui_project_removal_requested" not in platform_tools


def test_refuses_the_markers_own_project_with_no_platform_root_bound(
    client, tmp_path, tmp_path_factory, monkeypatch,
):
    """A guard: the marker's own project's removal request, with no platform root bound, refuses
    naming the marker's own text ("opens by default") rather than the generic "no project is
    open" string. The first request runs before the marker is written, so the backend's own
    startup binding falls to the inherited (unrelated) root rather than adopting the marker's own
    project, keeping "no platform root bound" genuinely true afterward."""
    ws = tmp_path.parent
    ws.mkdir(exist_ok=True)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path_factory.mktemp("unrelated_root")))
    assert client.get("/health").status_code == 200
    _init(ws, "sample_plot_marker")
    ts.replace(workspace.active_project_key(), "sample_plot_marker")

    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_marker", "confirm_name": "sample_plot_marker", "user": "t"},
    )
    assert resp.status_code == 409
    assert "opens by default" in resp.json()["detail"]


def test_refuses_the_canvas_bound_project_with_no_platform_root_bound(
    client, tmp_path, tmp_path_factory, monkeypatch,
):
    """A guard through the message alone: the canvas-bound project's removal request refuses
    naming "the GUI has open" in the detail, not just the same 409 status."""
    ws = tmp_path.parent
    ws.mkdir(exist_ok=True)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path_factory.mktemp("unrelated_root")))
    target = _init(ws, "sample_plot_canvas")
    _add_image(target)

    sel = client.post(
        "/api/dataset/select", json={"project_root": str(target), "dataset_root": str(target)},
    )
    assert sel.status_code == 200

    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_canvas", "confirm_name": "sample_plot_canvas", "user": "t"},
    )
    assert resp.status_code == 409
    assert "the GUI has open" in resp.json()["detail"]


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


def test_refuses_a_platform_root_bound_to_a_project_the_marker_does_not_name(client, tmp_path):
    """The platform-root spelling of "the open project" fires on its own, independent of the
    marker: this process's own bound root answers with its own text even when the marker names
    a different project."""
    from tcip_mcp import project_paths

    ws = tmp_path.parent
    open_project, target = _seed(ws)
    second = _init(ws, "sample_plot_second")

    before = project_paths.root_binding()
    project_paths.restore_binding(
        project_paths.RootBinding(
            root=second, source="adopted", inherited_root=None, marker_problem=None,
        )
    )
    try:
        resp = client.post(
            "/api/projects/remove",
            json={"name": "sample_plot_second", "confirm_name": "sample_plot_second", "user": "t"},
        )
        assert resp.status_code == 409
        assert "restart the backend first" in resp.json()["detail"]
        assert "state root naming this project" in resp.json()["detail"]
        assert "TCIP_STATE_ROOT" not in resp.json()["detail"]
    finally:
        project_paths.restore_binding(before)


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
    assert entry_before.get("present") is True

    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_external", "confirm_name": "sample_plot_external", "user": "t"},
    )
    assert resp.status_code == 200
    project_removal.complete_pending_removals(ws)
    assert not external.exists()

    preview_after = client.get("/api/projects/sample_plot_target/removal-preview").json()
    entry_after = next(r for r in preview_after["external_roots"] if r["path"] == str(external))
    assert entry_after.get("present") is False


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


def test_a_dependents_own_log_carries_the_dependency_line(client, tmp_path):
    """A guard: the dependent's own log carries a dependency_pending_removal line for the
    request."""
    from tcip_mcp.tools.project_tools import register_dataset

    ws = tmp_path.parent
    open_project, target = _seed(ws)
    dependent = _init(ws, "sample_plot_dependent")
    reg = register_dataset(str(target), crop="black locust", project_root=str(dependent))
    assert "error" not in reg, reg

    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 200, resp.text

    dep_lines = _audit_lines(dependent)
    line = next(l for l in dep_lines if l["tool"] == "dependency_pending_removal")
    assert line["arguments"]["target"] == "sample_plot_target"
    assert reg["id"] in line["arguments"]["dataset_ids"]


def test_a_dependent_registering_two_datasets_gets_one_line(client, tmp_path):
    from tcip_mcp.tools.project_tools import register_dataset

    ws = tmp_path.parent
    open_project, target = _seed(ws)
    sub = target / "sub_dataset"
    sub.mkdir()
    dependent = _init(ws, "sample_plot_two-datasets")

    reg1 = register_dataset(str(target), crop="black locust", project_root=str(dependent))
    reg2 = register_dataset(str(sub), crop="black locust", project_root=str(dependent))
    assert "error" not in reg1, reg1
    assert "error" not in reg2, reg2

    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 200, resp.text

    dep_lines = [l for l in _audit_lines(dependent) if l["tool"] == "dependency_pending_removal"]
    assert len(dep_lines) == 1
    assert set(dep_lines[0]["arguments"]["dataset_ids"]) == {reg1["id"], reg2["id"]}


def test_a_pending_dependent_still_gets_its_own_line(client, tmp_path):
    from tcip_mcp.tools.project_tools import register_dataset

    ws = tmp_path.parent
    open_project, target = _seed(ws)
    dependent = _init(ws, "sample_plot_dep-pending")
    reg = register_dataset(str(target), crop="black locust", project_root=str(dependent))
    assert "error" not in reg, reg

    pending_resp = client.post(
        "/api/projects/remove",
        json={
            "name": "sample_plot_dep-pending", "confirm_name": "sample_plot_dep-pending",
            "user": "t",
        },
    )
    assert pending_resp.status_code == 200, pending_resp.text

    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 200, resp.text
    dep = next(
        d for d in resp.json()["dependent_projects"] if d["project"] == "sample_plot_dep-pending"
    )
    assert dep["pending"] is True

    dep_lines = _audit_lines(dependent)
    assert any(l["tool"] == "dependency_pending_removal" for l in dep_lines)


def test_an_unreadable_dependent_gets_no_dependency_line(client, tmp_path):
    """coverage: a registry the door cannot read never gets a dependency line either."""
    from tests._record_damage_fixtures import damage_record

    from tcip_mcp.tools.project_tools import dataset_registry_key, register_dataset

    ws = tmp_path.parent
    open_project, target = _seed(ws)
    damaged = _init(ws, "sample_plot_damaged-dep")
    reg = register_dataset(str(target), crop="black locust", project_root=str(damaged))
    assert "error" not in reg, reg
    damage_record(dataset_registry_key(damaged), b"not json")

    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 200, resp.text
    unreadable_names = {
        d.get("project") for d in resp.json()["dependent_projects"] if d.get("unreadable")
    }
    assert "sample_plot_damaged-dep" in unreadable_names
    damaged_lines = _audit_lines(damaged)
    assert not any(l["tool"] == "dependency_pending_removal" for l in damaged_lines)


def test_an_entry_with_a_path_but_no_id_is_listed_as_unreadable_with_no_line(client, tmp_path):
    from tcip_mcp.tools.project_tools import registry_path_for, upsert_dataset

    ws = tmp_path.parent
    open_project, target = _seed(ws)
    malformed = _init(ws, "sample_plot_malformed")
    upsert_dataset(
        malformed, {"path": registry_path_for(target, malformed), "crop": "black locust"},
    )

    preview = client.get("/api/projects/sample_plot_target/removal-preview").json()
    entry = next(d for d in preview["dependent_projects"] if d["project"] == "sample_plot_malformed")
    assert entry.get("unreadable") is not None
    assert "no id" in entry["unreadable"]

    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 200, resp.text
    malformed_lines = _audit_lines(malformed)
    assert not any(l["tool"] == "dependency_pending_removal" for l in malformed_lines)


def test_a_dependents_line_refused_by_a_raising_append_names_it_and_leaves_the_others(
    client, tmp_path, monkeypatch,
):
    import tcip_store
    from tcip_mcp import audit
    from tcip_mcp.tools.project_tools import register_dataset

    ws = tmp_path.parent
    open_project, target = _seed(ws)
    ok_dependent = _init(ws, "sample_plot_dep-ok")
    bad_dependent = _init(ws, "sample_plot_dep-bad")
    reg_ok = register_dataset(str(target), crop="black locust", project_root=str(ok_dependent))
    reg_bad = register_dataset(str(target), crop="black locust", project_root=str(bad_dependent))
    assert "error" not in reg_ok, reg_ok
    assert "error" not in reg_bad, reg_bad

    real_append = tcip_store.append
    bad_root = str(bad_dependent.resolve())

    def _flaky_append(key, record):
        if record.get("tool") == "dependency_pending_removal" and key.root == bad_root:
            raise RuntimeError("simulated append failure")
        return real_append(key, record)

    monkeypatch.setattr(audit, "append", _flaky_append)

    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert "sample_plot_dep-bad" in detail
    assert "marker" in detail

    ok_lines = _audit_lines(ok_dependent)
    assert any(l["tool"] == "dependency_pending_removal" for l in ok_lines)
    bad_lines = _audit_lines(bad_dependent)
    assert not any(l["tool"] == "dependency_pending_removal" for l in bad_lines)
    target_lines = _audit_lines(target)
    assert target_lines[-1]["tool"] == "project_removal_requested"


def test_the_routes_own_line_and_a_dependents_both_refused_answer_one_409(
    client, tmp_path, monkeypatch,
):
    import tcip_store
    from tcip_mcp import audit
    from tcip_mcp.tools.project_tools import register_dataset

    ws = tmp_path.parent
    open_project, target = _seed(ws)
    dependent = _init(ws, "sample_plot_dep-both")
    reg = register_dataset(str(target), crop="black locust", project_root=str(dependent))
    assert "error" not in reg, reg

    real_append = tcip_store.append

    def _flaky_append(key, record):
        if record.get("tool") in ("dependency_pending_removal", "gui_project_removal_requested"):
            raise RuntimeError("simulated append failure")
        return real_append(key, record)

    monkeypatch.setattr(audit, "append", _flaky_append)

    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert "sample_plot_dep-both" in detail
    assert "route" in detail
    assert "''s" not in detail


def test_a_refused_preview_skips_the_dependent_and_external_root_scan(client, tmp_path):
    """A refused request never walks the sibling registries: a dependent registered against the
    marker's own project (refused before any scan) is never found."""
    from tcip_mcp.tools.project_tools import register_dataset

    ws = tmp_path.parent
    open_project, target = _seed(ws)
    dependent = _init(ws, "sample_plot_dependent")
    reg = register_dataset(str(open_project), crop="black locust", project_root=str(dependent))
    assert "error" not in reg, reg

    preview = client.get("/api/projects/sample_plot_open/removal-preview").json()
    assert preview["refusal"] is not None
    assert preview["dependent_projects"] == []
    assert preview["external_roots"] == []


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


# ── dependency_warnings and _workspace_child_of ──────────────────────────────


def test_workspace_child_of_a_missing_path_the_workspace_an_external_root_and_the_holding_dir(
    tmp_path, tmp_path_factory,
):
    from tcip_mcp.project_removal import _workspace_child_of

    ws = tmp_path.parent
    ws.mkdir(exist_ok=True)
    child = _init(ws, "sample_plot_child")

    missing = child / "gone" / "deeper"
    assert _workspace_child_of(missing, ws) == child
    assert _workspace_child_of(ws, ws) is None

    external = tmp_path_factory.mktemp("external_root")
    assert _workspace_child_of(external, ws) is None

    holding = ws / ".removed" / "sample_plot_child-20260304T120000Z"
    holding.mkdir(parents=True)
    assert _workspace_child_of(holding, ws) is None


def test_dependency_warnings_present_true_while_pending_then_false_after_the_move(
    client, tmp_path,
):
    """coverage: dependency_warnings answers present while the target is pending removal, false
    once the holding move completes, and stays that way even if the holding directory is renamed
    by hand afterward."""
    from tcip_mcp.project_removal import dependency_warnings
    from tcip_mcp.tools.project_tools import register_dataset

    ws = tmp_path.parent
    open_project, target = _seed(ws)
    dependent = _init(ws, "sample_plot_warn-dep")
    reg = register_dataset(str(target), crop="black locust", project_root=str(dependent))
    assert "error" not in reg, reg

    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 200, resp.text

    warnings, problem = dependency_warnings(dependent)
    assert problem is None
    assert len(warnings) == 1
    assert warnings[0]["present"] is True
    assert warnings[0]["target"] == "sample_plot_target"
    assert warnings[0]["dataset_id"] == reg["id"]

    project_removal.complete_pending_removals(ws)

    warnings2, _ = dependency_warnings(dependent)
    assert len(warnings2) == 1
    assert warnings2[0]["present"] is False

    holding = Path(resp.json()["holding_dir"])
    renamed = holding.parent / "renamed_by_hand"
    holding.rename(renamed)
    warnings3, _ = dependency_warnings(dependent)
    assert warnings3 == warnings2


def test_dependency_warnings_empty_for_an_entry_under_a_live_project(tmp_path):
    from tcip_mcp.project_removal import dependency_warnings
    from tcip_mcp.tools.project_tools import register_dataset

    ws = tmp_path.parent
    ws.mkdir(exist_ok=True)
    live = _init(ws, "sample_plot_live")
    dependent = _init(ws, "sample_plot_live-dep")
    reg = register_dataset(str(live), crop="black locust", project_root=str(dependent))
    assert "error" not in reg, reg

    warnings, problem = dependency_warnings(dependent)
    assert warnings == []
    assert problem is None


def test_dependency_warnings_clears_once_reregistered_from_the_moved_tree(client, tmp_path):
    from tcip_mcp.project_removal import dependency_warnings
    from tcip_mcp.tools.project_tools import register_dataset

    ws = tmp_path.parent
    open_project, target = _seed(ws)
    dependent = _init(ws, "sample_plot_reregister-dep")
    reg = register_dataset(str(target), crop="black locust", project_root=str(dependent))
    assert "error" not in reg, reg

    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 200, resp.text
    project_removal.complete_pending_removals(ws)
    holding_dir = Path(resp.json()["holding_dir"])

    reg2 = register_dataset(str(holding_dir), crop="black locust", project_root=str(dependent))
    assert "error" not in reg2, reg2

    warnings, _ = dependency_warnings(dependent)
    assert warnings == []


def test_dependency_problem_named_for_a_damaged_registry(tmp_path):
    from tests._record_damage_fixtures import damage_record

    from tcip_mcp.project_removal import dependency_warnings
    from tcip_mcp.tools.project_tools import dataset_registry_key, register_dataset

    ws = tmp_path.parent
    ws.mkdir(exist_ok=True)
    project = _init(ws, "sample_plot_damaged-self")
    reg = register_dataset(str(project), crop="black locust", project_root=str(project))
    assert "error" not in reg, reg
    damage_record(dataset_registry_key(project), b"not json")

    warnings, problem = dependency_warnings(project)
    assert warnings == []
    assert problem is not None


def test_dependency_warnings_names_a_no_id_entry_as_a_problem_not_a_null_id_warning(
    client, tmp_path,
):
    """coverage: a no-id entry already routes to the registry problem. The entry names a target
    already pending removal, the one state a no-id entry would otherwise reach the warning
    branch through, so this exercises the malformed-entry rule rather than a state (a live
    target) that never turns into a warning."""
    from tcip_mcp.project_removal import dependency_warnings
    from tcip_mcp.tools.project_tools import registry_path_for, upsert_dataset

    ws = tmp_path.parent
    open_project, target = _seed(ws)
    dependent = _init(ws, "sample_plot_no-id")
    upsert_dataset(
        dependent, {"path": registry_path_for(target, dependent), "crop": "black locust"},
    )

    resp = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 200, resp.text

    warnings, problem = dependency_warnings(dependent)
    assert warnings == []
    assert problem is not None
    assert "no id" in problem

    listing = client.get("/api/projects").json()
    entry = next(p for p in listing["projects"] if p["name"] == "sample_plot_no-id")
    assert entry["dependency_warnings"] == []
    assert entry["dependency_problem"] == problem


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


# ── the release door ──────────────────────────────────────────────────────────


def test_release_clears_the_marker_and_the_canvas_binding_both_naming_the_project(
    client, tmp_path,
):
    """coverage: release clears both the marker and the canvas binding, reports both in the
    route's own response, bumps the canvas binding's generation and records its audit line."""
    from tcip_mcp.web_client import read_canvas_binding

    ws = tmp_path.parent
    _seed(ws)
    target = _init(ws, "sample_plot_release")
    workspace.activate_project("sample_plot_release")
    sel = client.post(
        "/api/dataset/select", json={"project_root": str(target), "dataset_root": str(target)},
    )
    assert sel.status_code == 200
    before = read_canvas_binding()

    resp = client.post(
        "/api/projects/sample_plot_release/release-binding", json={"user": "t"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["marker_cleared"] is True
    assert body["canvas_binding_released"] is True
    assert body["releasable"] is False

    assert workspace.read_active_project() is None
    after = read_canvas_binding()
    assert after["generation"] == before["generation"] + 1
    assert after["released"] is True

    lines = _audit_lines(target)
    line = next(l for l in lines if l["tool"] == "project_binding_released")
    assert line["arguments"]["marker_cleared"] is True
    assert line["arguments"]["canvas_binding_released"] is True


def test_release_clears_the_marker_alone(client, tmp_path):
    ws = tmp_path.parent
    _seed(ws)
    _init(ws, "sample_plot_marker-only")
    workspace.activate_project("sample_plot_marker-only")

    resp = client.post(
        "/api/projects/sample_plot_marker-only/release-binding", json={"user": "t"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["marker_cleared"] is True
    assert body["canvas_binding_released"] is False
    assert workspace.read_active_project() is None


def test_release_line_not_written_answers_409_naming_the_plain_project_with_no_doubled_apostrophe(
    client, tmp_path, monkeypatch,
):
    """coverage: the failed-append message already names the plain project rather than a
    quoted repr; this pins that the release door's own line keeps that shape too."""
    from tcip_mcp import audit

    ws = tmp_path.parent
    _seed(ws)
    _init(ws, "sample_plot_release-line-fails")
    workspace.activate_project("sample_plot_release-line-fails")

    real_append = ts.append

    def _flaky_append(key, record):
        if record.get("tool") == "project_binding_released":
            raise RuntimeError("simulated append failure")
        return real_append(key, record)

    monkeypatch.setattr(audit, "append", _flaky_append)

    resp = client.post(
        "/api/projects/sample_plot_release-line-fails/release-binding", json={"user": "t"},
    )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert "sample_plot_release-line-fails's binding was cleared" in detail
    assert "''" not in detail


def test_release_when_neither_names_the_project_answers_both_false_with_no_line(
    client, tmp_path,
):
    ws = tmp_path.parent
    _seed(ws)
    _init(ws, "sample_plot_unbound")

    resp = client.post(
        "/api/projects/sample_plot_unbound/release-binding", json={"user": "t"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["marker_cleared"] is False
    assert body["canvas_binding_released"] is False
    lines = _audit_lines(Path(tmp_path.parent) / "sample_plot_unbound")
    assert not any(l["tool"] == "project_binding_released" for l in lines)


def test_release_leaves_the_marker_in_place_once_it_has_moved_elsewhere(client, tmp_path):
    """coverage of the read-back branch, not a race proof: the marker is moved to another
    project before the call, and release_project_binding reads it back rather than trusting an
    earlier read, so it finds no match and clears nothing."""
    ws = tmp_path.parent
    _seed(ws)
    _init(ws, "sample_plot_moved-away")
    workspace.activate_project("sample_plot_moved-away")
    workspace.activate_project("sample_plot_open")

    resp = client.post(
        "/api/projects/sample_plot_moved-away/release-binding", json={"user": "t"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["marker_cleared"] is False
    assert workspace.read_active_project() == "sample_plot_open"


def test_release_canvas_failure_after_the_marker_cleared_names_both_in_the_line_and_the_409(
    client, tmp_path, monkeypatch,
):
    import tcip_store
    from tcip_mcp import project_removal as pr_module

    ws = tmp_path.parent
    _seed(ws)
    target = _init(ws, "sample_plot_canvas-fail")
    workspace.activate_project("sample_plot_canvas-fail")
    sel = client.post(
        "/api/dataset/select", json={"project_root": str(target), "dataset_root": str(target)},
    )
    assert sel.status_code == 200

    real_transaction = tcip_store.transaction

    def _flaky_transaction(*keys, **kwargs):
        raise tcip_store.StoreBusy(keys, keys[0], 5.0)

    monkeypatch.setattr(pr_module.tcip_store, "transaction", _flaky_transaction)

    resp = client.post(
        "/api/projects/sample_plot_canvas-fail/release-binding", json={"user": "t"},
    )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert "sample_plot_canvas-fail" in detail
    assert "marker_cleared=True" in detail

    monkeypatch.setattr(pr_module.tcip_store, "transaction", real_transaction)
    lines = _audit_lines(target)
    line = next(l for l in lines if l["tool"] == "project_binding_released")
    assert line["arguments"]["marker_cleared"] is True
    assert line["arguments"]["canvas_binding_released"] is False


def test_release_canvas_record_decode_error_after_the_marker_cleared_still_records_the_line(
    client, tmp_path,
):
    """A guard: a DecodeError from the canvas record's read is folded into the 409, with the
    marker already cleared and the project_binding_released line recorded naming it, rather than
    escaping the route uncaught."""
    from tests._record_damage_fixtures import damage_record
    from tcip_mcp.web_client import canvas_open_binding_key

    ws = tmp_path.parent
    _seed(ws)
    target = _init(ws, "sample_plot_canvas-decode")
    workspace.activate_project("sample_plot_canvas-decode")
    sel = client.post(
        "/api/dataset/select", json={"project_root": str(target), "dataset_root": str(target)},
    )
    assert sel.status_code == 200

    damage_record(canvas_open_binding_key(), b"not json")

    resp = client.post(
        "/api/projects/sample_plot_canvas-decode/release-binding", json={"user": "t"},
    )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert "sample_plot_canvas-decode's canvas-open binding failed" in detail
    assert "''" not in detail
    assert "marker_cleared=True" in detail

    lines = _audit_lines(target)
    line = next(l for l in lines if l["tool"] == "project_binding_released")
    assert line["arguments"]["marker_cleared"] is True
    assert line["arguments"]["canvas_binding_released"] is False


def test_release_canvas_record_missing_generation_after_the_marker_cleared_still_records_the_line(
    client, tmp_path,
):
    """A guard: a KeyError on the record's missing generation field is folded into the 409, with
    the marker already cleared and the project_binding_released line recorded naming it, rather
    than escaping the route uncaught."""
    from tcip_mcp.web_client import canvas_open_binding_key

    ws = tmp_path.parent
    _seed(ws)
    target = _init(ws, "sample_plot_canvas-no-gen")
    workspace.activate_project("sample_plot_canvas-no-gen")

    ts.replace(
        canvas_open_binding_key(),
        {"root": str(target), "project_name": "sample_plot_canvas-no-gen",
         "issued_at": "2026-03-04T12:00:00+00:00"},
        expect=ts.Version.ABSENT,
    )

    resp = client.post(
        "/api/projects/sample_plot_canvas-no-gen/release-binding", json={"user": "t"},
    )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert "sample_plot_canvas-no-gen" in detail
    assert "marker_cleared=True" in detail

    lines = _audit_lines(target)
    line = next(l for l in lines if l["tool"] == "project_binding_released")
    assert line["arguments"]["marker_cleared"] is True
    assert line["arguments"]["canvas_binding_released"] is False


@pytest.mark.parametrize(
    "record",
    [
        {"generation": "3", "root": None, "project_name": None, "issued_at": "2026-03-04T12:00:00+00:00"},
        {"generation": 3, "root": ["not", "a", "path"], "issued_at": "2026-03-04T12:00:00+00:00"},
        ["not", "a", "mapping"],
    ],
    ids=["generation_not_an_integer", "root_not_a_path", "record_not_a_mapping"],
)
def test_release_canvas_record_of_a_malformed_shape_after_the_marker_cleared_still_records_the_line(
    client, tmp_path, record,
):
    """A guard: a string generation, a root that is not a path, or a record that is not a
    mapping is folded into the 409 the same way, with the marker already cleared and the
    project_binding_released line recorded naming it, rather than raising a TypeError or an
    AttributeError out of the route."""
    from tcip_mcp.web_client import canvas_open_binding_key

    ws = tmp_path.parent
    _seed(ws)
    target = _init(ws, "sample_plot_canvas-shape")
    workspace.activate_project("sample_plot_canvas-shape")
    if isinstance(record, dict) and record.get("root") is None:
        record = {**record, "root": str(target)}

    ts.replace(canvas_open_binding_key(), record, expect=ts.Version.ABSENT)

    resp = client.post(
        "/api/projects/sample_plot_canvas-shape/release-binding", json={"user": "t"},
    )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert "marker_cleared=True" in detail

    lines = _audit_lines(target)
    line = next(l for l in lines if l["tool"] == "project_binding_released")
    assert line["arguments"]["marker_cleared"] is True
    assert line["arguments"]["canvas_binding_released"] is False


def test_release_with_an_unreadable_marker_clears_nothing_for_it_and_does_not_500(
    client, tmp_path,
):
    """A guard: a damaged marker folds to "no marker" the way its sibling reader does, rather
    than raising out of the route as an unhandled 500."""
    from tests._record_damage_fixtures import damage_record

    ws = tmp_path.parent
    _seed(ws)
    target = _init(ws, "sample_plot_marker-damaged")
    sel = client.post(
        "/api/dataset/select", json={"project_root": str(target), "dataset_root": str(target)},
    )
    assert sel.status_code == 200
    damage_record(workspace.active_project_key(), b"not utf-8 or a name at all \xff\xfe")

    resp = client.post(
        "/api/projects/sample_plot_marker-damaged/release-binding", json={"user": "t"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["marker_cleared"] is False
    assert body["canvas_binding_released"] is True


def test_release_then_a_select_bumps_the_generation_again(client, tmp_path):
    from tcip_mcp.web_client import read_canvas_binding

    ws = tmp_path.parent
    _seed(ws)
    target = _init(ws, "sample_plot_reselect")
    sel = client.post(
        "/api/dataset/select", json={"project_root": str(target), "dataset_root": str(target)},
    )
    assert sel.status_code == 200
    before = read_canvas_binding()

    resp = client.post(
        "/api/projects/sample_plot_reselect/release-binding", json={"user": "t"},
    )
    assert resp.status_code == 200, resp.text
    after_release = read_canvas_binding()
    assert after_release["generation"] == before["generation"] + 1

    sel2 = client.post(
        "/api/dataset/select", json={"project_root": str(target), "dataset_root": str(target)},
    )
    assert sel2.status_code == 200
    after_select = read_canvas_binding()
    assert after_select["generation"] == after_release["generation"] + 1
    assert not after_select.get("released")


def test_canvas_push_carrying_the_pre_release_generation_is_refused(client, tmp_path):
    ws = tmp_path.parent
    _seed(ws)
    target = _init(ws, "sample_plot_push-refused")
    _add_image(target)
    sel = client.post(
        "/api/dataset/select", json={"project_root": str(target), "dataset_root": str(target)},
    )
    assert sel.status_code == 200
    pre_release_generation = sel.json()["generation"]

    resp = client.post(
        "/api/projects/sample_plot_push-refused/release-binding", json={"user": "t"},
    )
    assert resp.status_code == 200, resp.text

    push = client.post(
        "/api/canvas/state",
        json={
            "binding_generation": pre_release_generation, "tab": "annotate",
            "image_path": str(target / "images" / "2026-03-04" / "img.jpg"), "image": "img.jpg",
        },
    )
    assert push.status_code == 409


def test_gui_binding_matches_is_false_on_a_released_record(client, tmp_path):
    from tcip_mcp.web_client import gui_binding_matches

    ws = tmp_path.parent
    _seed(ws)
    target = _init(ws, "sample_plot_matches-false")
    sel = client.post(
        "/api/dataset/select", json={"project_root": str(target), "dataset_root": str(target)},
    )
    assert sel.status_code == 200

    resp = client.post(
        "/api/projects/sample_plot_matches-false/release-binding", json={"user": "t"},
    )
    assert resp.status_code == 200, resp.text

    matches, binding = gui_binding_matches(str(target))
    assert matches is False
    assert binding is not None
    assert binding["released"] is True


def test_release_route_answers_404_for_an_unknown_project(client, tmp_path):
    """coverage: the route answers 404 for an unknown project, the same status FastAPI's own
    unmatched-route 404 would give."""
    ws = tmp_path.parent
    _seed(ws)
    resp = client.post(
        "/api/projects/sample_plot_absent/release-binding", json={"user": "t"},
    )
    assert resp.status_code == 404


def test_release_response_refusal_names_the_bound_root_for_the_backends_own_project(
    client, tmp_path,
):
    """After release, the backend's own bound project (the process was started on it) still
    refuses removal through the bound-root spelling naming a restart; a different project's
    fresh state answers refusal null."""
    from tcip_mcp import project_paths

    ws = tmp_path.parent
    open_project, target = _seed(ws)
    second = _init(ws, "sample_plot_second-bound")

    before = project_paths.root_binding()
    project_paths.restore_binding(
        project_paths.RootBinding(
            root=second, source="adopted", inherited_root=None, marker_problem=None,
        )
    )
    try:
        resp = client.post(
            "/api/projects/sample_plot_second-bound/release-binding", json={"user": "t"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["refusal"] is not None
        assert "restart the backend first" in body["refusal"]
        assert body["releasable"] is False

        resp2 = client.post(
            "/api/projects/sample_plot_target/release-binding", json={"user": "t"},
        )
        assert resp2.status_code == 200, resp2.text
        assert resp2.json()["refusal"] is None
    finally:
        project_paths.restore_binding(before)


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
    assert open_lines[-1]["status"] == "ok"


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
    """A second process holding the target's database while phase two runs. Only Windows denies
    the rename of a directory with an open handle inside it, so the denied outcome (`blocked_by`
    and `blocked_errno`) has content there alone, where CI's own job selects this test by node
    id; on POSIX the same held handle does not stop the rename, and the test asserts that the
    move lands under the holder instead, as the module docstring states."""
    import errno
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
            if os.name == "nt":
                assert "blocked_by" in outcomes[0]
                # Windows denies the rename of a directory holding an open sqlite handle with
                # a sharing-violation OSError that Python maps to EACCES, not EPERM.
                assert outcomes[0]["blocked_errno"] == errno.EACCES
                assert target.is_dir()
            else:
                assert outcomes[0].get("moved_to") is not None
                assert not target.exists()
        finally:
            held.close()

        if os.name == "nt":
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

"""Tests for project rename: ``tcip_mcp.project_rename`` and the ``/api/projects`` rename routes.

Every project a test needs is seeded through the platform's own producers
(``initialize_project``, ``register_dataset``), through the app's test client with
``TCIP_WORKSPACE``/``TCIP_STATE_ROOT`` under ``tmp_path`` (overridden in ``conftest.py`` to
``<workspace>/project``); these tests use ``tmp_path.parent`` as the workspace and name their own
projects under it, the same pattern ``test_project_removal.py`` uses.
"""

from __future__ import annotations

import ast
import os
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import tcip_store as ts
from tcip_mcp import project_removal, project_rename, workspace
from tcip_mcp.tools.project_tools import initialize_project, register_dataset
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
    """An open project (activated) and a renamable target project, both under ``ws``."""
    open_project = _init(ws, "sample_plot_open")
    target = _init(ws, "sample_plot_target")
    _add_image(target)
    workspace.activate_project("sample_plot_open")
    return open_project, target


def _audit_lines(root: Path) -> list[dict]:
    from tcip_mcp import audit

    return list(ts.read_log(audit.audit_log_key(root)).records)


# ── the request: admits valid work end to end ────────────────────────────────


def test_request_marks_hides_and_a_completed_walk_renames_the_project(client, tmp_path):
    ws = tmp_path.parent
    _, target = _seed(ws)

    resp = client.post(
        "/api/projects/rename",
        json={"name": "sample_plot_target", "new_name": "sample_plot_renamed",
              "confirm_name": "sample_plot_target", "user": "tester"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert target.is_dir()  # nothing moved yet: phase one only marks
    assert body["new_name"] == "sample_plot_renamed"
    assert body["recorded_in_open_project"] is True
    assert "sample_plot_open" in body["audit_note"]

    pending = workspace.pending_rename_record(target)
    assert pending is not None
    assert pending["new_name"] == "sample_plot_renamed"

    with pytest.raises(workspace.ProjectPendingRename):
        workspace.adoptable_project_root("sample_plot_target")
    assert workspace.workspace_project_name(target) == "sample_plot_target"

    listing = client.get("/api/projects").json()
    names = {p["name"] for p in listing["projects"]}
    assert "sample_plot_target" not in names
    assert listing["pending_rename"][0] == {
        "name": "sample_plot_target", "new_name": "sample_plot_renamed",
        "requested_at": pending["requested_at"],
    }

    outcomes = project_rename.complete_pending_renames(ws)
    by_name = {o["name"]: o for o in outcomes}
    assert by_name["sample_plot_target"]["new_name"] == "sample_plot_renamed"
    assert not target.exists()
    renamed = ws / "sample_plot_renamed"
    assert renamed.is_dir()
    assert workspace.pending_rename_record(renamed) is None

    completion_lines = _audit_lines(renamed)
    assert completion_lines[-1]["tool"] == "project_rename_completed"
    assert completion_lines[-1]["arguments"]["old_root"] == str(target)

    listing2 = client.get("/api/projects").json()
    names2 = {p["name"] for p in listing2["projects"]}
    assert "sample_plot_renamed" in names2
    assert "sample_plot_target" not in names2


def test_rename_preview_matches_the_doors_own_refusal(client, tmp_path):
    ws = tmp_path.parent
    _seed(ws)

    preview = client.get("/api/projects/sample_plot_target/rename-preview").json()
    assert preview["refusal"] is None
    assert preview["dependent_projects"] == []
    assert preview["records_present"] == []


# ── the rename's own refusals ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "new_name,expected_fragment",
    [
        ("not/valid", "invalid project name"),
        ("sample_plot_target", "own current name"),
        ("not-three-segments", "crop_subject_phenotype"),
    ],
)
def test_new_name_shape_refusals(client, tmp_path, new_name, expected_fragment):
    ws = tmp_path.parent
    _seed(ws)

    resp = client.post(
        "/api/projects/rename",
        json={"name": "sample_plot_target", "new_name": new_name,
              "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 400
    assert expected_fragment in resp.json()["detail"]


def test_taken_new_name_refuses_409(client, tmp_path):
    ws = tmp_path.parent
    _seed(ws)
    _init(ws, "sample_plot_taken")

    resp = client.post(
        "/api/projects/rename",
        json={"name": "sample_plot_target", "new_name": "sample_plot_taken",
              "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 409
    assert "already taken" in resp.json()["detail"]


def test_mistyped_confirm_refuses_400(client, tmp_path):
    ws = tmp_path.parent
    _seed(ws)

    resp = client.post(
        "/api/projects/rename",
        json={"name": "sample_plot_target", "new_name": "sample_plot_renamed",
              "confirm_name": "not-the-name", "user": "t"},
    )
    assert resp.status_code == 400
    assert "does not match" in resp.json()["detail"]


# ── the shared refusal chain (project_removal._ordered_refusal) ──────────────


def test_the_open_project_refuses_by_the_shared_chain(client, tmp_path):
    ws = tmp_path.parent
    _seed(ws)

    resp = client.post(
        "/api/projects/rename",
        json={"name": "sample_plot_open", "new_name": "sample_plot_open_2",
              "confirm_name": "sample_plot_open", "user": "t"},
    )
    assert resp.status_code == 409
    assert "opens by default" in resp.json()["detail"]


def test_a_live_run_refuses_through_the_shared_chain(client, tmp_path):
    import datetime

    from tcip_mcp import experiments

    ws = tmp_path.parent
    _, target = _seed(ws)
    ts.replace(
        experiments.status_key("exp1", root=target),
        {"experiment_id": "exp1", "state": "created",
         "heartbeat": datetime.datetime.now(datetime.timezone.utc).isoformat()},
        expect=ts.Version.ABSENT,
    )

    resp = client.post(
        "/api/projects/rename",
        json={"name": "sample_plot_target", "new_name": "sample_plot_renamed",
              "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 409
    assert "running" in resp.json()["detail"]


def test_a_project_pending_removal_refuses_rename_and_names_when_it_completes(client, tmp_path):
    ws = tmp_path.parent
    _, target = _seed(ws)
    remove = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert remove.status_code == 200

    resp = client.post(
        "/api/projects/rename",
        json={"name": "sample_plot_target", "new_name": "sample_plot_renamed",
              "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 409
    assert "is pending removal" in resp.json()["detail"]
    assert "tcip complete-removals" in resp.json()["detail"]


def test_a_project_pending_rename_refuses_a_second_rename_request(client, tmp_path):
    ws = tmp_path.parent
    _seed(ws)
    first = client.post(
        "/api/projects/rename",
        json={"name": "sample_plot_target", "new_name": "sample_plot_first",
              "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert first.status_code == 200

    second = client.post(
        "/api/projects/rename",
        json={"name": "sample_plot_target", "new_name": "sample_plot_second",
              "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert second.status_code == 409
    assert "already has a pending-rename marker" in second.json()["detail"]


def test_the_removal_door_refuses_a_project_pending_rename(client, tmp_path):
    ws = tmp_path.parent
    _seed(ws)
    rename = client.post(
        "/api/projects/rename",
        json={"name": "sample_plot_target", "new_name": "sample_plot_renamed",
              "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert rename.status_code == 200

    remove = client.post(
        "/api/projects/remove",
        json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert remove.status_code == 409
    assert "is pending rename" in remove.json()["detail"]
    assert "can be removed then" in remove.json()["detail"]


def test_the_shared_lock_serializes_a_removal_and_a_rename_request(client, tmp_path):
    ws = tmp_path.parent
    _seed(ws)

    responses: list = []

    def _remove() -> None:
        responses.append(client.post(
            "/api/projects/remove",
            json={"name": "sample_plot_target", "confirm_name": "sample_plot_target", "user": "t"},
        ))

    def _rename() -> None:
        responses.append(client.post(
            "/api/projects/rename",
            json={"name": "sample_plot_target", "new_name": "sample_plot_renamed",
                  "confirm_name": "sample_plot_target", "user": "t"},
        ))

    threads = [threading.Thread(target=_remove), threading.Thread(target=_rename)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    statuses = sorted(resp.status_code for resp in responses)
    assert statuses == [200, 409]
    # Exactly one marker was written, never both.
    target = ws / "sample_plot_target"
    has_removal = workspace.pending_removal_or_none(target) is not None
    has_rename = workspace.pending_rename_or_none(target) is not None
    assert has_removal != has_rename


# ── decision 4: the records refusal ───────────────────────────────────────────


def test_records_refusal_names_experiments(client, tmp_path):
    from tcip_mcp import experiments

    ws = tmp_path.parent
    _, target = _seed(ws)
    ts.replace(
        experiments.status_key("exp1", root=target),
        {"experiment_id": "exp1", "state": "created"},
        expect=ts.Version.ABSENT,
    )

    resp = client.post(
        "/api/projects/rename",
        json={"name": "sample_plot_target", "new_name": "sample_plot_renamed",
              "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 409
    assert "experiments" in resp.json()["detail"]


def test_records_refusal_names_plant_mapping(client, tmp_path):
    from tcip_mcp.pipelines.postprocessing.plant_mapping import plant_mapping_key

    ws = tmp_path.parent
    _, target = _seed(ws)
    ts.replace(plant_mapping_key(target, "a-mapping"), {"name": "a-mapping"},
               expect=ts.Version.ABSENT)

    resp = client.post(
        "/api/projects/rename",
        json={"name": "sample_plot_target", "new_name": "sample_plot_renamed",
              "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 409
    assert "plant_mapping" in resp.json()["detail"]


def test_records_refusal_names_plant_registries(client, tmp_path):
    from tcip_mcp.pipelines.postprocessing import plant_mapping

    ws = tmp_path.parent
    _, target = _seed(ws)
    csv_path = target / "plants.csv"
    csv_path.write_text(
        "plot_name,accession_name,WGS84_centroid_x,WGS84_centroid_y\n"
        "p1,a1,-88.0,43.0\n", encoding="utf-8",
    )
    record = plant_mapping.register_plant_registry_record(
        target, "valley-plants", [csv_path], crop="black locust", site="north",
        registered_by="user:t",
    )
    assert record["n_plants"] == 1

    resp = client.post(
        "/api/projects/rename",
        json={"name": "sample_plot_target", "new_name": "sample_plot_renamed",
              "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 409
    assert "plant_registries" in resp.json()["detail"]


def test_records_refusal_names_delivery_events(client, tmp_path):
    from tcip_mcp.pipelines.resolution import delivery_event_key, delivery_events_scope

    ws = tmp_path.parent
    _, target = _seed(ws)
    ts.replace(
        delivery_event_key(delivery_events_scope(target), "event1"),
        {"event_id": "event1", "output_path": str(target / "out.csv")},
        expect=ts.Version.ABSENT,
    )

    resp = client.post(
        "/api/projects/rename",
        json={"name": "sample_plot_target", "new_name": "sample_plot_renamed",
              "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 409
    assert "delivery_events" in resp.json()["detail"]


def test_records_refusal_names_job_registry(client, tmp_path):
    from tcip_mcp.web_client import INFERENCE_JOBS, job_registry_key

    ws = tmp_path.parent
    _, target = _seed(ws)
    ts.replace(job_registry_key(INFERENCE_JOBS, root=target), [{"job_id": "j1"}])

    resp = client.post(
        "/api/projects/rename",
        json={"name": "sample_plot_target", "new_name": "sample_plot_renamed",
              "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 409
    assert "job_registry" in resp.json()["detail"]


def test_records_refusal_names_hpo_sweep_manifest_and_an_empty_sweep_dir_refuses_nothing(
    client, tmp_path,
):
    from tcip_mcp.tools import training_tools

    ws = tmp_path.parent
    _, target = _seed(ws)
    hpo_root = training_tools.hpo_root(root=target)
    (hpo_root / "empty-sweep").mkdir(parents=True)
    # A real sweep's own directory exists on disk (Ray's trial storage) whichever backend
    # the manifest record itself binds to.
    (hpo_root / "real-sweep").mkdir(parents=True)
    ts.replace(
        training_tools.sweep_manifest_key("real-sweep", root=target),
        {"study_name": "real-sweep", "status": "running"},
        expect=ts.Version.ABSENT,
    )

    resp = client.post(
        "/api/projects/rename",
        json={"name": "sample_plot_target", "new_name": "sample_plot_renamed",
              "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 409
    assert "hpo_sweep_manifest" in resp.json()["detail"]


def test_a_released_canvas_binding_does_not_refuse(client, tmp_path):
    """canvas_open_binding is deliberately not in the records bound: a released binding naming
    this project (written through its own door) admits the rename."""
    from tcip_mcp.web_client import canvas_open_binding_key

    ws = tmp_path.parent
    _, target = _seed(ws)
    ts.replace(canvas_open_binding_key(), {
        "generation": 1, "root": str(target), "project_name": "sample_plot_target",
        "issued_at": "2026-03-04T00:00:00+00:00",
    }, expect=ts.Version.ABSENT)
    released = project_removal.release_project_binding("sample_plot_target", released_by="t")
    assert "error" not in released
    assert released["canvas_binding_released"] is True

    resp = client.post(
        "/api/projects/rename",
        json={"name": "sample_plot_target", "new_name": "sample_plot_renamed",
              "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 200


def test_a_fresh_project_with_images_and_a_dataset_registers_no_record(client, tmp_path):
    """The admits-valid-work case decision 4 states: a project that has only ingested and
    registered its own dataset holds none of the six stores' records."""
    ws = tmp_path.parent
    _, target = _seed(ws)
    reg = register_dataset(str(target), crop="black locust", project_root=str(target))
    assert "error" not in reg, reg

    preview = client.get("/api/projects/sample_plot_target/rename-preview").json()
    assert preview["records_present"] == []
    assert preview["refusal"] is None


# ── dependents: warned, never refused ─────────────────────────────────────────


def test_dependent_project_is_listed_never_refused_and_gets_its_own_audit_line(client, tmp_path):
    ws = tmp_path.parent
    _, target = _seed(ws)
    dependent = _init(ws, "sample_plot_dependent")
    reg = register_dataset(str(target), crop="black locust", project_root=str(dependent))
    assert "error" not in reg, reg

    preview = client.get("/api/projects/sample_plot_target/rename-preview").json()
    assert preview["refusal"] is None
    names = {d["project"] for d in preview["dependent_projects"]}
    assert "sample_plot_dependent" in names

    resp = client.post(
        "/api/projects/rename",
        json={"name": "sample_plot_target", "new_name": "sample_plot_renamed",
              "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 200
    names2 = {d["project"] for d in resp.json()["dependent_projects"]}
    assert "sample_plot_dependent" in names2

    dep_lines = _audit_lines(dependent)
    assert dep_lines[-1]["tool"] == "dependency_pending_rename"
    assert dep_lines[-1]["arguments"]["new_name"] == "sample_plot_renamed"

    listing = client.get("/api/projects").json()
    by_name = {p["name"]: p for p in listing["projects"]}
    warning = by_name["sample_plot_dependent"]["dependency_warnings"][0]
    assert warning["pending_kind"] == "rename"
    assert warning["archive_path"] is None
    assert warning["holding_dir"] is None

    project_rename.complete_pending_renames(ws)
    listing2 = client.get("/api/projects").json()
    by_name2 = {p["name"]: p for p in listing2["projects"]}
    warning2 = by_name2["sample_plot_dependent"]["dependency_warnings"][0]
    assert warning2["present"] is False

    remedy = register_dataset(
        str(ws / "sample_plot_renamed"), crop="black locust", project_root=str(dependent),
    )
    assert "error" not in remedy
    assert remedy["id"] == reg["id"]
    listing3 = client.get("/api/projects").json()
    by_name3 = {p["name"]: p for p in listing3["projects"]}
    assert by_name3["sample_plot_dependent"]["dependency_warnings"] == []


# ── phase two: resume, blocked, withdraw, two destinations ────────────────────


def test_resume_after_a_crash_between_the_rename_and_the_marker_delete(tmp_path):
    """The child's own name already equals the marker's new_name: no rename, the completion
    line writes naming old_name as old_root, and the marker is deleted."""
    from tcip_store import Version, replace

    ws = tmp_path.parent
    _, target = _seed(ws)
    already_renamed = ws / "sample_plot_renamed"
    ts.close_connections()  # this process's own cached handle would otherwise deny the rename
    target.rename(already_renamed)
    replace(workspace.pending_rename_key(already_renamed), {
        "requested_at": "20260304T120000Z", "requested_by": "user:tester",
        "old_name": "sample_plot_target", "new_name": "sample_plot_renamed",
    }, expect=Version.ABSENT)

    outcomes = project_rename.complete_pending_renames(ws)
    outcome = next(o for o in outcomes if o["new_name"] == "sample_plot_renamed")
    assert outcome["already_renamed"] is True
    assert outcome["name"] == "sample_plot_target"
    assert workspace.pending_rename_record(already_renamed) is None

    lines = _audit_lines(already_renamed)
    completed = next(line for line in lines if line["tool"] == "project_rename_completed")
    assert completed["arguments"]["old_root"] == str(target)


def test_blocked_when_the_destination_is_taken_and_withdraw_clears_it(client, tmp_path):
    ws = tmp_path.parent
    _seed(ws)
    resp = client.post(
        "/api/projects/rename",
        json={"name": "sample_plot_target", "new_name": "sample_plot_taken",
              "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 200
    # Something else takes the destination name between request and phase two.
    _init(ws, "sample_plot_taken")

    outcomes = project_rename.complete_pending_renames(ws)
    outcome = next(o for o in outcomes if o["name"] == "sample_plot_target")
    assert "blocked_by" in outcome
    assert "already taken" in outcome["blocked_by"]
    assert workspace.pending_rename_record(ws / "sample_plot_target") is not None

    listing = client.get("/api/projects").json()
    assert not any(p["name"] == "sample_plot_target" for p in listing["projects"])

    withdraw = client.post(
        "/api/projects/rename/withdraw", json={"name": "sample_plot_target", "user": "t"},
    )
    assert withdraw.status_code == 200, withdraw.text
    assert withdraw.json()["withdrawn"] is True
    assert workspace.pending_rename_record(ws / "sample_plot_target") is None

    listing2 = client.get("/api/projects").json()
    assert any(p["name"] == "sample_plot_target" for p in listing2["projects"])


def test_two_pending_renames_to_one_destination_both_admitted_second_blocks(client, tmp_path):
    ws = tmp_path.parent
    _seed(ws)
    second_target = _init(ws, "sample_plot_second")
    _add_image(second_target)

    r1 = client.post(
        "/api/projects/rename",
        json={"name": "sample_plot_target", "new_name": "sample_plot_shared",
              "confirm_name": "sample_plot_target", "user": "t"},
    )
    assert r1.status_code == 200
    r2 = client.post(
        "/api/projects/rename",
        json={"name": "sample_plot_second", "new_name": "sample_plot_shared",
              "confirm_name": "sample_plot_second", "user": "t"},
    )
    assert r2.status_code == 200

    outcomes = project_rename.complete_pending_renames(ws)
    by_name = {o["name"]: o for o in outcomes}
    winner_blocked = "blocked_by" in by_name["sample_plot_target"]
    loser_blocked = "blocked_by" in by_name["sample_plot_second"]
    assert winner_blocked != loser_blocked


def test_withdraw_refuses_a_project_with_no_pending_rename(client, tmp_path):
    ws = tmp_path.parent
    _seed(ws)

    resp = client.post(
        "/api/projects/rename/withdraw", json={"name": "sample_plot_target", "user": "t"},
    )
    assert resp.status_code == 404


# ── startup order ──────────────────────────────────────────────────────────────


def test_bind_startup_root_runs_renames_before_removals(tmp_path, monkeypatch):
    from tcip_mcp import project_paths
    from tcip_web import app as app_module

    ws = tmp_path.parent
    _init(ws, "sample_plot_open")
    workspace.activate_project("sample_plot_open")
    order: list[str] = []

    real_renames = project_rename.complete_pending_renames
    real_removals = project_removal.complete_pending_removals

    def _renames(root):
        order.append("renames")
        return real_renames(root)

    def _removals(root):
        order.append("removals")
        return real_removals(root)

    monkeypatch.setattr(project_rename, "complete_pending_renames", _renames)
    monkeypatch.setattr(project_removal, "complete_pending_removals", _removals)
    project_paths.restore_binding(None)
    ts.close_connections()

    with TestClient(app_module.app, base_url="http://127.0.0.1") as fresh_client:
        health = fresh_client.get("/health")
        assert health.status_code == 200

    assert order == ["renames", "removals"]


# ── the CLI ───────────────────────────────────────────────────────────────────


def test_the_complete_renames_command_renames_a_pending_project(client, tmp_path):
    import subprocess
    import sys

    from tcip_store import Version, replace

    ws = tmp_path.parent
    _, target = _seed(ws)
    replace(workspace.pending_rename_key(target), {
        "requested_at": "20260304T120000Z", "requested_by": "user:tester",
        "old_name": "sample_plot_target", "new_name": "sample_plot_renamed",
    }, expect=Version.ABSENT)

    ts.close_connections()  # this process's own cached handle would otherwise deny the rename
    proc = subprocess.run(
        [sys.executable, "-m", "tcip_web.cli", "complete-renames", "--workspace", str(ws)],
        env=dict(os.environ), capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "sample_plot_target" in proc.stdout
    assert not target.exists()
    assert (ws / "sample_plot_renamed").is_dir()


# ── coverage: the predicate's caller set ──────────────────────────────────────

_PACKAGES = Path(__file__).resolve().parent.parent / "packages"

_ALLOWED_CALL_SITES = {
    "pending_removal_or_none": {
        ("tcip-mcp/src/tcip_mcp/workspace.py", "pending_marker_or_none"),
    },
    "pending_rename_or_none": {
        ("tcip-mcp/src/tcip_mcp/workspace.py", "pending_marker_or_none"),
    },
    "pending_removal_record": {
        ("tcip-mcp/src/tcip_mcp/workspace.py", "pending_removal_or_none"),
        ("tcip-mcp/src/tcip_mcp/project_removal.py", "_ordered_refusal"),
    },
    "pending_rename_record": {
        ("tcip-mcp/src/tcip_mcp/workspace.py", "pending_rename_or_none"),
        ("tcip-mcp/src/tcip_mcp/project_removal.py", "_ordered_refusal"),
    },
    "pending_removal_key": {
        ("tcip-mcp/src/tcip_mcp/workspace.py", "pending_removal_record"),
        ("tcip-mcp/src/tcip_mcp/project_removal.py", "request_project_removal"),
        ("tcip-mcp/src/tcip_mcp/project_removal.py", "complete_pending_removals"),
    },
    "pending_rename_key": {
        ("tcip-mcp/src/tcip_mcp/workspace.py", "pending_rename_record"),
        ("tcip-mcp/src/tcip_mcp/project_rename.py", "request_project_rename"),
        ("tcip-mcp/src/tcip_mcp/project_rename.py", "withdraw_project_rename"),
        ("tcip-mcp/src/tcip_mcp/project_rename.py", "complete_pending_renames"),
    },
}


class _CallSiteVisitor(ast.NodeVisitor):
    def __init__(self, relative_path: str, wanted: set[str]) -> None:
        self.relative_path = relative_path
        self.wanted = wanted
        self.func_stack: list[str] = []
        self.found: set[tuple[str, str, str]] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802 - ast API
        self.func_stack.append(node.name)
        self.generic_visit(node)
        self.func_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815 - ast API

    def visit_Call(self, node: ast.Call) -> None:
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name in self.wanted:
            caller = self.func_stack[-1] if self.func_stack else "<module>"
            self.found.add((name, self.relative_path, caller))
        self.generic_visit(node)


def test_the_six_marker_patterns_call_sites_are_the_complete_named_set():
    """Coverage: every call of the six ``pending_removal``/``pending_rename`` patterns (the key
    builders included) is one of the sites decision 1 names; a new site touching one marker
    without the other is exactly what this catches."""
    wanted = set(_ALLOWED_CALL_SITES)
    found: dict[str, set[tuple[str, str]]] = {name: set() for name in wanted}

    for path in sorted(_PACKAGES.glob("*/src/**/*.py")):
        relative = str(path.relative_to(_PACKAGES)).replace(os.sep, "/")
        text = path.read_text(encoding="utf-8")
        if not any(w in text for w in wanted):
            continue
        tree = ast.parse(text, filename=str(path))
        visitor = _CallSiteVisitor(relative, wanted)
        visitor.visit(tree)
        for name, rel, caller in visitor.found:
            found[name].add((rel, caller))

    for name in wanted:
        assert found[name] == _ALLOWED_CALL_SITES[name], (name, found[name])


# ── the re-pointed readers' own rename branches ──────────────────────────────


def test_ingest_images_refuses_a_project_pending_rename_by_name(client, tmp_path):
    """coverage. ingest_images reads the shared marker predicate, so a project pending rename
    refuses by name rather than taking images into a tree about to move. The branch landed with
    the door; nothing held it before this."""
    from tcip_mcp.tools.ingest_tools import ingest_images

    ws = tmp_path.parent
    _open_project, target = _seed(ws)
    raw = ws / "raw"
    raw.mkdir(exist_ok=True)
    Image.new("RGB", (8, 8), (1, 2, 3)).save(raw / "new.jpg")

    ok = client.post("/api/projects/rename", json={
        "name": target.name, "new_name": "sample_plot_renamed", "confirm_name": target.name})
    assert ok.status_code == 200, ok.text

    result = ingest_images(source=str(raw / "*.jpg"), name=target.name, site="a site")

    assert "error" in result
    assert "pending rename" in result["error"]
    assert "sample_plot_renamed" in result["error"]


def test_allowed_roots_refuses_a_project_pending_rename_by_identity(client, tmp_path):
    """coverage. allowed_roots names a project pending rename in its refused-by-identity list,
    the same place a project pending removal lands, so no route opens a tree about to move. It
    stays in the roots list, which is what keeps the filesystem listing showing its directory.
    The branch landed with the door; nothing held it before this."""
    from tcip_web import paths

    ws = tmp_path.parent
    _open_project, target = _seed(ws)

    _roots_before, excluded_before = paths.allowed_roots()
    assert not any(str(r) == str(target.resolve()) for r in excluded_before)

    ok = client.post("/api/projects/rename", json={
        "name": target.name, "new_name": "sample_plot_renamed", "confirm_name": target.name})
    assert ok.status_code == 200, ok.text

    roots_after, excluded_after = paths.allowed_roots()
    assert any(str(r) == str(target.resolve()) for r in excluded_after)
    assert any(str(r) == str(target.resolve()) for r in roots_after)


def test_select_dataset_refuses_a_project_pending_rename_naming_the_new_name(client, tmp_path):
    """coverage. The dataset route's own marker message branches on the kind, so a project
    pending rename is refused with the rename sentence and the name it takes, never a removal
    sentence about a holding directory. The branch landed with the door; nothing held it
    before this."""
    ws = tmp_path.parent
    _open_project, target = _seed(ws)

    ok = client.post("/api/projects/rename", json={
        "name": target.name, "new_name": "sample_plot_renamed", "confirm_name": target.name})
    assert ok.status_code == 200, ok.text

    response = client.post("/api/dataset/select", json={
        "project_root": str(target), "dataset_root": str(target)})

    assert response.status_code in (403, 409), response.text
    body = response.json()["detail"]
    assert "pending rename" in body and "sample_plot_renamed" in body
    assert "holding directory" not in body


# ── the fix-up's own corrections ─────────────────────────────────────────────


def test_withdraw_deletes_the_marker_before_it_writes_its_own_line(client, tmp_path, monkeypatch):
    """guard. The withdraw door deletes the marker first and logs second, so a log that cannot be
    written never leaves a line saying a rename was withdrawn while the marker still stands and
    the project still renames at the next start."""
    from tcip_mcp import audit

    ws = tmp_path.parent
    _open_project, target = _seed(ws)
    ok = client.post("/api/projects/rename", json={
        "name": target.name, "new_name": "sample_plot_renamed", "confirm_name": target.name})
    assert ok.status_code == 200, ok.text

    def _refuse(*args, **kwargs):
        raise audit.AuditEntryNotWritten("project_rename_withdrawn", RuntimeError("no log"))

    monkeypatch.setattr(project_rename.audit, "record_event_or_raise", _refuse)
    result = project_rename.withdraw_project_rename(target.name, requested_by="user:test")

    assert result["status"] == 409
    assert "marker is gone" in result["error"]
    assert workspace.pending_rename_record(target) is None


def test_withdraw_refuses_a_tree_already_at_its_new_name(client, tmp_path):
    """guard. A marker whose project already sits at its new name is in the crash window phase
    two's resume branch finishes; withdrawing there would record a withdrawal of a rename that
    landed, so the door refuses and names the resume instead."""
    ws = tmp_path.parent
    _open_project, target = _seed(ws)
    ok = client.post("/api/projects/rename", json={
        "name": target.name, "new_name": "sample_plot_renamed", "confirm_name": target.name})
    assert ok.status_code == 200, ok.text

    ts.close_connections()
    os.rename(str(target), str(ws / "sample_plot_renamed"))

    result = project_rename.withdraw_project_rename("sample_plot_renamed",
                                                    requested_by="user:test")

    assert result["status"] == 409
    assert "already at its new name" in result["error"]
    assert workspace.pending_rename_record(ws / "sample_plot_renamed") is not None


def test_phase_two_keeps_the_marker_when_its_completion_line_does_not_write(
        client, tmp_path, monkeypatch):
    """guard. Phase two deletes the marker only once the completion line has landed, so a log the
    walk could not append to leaves the rename outstanding for the next start rather than
    finishing it with no record that it happened."""
    from tcip_mcp import audit

    ws = tmp_path.parent
    _open_project, target = _seed(ws)
    ok = client.post("/api/projects/rename", json={
        "name": target.name, "new_name": "sample_plot_renamed", "confirm_name": target.name})
    assert ok.status_code == 200, ok.text

    def _refuse(*args, **kwargs):
        raise audit.AuditEntryNotWritten("project_rename_completed", RuntimeError("no log"))

    monkeypatch.setattr(project_rename.audit, "record_event_or_raise", _refuse)
    outcomes = project_rename.complete_pending_renames(ws)

    assert len(outcomes) == 1
    assert outcomes[0]["new_name"] == "sample_plot_renamed"
    assert "marker stands" in outcomes[0]["note"]
    assert (ws / "sample_plot_renamed").is_dir()
    assert workspace.pending_rename_record(ws / "sample_plot_renamed") is not None


def test_a_sweep_manifest_that_cannot_be_read_refuses_the_rename(client, tmp_path, monkeypatch):
    """guard. The records bound fails closed on a sweep manifest the store refuses to read: a
    manifest that cannot be read is not proof that no sweep names this project's path."""
    ws = tmp_path.parent
    _open_project, target = _seed(ws)

    from tcip_mcp.tools import training_tools

    sweep_dir = training_tools.hpo_root(root=target) / "hpo_probe"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    real_read = ts.read

    def _refuse_manifest(key, default=None):
        if key.store == "hpo_sweep_manifest":
            raise ts.DecodeError("the manifest is not readable")
        return real_read(key, default=default)

    monkeypatch.setattr(project_rename.tcip_store, "read", _refuse_manifest)
    present = project_rename.project_records_present(target)

    assert "hpo_sweep_manifest" in present


def test_the_records_refusal_leads_with_the_breeders_own_sentence(client, tmp_path):
    """coverage. The dialog renders the records refusal verbatim, so its first sentence says what
    is wrong in words a breeder follows and names the record in plain terms; the store names
    follow for the agent."""
    ws = tmp_path.parent
    _open_project, target = _seed(ws)
    from tcip_mcp.pipelines.postprocessing import plant_mapping

    csv_path = ws / "plants.csv"
    csv_path.write_text(
        "plot_name,accession_name,plot_number,row_number,col_number,"
        "WGS84_centroid_x,WGS84_centroid_y\nP1,acc-A,1,1,1,-90.058,43.197\n",
        encoding="utf-8",
    )
    plant_mapping.register_plant_registry_record(
        target, "reg", [csv_path], crop="black locust", site="a site",
        registered_by="user:test",
    )

    refusal = project_rename._records_refusal(target)

    assert refusal is not None
    assert refusal.startswith(f"{target.name} has already been worked on")
    assert "a plant location list" in refusal
    assert "plant_registries" in refusal


def test_a_rename_warning_carries_the_name_the_target_takes(client, tmp_path):
    """guard. A dependent's own warning carries the new name, so the card can name the folder its
    owner has to re-register at rather than saying only that the target is being renamed."""
    ws = tmp_path.parent
    _open_project, target = _seed(ws)
    dependent = _init(ws, "sample_plot_dependent")
    registered = register_dataset(dataset_root=str(target), crop="black locust",
                                  project_root=str(dependent))
    assert "error" not in registered, registered

    ok = client.post("/api/projects/rename", json={
        "name": target.name, "new_name": "sample_plot_renamed", "confirm_name": target.name})
    assert ok.status_code == 200, ok.text

    warnings, problem = project_removal.dependency_warnings(dependent)

    assert problem is None
    assert len(warnings) == 1
    assert warnings[0]["pending_kind"] == "rename"
    assert warnings[0]["new_name"] == "sample_plot_renamed"

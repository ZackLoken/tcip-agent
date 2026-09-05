"""Tests for the workspace project front-door routes."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_web.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


@pytest.fixture
def workspace_dir(tmp_path: Path, monkeypatch) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    return ws


def _make_project(ws: Path, name: str, *, dates=(), subjects=(), models=()) -> Path:
    proj = ws / name
    (proj / ".tcip").mkdir(parents=True)
    from PIL import Image

    for d in dates:
        ddir = proj / "images" / d
        ddir.mkdir(parents=True)
        Image.new("RGB", (8, 8), (0, 0, 0)).save(ddir / "img.png")
    if subjects:
        # Subjects live in the dataset's nested registry now, not as child dirs of annotations/.
        from tcip_mcp.class_registry import ClassRegistry, Subject, write_registry

        write_registry(proj / "classes.json",
                       ClassRegistry(tuple(Subject(s) for s in sorted(subjects))))
    for m in models:
        (proj / "predictions" / m).mkdir(parents=True)
    return proj


def test_list_projects_lists_workspace_projects(client, workspace_dir):
    _make_project(workspace_dir, "currant_bud_valley-farm", dates=["2026-02-11"], subjects=["bud", "bush"], models=["baseline"])
    _make_project(workspace_dir, "chestnut_burr_site-b", dates=["2026-03-01"])

    resp = client.get("/api/projects")
    assert resp.status_code == 200
    body = resp.json()
    assert body["workspace"] == str(workspace_dir.resolve())
    names = {p["name"] for p in body["projects"]}
    assert names == {"currant_bud_valley-farm", "chestnut_burr_site-b"}

    hz = next(p for p in body["projects"] if p["name"] == "currant_bud_valley-farm")
    assert hz["dates"] == ["2026-02-11"]
    assert hz["subjects"] == ["bud", "bush"]  # sorted
    assert hz["models"] == ["baseline"]
    assert hz["image_count"] == 1


def test_projects_report_per_date_subject_model_availability(client, workspace_dir):
    # bud labelled on 02-11 (+ baseline predictions there); bush labelled on 03-02;
    # 03-24 has images but nothing labelled. One name-based label file per image.
    from tcip_annotation.json_io import write_annotations
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.dataset_layout import annotation_dir, prediction_dir

    proj = _make_project(
        workspace_dir,
        "currant_bud_valley-farm",
        dates=["2026-02-11", "2026-03-02", "2026-03-24"],
        subjects=["bud", "bush"],
        models=["baseline"],
    )
    ad = annotation_dir(proj, "2026-02-11")
    ad.mkdir(parents=True, exist_ok=True)
    write_annotations(str(ad / "img.json"), [Annotation(subject="bud", geometry=BBox(1, 1, 7, 7))],
                      8, 8)
    ad2 = annotation_dir(proj, "2026-03-02")
    ad2.mkdir(parents=True, exist_ok=True)
    write_annotations(str(ad2 / "img.json"), [Annotation(subject="bush", geometry=BBox(2, 2, 6, 6))],
                      8, 8)
    pd = prediction_dir(proj, "baseline", "2026-02-11")
    pd.mkdir(parents=True, exist_ok=True)
    write_annotations(str(pd / "img.json"),
                      [Annotation(subject="bud", geometry=BBox(1, 1, 7, 7), score=0.9)], 8, 8)

    hz = next(
        p
        for p in client.get("/api/projects").json()["projects"]
        if p["name"] == "currant_bud_valley-farm"
    )
    # Flat lists still list everything present anywhere.
    assert hz["subjects"] == ["bud", "bush"]
    # Per-date maps reflect where labels/predictions actually are.
    assert hz["subjects_by_date"]["2026-02-11"] == ["bud"]
    assert hz["subjects_by_date"]["2026-03-02"] == ["bush"]
    assert hz["subjects_by_date"]["2026-03-24"] == []  # images but no labels
    assert hz["models_by_date"]["2026-02-11"] == ["baseline"]
    assert hz["models_by_date"]["2026-03-02"] == []
    assert hz["models_by_date"]["2026-03-24"] == []
    assert hz["label_problem"] is None


def test_projects_report_a_label_problem_and_still_list(client, workspace_dir):
    """A corrupt label under one project must not 500 the whole listing (mirrors site_problem):
    the project still lists, its other dates are unaffected, and the file is named."""
    from tcip_mcp.dataset_layout import annotation_dir

    proj = _make_project(
        workspace_dir, "currant_bud_valley-farm",
        dates=["2026-02-11", "2026-03-02"], subjects=["bud"],
    )
    bad = annotation_dir(proj, "2026-02-11")
    bad.mkdir(parents=True, exist_ok=True)
    (bad / "img.json").write_text("not json {][", encoding="utf-8")

    resp = client.get("/api/projects")
    assert resp.status_code == 200
    hz = next(p for p in resp.json()["projects"] if p["name"] == "currant_bud_valley-farm")
    assert hz["subjects_by_date"]["2026-02-11"] == []
    assert hz["subjects_by_date"]["2026-03-02"] == []
    assert hz["label_problem"] is not None
    assert str(bad / "img.json") in hz["label_problem"]


def test_list_ignores_dirs_without_tcip(client, workspace_dir):
    _make_project(workspace_dir, "real_project_site", dates=["2026-02-11"])
    (workspace_dir / "not_a_project").mkdir()  # no .tcip/

    names = {p["name"] for p in client.get("/api/projects").json()["projects"]}
    assert names == {"real_project_site"}


def test_list_sorted_by_modified_desc(client, workspace_dir):
    older = _make_project(workspace_dir, "old_project_site", dates=["2026-02-11"])
    newer = _make_project(workspace_dir, "new_project_site", dates=["2026-03-01"])
    os.utime(older, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))

    order = [p["name"] for p in client.get("/api/projects").json()["projects"]]
    assert order == ["new_project_site", "old_project_site"]


def test_active_marker_round_trip(client, workspace_dir):
    _make_project(workspace_dir, "currant_bud_valley-farm", dates=["2026-02-11"])

    # Unset initially
    unset = client.get("/api/projects").json()
    assert unset["active"] is None
    assert unset["active_path"] is None

    # Set it
    resp = client.post("/api/projects/active", json={"name": "currant_bud_valley-farm"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "currant_bud_valley-farm"

    # Read it back through the list route's active/active_path fields
    listed = client.get("/api/projects").json()
    assert listed["active"] == "currant_bud_valley-farm"
    assert listed["active_path"] == str((workspace_dir / "currant_bud_valley-farm").resolve())

    # is_active flag surfaces in the list
    assert next(p for p in listed["projects"] if p["name"] == "currant_bud_valley-farm")[
        "is_active"
    ]


def test_set_active_rejects_traversal(client, workspace_dir):
    resp = client.post("/api/projects/active", json={"name": "../escape"})
    assert resp.status_code == 400


def test_set_active_rejects_unknown_project(client, workspace_dir):
    resp = client.post("/api/projects/active", json={"name": "does_not_exist"})
    assert resp.status_code == 404


def test_wrong_encoding_marker_does_not_break_the_front_door(client, workspace_dir):
    # Bound to the file backend: the claim is about raw bytes on disk no text codec decodes.
    import tcip_store
    from tcip_store.file_backend import FileBackend

    tcip_store.bind(FileBackend())
    _make_project(workspace_dir, "currant_bud_valley-farm", dates=["2026-02-11"])
    (workspace_dir / ".active").write_bytes("currant_bud_valley-farm".encode("utf-16"))

    resp = client.get("/api/projects")
    assert resp.status_code == 200
    body = resp.json()
    assert {p["name"] for p in body["projects"]} == {"currant_bud_valley-farm"}
    # The undecodable marker is treated as unset.
    assert body["active"] is None
    assert body["active_path"] is None


def test_active_returns_null_when_marker_names_a_traversal(client, workspace_dir):
    """activate_project refuses a traversal name, so this writes the marker directly
    through the store, the shape a corrupted or hand-edited marker would take. A real
    .tcip sits where the traversal points, so an unsafe reader would actually find it."""
    import tcip_store
    from tcip_mcp import workspace

    (workspace_dir.parent / "escapee" / ".tcip").mkdir(parents=True)
    tcip_store.replace(workspace.active_project_key(), "../escapee")

    body = client.get("/api/projects").json()
    assert body["active"] is None
    assert body["active_path"] is None


def test_list_reports_site_fields_across_four_project_states(client, workspace_dir):
    """A project with a record, one with ``.tcip`` and nothing else, one whose record is
    undecodable, and one whose record decodes to something else: all four list, each with the
    right ``site``/``site_problem`` pair. The recordless one gets no database published under
    it by the listing, the store's own guarantee at this surface."""
    import tcip_store
    from tcip_store.file_backend import database_file

    from tcip_mcp.project_record import project_record_key
    from tcip_mcp.tools.project_tools import initialize_project
    from tests._record_damage_fixtures import damage_record

    recorded = workspace_dir / "currant_bud_recorded"
    initialize_project(str(recorded), site="north orchard")

    recordless = _make_project(workspace_dir, "currant_bud_recordless", dates=["2026-02-11"])

    undecodable = workspace_dir / "currant_bud_undecodable"
    initialize_project(str(undecodable), site="north orchard")
    damage_record(project_record_key(str(undecodable)), b"{not valid json")

    invalid = workspace_dir / "currant_bud_invalid"
    initialize_project(str(invalid), site="north orchard")
    key = project_record_key(str(invalid))
    current = tcip_store.read_versioned(key).version
    tcip_store.replace(key, {"not_site": "x"}, expect=current)

    body = client.get("/api/projects").json()
    by_name = {p["name"]: p for p in body["projects"]}
    assert set(by_name) == {
        "currant_bud_recorded", "currant_bud_recordless",
        "currant_bud_undecodable", "currant_bud_invalid",
    }

    assert by_name["currant_bud_recorded"]["site"] == "north orchard"
    assert by_name["currant_bud_recorded"]["site_problem"] is None

    assert by_name["currant_bud_recordless"]["site"] is None
    assert "initialize_project" in by_name["currant_bud_recordless"]["site_problem"]
    assert not database_file(str(recordless)).is_file()

    assert by_name["currant_bud_undecodable"]["site"] is None
    assert "does not decode" in by_name["currant_bud_undecodable"]["site_problem"]

    assert by_name["currant_bud_invalid"]["site"] is None
    assert "does not hold a site" in by_name["currant_bud_invalid"]["site_problem"]


def test_list_route_reports_the_recorded_site(client, workspace_dir):
    """Minimal, single-claim sibling of the four-state test above: only ``initialize_project``, so a
    fail-before run against a tree that predates the ``site`` parameter observes a real
    behavioral gap (the key absent) rather than the whole test file failing to import."""
    from tcip_mcp.tools.project_tools import initialize_project

    recorded = workspace_dir / "currant_bud_recorded"
    initialize_project(str(recorded), site="north orchard")

    body = client.get("/api/projects").json()
    by_name = {p["name"]: p for p in body["projects"]}

    assert by_name["currant_bud_recorded"]["site"] == "north orchard"


def test_list_reports_the_current_platform_root_after_a_repin(client, workspace_dir):
    """platform_root/platform_root_source answer the root this backend just repinned to, not
    whatever pin_platform_root last decided at process startup."""
    from tcip_mcp import workspace

    proj = _make_project(workspace_dir, "currant_bud_valley", dates=["2026-02-11"])

    workspace.activate_project("currant_bud_valley")
    after = client.get("/api/projects").json()
    assert after["platform_root"] == str(proj.resolve())
    assert after["platform_root_source"] == "adopted"


def test_active_returns_null_when_marker_points_at_missing_project(client, workspace_dir):
    from tcip_mcp import workspace

    _make_project(workspace_dir, "temp_project_site", dates=["2026-02-11"])
    workspace.activate_project("temp_project_site")
    # Now remove the project's .tcip so the marker dangles.
    import shutil

    shutil.rmtree(workspace_dir / "temp_project_site")

    body = client.get("/api/projects").json()
    assert body["active"] is None
    assert body["active_path"] is None

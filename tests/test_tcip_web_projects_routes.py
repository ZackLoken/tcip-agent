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
    _make_project(workspace_dir, "hazelnut_catkin_valley-farm", dates=["2026-02-11"], subjects=["catkin", "bush"], models=["baseline"])
    _make_project(workspace_dir, "chestnut_burr_site-b", dates=["2026-03-01"])

    resp = client.get("/api/projects")
    assert resp.status_code == 200
    body = resp.json()
    assert body["workspace"] == str(workspace_dir.resolve())
    names = {p["name"] for p in body["projects"]}
    assert names == {"hazelnut_catkin_valley-farm", "chestnut_burr_site-b"}

    hz = next(p for p in body["projects"] if p["name"] == "hazelnut_catkin_valley-farm")
    assert hz["dates"] == ["2026-02-11"]
    assert hz["subjects"] == ["bush", "catkin"]  # sorted
    assert hz["models"] == ["baseline"]
    assert hz["image_count"] == 1


def test_projects_report_per_date_subject_model_availability(client, workspace_dir):
    # catkin labelled on 02-11 (+ baseline predictions there); bush labelled on 03-02;
    # 03-24 has images but nothing labelled. One name-based label file per image.
    from tcip_annotation.json_io import write_annotations
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.dataset_layout import annotation_dir, prediction_dir

    proj = _make_project(
        workspace_dir,
        "hazelnut_catkin_valley-farm",
        dates=["2026-02-11", "2026-03-02", "2026-03-24"],
        subjects=["catkin", "bush"],
        models=["baseline"],
    )
    ad = annotation_dir(proj, "2026-02-11")
    ad.mkdir(parents=True, exist_ok=True)
    write_annotations(str(ad / "img.json"), [Annotation(subject="catkin", geometry=BBox(1, 1, 7, 7))],
                      8, 8)
    ad2 = annotation_dir(proj, "2026-03-02")
    ad2.mkdir(parents=True, exist_ok=True)
    write_annotations(str(ad2 / "img.json"), [Annotation(subject="bush", geometry=BBox(2, 2, 6, 6))],
                      8, 8)
    pd = prediction_dir(proj, "baseline", "2026-02-11")
    pd.mkdir(parents=True, exist_ok=True)
    write_annotations(str(pd / "img.json"),
                      [Annotation(subject="catkin", geometry=BBox(1, 1, 7, 7), score=0.9)], 8, 8)

    hz = next(
        p
        for p in client.get("/api/projects").json()["projects"]
        if p["name"] == "hazelnut_catkin_valley-farm"
    )
    # Flat lists still list everything present anywhere.
    assert hz["subjects"] == ["bush", "catkin"]
    # Per-date maps reflect where labels/predictions actually are.
    assert hz["subjects_by_date"]["2026-02-11"] == ["catkin"]
    assert hz["subjects_by_date"]["2026-03-02"] == ["bush"]
    assert hz["subjects_by_date"]["2026-03-24"] == []  # images but no labels
    assert hz["models_by_date"]["2026-02-11"] == ["baseline"]
    assert hz["models_by_date"]["2026-03-02"] == []
    assert hz["models_by_date"]["2026-03-24"] == []


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
    _make_project(workspace_dir, "hazelnut_catkin_valley-farm", dates=["2026-02-11"])

    # Unset initially
    unset = client.get("/api/projects").json()
    assert unset["active"] is None
    assert unset["active_path"] is None

    # Set it
    resp = client.post("/api/projects/active", json={"name": "hazelnut_catkin_valley-farm"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "hazelnut_catkin_valley-farm"

    # Read it back through the list route's active/active_path fields
    listed = client.get("/api/projects").json()
    assert listed["active"] == "hazelnut_catkin_valley-farm"
    assert listed["active_path"] == str((workspace_dir / "hazelnut_catkin_valley-farm").resolve())

    # is_active flag surfaces in the list
    assert next(p for p in listed["projects"] if p["name"] == "hazelnut_catkin_valley-farm")[
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
    _make_project(workspace_dir, "hazelnut_catkin_valley-farm", dates=["2026-02-11"])
    (workspace_dir / ".active").write_bytes("hazelnut_catkin_valley-farm".encode("utf-16"))

    resp = client.get("/api/projects")
    assert resp.status_code == 200
    body = resp.json()
    assert {p["name"] for p in body["projects"]} == {"hazelnut_catkin_valley-farm"}
    # The undecodable marker is treated as unset.
    assert body["active"] is None
    assert body["active_path"] is None


def test_active_returns_null_when_marker_names_a_traversal(client, workspace_dir):
    """set_active_project refuses a traversal name, so this writes the marker directly
    through the store, the shape a corrupted or hand-edited marker would take. A real
    .tcip sits where the traversal points, so an unsafe reader would actually find it."""
    import tcip_store
    from tcip_mcp import workspace

    (workspace_dir.parent / "escapee" / ".tcip").mkdir(parents=True)
    tcip_store.replace(workspace.active_project_key(), "../escapee")

    body = client.get("/api/projects").json()
    assert body["active"] is None
    assert body["active_path"] is None


def test_active_returns_null_when_marker_points_at_missing_project(client, workspace_dir):
    from tcip_mcp import workspace

    _make_project(workspace_dir, "temp_project_site", dates=["2026-02-11"])
    workspace.set_active_project("temp_project_site")
    # Now remove the project's .tcip so the marker dangles.
    import shutil

    shutil.rmtree(workspace_dir / "temp_project_site")

    body = client.get("/api/projects").json()
    assert body["active"] is None
    assert body["active_path"] is None

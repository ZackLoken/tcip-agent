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
def workspace_dir(tmp_path: Path) -> Path:
    """A fresh workspace the backend is started with."""
    from tcip_web.state import store

    ws = (tmp_path / "ws").resolve()
    ws.mkdir()
    store.configure(ws, ())
    return ws


def _make_project(ws: Path, name: str, *, dates=(), subjects=(), models=()) -> Path:
    """A workspace project created through ``initialize_project``, its display name ``name``,
    holding one image per date, the named subjects and a prediction directory per model."""
    from PIL import Image

    from tcip_mcp.tools.project_tools import initialize_project

    proj = ws / name
    created = initialize_project(str(proj), name, "north orchard")
    assert "error" not in created, created
    for d in dates:
        ddir = proj / "images" / d
        ddir.mkdir(parents=True)
        Image.new("RGB", (8, 8), (0, 0, 0)).save(ddir / "img.png")
    if subjects:
        from tcip_mcp.subject_registry import SubjectRegistry, Subject
        from tests._producer_fixtures import registry_over

        registry_over(proj, SubjectRegistry(tuple(Subject(s) for s in sorted(subjects))))
    for m in models:
        (proj / "predictions" / m).mkdir(parents=True)
    return proj


def _listed(client: TestClient) -> dict[str, dict]:
    return {p["display_name"]: p for p in client.get("/api/projects").json()["projects"]}


def test_list_projects_lists_workspace_projects(client, workspace_dir):
    _make_project(workspace_dir, "currant_bud_valley-farm", dates=["2026-02-11"], subjects=["bud", "bush"], models=["baseline"])
    _make_project(workspace_dir, "chestnut_burr_site-b", dates=["2026-03-01"])

    resp = client.get("/api/projects")
    assert resp.status_code == 200
    body = resp.json()
    assert body["workspace"] == str(workspace_dir.resolve())
    names = {p["display_name"] for p in body["projects"]}
    assert names == {"currant_bud_valley-farm", "chestnut_burr_site-b"}

    hz = _listed(client)["currant_bud_valley-farm"]
    assert hz["dates"] == ["2026-02-11"]
    assert hz["subjects"] == ["bud", "bush"]  # sorted
    assert hz["prediction_dirs"] == {"2026-02-11": {}}  # predictions/baseline publishes nothing
    assert hz["image_count"] == 1


def test_projects_report_per_date_subject_model_availability(client, workspace_dir):
    # bud labeled on 02-11 (+ a bucket published over it); bush labeled on 03-02;
    # 03-24 has images but nothing labeled. One name-based label file per image.
    pytest.importorskip("torch")
    from tcip_annotation.json_io import write_annotations
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.dataset_layout import annotation_dir, prediction_root
    from tests._chain_fixtures import published

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
    pd = prediction_root(proj) / "baseline" / "2026-02-11"
    published(proj, pd, [{"image": str(proj / "images" / "2026-02-11" / "img.png"), "width": 8,
                          "height": 8, "boxes": [[1.0, 1.0, 7.0, 7.0]], "scores": [0.9],
                          "labels": [1]}], scope={"subject": "bud", "id_map": {"bud": 0}})

    hz = _listed(client)["currant_bud_valley-farm"]
    # Flat lists still list everything present anywhere.
    assert hz["subjects"] == ["bud", "bush"]
    # Per-date maps reflect where labels/predictions actually are.
    assert hz["subjects_by_date"]["2026-02-11"] == ["bud"]
    assert hz["subjects_by_date"]["2026-03-02"] == ["bush"]
    assert hz["subjects_by_date"]["2026-03-24"] == []  # images but no labels
    assert hz["prediction_dirs"]["2026-02-11"] == {"baseline/2026-02-11": str(pd)}
    assert hz["prediction_dirs"]["2026-03-02"] == {}
    assert hz["prediction_dirs"]["2026-03-24"] == {}
    assert hz["label_problem"] is None


def test_projects_report_a_label_problem_and_still_list(client, workspace_dir):
    """A corrupt label under one project must not 500 the whole listing (mirrors
    record_problem): the project still lists, its other dates are unaffected, and the file is
    named."""
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
    hz = _listed(client)["currant_bud_valley-farm"]
    assert hz["subjects_by_date"]["2026-02-11"] == []
    assert hz["subjects_by_date"]["2026-03-02"] == []
    assert hz["label_problem"] is not None
    assert str(bad / "img.json") in hz["label_problem"]


def test_list_ignores_dirs_without_tcip(client, workspace_dir):
    _make_project(workspace_dir, "real_project_site", dates=["2026-02-11"])
    (workspace_dir / "not_a_project").mkdir()  # no .tcip/

    assert set(_listed(client)) == {"real_project_site"}


def test_list_sorted_by_modified_desc(client, workspace_dir):
    older = _make_project(workspace_dir, "old_project_site", dates=["2026-02-11"])
    newer = _make_project(workspace_dir, "new_project_site", dates=["2026-03-01"])
    os.utime(older, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))

    order = [p["display_name"] for p in client.get("/api/projects").json()["projects"]]
    assert order == ["new_project_site", "old_project_site"]


def test_opening_a_project_by_id_marks_it_open_in_the_list(client, workspace_dir):
    _make_project(workspace_dir, "currant_bud_valley-farm", dates=["2026-02-11"])
    listed = _listed(client)["currant_bud_valley-farm"]
    assert client.get("/api/projects").json()["open_id"] is None
    assert listed["is_open"] is False

    resp = client.post("/api/projects/open", json={"id": listed["id"]})
    assert resp.status_code == 200
    assert resp.json() == {"id": listed["id"], "display_name": "currant_bud_valley-farm",
                           "path": str((workspace_dir / "currant_bud_valley-farm").resolve())}

    assert client.get("/api/projects").json()["open_id"] == listed["id"]
    assert _listed(client)["currant_bud_valley-farm"]["is_open"] is True


def test_opening_an_id_no_project_holds_is_404(client, workspace_dir):
    _make_project(workspace_dir, "currant_bud_valley-farm")
    resp = client.post("/api/projects/open", json={"id": "000000000000"})
    assert resp.status_code == 404
    assert "000000000000" in resp.json()["detail"]
    assert client.get("/api/projects").json()["open_id"] is None


def test_list_reports_record_fields_across_four_project_states(client, workspace_dir):
    """A project with a record, one with ``.tcip`` and nothing else, one whose record is
    undecodable, and one whose record decodes to something else: all four list, each with its
    record's fields or the ``record_problem`` naming why they are absent. The recordless one gets
    no database published under it by the listing, the store's own guarantee at this surface."""
    import tcip_store
    from tcip_store.file_backend import database_file

    from tcip_mcp.project_record import project_record_key
    from tests._record_damage_fixtures import damage_record

    recorded = _make_project(workspace_dir, "currant_bud_recorded")

    recordless = workspace_dir / "currant_bud_recordless"
    (recordless / ".tcip").mkdir(parents=True)

    undecodable = _make_project(workspace_dir, "currant_bud_undecodable")
    damage_record(project_record_key(str(undecodable)), b"{not valid json")

    invalid = _make_project(workspace_dir, "currant_bud_invalid")
    key = project_record_key(str(invalid))
    current = tcip_store.read_versioned(key).version
    tcip_store.replace(key, {"not_site": "x"}, expect=current)

    by_path = {p["path"]: p for p in client.get("/api/projects").json()["projects"]}
    assert set(by_path) == {str(p.resolve()) for p in (recorded, recordless, undecodable, invalid)}

    good = by_path[str(recorded.resolve())]
    assert (good["display_name"], good["site"], good["record_problem"]) == (
        "currant_bud_recorded", "north orchard", None)
    assert len(good["id"]) == 12

    missing = by_path[str(recordless.resolve())]
    assert (missing["id"], missing["display_name"], missing["site"]) == (None, None, None)
    assert "initialize_project" in missing["record_problem"]
    assert not database_file(str(recordless)).is_file()

    assert "does not decode" in by_path[str(undecodable.resolve())]["record_problem"]
    assert "does not hold an id, a display name and a site" in (
        by_path[str(invalid.resolve())]["record_problem"])


def test_a_last_opened_pointer_naming_a_missing_project_is_named_and_opens_nothing(
        client, workspace_dir, monkeypatch):
    from tcip_mcp import workspace

    _make_project(workspace_dir, "temp_project_site", dates=["2026-02-11"])
    workspace.write_last_opened(workspace_dir, "000000000000")
    monkeypatch.setenv("TCIP_WORKSPACE", str(workspace_dir))

    with TestClient(app, base_url="http://127.0.0.1") as started:
        body = started.get("/api/projects").json()
    assert body["open_id"] is None
    assert "000000000000" in body["last_opened_problem"]

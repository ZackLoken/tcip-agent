"""Tests for the meta routes (friction reports + retrospectives surfacing)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_web.app import app
from tcip_mcp.tools.meta_tools import report_friction, project_retrospective


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def test_reports_empty_when_dir_missing(client: TestClient, tmp_path: Path) -> None:
    resp = client.get("/api/meta/reports", params={"project_root": str(tmp_path)})
    assert resp.status_code == 200
    data = resp.json()
    assert data == {"reports": [], "count": 0, "total_available": 0}


def test_reports_surfaces_written_report(client: TestClient, tmp_path: Path) -> None:
    report_friction(
        str(tmp_path),
        category="ambiguous_data",
        detail="two plausible interpretations of the label dir",
        context={"crop": "currant"},
    )
    data = client.get(
        "/api/meta/reports", params={"project_root": str(tmp_path)}
    ).json()
    assert data["count"] == 1
    rep = data["reports"][0]
    assert rep["category"] == "ambiguous_data"
    assert rep["detail"] == "two plausible interpretations of the label dir"
    assert rep["context"]["crop"] == "currant"


def test_retrospectives_empty_when_dir_missing(client: TestClient, tmp_path: Path) -> None:
    data = client.get(
        "/api/meta/retrospectives", params={"project_root": str(tmp_path)}
    ).json()
    assert data == {"retrospectives": [], "count": 0, "total_available": 0}


def test_retrospectives_surfaces_written_retro(client: TestClient, tmp_path: Path) -> None:
    project_retrospective(
        str(tmp_path),
        project_id="elderberry-cluster",
        task="count fruit clusters",
        worked="instance seg held up",
        did_not_work="overlapping clusters merged",
    )
    data = client.get(
        "/api/meta/retrospectives", params={"project_root": str(tmp_path)}
    ).json()
    assert data["count"] == 1
    retro = data["retrospectives"][0]
    assert retro["project_id"] == "elderberry-cluster"
    assert "count fruit clusters" in retro["content"]


def test_reports_confines_project_root_to_allowed_roots(
    client: TestClient, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside = tmp_path_factory.mktemp("outside")
    resp = client.get("/api/meta/reports", params={"project_root": str(outside)})
    assert resp.status_code == 403


def test_retrospectives_confines_project_root_to_allowed_roots(
    client: TestClient, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside = tmp_path_factory.mktemp("outside")
    resp = client.get("/api/meta/retrospectives", params={"project_root": str(outside)})
    assert resp.status_code == 403

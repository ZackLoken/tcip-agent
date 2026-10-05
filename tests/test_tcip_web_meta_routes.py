"""Tests for the meta routes (friction reports + retrospectives surfacing)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from tcip_mcp.tools.meta_tools import report_friction, write_retrospective


def test_reports_empty_when_dir_missing(opened_client: TestClient) -> None:
    resp = opened_client.get("/api/meta/reports")
    assert resp.status_code == 200
    data = resp.json()
    assert data == {"reports": [], "count": 0, "total_available": 0}


def test_reports_surfaces_written_report(opened_client: TestClient, opened_project: Path) -> None:
    report_friction(
        opened_project,
        category="ambiguous_data",
        detail="two plausible interpretations of the label dir",
        context={"crop": "currant"},
    )
    data = opened_client.get("/api/meta/reports").json()
    assert data["count"] == 1
    rep = data["reports"][0]
    assert rep["category"] == "ambiguous_data"
    assert rep["detail"] == "two plausible interpretations of the label dir"
    assert rep["context"]["crop"] == "currant"


def test_retrospectives_empty_when_dir_missing(opened_client: TestClient) -> None:
    data = opened_client.get("/api/meta/retrospectives").json()
    assert data == {"retrospectives": [], "count": 0, "total_available": 0}


def test_retrospectives_surfaces_written_retro(
    opened_client: TestClient, opened_project: Path,
) -> None:
    write_retrospective(
        opened_project,
        project_id="elderberry-cluster",
        task="count fruit clusters",
        worked="instance seg held up",
        did_not_work="overlapping clusters merged",
    )
    data = opened_client.get("/api/meta/retrospectives").json()
    assert data["count"] == 1
    retro = data["retrospectives"][0]
    assert retro["project_id"] == "elderberry-cluster"
    assert "count fruit clusters" in retro["content"]


def test_the_meta_routes_refuse_while_no_project_is_open(client: TestClient) -> None:
    assert client.get("/api/meta/reports").status_code == 409
    assert client.get("/api/meta/retrospectives").status_code == 409

"""Tests for the meta routes (friction reports + retrospectives surfacing)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import tcip_store as ts
from tcip_mcp.tools.meta_tools import (
    MemoryKind,
    Retrospective,
    load_project_memory,
    memory_key,
    report_friction,
    write_retrospective,
)


def test_reports_empty_when_dir_missing(opened_client: TestClient) -> None:
    resp = opened_client.get("/api/meta/reports")
    assert resp.status_code == 200
    assert resp.json() == {"entries": [], "total_available": 0}


def test_reports_surfaces_written_report(
    opened_client: TestClient, opened_project: Path
) -> None:
    report_friction(
        opened_project,
        category="ambiguous_data",
        detail="two plausible interpretations of the label dir",
        context={"crop": "currant"},
    )
    (entry,) = opened_client.get("/api/meta/reports").json()["entries"]
    rep = entry["record"]
    assert rep["category"] == "ambiguous_data"
    assert rep["detail"] == "two plausible interpretations of the label dir"
    assert rep["context"]["crop"] == "currant"


def test_retrospectives_empty_when_dir_missing(
    opened_client: TestClient,
) -> None:
    data = opened_client.get("/api/meta/retrospectives").json()
    assert data == {"entries": [], "total_available": 0}


def test_retrospectives_surfaces_written_retro(
    opened_client: TestClient,
    opened_project: Path,
) -> None:
    write_retrospective(
        opened_project,
        project_id="elderberry-cluster",
        task="count fruit clusters",
        worked="instance seg held up",
        did_not_work="overlapping clusters merged",
    )
    (entry,) = opened_client.get("/api/meta/retrospectives").json()["entries"]
    assert entry["record"]["project_id"] == "elderberry-cluster"
    assert entry["record"]["task"] == "count fruit clusters"


@pytest.mark.parametrize("corpus", list(MemoryKind))
def test_a_route_refuses_a_page_limit_below_one(
    opened_client: TestClient, corpus: MemoryKind
) -> None:
    for limit in (0, -1):
        response = opened_client.get(
            f"/api/meta/{corpus}", params={"limit": limit}
        )
        assert response.status_code == 422, (limit, response.text)


@pytest.mark.parametrize("corpus", list(MemoryKind))
def test_a_route_and_load_project_memory_answer_one_page_of_the_same_records(
    opened_client: TestClient, opened_project: Path, corpus: MemoryKind
) -> None:
    """The route's page and the tool's page over the same records agree entry for entry."""
    for i in range(3):
        report_friction(
            opened_project, category="missing_tool", detail=f"report {i}"
        )
        write_retrospective(
            opened_project,
            project_id=f"line-{i}",
            task=f"task {i}",
            worked="w",
            did_not_work="d",
        )
    ts.replace(
        memory_key(Retrospective, str(opened_project), "unreadable"),
        "# a markdown document\n",
    )

    routed = opened_client.get(
        f"/api/meta/{corpus}", params={"limit": 2}
    ).json()

    assert routed == load_project_memory(opened_project, corpus, limit=2)
    assert len(routed["entries"]) == 2


def test_a_retrospective_that_will_not_read_is_answered_with_its_reason(
    opened_client: TestClient, opened_project: Path
) -> None:
    ts.replace(
        memory_key(Retrospective, str(opened_project), "unreadable"),
        "# a markdown document\n",
    )

    (entry,) = opened_client.get("/api/meta/retrospectives").json()["entries"]

    assert entry["id"] == "unreadable"
    assert "Retrospective" in entry["record"]["reason"]


def test_the_meta_routes_refuse_while_no_project_is_open(
    client: TestClient,
) -> None:
    assert client.get("/api/meta/reports").status_code == 409
    assert client.get("/api/meta/retrospectives").status_code == 409

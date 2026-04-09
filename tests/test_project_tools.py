"""Tests for project management tools."""

from __future__ import annotations

from pathlib import Path

from tcip_mcp.tools.project_tools import (
    init_project,
    create_session,
    append_session_event,
    list_sessions,
    get_session,
    get_project_status,
)


def test_init_project(tmp_path: Path):
    result = init_project(str(tmp_path))
    assert (tmp_path / ".tcip").is_dir()
    assert (tmp_path / ".tcip" / "sessions").is_dir()
    assert (tmp_path / ".tcip" / "config.toml").is_file()
    assert ".tcip/" in result["created"]


def test_create_and_get_session(tmp_path: Path):
    init_project(str(tmp_path))
    session = create_session(str(tmp_path), description="Test session")
    assert "session_id" in session

    sessions = list_sessions(str(tmp_path))
    assert sessions["count"] == 1
    assert sessions["sessions"][0]["description"] == "Test session"

    detail = get_session(str(tmp_path), session["session_id"])
    assert detail["count"] == 1
    assert detail["events"][0]["type"] == "session_start"


def test_append_session_event(tmp_path: Path):
    init_project(str(tmp_path))
    session = create_session(str(tmp_path))
    result = append_session_event(
        str(tmp_path), session["session_id"], "tool_call",
        {"tool": "list_crops", "result": "ok"},
    )
    assert result["event_type"] == "tool_call"

    detail = get_session(str(tmp_path), session["session_id"])
    assert detail["count"] == 2


def test_get_project_status(tmp_path: Path):
    status = get_project_status(str(tmp_path))
    assert status["initialized"] is False

    init_project(str(tmp_path))
    status = get_project_status(str(tmp_path))
    assert status["initialized"] is True
    assert status["has_config"] is True

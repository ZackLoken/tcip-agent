"""Tests for the set_active_project MCP tool (C3 loop-closer)."""

from __future__ import annotations

import pytest

from tcip_mcp import workspace
from tcip_mcp.tools.project_tools import set_active_project


@pytest.fixture(autouse=True)
def _isolate_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "ws"))


def test_set_active_writes_marker(monkeypatch):
    # Stub the GUI notification so the result is deterministic regardless of whether a
    # tcip-web backend happens to be listening on this machine.
    import tcip_mcp.web_client as web_client

    monkeypatch.setattr(
        web_client, "post_panel_event", lambda *a, **k: {"delivered": False}
    )
    result = set_active_project(name="hazelnut_catkin_valley-farm")
    assert "error" not in result
    assert result["name"] == "hazelnut_catkin_valley-farm"
    assert result["gui_notified"] is False
    assert result["project_path"].endswith("hazelnut_catkin_valley-farm")
    assert workspace.read_active_project() == "hazelnut_catkin_valley-farm"


def test_set_active_reports_gui_notified_when_delivered(monkeypatch):
    import tcip_mcp.web_client as web_client

    monkeypatch.setattr(web_client, "post_panel_event", lambda *a, **k: {"delivered": True})
    result = set_active_project(name="chestnut_burr_site-a")
    assert result["gui_notified"] is True


def test_set_active_rejects_traversal():
    result = set_active_project(name="../escape")
    assert "error" in result
    assert workspace.read_active_project() is None

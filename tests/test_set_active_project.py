"""Tests for the set_active_project MCP tool (C3 loop-closer)."""

from __future__ import annotations

import pytest

import tcip_store
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
    (workspace.project_path("hazelnut_catkin_valley-farm") / ".tcip").mkdir(parents=True)
    result = set_active_project(name="hazelnut_catkin_valley-farm")
    assert "error" not in result
    assert result["name"] == "hazelnut_catkin_valley-farm"
    assert result["gui_notified"] is False
    assert result["project_path"].endswith("hazelnut_catkin_valley-farm")
    assert workspace.read_active_project() == "hazelnut_catkin_valley-farm"


def test_set_active_reports_gui_notified_when_delivered(monkeypatch):
    import tcip_mcp.web_client as web_client

    monkeypatch.setattr(web_client, "post_panel_event", lambda *a, **k: {"delivered": True})
    (workspace.project_path("chestnut_burr_site-a") / ".tcip").mkdir(parents=True)
    result = set_active_project(name="chestnut_burr_site-a")
    assert result["gui_notified"] is True


def test_a_marker_that_does_not_decode_reads_as_no_active_project():
    """The marker is the front door's first read, so bytes no encoding of it can turn into a
    name are treated as unset rather than raised at every caller that asks which project is
    open. An ordinary marker still names its project, in the tests above.

    Bound to the file backend on purpose: the claim is about surviving bytes on disk that no
    text codec decodes, which only the file backend ever holds raw.
    """
    from tcip_store.file_backend import FileBackend

    tcip_store.bind(FileBackend())
    marker = workspace.active_marker_path()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_bytes("hazelnut_catkin_valley-farm\n".encode("utf-16"))

    assert workspace.read_active_project() is None


def test_set_active_rejects_traversal():
    result = set_active_project(name="../escape")
    assert "error" in result
    assert workspace.read_active_project() is None


def test_set_active_refuses_a_name_with_no_tcip_and_leaves_no_directory():
    result = set_active_project(name="no_such_project")
    assert "error" in result
    assert workspace.read_active_project() is None
    assert not workspace.project_path("no_such_project").exists()

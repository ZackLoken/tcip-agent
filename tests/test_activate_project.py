"""Tests for the activate_project MCP tool (C3 loop-closer)."""

from __future__ import annotations

import pytest

import tcip_store
from tcip_mcp import workspace
from tcip_mcp.tools.project_tools import activate_project


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
    (workspace.project_path("currant_bud_valley-farm") / ".tcip").mkdir(parents=True)
    result = activate_project(name="currant_bud_valley-farm")
    assert "error" not in result
    assert result["name"] == "currant_bud_valley-farm"
    assert result["gui_notified"] is False
    assert result["project_path"].endswith("currant_bud_valley-farm")
    assert workspace.read_active_project() == "currant_bud_valley-farm"
    # No backend answered at all: the docstring's down-backend case.
    assert result["backend_repinned"] is False
    assert result["backend_platform_root"] is None
    assert result["backend_root_problem"] is None


def test_set_active_reports_gui_notified_when_delivered(monkeypatch):
    import tcip_mcp.web_client as web_client

    monkeypatch.setattr(web_client, "post_panel_event", lambda *a, **k: {"delivered": True})
    (workspace.project_path("chestnut_burr_site-a") / ".tcip").mkdir(parents=True)
    result = activate_project(name="chestnut_burr_site-a")
    assert result["gui_notified"] is True
    # Delivered, but with no response body: still the down-backend shape for the three fields.
    assert result["backend_repinned"] is False
    assert result["backend_platform_root"] is None
    assert result["backend_root_problem"] is None


def test_set_active_reports_the_root_the_backend_repinned_to(monkeypatch):
    import tcip_mcp.web_client as web_client

    monkeypatch.setattr(
        web_client, "post_panel_event",
        lambda *a, **k: {"delivered": True, "response": {"platform_root": "/repinned/root"}},
    )
    (workspace.project_path("elderberry_cyme_bloom-site") / ".tcip").mkdir(parents=True)
    result = activate_project(name="elderberry_cyme_bloom-site")
    assert result["backend_repinned"] is True
    assert result["backend_platform_root"] == "/repinned/root"
    assert result["backend_root_problem"] is None


def test_set_active_reports_why_the_backend_could_not_repin(monkeypatch):
    import tcip_mcp.web_client as web_client

    monkeypatch.setattr(
        web_client, "post_panel_event",
        lambda *a, **k: {
            "delivered": True,
            "response": {"platform_root_problem": "the marker's project has no .tcip"},
        },
    )
    (workspace.project_path("persimmon_calyx_site-a") / ".tcip").mkdir(parents=True)
    result = activate_project(name="persimmon_calyx_site-a")
    assert result["backend_repinned"] is False
    assert result["backend_platform_root"] is None
    assert result["backend_root_problem"] == "the marker's project has no .tcip"


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
    marker.write_bytes("currant_bud_valley-farm\n".encode("utf-16"))

    assert workspace.read_active_project() is None


def test_set_active_rejects_traversal():
    result = activate_project(name="../escape")
    assert "error" in result
    assert workspace.read_active_project() is None


def test_set_active_refuses_a_name_with_no_tcip_and_leaves_no_directory():
    result = activate_project(name="no_such_project")
    assert "error" in result
    assert workspace.read_active_project() is None
    assert not workspace.project_path("no_such_project").exists()


def test_active_project_if_present_treats_a_traversal_marker_as_no_project(tmp_path):
    """activate_project refuses a traversal name, so this writes the marker directly
    through the store, the shape a corrupted or hand-edited marker would take. A real
    .tcip sits where the traversal points, so an unsafe reader would actually find it."""
    (tmp_path / "escapee" / ".tcip").mkdir(parents=True)
    tcip_store.replace(workspace.active_project_key(), "../escapee")

    assert workspace.active_project_if_present() is None

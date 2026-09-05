"""view_gui_state + active-project anchoring: the agent reads the live GUI session."""

from __future__ import annotations

from pathlib import Path


def _setup(tmp_path, monkeypatch, *, with_gui=True):
    import tcip_store
    from tcip_mcp.web_client import gui_snapshot_key
    from tcip_mcp.workspace import active_project_key

    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    proj = ws / "currant_bud_valley"
    (proj / ".tcip" / "state").mkdir(parents=True)
    tcip_store.replace(active_project_key(), "currant_bud_valley")
    if with_gui:
        gui = {
            "active_tab": "annotate",
            "mode": "box",
            "dataset": {
                "project_root": str(proj),
                "dataset_root": str(proj),
                "subject": "bud",
                "date": "2026-02-11",
                "image_list": ["IMG_0132.JPG", "IMG_0133.JPG"],
                "current_image_index": 1,
            },
        }
        tcip_store.replace(gui_snapshot_key(str(proj)), gui)
    return proj


def test_view_gui_state_none_when_no_project(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    from tcip_mcp.tools.project_tools import view_gui_state
    assert view_gui_state()["active_project"] is None


def test_view_gui_state_reads_gui(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from tcip_mcp.tools.project_tools import view_gui_state
    ctx = view_gui_state()
    assert ctx["active_project"] == "currant_bud_valley"
    assert ctx["subject"] == "bud"
    assert ctx["date"] == "2026-02-11"
    assert ctx["active_tab"] == "annotate"
    assert ctx["current_image_index"] == 1
    assert Path(ctx["current_image"]).name == "IMG_0133.JPG"
    assert "2026-02-11" in ctx["current_image"]


def test_view_gui_state_no_gui_yet(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, with_gui=False)
    from tcip_mcp.tools.project_tools import view_gui_state
    ctx = view_gui_state()
    assert ctx["active_project"] == "currant_bud_valley"
    assert "note" in ctx


def test_inspect_project_anchors_to_active_project(tmp_path, monkeypatch):
    proj = _setup(tmp_path, monkeypatch)
    from tcip_mcp.tools.project_tools import inspect_project
    st = inspect_project()  # empty project_path -> the active project
    assert st["project_path"] == str(proj)
    assert st["initialized"] is True

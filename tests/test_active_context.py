"""get_active_context + active-project anchoring — the agent reads the live GUI session (Root A)."""

from __future__ import annotations

import json
from pathlib import Path


def _setup(tmp_path, monkeypatch, *, with_gui=True):
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    proj = ws / "hazelnut_catkin_valley"
    (proj / ".tcip" / "state").mkdir(parents=True)
    (ws / ".active").write_text("hazelnut_catkin_valley\n", encoding="utf-8")
    if with_gui:
        gui = {
            "active_tab": "annotate",
            "mode": "box",
            "dataset": {
                "project_root": str(proj),
                "dataset_root": str(proj),
                "annotation_type": "catkin",
                "date": "2026-02-11",
                "image_list": ["IMG_0132.JPG", "IMG_0133.JPG"],
                "current_image_index": 1,
            },
        }
        (proj / ".tcip" / "state" / "gui.json").write_text(json.dumps(gui), encoding="utf-8")
    return proj


def test_get_active_context_none_when_no_project(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    from tcip_mcp.tools.project_tools import get_active_context
    assert get_active_context()["active_project"] is None


def test_get_active_context_reads_gui(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from tcip_mcp.tools.project_tools import get_active_context
    ctx = get_active_context()
    assert ctx["active_project"] == "hazelnut_catkin_valley"
    assert ctx["annotation_type"] == "catkin"
    assert ctx["date"] == "2026-02-11"
    assert ctx["active_tab"] == "annotate"
    assert ctx["current_image_index"] == 1
    assert Path(ctx["current_image"]).name == "IMG_0133.JPG"
    assert "2026-02-11" in ctx["current_image"]


def test_get_active_context_no_gui_yet(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, with_gui=False)
    from tcip_mcp.tools.project_tools import get_active_context
    ctx = get_active_context()
    assert ctx["active_project"] == "hazelnut_catkin_valley"
    assert "note" in ctx


def test_get_project_status_anchors_to_active_project(tmp_path, monkeypatch):
    proj = _setup(tmp_path, monkeypatch)
    from tcip_mcp.tools.project_tools import get_project_status
    st = get_project_status()  # empty project_path -> the active project
    assert st["project_path"] == str(proj)
    assert st["initialized"] is True

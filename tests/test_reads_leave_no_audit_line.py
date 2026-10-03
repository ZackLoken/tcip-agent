"""A render and a panel push are reads: neither leaves a line in the project's audit log."""

from __future__ import annotations

from pathlib import Path

from tests._audit_fixtures import audit_rows


def test_a_dataset_render_leaves_no_audit_line(project: Path) -> None:
    from PIL import Image

    from tcip_mcp.tools.vision_tools import visualize

    images = project / "dataset" / "images" / "2026-04-01"
    images.mkdir(parents=True)
    Image.new("RGB", (32, 24), (40, 90, 60)).save(images / "a.png")
    before = audit_rows(project)

    result = visualize(project, source="dataset", path=str(project / "dataset"), n=1)

    assert "error" not in result, result
    assert audit_rows(project) == before


def test_a_panel_push_leaves_no_audit_line(project: Path) -> None:
    from tcip_mcp.tools.gui_tools import push_panel_event

    before = audit_rows(project)

    push_panel_event(project, project.parent, "meta", "probe", {"n": 1})

    assert audit_rows(project) == before

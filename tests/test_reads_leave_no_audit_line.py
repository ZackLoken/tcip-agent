"""A live canvas render and a panel push, each checked to leave the project's log unchanged."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tests._audit_fixtures import audit_rows

if TYPE_CHECKING:
    from tcip_mcp.workspace import BoundProject


def test_a_dataset_render_leaves_no_audit_line(project: Path) -> None:
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.tools.vision_tools import DatasetRender, visualize
    from tests._producer_fixtures import write_image

    images = image_dir(project / "dataset", "2026-04-01")
    write_image(images / "a.png", (32, 24), (40, 90, 60))
    before = audit_rows(project)

    result = visualize(
        project,
        DatasetRender(source="dataset", path=str(project / "dataset"), n=1),
    )

    assert "error" not in result, result
    assert audit_rows(project) == before


def test_a_live_canvas_render_leaves_no_audit_line(
    project: Path, bound: BoundProject
) -> None:
    from tcip_mcp.tools.vision_tools import capture_live_canvas
    from tests.test_canvas_liveview import _make_image, _write_state

    _write_state(project, _make_image(project))
    before = audit_rows(project)

    result = capture_live_canvas(bound, refresh=False)

    assert "error" not in result, result
    assert audit_rows(project) == before


def test_a_panel_push_leaves_no_audit_line(
    project: Path, bound: BoundProject
) -> None:
    from tcip_mcp.tools.gui_tools import push_panel_event

    before = audit_rows(project)

    push_panel_event(bound, "meta", "probe", {"n": 1})

    assert audit_rows(project) == before

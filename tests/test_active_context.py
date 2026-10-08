"""view_gui_state: the agent reads the GUI state the web backend persisted for the project it acts
on, through the same decoder the backend reopens it with."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tcip_web.state import OpenProject


def _persist_a_selection(project: OpenProject) -> None:
    """The backend's own producer: open ``project``, select a date's second image, persist."""
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.web_client import selection_for
    from tcip_web.state import StateStore

    images = image_dir(project.root, "2026-02-11")
    images.mkdir(parents=True)
    for name in ("IMG_0132.JPG", "IMG_0133.JPG"):
        (images / name).write_bytes(b"")
    store = StateStore()
    asyncio.run(store.open_project(project))
    asyncio.run(store.mutate(
        {"active_tab": "annotate",
         "dataset": selection_for(project.root, "bud", "2026-02-11", None, 1)},
        project=project))


def test_view_gui_state_reads_what_the_backend_persisted_and_reopens(made):
    from tcip_mcp.tools.project_tools import view_gui_state
    from tcip_web.state import StateStore

    _persist_a_selection(made)
    ctx = view_gui_state(made.root)
    reopened = StateStore()
    asyncio.run(reopened.open_project(made))

    assert ctx["dataset"] == reopened.state.dataset.model_dump(mode="json")
    assert ctx["dataset"]["subject"] == "bud" and ctx["dataset"]["date"] == "2026-02-11"
    assert ctx["active_tab"] == "annotate"
    assert Path(ctx["current_image"]).name == "IMG_0133.JPG"
    assert "2026-02-11" in ctx["current_image"]


def test_view_gui_state_with_no_snapshot_carries_a_note(project):
    from tcip_mcp.tools.project_tools import view_gui_state

    assert set(view_gui_state(project)) == {"note"}


def test_view_gui_state_refuses_a_snapshot_that_does_not_decode(project):
    import tcip_store
    from pydantic import ValidationError

    from tcip_mcp.tools.project_tools import view_gui_state
    from tcip_mcp.web_client import gui_snapshot_key

    tcip_store.replace(gui_snapshot_key(project), {"active_tab": "annotate"})

    with pytest.raises(ValidationError):
        view_gui_state(project)


def test_inspect_project_reports_the_project_it_is_given(project):
    from tcip_mcp.tools.project_tools import inspect_project
    st = inspect_project(project)
    assert st["project_path"] == str(project)
    assert st["display_name"] == "Test project"

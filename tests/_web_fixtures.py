"""Making a test directory a project, and opening it in the web backend the way the picker does."""

from __future__ import annotations

import asyncio
from pathlib import Path


def new_project(path: Path) -> Path:
    """Make ``path`` a project through the platform's own creation door (``initialize_project``),
    its record holding an id, a display name and a site; returns ``path``. Refuses (by assertion)
    when ``path`` already records another name or site."""
    from tcip_mcp.tools.project_tools import initialize_project

    result = initialize_project(str(path), "Test project", "north orchard")
    assert "error" not in result, result
    return path


def open_new_project(path: Path) -> Path:
    """Make ``path`` a project (:func:`new_project`) and open it in the web backend; returns
    ``path``."""
    from tcip_web.state import store

    asyncio.run(store.open_project(new_project(path)))
    return path

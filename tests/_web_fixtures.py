"""Making a test directory a project, and opening it in the web backend the way the picker does."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tcip_mcp.workspace import BoundProject
    from tcip_web.state import OpenProject


def named_project(path: Path, display_name: str) -> OpenProject:
    """Make ``path`` a project named ``display_name`` through the platform's own creation door
    (``initialize_project``), its record holding an id, that name and a site; the project as
    the backend's open takes it. Refuses (by assertion) when ``path`` already records another
    name or site."""
    from tcip_mcp.tools.project_tools import initialize_project
    from tcip_web.state import OpenProject

    result = initialize_project(str(path), display_name, "north orchard")
    assert "error" not in result, result
    return OpenProject(path, result["id"])


def bound_to(project: OpenProject) -> BoundProject:
    """``project`` as an MCP server started for it binds it, under the workspace this process
    names (``TCIP_WORKSPACE``)."""
    from tcip_mcp.workspace import BoundProject, workspace_from_environment

    return BoundProject(project.root, project.id, workspace_from_environment())


def new_project(path: Path) -> OpenProject:
    """Make ``path`` a project named "Test project" (:func:`named_project`)."""
    return named_project(path, "Test project")


def open_new_project(path: Path) -> OpenProject:
    """Make ``path`` a project (:func:`new_project`) and open it in the web backend."""
    from tcip_web.state import store

    project = new_project(path)
    asyncio.run(store.open_project(project))
    return project


BROWSER = {"Origin": "http://127.0.0.1"}
"""The Origin a browser on the backend's own page sends, which an acknowledgment requires."""


def acknowledged_post(client, url: str, body: dict, *, reason: str, user: str = "breeder"):
    """Post ``body`` to ``url`` as ``user`` ships an unvalidated result from the screen: refused
    first, then posted again from the browser acknowledging, for ``reason``, the result the
    refusal names; the second response."""
    body = {**body, "user": user}
    refused = client.post(url, json=body)
    assert refused.status_code == 400, refused.text
    digest = refused.json()["detail"]["result_sha256"]
    assert digest is not None, refused.text
    return client.post(url, headers=BROWSER, json={**body, "acknowledgment": {
        "reason": reason, "result_sha256": digest}})

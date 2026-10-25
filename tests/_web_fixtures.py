"""Web backend fixtures over a scratch workspace.

They start the web backend over a scratch workspace, make a test directory a project, and open it
in the web backend.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from tcip_mcp.project_record import CaptureInvariances

if TYPE_CHECKING:
    from fastapi.testclient import TestClient
    from httpx import Response

    from tcip_mcp.workspace import BoundProject
    from tcip_web.state import OpenProject

INVARIANCES = CaptureInvariances(
    horizontal_flip=True,
    vertical_flip=True,
    rotation=180,
    color=True,
    intensity=True,
    scale=True,
)
"""A sample breeder's answer: every capture transform leaves the trait's measurement unchanged."""


def named_project(
    path: Path,
    display_name: str,
    *,
    invariances: CaptureInvariances = INVARIANCES,
) -> OpenProject:
    """Make ``path`` a project named ``display_name`` through the platform's own creation door.

    ``initialize_project`` writes its record, holding an id, that name, the site "north orchard"
    and ``invariances`` (the sample :data:`INVARIANCES` unless stated), and the call is asserted
    to return no error. Returns an ``OpenProject`` of ``path`` and the id.
    """
    from tcip_mcp.tools.project_tools import initialize_project
    from tcip_web.state import OpenProject

    result = initialize_project(
        str(path), display_name, "north orchard", invariances, actor=None
    )
    assert "error" not in result, result
    return OpenProject(path, result["id"])


@contextmanager
def backend_session(workspace: Path) -> Iterator[TestClient]:
    """Start the web backend serving ``workspace`` and yield a loopback client of it.

    The backend starts with the store bound, ``TCIP_WORKSPACE`` naming ``workspace`` and
    ``TCIP_IMAGE_ROOTS`` unset, configured with the workspace and no additive image roots. On
    exit, by return or by exception, the open project is closed, the bound store's connections
    are closed and both environment variables are restored.
    """
    from fastapi.testclient import TestClient

    import tcip_store
    from tcip_web.app import app
    from tcip_web.state import store

    names = ("TCIP_WORKSPACE", "TCIP_IMAGE_ROOTS")
    saved = {name: os.environ.get(name) for name in names}
    backend = tcip_store.bind()
    os.environ["TCIP_WORKSPACE"] = str(workspace)
    os.environ.pop("TCIP_IMAGE_ROOTS", None)
    store.configure(workspace, ())
    try:
        yield TestClient(app, base_url="http://127.0.0.1")
    finally:
        try:
            opened = store.opened
            if opened is not None:
                asyncio.run(store.close_project(opened.id))
        finally:
            backend.close()
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def bound_to(project: OpenProject) -> BoundProject:
    """Return a ``BoundProject`` of ``project``'s root and id.

    The binding is under the workspace this process names (``TCIP_WORKSPACE``).
    """
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


def acknowledged_post(
    client: TestClient,
    url: str,
    body: dict,
    *,
    reason: str,
    user: str = "breeder",
) -> Response:
    """Post ``body`` to ``url`` as ``user``, refused, then again under an acknowledgment.

    The first post is asserted to answer 400 naming a result hash; the second carries the
    browser headers and an acknowledgment of that hash for ``reason``. Returns the second
    response.
    """
    body = {**body, "user": user}
    refused = client.post(url, json=body)
    assert refused.status_code == 400, refused.text
    digest = refused.json()["detail"]["result_sha256"]
    assert digest is not None, refused.text
    return client.post(
        url,
        headers=BROWSER,
        json={
            **body,
            "acknowledgment": {"reason": reason, "result_sha256": digest},
        },
    )

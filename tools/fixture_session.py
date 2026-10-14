"""The scratch workspace the frontend fixture generators make their scenes in, the opening of a
project through the picker's route, the one walk that replaces what a run fills differently in
what a scene serves, and the write of a fixture file.

A generator creates its projects through ``tests._web_fixtures.named_project`` inside
:func:`scratch_session`, sends its own requests through the session's client, states its own
replacement policy to :func:`replaced` and writes the result with :func:`write_fixture`.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tests._web_fixtures import backend_session

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from tcip_web.state import OpenProject


@contextmanager
def scratch_session() -> Iterator[tuple[Path, TestClient]]:
    """A fresh workspace inside a scratch directory and a client of the web backend serving it
    (``tests._web_fixtures.backend_session``); the directory is removed on exit, after the
    backend's session has closed.
    """
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory).resolve() / "workspace"
        workspace.mkdir()
        with backend_session(workspace) as client:
            yield workspace, client


def opened(client: TestClient, project: OpenProject) -> None:
    """Open ``project`` through ``POST /api/projects/open``; raises on a refusal."""
    response = client.post(
        "/api/projects/open", json={"id": project.id, "user": "grower"}
    )
    response.raise_for_status()


def replaced(
    value: Any, leaf: Callable[[str | None, Any], Any], key: str | None = None
) -> Any:
    """``value`` with each leaf (anything not a dict or a list) replaced by ``leaf(key, leaf
    value)``, ``key`` being the dict key the leaf sits under, ``None`` for a list item or the
    root.
    """
    if isinstance(value, dict):
        return {k: replaced(v, leaf, k) for k, v in value.items()}
    if isinstance(value, list):
        return [replaced(v, leaf) for v in value]
    return leaf(key, value)


def write_fixture(path: Path, content: Any) -> int:
    """Write ``content`` to ``path`` as indented JSON with LF line endings and say so; returns
    the exit code 0.
    """
    path.write_text(
        json.dumps(content, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {path}")
    return 0

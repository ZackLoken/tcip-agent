"""Scratch-session utilities for producing served JSON as fixture files.

They are a scratch workspace with a web-backend client serving it (:func:`scratch_session`), the
opening of a project through the picker's route (:func:`opened`), a walk that replaces each leaf
of a served value (:func:`replaced`), and the write of a fixture file (:func:`write_fixture`).
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from tests._web_fixtures import backend_session

if TYPE_CHECKING:
    from fastapi.testclient import TestClient
    from pydantic import JsonValue

    from tcip_web.state import OpenProject


@contextmanager
def scratch_session() -> Iterator[tuple[Path, TestClient]]:
    """Yield a fresh scratch workspace and a client of the web backend serving it.

    The client is ``tests._web_fixtures.backend_session``'s; the directory is removed on exit,
    after the backend's session has closed.
    """
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory).resolve() / "workspace"
        workspace.mkdir()
        with backend_session(workspace) as client:
            yield workspace, client


def opened(client: TestClient, project: OpenProject) -> JsonValue:
    """Open ``project`` through ``POST /api/projects/open`` and return its answer.

    Raises on a refusal.
    """
    from tcip_web.app import app

    response = client.post(
        app.url_path_for("open_project"),
        json={"id": project.id, "user": "grower"},
    )
    response.raise_for_status()
    return response.json()


type Leaf = Callable[[str | None, JsonValue], JsonValue]
"""A replacement of one leaf of a served value, given the dict key it sits under."""


def replaced(
    served: Mapping[str, JsonValue], leaf: Leaf
) -> dict[str, JsonValue]:
    """Return the served mapping with each leaf (anything not a dict or a list) through ``leaf``.

    A leaf becomes ``leaf(key, leaf value)``, ``key`` being the dict key the leaf sits under,
    ``None`` for a list item.
    """
    return {k: _replaced(v, leaf, k) for k, v in served.items()}


def _replaced(value: JsonValue, leaf: Leaf, key: str | None) -> JsonValue:
    """Return ``value``, under ``key``, with each of its leaves replaced through ``leaf``."""
    if isinstance(value, dict):
        return replaced(value, leaf)
    if isinstance(value, list):
        return [_replaced(v, leaf, None) for v in value]
    return leaf(key, value)


def write_fixture(path: Path, content: JsonValue) -> int:
    """Write ``content`` to ``path`` as indented JSON with LF line endings and say so.

    Returns the exit code 0.
    """
    path.write_text(
        json.dumps(content, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {path}")
    return 0

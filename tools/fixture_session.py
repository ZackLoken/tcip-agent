"""Scratch-session utilities for producing served JSON as fixture files.

They are a scratch workspace with a web-backend client serving it (:func:`scratch_session`), the
opening of a project through the picker's route (:func:`opened`), a walk that replaces each leaf
of a served value (:func:`replaced`), the normalization of what a run or a platform fills
differently in a served value (:func:`normalized`), and the write of a fixture file
(:func:`write_fixture`).
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from tests import formatted, write_generated
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


SCRATCH_ROOT = "/scratch"
"""What a scratch session's directory reads as in a served string."""

FLOAT_PLACES = 6
"""The decimal places a served float is compared to: a computed value's last bits differ between
the platforms a scene runs on, and a delivered meter value carries no more."""

type Stamp = tuple[type, JsonValue]
"""The exact type of the value a producer writes under a key, and what that value reads as."""

_NOTHING: Mapping = MappingProxyType({})


def normalized(
    served: Mapping[str, JsonValue],
    *,
    stamped: Mapping[str, Stamp] = _NOTHING,
    numbered: frozenset[str] = frozenset(),
    minted: Mapping[str, str] = _NOTHING,
    places: Mapping[Path, str] = _NOTHING,
    rounded: bool = False,
) -> dict[str, JsonValue]:
    """Return ``served`` with each value a run or a platform fills differently read as its stand-in.

    A value under a ``stamped`` key whose type is exactly the one its :data:`Stamp` names becomes
    the placeholder beside it; a string under a ``numbered`` key becomes ``<key n>``, ``n`` counting
    the distinct such strings in order of first appearance, so two places sharing a value still
    share one; with ``rounded``, each float is rounded to :data:`FLOAT_PLACES`; and in each other
    string each id ``minted`` names becomes its name and each spelling of each directory
    ``places`` names its placeholder, a string naming one then reading with forward slashes
    alone. Every other value, a wrong-typed one under a ``stamped`` key among them, is kept as
    served.
    """
    seen: dict[str, str] = {}

    def leaf(key: str | None, value: JsonValue) -> JsonValue:
        if key is not None and key in stamped:
            kind, placeholder = stamped[key]
            return placeholder if type(value) is kind else value
        if key in numbered and isinstance(value, str):
            return seen.setdefault(value, f"<{key} {len(seen)}>")
        if rounded and isinstance(value, float):
            return round(value, FLOAT_PLACES)
        if not isinstance(value, str):
            return value
        for minted_id, name in minted.items():
            value = value.replace(minted_id, name)
        spelled = value
        for directory, placeholder in places.items():
            for spelling in (str(directory), directory.as_posix()):
                spelled = spelled.replace(spelling, placeholder)
        return value if spelled == value else spelled.replace("\\", "/")

    return replaced(served, leaf)


def write_fixture(path: Path, content: JsonValue) -> int:
    """Write ``content`` to ``path`` as JSON the frontend's Prettier formats; return exit code 0."""
    return write_generated(path, formatted(json.dumps(content), path))

"""Write a project's open answer and the state envelope that follows it as a fixture file.

In a scratch session (``tools/fixture_session.py``), a project created through
``tests._web_fixtures.named_project`` is opened through the picker's route; the route's answer
(``opened``) and the envelope ``/ws/state`` then sends on connect
(``tcip_web.app.state_snapshot_message``, ``envelope``) are written, :func:`normalized`, to
``frontend/src/test/openFixtures.json``.

    python -m tools.generate_open_fixture
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from tests import FRONTEND_SRC
from tools.fixture_session import (
    opened,
    replaced,
    scratch_session,
    write_fixture,
)

if TYPE_CHECKING:
    from pydantic import JsonValue

GENERATED_PATH = FRONTEND_SRC / "test" / "openFixtures.json"


def normalized(
    served: dict, workspace: Path, project_id: str
) -> dict[str, JsonValue]:
    """Return ``served`` with the values a run fills differently replaced by what they stand for.

    The project's minted id becomes ``<valley id>``, its directory the workspace placeholder and
    its folder name, the process's epoch ``<epoch>`` and the version, which counts every state
    the process has held, ``0``. A path ``Path`` cannot read raises its ``TypeError``.
    """

    def leaf(key: str | None, value: JsonValue) -> JsonValue:
        if key == "path":
            relative = Path(cast("str", value)).relative_to(workspace)
            return f"<workspace>/{relative.as_posix()}"
        if key == "epoch":
            return "<epoch>"
        if key == "version":
            return 0
        if value == project_id:
            return "<valley id>"
        return value

    return replaced(served, leaf)


def produce() -> dict[str, JsonValue]:
    """Open the fixture project; return its open answer and the envelope after it, normalized."""
    from tcip_web.app import app
    from tests._web_fixtures import named_project

    with scratch_session() as (workspace, client):
        valley = named_project(workspace / "valley", "valley")
        answer = opened(client, valley)
        socket = client.base_url.join(app.url_path_for("state_ws"))
        with client.websocket_connect(
            str(socket.copy_with(scheme="ws"))
        ) as ws:
            envelope = ws.receive_json()
        return normalized(
            {"opened": answer, "envelope": envelope}, workspace, valley.id
        )


if __name__ == "__main__":
    raise SystemExit(write_fixture(GENERATED_PATH, produce()))

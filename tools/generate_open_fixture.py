"""Write a project's open answer and the state envelope that follows it as a fixture file.

In a scratch session (``tools/fixture_session.py``), a project created through
``tests._web_fixtures.named_project`` is opened through the picker's route; the route's answer
(``opened``) and the envelope ``/ws/state`` then sends on connect
(``tcip_web.app.state_snapshot_message``, ``envelope``) are written, normalized
(``fixture_session.normalized``), to ``frontend/src/test/openFixtures.json``.

    python -m tools.generate_open_fixture
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests import FRONTEND_SRC
from tools.fixture_session import (
    normalized,
    opened,
    scratch_session,
    write_fixture,
)

if TYPE_CHECKING:
    from pydantic import JsonValue

    from tools.fixture_session import Stamp

GENERATED_PATH = FRONTEND_SRC / "test" / "openFixtures.json"

STAMPED: dict[str, Stamp] = {"epoch": (str, "<epoch>"), "version": (int, 0)}
"""The process's epoch, read as ``<epoch>``, and the version, which counts every state the
process has held, read as ``0``."""


def produce() -> dict[str, JsonValue]:
    """Open the fixture project; return its open answer and the envelope after it.

    Both are normalized by :data:`STAMPED`, the project's minted id read as ``<valley id>`` and
    the workspace as ``<workspace>``.
    """
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
            {"opened": answer, "envelope": envelope},
            stamped=STAMPED,
            minted={valley.id: "<valley id>"},
            places={workspace: "<workspace>"},
        )


if __name__ == "__main__":
    raise SystemExit(write_fixture(GENERATED_PATH, produce()))

"""``tools/smoke_terminal_e2e.py`` runs against the workspace it is given.

It sets ``TCIP_WORKSPACE`` to the workspace it is given and opens nothing for a row that cannot
launch.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import sys
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from tcip_web import terminal as pty_host
from tests import REPO_ROOT

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

    import pytest
    from fastapi.testclient import TestClient

SCRIPT = REPO_ROOT / "tools" / "smoke_terminal_e2e.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("smoke_terminal_e2e", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["smoke_terminal_e2e"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_runs_under_the_given_workspace_and_opens_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The smoke run binds the workspace it is given and opens nothing for a row that fails."""
    from tcip_web.state import store

    monkeypatch.delenv(pty_host.TERMINAL_CMD_ENV, raising=False)
    row = dataclasses.replace(
        pty_host.PROVIDERS[0], executable="tcip-smoke-test-nonexistent-cli"
    )
    monkeypatch.setattr(pty_host, "PROVIDERS", (row,))

    scratch_ws = tmp_path.parent.parent / "scratch-workspace"
    mod = _load()
    result = mod.main(row.id, workspace=str(scratch_ws))

    assert result == 1
    assert os.environ["TCIP_WORKSPACE"] == str(scratch_ws)
    assert store.workspace == scratch_ws.resolve()
    assert store.opened is None


def test_the_signal_is_a_friction_report_of_the_open_project_carrying_the_token(
    opened_project: Path, opened_client: TestClient
) -> None:
    """The smoke signal is one friction report of the open project carrying the token.

    ``reported`` answers only while ``report_friction`` has recorded, in the open project,
    exactly one report whose detail carries the token, of the requested category: the token
    anywhere else, a report of another category, or a second call is not it.
    """
    from tcip_mcp.tools.meta_tools import report_friction

    mod = _load()
    token = "ack-0123456789ab"
    report_friction(
        opened_project,
        mod.REQUESTED_CATEGORY,
        "an unrelated report",
        context={"note": token},
    )
    assert not mod.reported(opened_client, token)

    report_friction(opened_project, mod.REQUESTED_CATEGORY, f"smoke {token}")
    assert mod.reported(opened_client, token)

    report_friction(opened_project, mod.REQUESTED_CATEGORY, f"again {token}")
    assert not mod.reported(opened_client, token)

    other = "ack-ba9876543210"
    report_friction(opened_project, "missing_tool", other)
    assert not mod.reported(opened_client, other)


def test_the_websocket_url_is_absolute_and_carries_the_served_host(
    tmp_path: Path, client: TestClient
) -> None:
    """``terminal_ws_url`` is absolute, names the client's own host and carries the socket path.

    The path is the one the app reverses for a session's socket, as the smoke produces it.
    """
    from tcip_web.app import app

    mod = _load()
    path = app.url_path_for("terminal_ws", session_id="abc123")

    url = mod.terminal_ws_url(client, path)

    parsed = urlsplit(url)
    assert parsed.scheme == "ws"
    assert parsed.netloc == "127.0.0.1"
    assert parsed.path == path == "/api/terminal/ws/abc123"

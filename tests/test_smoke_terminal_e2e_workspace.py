"""``tools/smoke_terminal_e2e.py`` runs against the workspace it is given.

It sets ``TCIP_WORKSPACE`` to the workspace it is given, and a run beside a live project opens
nothing and adds nothing to that project's log.
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


def test_runs_under_the_given_workspace_never_the_machines_last_opened_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The smoke run uses the workspace it is given, not the last-opened project.

    Beside a workspace whose last-opened pointer names a project, the run binds the workspace
    it was given, opens nothing for a row that cannot launch, and leaves that project's audit
    record count unchanged.
    """
    import tcip_store
    from tcip_mcp import workspace
    from tcip_mcp.audit import audit_log_key
    from tcip_web.state import store
    from tests._web_fixtures import new_project

    live_project = new_project(tmp_path)
    workspace.write_last_opened(tmp_path.parent, live_project.id)
    live_lines = len(
        tcip_store.read_log(audit_log_key(live_project.root)).records
    )
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
    assert (
        len(tcip_store.read_log(audit_log_key(live_project.root)).records)
        == live_lines
    )


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
    """``terminal_ws_url`` is absolute and names the client's own host."""
    mod = _load()

    url = mod.terminal_ws_url(client, "abc123")

    parsed = urlsplit(url)
    assert parsed.scheme == "ws"
    assert parsed.netloc == "127.0.0.1"
    assert parsed.path == "/api/terminal/ws/abc123"

"""``tools/smoke_terminal_e2e.py`` must never run against the machine's own workspace: it sets
``TCIP_WORKSPACE`` to a workspace it is given before the served app resolves anything, so a run
beside a developer's real projects never opens or audits into one.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

from tcip_web import terminal as pty_host

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "smoke_terminal_e2e.py"


def _load():
    spec = importlib.util.spec_from_file_location("smoke_terminal_e2e", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["smoke_terminal_e2e"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_runs_under_the_given_workspace_never_the_machines_last_opened_project(
        tmp_path, monkeypatch):
    """A live workspace whose last-opened pointer names a real project must not decide what this
    run touches: the run sees the workspace it was given, opens nothing, and leaves the live
    project's log as it was."""
    import tcip_store

    from tcip_mcp import workspace
    from tcip_mcp.audit import audit_log_key
    from tcip_mcp.project_record import read_record
    from tcip_web.state import store
    from tests._web_fixtures import new_project

    live_project = new_project(tmp_path)
    workspace.write_last_opened(tmp_path.parent, read_record(live_project)["id"])
    live_lines = len(tcip_store.read_log(audit_log_key(live_project)).records)
    monkeypatch.delenv(pty_host.TERMINAL_CMD_ENV, raising=False)
    row = dataclasses.replace(pty_host.PROVIDERS[0], executable="tcip-smoke-test-nonexistent-cli")
    monkeypatch.setattr(pty_host, "PROVIDERS", (row,))

    scratch_ws = tmp_path.parent.parent / "scratch-workspace"
    mod = _load()
    result = mod.main(row.id, workspace=str(scratch_ws))

    assert result == 1
    assert os.environ["TCIP_WORKSPACE"] == str(scratch_ws)
    assert store.workspace == scratch_ws.resolve()
    assert store.project_id is None
    assert len(tcip_store.read_log(audit_log_key(live_project)).records) == live_lines


def test_the_websocket_url_is_absolute_and_carries_the_served_host(tmp_path, monkeypatch):
    """``terminal_ws_url`` is absolute and names the client's own host."""
    from fastapi.testclient import TestClient

    from tcip_web.app import app

    mod = _load()
    client = TestClient(app, base_url="http://127.0.0.1")

    url = mod.terminal_ws_url(client, "abc123")

    parsed = urlsplit(url)
    assert parsed.scheme == "ws"
    assert parsed.netloc == "127.0.0.1"
    assert parsed.path == "/api/terminal/ws/abc123"

"""Live smoke of one provider row's real harness through the embedded agent terminal.

Launches the row, submits a prompt to the session the way the rail submits a staged request,
and passes only when the harness answers with a token the prompt asks it to compute, which its
echoed input never contains.

Usage (from the repo root, tcip-agent env):
    python tools/smoke_terminal_e2e.py <provider id>
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-web" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-mcp" / "src"))

PROMPT = ("After that, reply with the word SMOKE immediately followed by the product of six and "
          "seven, written as one token, and use no tools.")
ANSWER = "SMOKE42"
ANSWER_TIMEOUT_S = 240
ANSI = re.compile(r"\x1b\[[0-9;?>]*[A-Za-z]|\x1b\][^\x07]*?(?:\x07|\x1b\\)|\x1b[>=()][0-9A-Za-z]?")


def terminal_ws_url(client: Any, session_id: str) -> str:
    """The absolute ``ws://`` URL of ``session_id``'s socket on the host ``client`` serves."""
    parts = urlsplit(str(client.base_url))
    scheme = "wss" if parts.scheme == "https" else "ws"
    return urlunsplit((scheme, parts.netloc, f"/api/terminal/ws/{session_id}", "", ""))


def main(provider: str, workspace: str | None = None) -> int:
    """Drive the smoke for the ``provider`` row with ``TCIP_WORKSPACE`` set to ``workspace`` (a
    fresh temp directory when omitted) and the backend started on it. Returns 0 on a pass."""
    os.environ["TCIP_WORKSPACE"] = workspace or tempfile.mkdtemp(prefix="terminal-smoke-ws-")

    from fastapi.testclient import TestClient

    from tcip_mcp.workspace import workspace_from_environment
    from tcip_web.app import app
    from tcip_web.routes import terminal as terminal_routes
    from tcip_web.state import store

    store.configure(workspace_from_environment(), ())
    client = TestClient(app, base_url="http://127.0.0.1")

    status = client.get("/api/terminal/status").json()
    print(f"[1] status: {status}")
    reasons = {row["id"]: row["unavailable_reason"] for row in status["providers"]}
    if provider not in reasons or reasons[provider] is not None:
        print(f"FAIL: provider {provider!r} cannot launch here: {reasons.get(provider, reasons)}")
        return 1

    created = client.post(
        "/api/terminal/sessions", json={"provider": provider, "rows": 35, "cols": 120}).json()
    sid = created["session_id"]
    print(f"[2] session spawned: {sid} {created['launched']}")
    session = terminal_routes._SESSIONS[sid]

    answered = False
    try:
        with client.websocket_connect(terminal_ws_url(client, sid)) as ws:
            ws.send_json({"type": "input", "data": "\x1b[?1;2c"})  # a terminal's DA reply
            client.post(f"/api/terminal/sessions/{sid}/submit", json={"text": PROMPT})
            print("[3] prompt submitted; waiting for the answer...")
            deadline = time.time() + ANSWER_TIMEOUT_S
            while not answered and time.time() < deadline:
                time.sleep(1.0)
                answered = ANSWER in ANSI.sub("", session.scrollback_snapshot())
            if not answered:
                print(f"    stream tail: {ascii(ANSI.sub('', session.scrollback_snapshot())[-800:])}")
    finally:
        terminal_routes.shutdown_all()
        print("[4] session terminated (shutdown_all)")

    print("SMOKE PASS" if answered else "SMOKE FAIL")
    return 0 if answered else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("provider", help="the id of the provider row to launch")
    raise SystemExit(main(parser.parse_args().provider))

"""Live smoke of one provider row's real harness through the embedded agent terminal.

Creates and opens a scratch project, launches the row for it, submits a request the way the rail
submits a staged one, asking for one ``report_friction`` call carrying a token generated once the
project is open, so only the request names it, and passes only when the open project's friction
reports, read through the platform's own route, hold exactly one report carrying that token, of
the requested category, at the first poll that sees one; a later second call is not watched for.

Usage (from the repo root, tcip-agent env):
    python tools/smoke_terminal_e2e.py <provider id>
"""

from __future__ import annotations

import argparse
import getpass
import os
import secrets
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-web" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-mcp" / "src"))

ANSWER_TIMEOUT_S = 240


def terminal_ws_url(client: Any, session_id: str) -> str:
    """The absolute ``ws://`` URL of ``session_id``'s socket on the host ``client`` serves."""
    parts = urlsplit(str(client.base_url))
    scheme = "wss" if parts.scheme == "https" else "ws"
    return urlunsplit((scheme, parts.netloc, f"/api/terminal/ws/{session_id}", "", ""))


REQUESTED_CATEGORY = "unexpected_behavior"
"""The ``report_friction`` category the request names, one of that tool's own categories."""


def request_for(token: str) -> str:
    """The request asking the harness for one ``report_friction`` call carrying ``token``."""
    return (f"This is a smoke test of the agent terminal. Call the tcip report_friction tool "
            f"exactly once, with category {REQUESTED_CATEGORY} and detail {token}, then stop.")


def reported(client: Any, token: str) -> bool:
    """Whether exactly one friction report of the open project carries ``token`` in its detail,
    and that report has the requested category."""
    reports = client.get("/api/meta/reports").json()["reports"]
    matching = [row for row in reports if "malformed" not in row and token in row["detail"]]
    return len(matching) == 1 and matching[0]["category"] == REQUESTED_CATEGORY


def main(provider: str, workspace: str | None = None,
         answer_timeout_s: float = ANSWER_TIMEOUT_S) -> int:
    """Drive the smoke for the ``provider`` row with ``TCIP_WORKSPACE`` set to ``workspace`` (a
    fresh temp directory when omitted) and the backend started on it, waiting
    ``answer_timeout_s`` for the report. Returns 0 on a pass."""
    os.environ["TCIP_WORKSPACE"] = workspace or tempfile.mkdtemp(prefix="terminal-smoke-ws-")

    from fastapi.testclient import TestClient

    from tcip_mcp.tools.project_tools import initialize_project
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

    project = tempfile.mkdtemp(prefix="terminal-smoke-project-", dir=store.workspace)
    record = initialize_project(project, "Terminal smoke", "the terminal smoke's scratch workspace")
    if "error" in record:
        print(f"FAIL: the scratch project was refused: {record['error']}")
        return 1
    user = getpass.getuser()
    opened = client.post("/api/projects/open", json={"id": record["id"], "user": user})
    if opened.status_code != 200:
        print(f"FAIL: the scratch project did not open ({opened.status_code}): {opened.text}")
        return 1
    print(f"[2] project opened: {opened.json()}")
    token = f"ack-{secrets.token_hex(6)}"

    resp = client.post("/api/terminal/sessions", json={
        "provider": provider, "rows": 35, "cols": 120, "user": user})
    if resp.status_code != 200:
        print(f"FAIL: the launch was refused ({resp.status_code}): {resp.text}")
        return 1
    created = resp.json()
    sid = created["session_id"]
    print(f"[3] session spawned: {sid} {created['launched']}")
    session = terminal_routes._SESSIONS[sid]

    answered = False
    started = time.time()
    try:
        with client.websocket_connect(terminal_ws_url(client, sid)) as ws:
            ws.send_json({"type": "input", "data": "\x1b[?1;2c"})  # a terminal's DA reply
            client.post(f"/api/terminal/sessions/{sid}/submit", json={"text": request_for(token)})
            print(f"[4] request submitted for {token}; waiting for its friction report...")
            while not answered and time.time() - started < answer_timeout_s:
                time.sleep(1.0)
                answered = reported(client, token)
            if not answered:
                launch = session._launch
                print(f"    bracketed paste on: {launch.ready}; ritual delivered: "
                      f"{launch.ritual_sent}; requests still queued: {len(launch.queued)}")
                print(f"    stream tail: {ascii(session.scrollback_snapshot()[-1500:])}")
    finally:
        terminal_routes.shutdown_all()
        print(f"[5] session terminated (shutdown_all) after {time.time() - started:.0f} s")

    print("SMOKE PASS" if answered else "SMOKE FAIL")
    return 0 if answered else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("provider", help="the id of the provider row to launch")
    parser.add_argument("--answer-timeout", type=float, default=ANSWER_TIMEOUT_S,
                        help=f"seconds to wait for the report (default {ANSWER_TIMEOUT_S})")
    args = parser.parse_args()
    raise SystemExit(main(args.provider, answer_timeout_s=args.answer_timeout))

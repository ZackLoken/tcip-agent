"""Live smoke of one provider row's real harness through the embedded agent terminal.

Creates and opens a scratch project under a display name generated for the run, launches the row
for it, submits a request the way the rail submits a staged one, asking for one
``report_friction`` call carrying a token generated once the project is open, so only the request
names it, followed by the project's display name, which with no other tool called only the
agent's orientation states, and passes only when the open project's friction reports, read
through the platform's own route, hold exactly one report carrying that token and that name, of
the requested category, at the first poll that sees one; a later second call is not watched for.
The backend is also served on a loopback port recorded under the workspace, so the session hook
the harness runs reports its conversation as it does under ``python -m tcip_web``. For a row whose
conversations are resumed, the agent is then ended by the backend's shutdown once its harness has
reported its conversation, the row is launched again, and the smoke passes only when that launch
resumed the conversation and its agent reports the token, continued, though no request of the
resumed launch names it.

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
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

sys.path.insert(
    0, str(Path(__file__).parent.parent / "packages" / "tcip-web" / "src")
)
sys.path.insert(
    0, str(Path(__file__).parent.parent / "packages" / "tcip-mcp" / "src")
)

ANSWER_TIMEOUT_S = 240


def terminal_ws_url(client: TestClient, path: str) -> str:
    """Return the absolute ``ws://`` URL of the socket ``path`` on ``client``'s host."""
    parts = urlsplit(str(client.base_url))
    scheme = "wss" if parts.scheme == "https" else "ws"
    return urlunsplit((scheme, parts.netloc, path, "", ""))


REQUESTED_CATEGORY = "unexpected_behavior"
"""The ``report_friction`` category the request names, one of that tool's own categories."""


def request_for(token: str) -> str:
    """Return the request asking for one ``report_friction`` call with ``token`` and a name.

    The name is the open project's display name, which the request does not state.
    """
    return (
        "This is a smoke test of the agent terminal. Without calling any other tool, call "
        f"the tcip report_friction tool exactly once, with category {REQUESTED_CATEGORY} and, "
        f"as detail, {token} followed by this session's project's display name, then stop."
    )


CONTINUED_SUFFIX = "-continued"
"""What the resumed agent adds to the token of its earlier report."""


def continuation_request() -> str:
    """Return the request asking a resumed harness to report its earlier token, continued.

    It names no token.
    """
    return (
        "This is a smoke test of a resumed conversation. Without calling any other tool, call "
        f"the tcip report_friction tool exactly once, with category {REQUESTED_CATEGORY} and, "
        "as detail, the detail of the report_friction call you made earlier in this "
        f"conversation followed by {CONTINUED_SUFFIX}, then stop."
    )


def reported(client: TestClient, token: str, *facts: str) -> bool:
    """Return whether exactly one friction report of the open project carries ``token``.

    The token is looked for in its detail, and that report must have the requested category
    and carry each of ``facts`` in its detail too.
    """
    entries = client.get("/api/meta/reports").json()["entries"]
    matching = [
        entry["record"]
        for entry in entries
        if "reason" not in entry["record"]
        and token in entry["record"]["detail"]
    ]
    return (
        len(matching) == 1
        and matching[0]["category"] == REQUESTED_CATEGORY
        and all(fact in matching[0]["detail"] for fact in facts)
    )


SERVE_TIMEOUT_S = 10
"""Provisional: seconds the served backend is given to start listening."""


@contextmanager
def served(app: FastAPI, workspace: Path) -> Iterator[None]:
    """Serve ``app`` on a free loopback port recorded for ``workspace``, until the block ends.

    The port is :func:`~tcip_mcp.web_client.free_port`'s, recorded as ``python -m tcip_web``
    records it, so any process under the workspace (a harness's hook, an MCP server's push)
    reaches this backend through ``backend_url``. Raises ``RuntimeError`` when the server does not
    start listening within :data:`SERVE_TIMEOUT_S`.
    """
    import uvicorn

    from tcip_mcp.web_client import (
        LOOPBACK_HOST,
        free_port,
        record_backend_port,
    )

    port = free_port(0)
    record_backend_port(workspace, port)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=LOOPBACK_HOST,
            port=port,
            log_level="error",
            lifespan="off",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + SERVE_TIMEOUT_S
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        if not server.started:
            raise RuntimeError(f"the backend did not listen on port {port}")
        yield
    finally:
        server.should_exit = True
        thread.join(timeout=SERVE_TIMEOUT_S)


def main(
    provider: str,
    workspace: str | None = None,
    answer_timeout_s: float = ANSWER_TIMEOUT_S,
) -> int:
    """Drive the smoke for the ``provider`` row and return 0 on a pass.

    ``TCIP_WORKSPACE`` is set to ``workspace`` (a fresh temp directory when omitted) and the
    backend started on it and :func:`served` for the harness's own processes, waiting
    ``answer_timeout_s`` for each report.
    """
    os.environ["TCIP_WORKSPACE"] = workspace or tempfile.mkdtemp(
        prefix="terminal-smoke-ws-"
    )

    from fastapi.testclient import TestClient

    from tcip_mcp.workspace import workspace_from_environment
    from tcip_web.app import app
    from tcip_web.state import store

    store.configure(workspace_from_environment(), ())
    client = TestClient(app, base_url="http://127.0.0.1")
    with served(app, store.workspace):
        return _smoke(client, provider, answer_timeout_s)


def _smoke(client: TestClient, provider: str, answer_timeout_s: float) -> int:
    """Run the smoke for the ``provider`` row through ``client``; return 0 on a pass."""
    from tcip_mcp.audit import acts_of
    from tcip_mcp.project_record import CaptureInvariances
    from tcip_mcp.tools.project_tools import initialize_project
    from tcip_web.app import app
    from tcip_web.routes import terminal as terminal_routes
    from tcip_web.routes.terminal import TerminalSession
    from tcip_web.state import store

    status = client.get("/api/terminal/status").json()
    print(f"[1] status: {status}")
    reasons = {
        row["id"]: row["unavailable_reason"] for row in status["providers"]
    }
    if provider not in reasons or reasons[provider] is not None:
        print(
            f"FAIL: provider {provider!r} cannot launch here: {reasons.get(provider, reasons)}"
        )
        return 1

    project = tempfile.mkdtemp(
        prefix="terminal-smoke-project-", dir=store.workspace
    )
    # A sample answer for the scratch project; nothing in the smoke trains under it.
    invariances = CaptureInvariances(
        horizontal_flip=True,
        vertical_flip=True,
        rotation=180,
        color=True,
        intensity=True,
        scale=True,
    )
    name = f"Terminal smoke {secrets.token_hex(4)}"
    record = initialize_project(
        project,
        name,
        "the terminal smoke's scratch workspace",
        invariances,
        actor=None,
    )
    if "error" in record:
        print(f"FAIL: the scratch project was refused: {record['error']}")
        return 1
    user = getpass.getuser()
    opened = client.post(
        "/api/projects/open", json={"id": record["id"], "user": user}
    )
    if opened.status_code != 200:
        print(
            f"FAIL: the scratch project did not open ({opened.status_code}): {opened.text}"
        )
        return 1
    print(f"[2] project opened: {opened.json()}")
    token = f"ack-{secrets.token_hex(6)}"

    def launched() -> TerminalSession | None:
        """Launch the row by ``user``; return its session, or ``None`` printing the refusal."""
        resp = client.post(
            "/api/terminal/sessions",
            json={"provider": provider, "rows": 35, "cols": 120, "user": user},
        )
        if resp.status_code != 200:
            print(
                f"FAIL: the launch was refused ({resp.status_code}): {resp.text}"
            )
            return None
        created = resp.json()
        print(
            f"    session spawned: {created['session_id']} {created['launched']}"
        )
        return terminal_routes._SESSIONS[created["session_id"]]

    def answered(
        session: TerminalSession, request: str, signal: str, *facts: str
    ) -> bool:
        """Submit ``request`` to ``session``; return whether ``signal`` with ``facts`` came in time.

        The signal and facts are what :func:`reported` looks for.
        """
        started = time.time()
        socket_path = app.url_path_for("terminal_ws", session_id=session.id)
        with client.websocket_connect(
            terminal_ws_url(client, socket_path)
        ) as ws:
            ws.send_json(
                {"type": "input", "data": "\x1b[?1;2c"}
            )  # a terminal's DA reply
            client.post(
                f"/api/terminal/sessions/{session.id}/submit",
                json={"text": request},
            )
            print(
                f"    request submitted; waiting for a report of {signal}..."
            )
            while time.time() - started < answer_timeout_s:
                time.sleep(1.0)
                if reported(client, signal, *facts):
                    return True
            print(
                f"    bracketed paste on: {session._ready}; own text delivered: "
                f"{session._ahead is None}; requests still queued: "
                f"{len(session._queued)}"
            )
            print(
                f"    stream tail: {ascii(session.scrollback_snapshot()[-1500:])}"
            )
            return False

    def end_lines() -> list:
        """Return the scratch project's ``agent_terminal_ended`` lines, oldest first."""
        return acts_of(project, ("agent_terminal_ended",))[0]

    print("[3] the first launch")
    first = launched()
    if first is None:
        return 1
    try:
        passed = answered(first, request_for(token), token, name)
        print(f"    reported naming {name!r} from its orientation: {passed}")
        if passed and terminal_routes._provider(provider).resume is not None:
            started = time.time()
            while (
                first.conversation is None
                and time.time() - started < answer_timeout_s
            ):
                time.sleep(1.0)
            print(
                f"[4] the harness reported conversation {first.conversation}"
            )
            terminal_routes.shutdown_all()
            (ended,) = end_lines()
            print(f"    the shutdown's end line: {ended['arguments']}")
            print("[5] the launch after the shutdown")
            second = launched()
            passed = (
                second is not None
                and first.conversation is not None
                and second.launched.resumed == first.conversation
                and answered(
                    second, continuation_request(), token + CONTINUED_SUFFIX
                )
            )
    finally:
        terminal_routes.shutdown_all()
    print(
        f"[6] after shutdown_all: sessions still held {len(terminal_routes._SESSIONS)}, "
        f"end lines recorded {len(end_lines())}"
    )

    print("SMOKE PASS" if passed else "SMOKE FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "provider", help="the id of the provider row to launch"
    )
    parser.add_argument(
        "--answer-timeout",
        type=float,
        default=ANSWER_TIMEOUT_S,
        help=f"seconds to wait for the report (default {ANSWER_TIMEOUT_S})",
    )
    args = parser.parse_args()
    raise SystemExit(main(args.provider, answer_timeout_s=args.answer_timeout))

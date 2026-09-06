"""One-shot smoke: the embedded agent terminal against the real `claude` CLI.

Exercises the exact flow the GUI uses: POST /api/terminal/sessions (spawns claude.exe
in a ConPTY, cwd = repo root), attach the WebSocket, answer the terminal's
Device-Attributes query like xterm.js would, type a prompt, and assert a real model
response streams back. This is the scenario that silently failed in the old chat
implementation; run it after any change to the terminal stack.

Usage (from the repo root, tcip-agent env):
    python tools/smoke_terminal_e2e.py

Costs one trivial model turn on the machine's Claude Code account.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-web" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-mcp" / "src"))

MARKER = "SMOKE_OK"
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*(?:\x07|\x1b\\)|\x1b[>=()][0-9A-Za-z]?")


def main(workspace: str | None = None) -> int:
    """Drive the smoke flow against ``workspace`` (a fresh temp directory when omitted),
    never the machine's own workspace: the first request this process issues binds the served
    app's platform-state root from whatever project that workspace's active-project marker
    names, and every audit line the session writes lands there.
    """
    os.environ["TCIP_WORKSPACE"] = workspace or tempfile.mkdtemp(prefix="terminal-smoke-ws-")

    from fastapi.testclient import TestClient

    from tcip_web.app import app
    from tcip_web.routes import terminal as terminal_routes

    client = TestClient(app, base_url="http://127.0.0.1")

    status = client.get("/api/terminal/status").json()
    print(f"[1] status: {status}")
    if not status.get("available"):
        print("FAIL: claude CLI not available on this machine")
        return 1

    sid = client.post("/api/terminal/sessions", json={"rows": 35, "cols": 120}).json()[
        "session_id"
    ]
    print(f"[2] session spawned: {sid}")
    session = terminal_routes._SESSIONS[sid]

    try:
        with client.websocket_connect(f"/api/terminal/ws/{sid}") as ws:
            # Answer the DA query the way a real terminal (xterm.js) does, so ConPTY
            # flushes the first paint immediately.
            ws.send_json({"type": "input", "data": "\x1b[?1;2c"})

            # Wait for the interactive TUI to paint (bounded).
            deadline = time.time() + 30
            while time.time() < deadline:
                clean = ANSI.sub("", session.scrollback_snapshot())
                if "?" in clean and len(clean) > 100:  # the TUI footer/hints rendered
                    break
                time.sleep(0.5)
            print(f"[3] TUI painted ({len(session.scrollback_snapshot())} raw chars)")

            ws.send_json(
                {"type": "input", "data": f"Reply with exactly {MARKER} and use no tools."}
            )
            time.sleep(0.5)
            ws.send_json({"type": "input", "data": "\r"})
            print("[4] prompt submitted; waiting for the model...")

            deadline = time.time() + 90
            seen = ""
            while time.time() < deadline:
                seen = ANSI.sub("", session.scrollback_snapshot())
                # Require the marker OUTSIDE our own echoed prompt line.
                if seen.count(MARKER) >= 2 or re.search(rf"[●>]\s*{MARKER}", seen):
                    break
                time.sleep(1.0)

            ok = seen.count(MARKER) >= 2 or bool(re.search(rf"[●>]\s*{MARKER}", seen))
            print(f"[5] model responded: {ok}")
            if not ok:
                tail = seen[-600:]
                print(f"    stream tail: {tail!r}")
    finally:
        terminal_routes.shutdown_all()
        print("[6] session terminated (shutdown_all)")

    print("SMOKE PASS" if ok else "SMOKE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

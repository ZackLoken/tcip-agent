"""SessionEnd capture hook: appends a session-boundary record to the session's project,
``<project>/.tcip/learning_capture.jsonl``.

The agent terminal launches this hook with ``--project <path>``, the project its MCP server was
started for; a session with no project records nothing. Each record holds the time and the
session's id.

Best-effort: any error is swallowed and the hook exits 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone


def main(argv: list[str] | None = None) -> None:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:
        payload = {}
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--project", default=None)
        project = parser.parse_args(argv).project
        if project is None:
            return
        from tcip_store import append
        from tcip_store.binding import bind_default

        from tcip_mcp.web_client import learning_capture_key

        bind_default()
        append(learning_capture_key(project), {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session_id": payload.get("session_id"),
        })
    except Exception:
        pass  # never let the capture backstop fail the session


if __name__ == "__main__":
    main()
    sys.exit(0)

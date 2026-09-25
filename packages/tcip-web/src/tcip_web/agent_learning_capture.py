"""SessionEnd capture hook: appends a session-boundary record to
``<cwd>/.tcip/learning_capture.jsonl``.

The terminal pins its sessions' cwd to the repo root, so records from every project's sessions pool
in one platform-level file. Each entry stamps the workspace's active project (when one is adopted).

Best-effort: any error is swallowed and the hook exits 0.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone



def main() -> None:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:
        payload = {}
    try:
        from tcip_store import append
        from tcip_store.binding import bind_default

        from tcip_mcp.web_client import learning_capture_key

        bind_default()
        try:
            from tcip_mcp.workspace import read_active_project

            active_project = read_active_project()
        except Exception:
            active_project = None
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session_id": payload.get("session_id"),
            "reason": payload.get("reason"),
            "active_project": active_project,
            "note": "session ended; run tcip distill-learnings to review the workspace "
                    "projects' reports and retrospectives",
        }
        append(learning_capture_key(payload.get("cwd") or "."), entry)
    except Exception:
        pass  # never let the capture backstop fail the session


if __name__ == "__main__":
    main()
    sys.exit(0)

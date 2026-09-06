"""SessionEnd capture hook: the soft backstop for the self-learning loop.

Appends a session-boundary record to ``<cwd>/.tcip/learning_capture.jsonl``, and the terminal
pins its sessions' cwd to the repo root, so records from every project's sessions pool in one
platform-level file: this capture feeds platform improvement, not any one project's record.
Each entry stamps the workspace's active project (when one is adopted) so a distill pass can
group entries by project without sharding the file. The genuine learnings still come from the
agent following the ``self-improvement`` skill and the per-project ``.tcip/reports/`` and
``.tcip/retrospectives/``, which ``tcip distill-learnings`` gathers from each project;
this only guarantees a record of the session exists.

Non-blocking + best-effort: any error is swallowed and the hook exits 0. A capture backstop must
never break the agent's session (a SessionEnd hook that errors would surface as a failure).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from tcip_store import LOG_JSON, Key, StoreDescriptor, append, register_store
from tcip_store.file_backend import RootedFileLocator

_CAPTURE_LOG = RootedFileLocator(prefix=(".tcip",), suffix=".jsonl")
"""The capture log under a root's own ``.tcip/``."""

LEARNING_CAPTURE_STORE = "learning_capture"
_CAPTURE_PARTS = ("learning_capture",)
register_store(
    StoreDescriptor(
        name=LEARNING_CAPTURE_STORE,
        kind="log",
        key_fields=("document",),
        frozen=True,
        codec=LOG_JSON,
        locator=_CAPTURE_LOG,
    )
)


def learning_capture_key(root: str | Path) -> Key:
    """The session-boundary log under ``root``.

    Every session's hook appends here, from its own process, so the entries are serialized and
    each one is on disk before the hook exits rather than buffered in a bare handle.
    """
    return Key(LEARNING_CAPTURE_STORE, str(Path(root).resolve()), _CAPTURE_PARTS)


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:
        payload = {}
    try:
        from tcip_store.binding import bind_default

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

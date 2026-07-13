"""SessionEnd capture hook — the soft backstop for the self-learning loop (governance Part 2).

Appends a session-boundary record to ``<project>/.tcip/learning_capture.jsonl`` so a session's
existence is captured even when the agent forgot to journal. It writes to machine-local ``.tcip/``
(not the committed journal) so there is no git churn; the distill worksheet
(``scripts/distill_learnings.py``) surfaces these for a human-in-the-loop review, which promotes
the real signal into the committed journal / skills. The genuine learnings still come from the
agent following the ``self-improvement`` skill — this only guarantees a record exists.

NON-blocking + best-effort: any error is swallowed and the hook exits 0. A capture backstop must
never break the agent's session (a SessionEnd hook that errors would surface as a failure).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:
        payload = {}
    try:
        cwd = payload.get("cwd") or "."
        d = Path(cwd) / ".tcip"
        d.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session_id": payload.get("session_id"),
            "reason": payload.get("reason"),
            "note": "session ended; run scripts/distill_learnings.py to review for the journal",
        }
        with (d / "learning_capture.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # never let the capture backstop fail the session


if __name__ == "__main__":
    main()
    sys.exit(0)

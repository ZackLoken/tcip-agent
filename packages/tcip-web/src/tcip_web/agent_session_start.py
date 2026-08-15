"""SessionStart ritual hook: inject the session-start ritual directive with live open-loop counts.

Fast and stdlib-only by design (Anthropic guidance: SessionStart hooks must be quick, they are for
context loading, never slow work). It spawns no subprocess and imports nothing heavy: it reads the
active-project marker, counts open reports/retrospectives, and injects an ``additionalContext``
directive telling the agent to run the ritual (``load_project_memory``/
``inspect_project``/``doctor.py``) as its first actions. A shell hook has no MCP client, so it
cannot run those calls itself, it makes them salient and dynamic, which prose in a large always-on
file does not. ``additionalContext`` lands as a fresh session-start reminder at the top of context.

Best-effort: every path swallows its error and exits 0. A session-start hook must never break the
session, and (the reason the earlier subprocess version was reverted) must never slow its spawn.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _active_project() -> str:
    """Active project's root via the workspace ``.active`` marker (stdlib only); "" when none.

    Deliberately does not import tcip_mcp or spawn anything: the hook must stay fast. The inherited
    ``TCIP_PROJECT_ROOT`` is not a reliable signal (tcip-web pins it to the repo root and the spawned
    terminal keeps that stale value), so the live marker is the trustworthy source.
    """
    try:
        raw = os.environ.get("TCIP_WORKSPACE", "").strip()
        ws = Path(raw).expanduser() if raw else Path.home() / "tcip-projects"
        name = (ws / ".active").read_text(encoding="utf-8").strip()
        return str(ws / name) if name else ""
    except Exception:
        return ""


def _count(proj: str, subdir: str, pattern: str) -> int:
    try:
        d = Path(proj) / ".tcip" / subdir
        return len(list(d.glob(pattern))) if d.is_dir() else 0
    except Exception:
        return 0


def _active_context(proj: str) -> str:
    nrep = _count(proj, "reports", "*.json")
    nret = _count(proj, "retrospectives", "*.md")
    name = Path(proj).name or proj
    return (
        "[TCIP session-start ritual, auto-injected by the SessionStart hook]\n"
        f"Active project: {name} ({proj}), {nrep} friction report(s), {nret} retrospective(s) open.\n\n"
        "If this session continues work on that project, run the ritual first: load_project_memory "
        "(kind='reports' and kind='retrospectives'), inspect_project, then python scripts/doctor.py <project_root>.\n"
        "If the user's task is to create or switch to a different project, do that first (init_project "
        "then set_active_project), then run the ritual on the project you end up in, do not run it on "
        "a stale active project.\n"
        "If any mandated action is blocked or errors, that itself is a claude_reports, never a silent skip."
    )


def _no_project_context() -> str:
    return (
        "[TCIP session-start ritual, auto-injected by the SessionStart hook]\n"
        "No active project yet (.active marker absent). Resolve by the user's intent:\n"
        "  • New project  → init_project(<path>) then set_active_project(<name>) to make it active "
        "(init_project only scaffolds; set_active_project sets the marker the GUI + ritual read).\n"
        "  • Resume existing work → set_active_project(<name>) (or open it in the GUI).\n"
        "Once a project is active, run the ritual: load_project_memory (kind='reports' and "
        "kind='retrospectives') + inspect_project, then python scripts/doctor.py <project_root>.\n"
        "If any mandated action is blocked or errors, that itself is a claude_reports, never a silent skip."
    )


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:
        payload = {}
    try:
        if payload.get("source") == "compact":
            return  # mid-session compaction; re-running the ritual is noise
        proj = _active_project()
        ctx = _active_context(proj) if proj else _no_project_context()
        print(json.dumps({
            "hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": ctx}
        }))
    except Exception:
        pass  # degrade to a bare exit(0); a session-start hook must never break the session


if __name__ == "__main__":
    main()
    sys.exit(0)

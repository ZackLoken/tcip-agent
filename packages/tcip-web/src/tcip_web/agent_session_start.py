"""SessionStart ritual hook: inject the session-start ritual directive naming the session's project.

The agent terminal launches this hook with ``--project <path>``, the project its MCP server was
started for, and it injects an ``additionalContext`` directive telling the agent to run the ritual
(``load_project_memory``/``inspect_project``/``tcip doctor``) as its first actions. Without
``--project`` the directive says the session has no project. It reads the project's record
through the platform's own storage seam with a short lock timeout, spawns no subprocess and counts
no reports or retrospectives.

Best-effort: every path swallows its error and exits 0.
"""

from __future__ import annotations

import argparse
import json
import sys

_LOCK_TIMEOUT_S = 2.0
"""Bounds how long a locked store can hold this hook, well under the store's own 30s default."""

_HEADER = "[TCIP session-start ritual, auto-injected by the SessionStart hook]\n"
_FRICTION = ("If any mandated action is blocked or errors, that itself is a report_friction, never "
             "a silent skip.")


def _project_context(project: str) -> str:
    """The directive for a session started for ``project``: its display name when its record
    reads, else the reason it does not."""
    try:
        from tcip_store.binding import bind_default

        from tcip_mcp.project_record import record_fields

        bind_default(lock_timeout_s=_LOCK_TIMEOUT_S)
        record = record_fields(project)
    except Exception as exc:  # noqa: BLE001, a store refusal or import failure is reported, not raised
        record = {"display_name": None, "record_problem": str(exc)}
    if record["display_name"] is None:
        return (f"{_HEADER}This session's project ({project}) has no readable record: "
                f"{record['record_problem']}\nFile this with report_friction before any project "
                f"work.\n{_FRICTION}")
    return (
        f"{_HEADER}Project: {record['display_name']} ({project}).\n"
        "Run the ritual first: load_project_memory (kind='reports' and kind='retrospectives'), "
        f"inspect_project, then tcip doctor {project}.\n{_FRICTION}"
    )


def _no_project_context() -> str:
    return (
        f"{_HEADER}This session has no project: the GUI had none open when the terminal started, "
        "so every tool that acts on a project refuses. Create one with initialize_project, or open "
        "one in the GUI, then restart the terminal to work on it.\n"
        f"{_FRICTION}"
    )


def main(argv: list[str] | None = None) -> None:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:
        payload = {}
    try:
        if payload.get("source") == "compact":
            return  # mid-session compaction; re-running the ritual is noise
        parser = argparse.ArgumentParser()
        parser.add_argument("--project", default=None)
        project = parser.parse_args(argv).project
        ctx = _no_project_context() if project is None else _project_context(project)
        print(json.dumps({
            "hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": ctx}
        }))
    except Exception:
        pass  # degrade to a bare exit(0); a session-start hook must never break the session


if __name__ == "__main__":
    main()
    sys.exit(0)

"""SessionStart ritual hook: inject the session-start ritual directive naming the active project.

Fast by design (Anthropic guidance: SessionStart hooks must be quick, they are for context
loading, never slow work). It spawns no subprocess and imports nothing costly: it reads the
active-project marker through the platform's own storage seam (``tcip_mcp.workspace``,
``tcip_store.binding``), not a loose file that the default backend may not even write, and
injects an ``additionalContext`` directive telling the agent to run the ritual
(``load_project_memory``/``inspect_project``/``doctor.py``) as its first actions. A shell hook
has no MCP client, so it cannot run those calls itself, it makes them salient and dynamic, which
prose in a large always-on file does not. ``additionalContext`` lands as a fresh session-start
reminder at the top of context.

It once also counted open reports/retrospectives by globbing the project's own directories, which
undercounts to zero once a project's state moves to a database backend. Re-pointing that count
through the storage seam was tried and rejected: the seam's own enumeration of those two record
kinds lives in ``tcip_mcp.tools.meta_tools``, and importing it pulls in the MCP server's full tool
registration (measured at several seconds, not the milliseconds a SessionStart hook gets), the
same regression this module's own import test stands guard against. Calling ``tcip_store.keys``
directly needs a store descriptor only that same module registers, so doing it here without that
import would mean re-declaring the report/retrospective file layout a second time, the drift this
platform's own seam discipline forbids. So the count is dropped rather than served wrong or
duplicated: this hook now only names the active project.

Measured on this machine: importing ``tcip_mcp.workspace`` (plus the ``tcip_store`` imports it
pulls in) costs ~57ms; a fresh process that imports it, binds the backend and reads the marker
costs 152-168ms wall clock over five runs, against 38-42ms for this hook with no seam read and
22-28ms for a bare interpreter. The lock timeout below is held well under the store's own
30-second default for the same reason: a store a writer is holding must not hold session start
for anywhere near that long.

Best-effort: every path swallows its error and exits 0. A session-start hook must never break the
session, and (the reason the earlier subprocess version was reverted) must never slow its spawn.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_LOCK_TIMEOUT_S = 2.0
"""Bounds how long a locked store can hold this hook, well under the store's own 30s default."""


def _resolve_active() -> tuple[str, str]:
    """What the workspace's active-project marker says, as ``(outcome, detail)``.

    ``outcome`` is one of:
      - ``"active"``: the marker names a project whose ``.tcip`` exists; ``detail`` is its root.
      - ``"none"``: no marker is set; ``detail`` is empty.
      - ``"unreadable"``: the marker could not be read (a store refusal, e.g. a workspace still
        holding loose files under the database backend) or names a project that is not
        adoptable; ``detail`` is ``workspace.marker_problem``'s own text, the one place that
        tells those cases apart, rather than this hook's own re-encoding of the same fold.
      - ``"import_error"``: the platform packages could not be imported from this interpreter;
        ``detail`` is the error.

    Imports ``tcip_mcp.workspace`` and ``tcip_store.binding`` inside this function, never at
    module scope, so an interpreter that cannot see those packages still reports that fact
    rather than a false no-project state. Binds the environment's own backend (the same rule
    every process follows) with a short lock timeout, and reads the marker with
    ``create=False`` so the read cannot create the workspace root itself. It can still touch
    disk under an existing ``<workspace>/.tcip/``: opening a database not yet in WAL mode
    creates that directory (if absent) and a lock file there, and the short timeout bounds
    only that transition lock, not SQLite's own busy wait on a database another writer holds.
    """
    try:
        from tcip_mcp import workspace
        from tcip_store.binding import bind_default
    except ImportError as exc:
        return "import_error", str(exc)
    try:
        bind_default(lock_timeout_s=_LOCK_TIMEOUT_S)
        found = workspace.active_project_if_present(create=False)
    except Exception as exc:  # noqa: BLE001, a store refusal or lock timeout is reported, not raised
        return "unreadable", str(exc)
    if found is not None:
        _, path = found
        return "active", str(path)
    try:
        problem = workspace.marker_problem(create=False)
    except Exception as exc:  # noqa: BLE001, same as above
        return "unreadable", str(exc)
    if problem is None:
        return "none", ""
    return "unreadable", problem


def _root_divergence_note(proj: str) -> str:
    """States it when this session's inherited platform-state root names a different
    project than the active marker: that env var is a copy taken when this terminal was
    spawned, not a live read (this module's own docstring already calls it unreliable on its
    own), so the two can disagree until a process that binds from the marker converges.

    The web backend already binds from the marker at its own startup; an MCP server this
    terminal launches binds from it too, but only the next time that process starts, so a
    server already running in this terminal keeps the root it inherited until then, and
    ``inspect_project`` reports that server's actual disagreement in the meantime.

    Reads the variable's name from ``tcip_mcp.project_paths.ENV_VAR`` rather than a literal,
    since that module is what declares it.
    """
    from tcip_mcp.project_paths import ENV_VAR

    inherited = os.environ.get(ENV_VAR)
    if not inherited or str(Path(inherited)) == str(Path(proj)):
        return ""
    return (
        f"This session's inherited {ENV_VAR} ({inherited}) names a different project "
        f"than the active marker ({proj}). The web backend already binds from the marker at "
        "its own startup; an MCP server launched in this terminal binds from it only at its "
        "next start, so restart it, or adopt the marker's project explicitly "
        "(activate_project) now, before running the ritual.\n\n"
    )


def _active_context(proj: str) -> str:
    name = Path(proj).name or proj
    return (
        "[TCIP session-start ritual, auto-injected by the SessionStart hook]\n"
        f"Active project: {name} ({proj}).\n\n"
        f"{_root_divergence_note(proj)}"
        "If this session continues work on that project, run the ritual first: load_project_memory "
        "(kind='reports' and kind='retrospectives'), inspect_project, then tcip doctor <project_root>.\n"
        "If the user's task is to create or switch to a different project, do that first "
        "(initialize_project(<path>, site=<site>) then activate_project), then run the ritual on the "
        "project you end up in, do not run it on a stale active project.\n"
        "If any mandated action is blocked or errors, that itself is a report_friction, never a silent skip."
    )


def _no_project_context() -> str:
    return (
        "[TCIP session-start ritual, auto-injected by the SessionStart hook]\n"
        "No active project yet (.active marker absent). Resolve by the user's intent:\n"
        "  • New project  → initialize_project(<path>, site=<site>) then activate_project(<name>) to "
        "make it active (activate_project sets the marker the GUI + ritual read).\n"
        "  • Resume existing work → activate_project(<name>) (or open it in the GUI).\n"
        "Once a project is active, run the ritual: load_project_memory (kind='reports' and "
        "kind='retrospectives') + inspect_project, then tcip doctor <project_root>.\n"
        "If any mandated action is blocked or errors, that itself is a report_friction, never a silent skip."
    )


def _unreadable_context(detail: str) -> str:
    return (
        "[TCIP session-start ritual, auto-injected by the SessionStart hook]\n"
        f"The active-project marker could not be adopted: {detail}\n"
        "This is a mandated action that failed, so file it with report_friction once an MCP client "
        "is available, rather than treating it as no active project. If the detail names a "
        "workspace holding loose files with no database, conform it with "
        "tcip adopt-store before trusting the marker again."
    )


def _import_error_context(detail: str) -> str:
    return (
        "[TCIP session-start ritual, auto-injected by the SessionStart hook]\n"
        f"The active-project marker could not be read from this interpreter: {detail}\n"
        "This session cannot see whether a project is active; do not assume there is none. "
        "File this with report_friction once an MCP client is available."
    )


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:
        payload = {}
    try:
        if payload.get("source") == "compact":
            return  # mid-session compaction; re-running the ritual is noise
        outcome, detail = _resolve_active()
        if outcome == "active":
            ctx = _active_context(detail)
        elif outcome == "unreadable":
            ctx = _unreadable_context(detail)
        elif outcome == "import_error":
            ctx = _import_error_context(detail)
        else:
            ctx = _no_project_context()
        print(json.dumps({
            "hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": ctx}
        }))
    except Exception:
        pass  # degrade to a bare exit(0); a session-start hook must never break the session


if __name__ == "__main__":
    main()
    sys.exit(0)

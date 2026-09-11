"""Rename every workspace project carrying a pending-rename marker onto its own new name: the
operator/agent entry point for ``tcip_mcp.project_rename.complete_pending_renames``, the same
walk a served backend runs once at its own startup, run by hand or on a schedule with no backend
running.

    tcip complete-renames [--workspace PATH]

``--workspace`` defaults to ``$TCIP_WORKSPACE`` (``tcip_mcp.workspace.workspace_root``'s own
default, ``~/tcip-projects``, when neither is set). Prints one line per outcome and exits 1
when any rename was blocked, 0 otherwise (a skipped project, its own marker unreadable, is
printed but does not fail the exit code: the walk still renamed every other pending project).
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, prog=prog)
    parser.add_argument("--workspace", default="",
                         help="Workspace root to walk; defaults to $TCIP_WORKSPACE, or "
                              "~/tcip-projects if that is unset too.")
    args = parser.parse_args(argv)

    if args.workspace:
        os.environ["TCIP_WORKSPACE"] = args.workspace

    # Its own process entry point, so it binds the storage backend the seam has no default for.
    from tcip_store.binding import bind_default

    from tcip_mcp import project_rename, workspace

    bind_default()

    outcomes = project_rename.complete_pending_renames(workspace.workspace_root(create=False))
    if not outcomes:
        print("no pending renames")
        return 0

    blocked = 0
    for outcome in outcomes:
        name = outcome["name"]
        if "skipped" in outcome:
            print(f"{name}: skipped ({outcome['skipped']})")
            continue
        new_name = outcome.get("new_name")
        if "blocked_by" in outcome:
            blocked += 1
            print(f"{name}: blocked ({outcome['blocked_by']})")
        elif outcome.get("already_renamed"):
            print(f"{name}: already renamed to {new_name}")
        else:
            print(f"{name}: renamed to {new_name}")
    return 1 if blocked else 0


if __name__ == "__main__":
    sys.exit(main())

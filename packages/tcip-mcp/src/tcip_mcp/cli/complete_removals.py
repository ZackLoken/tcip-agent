"""Move every workspace project carrying a pending-removal marker onto its own holding
directory: the operator/agent entry point for
``tcip_mcp.project_removal.complete_pending_removals``, the same walk a served backend runs
once at its own startup, run by hand or on a schedule with no backend running.

    tcip complete-removals [--workspace PATH]

``--workspace`` defaults to ``$TCIP_WORKSPACE`` (``tcip_mcp.workspace.workspace_root``'s own
default, ``~/tcip-projects``, when neither is set). Prints one line per outcome and exits 1
when any removal was blocked, 0 otherwise (a skipped project, its own marker unreadable, is
printed but does not fail the exit code: the walk still moved every other pending project).
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

    from tcip_mcp import project_removal, workspace

    bind_default()

    outcomes = project_removal.complete_pending_removals(workspace.workspace_root(create=False))
    if not outcomes:
        print("no pending removals")
        return 0

    blocked = 0
    for outcome in outcomes:
        name = outcome["name"]
        if "moved_to" in outcome:
            print(f"{name}: moved to {outcome['moved_to']}")
        elif "blocked_by" in outcome:
            blocked += 1
            print(f"{name}: blocked ({outcome['blocked_by']})")
        else:
            print(f"{name}: skipped ({outcome['skipped']})")
    return 1 if blocked else 0


if __name__ == "__main__":
    sys.exit(main())

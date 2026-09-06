#!/usr/bin/env python
"""Print the live MCP tool registry (count + names).

Single source of truth for "how many domain tools are there": docs reference
this script instead of hard-coding a number that drifts. Run from the repo root:

    python tools/list_tools.py

Every tool module imports without torch (a module that needs it imports torch inside its own
functions), so the registered count does not vary by environment.
"""

from __future__ import annotations

from tcip_mcp.server import list_registered_tools


def main() -> None:
    names = list_registered_tools()
    print(f"{len(names)} MCP tools registered:\n")
    for name in names:
        print(f"  {name}")


if __name__ == "__main__":
    main()

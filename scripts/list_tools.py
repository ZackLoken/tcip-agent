#!/usr/bin/env python
"""Print the live MCP tool registry (count + names).

Single source of truth for "how many domain tools are there" — docs reference
this script instead of hard-coding a number that drifts. Run from the repo root:

    python scripts/list_tools.py

The count reflects what imported in the current environment: torch-dependent
tool modules only register when their dependencies are installed.
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

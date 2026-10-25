#!/usr/bin/env python
"""Print the live MCP tool registry (count + names), as a server for no project lists it.

Run from the repo root as ``python tools/list_tools.py``.
"""

from __future__ import annotations

from tcip_web.cli import tool_listing


def main() -> None:
    """Print the count and the names of the registered MCP tools."""
    names = sorted(t.name for t in tool_listing())
    print(f"{len(names)} MCP tools registered:\n")
    for name in names:
        print(f"  {name}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Print the live MCP tool registry (count + names).

Run from the repo root as ``python tools/list_tools.py``.
"""

from __future__ import annotations

from tcip_mcp.server import list_registered_tools


def main() -> None:
    """Print the count and the names of the registered MCP tools."""
    names = list_registered_tools()
    print(f"{len(names)} MCP tools registered:\n")
    for name in names:
        print(f"  {name}")


if __name__ == "__main__":
    main()

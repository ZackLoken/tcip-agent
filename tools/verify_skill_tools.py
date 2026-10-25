#!/usr/bin/env python
"""Guardrail: hold every tool name in agent-facing prose to the registry.

Two independent checks:

- `fabricated_tool_names` (fabrication detection) reads every Tools table (a table whose
  header's first column is literally "Tool") and extracts the name from the first backticked
  token of each row's first cell (the identifier up to the first `(`, space or backtick, so a
  documented call signature or a keyword note still yields the plain name). A name that is not
  in the server's tool listing is fabricated or retired. A first cell whose backticked token
  is not a plain identifier (`tools/`) is skipped rather than reported.
- `orphan_tool_names` (rename/omission detection) searches every surface's whole text for each
  registered tool name as `` `name` `` or `` `name( `` (tools are documented with their call
  form). A registered tool no surface names this way is an orphan.

Reach, two gaps: the fabrication check only reads Tools tables, so a fabricated or retired name
in running prose outside a table is invisible to it; and a table documenting tools under another
header is also invisible to it, since header text is how a table is recognized as one at all.
The orphan check reads every surface's whole text, table or prose alike.

CLI: `python tools/verify_skill_tools.py`, exit 0 clean, 1 fabricated or orphaned names found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.types import Tool

REPO_ROOT = Path(__file__).resolve().parent.parent

# Registered tools deliberately undocumented anywhere, added here only after conscious review,
# never to silence a rename nobody wrote down. Empty: every registered tool is named somewhere.
ORPHAN_ALLOW: frozenset[str] = frozenset()

_TABLE_ROW = re.compile(r"^\|(.+)\|\s*$")
_BACKTICK_SPAN = re.compile(r"`([^`]*)`")
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def prose_surfaces() -> list[Path]:
    """Return every agent-facing surface a tool name is documented (or fabricated) in.

    A tool module's own docstrings and runtime return strings (`vision_tools.py`, `viz.py`) are
    not among these surfaces: a retired tool name surviving there is invisible to this checker.
    """
    from tcip_mcp.knowledge import document_paths

    surfaces = list(document_paths())
    surfaces += sorted((REPO_ROOT / "packages").glob("*/CLAUDE.md"))
    surfaces.append(REPO_ROOT / "CLAUDE.md")
    surfaces.append(REPO_ROOT / "README.md")
    return [p for p in surfaces if p.is_file()]


def tool_table_first_cells(md_text: str) -> list[str]:
    """Return the first cell of every data row in a table whose header's first column is "Tool"."""
    cells: list[str] = []
    in_tool_table = False
    for line in md_text.splitlines():
        m = _TABLE_ROW.match(line)
        if not m:
            in_tool_table = False
            continue
        first = m.group(1).split("|", 1)[0].strip()
        if first == "Tool":
            in_tool_table = True
            continue
        if not in_tool_table:
            continue
        if set(first) <= {"-", ":", " "}:
            continue  # the header/body separator row
        cells.append(first)
    return cells


def extract_tool_name(cell: str, commands: set[str]) -> str | None:
    """Return the plain identifier a Tools-table cell's first backticked token names, or ``None``.

    ``None`` when that token is not a snake_case identifier (a call signature or a keyword note
    still yields the name; a path fragment like `tools/` yields nothing to check).

    A `` `tcip <command>` `` console invocation is never a claimed MCP tool name, so it is
    validated against ``commands``, every command the console dispatcher runs
    (``tcip_web.cli.command_names``), instead of the tool registry: a real command yields
    ``None`` (nothing left to check), and one the dispatcher does not run is returned in full
    (``"tcip <command>"``).
    """
    m = _BACKTICK_SPAN.search(cell)
    if not m:
        return None
    token = m.group(1)
    if token.startswith("tcip "):
        command = (
            token[len("tcip ") :].split(None, 1)[0]
            if token[len("tcip ") :].strip()
            else ""
        )
        return None if command in commands else f"tcip {command}"
    for sep in ("(", " "):
        idx = token.find(sep)
        if idx != -1:
            token = token[:idx]
    return token if _IDENTIFIER.match(token) else None


def fabricated_tool_names(
    listing: list[Tool], surfaces: list[Path] | None = None
) -> dict[str, list[str]]:
    """Return every Tools-table cell naming a tool ``listing`` does not hold, by surface.

    ``listing`` is a server's tool listing; a console command is checked against the commands
    the dispatcher runs over it.
    """
    from tcip_web.cli import command_names

    registered = {t.name for t in listing}
    commands = set(command_names(listing))
    result: dict[str, list[str]] = {}
    for surface in surfaces if surfaces is not None else prose_surfaces():
        cells = tool_table_first_cells(surface.read_text(encoding="utf-8"))
        names = [extract_tool_name(c, commands) for c in cells]
        bad = sorted({n for n in names if n and n not in registered})
        if bad:
            try:
                key = str(surface.relative_to(REPO_ROOT))
            except ValueError:
                key = str(surface)
            result[key] = bad
    return result


def orphan_tool_names(
    listing: list[Tool], surfaces: list[Path] | None = None
) -> list[str]:
    """Return every tool of ``listing`` no surface names as `` `name` `` or `` `name( ``."""
    registered = {t.name for t in listing}
    text = "\n".join(
        s.read_text(encoding="utf-8")
        for s in (surfaces if surfaces is not None else prose_surfaces())
    )
    mentioned = {n for n in registered if f"`{n}`" in text or f"`{n}(" in text}
    return sorted(registered - mentioned - ORPHAN_ALLOW)


def main() -> int:
    """Print fabricated and orphaned tool names; return 0 when there are none, else 1.

    Both checks read the one listing this invocation takes.
    """
    from tcip_web.cli import tool_listing

    listing = tool_listing()
    fabricated = fabricated_tool_names(listing)
    orphans = orphan_tool_names(listing)
    ok = True
    if fabricated:
        ok = False
        print("FABRICATED (Tools-table name not in the registry):")
        for surface, names in fabricated.items():
            for name in names:
                print(f"  {surface}: {name}")
    else:
        print("OK: no fabricated tool names in any Tools table")
    if orphans:
        ok = False
        print(f"ORPHANED (registered tool named nowhere): {orphans}")
    else:
        print("OK: no orphaned registered tools")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

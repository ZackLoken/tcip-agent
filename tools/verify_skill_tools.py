#!/usr/bin/env python
"""Guardrail: hold every tool name in agent-facing prose to the registry.

Two independent checks, mirroring `tools/verify_skill_traits.py`'s split between a
fabrication check and a membership check, for tool names instead of trait names:

- `fabricated_tool_names` (fabrication detection) reads every Tools table (a table whose
  header's first column is literally "Tool") and extracts the name from the first backticked
  token of each row's first cell (the identifier up to the first `(`, space or backtick, so a
  documented call signature or a keyword note still yields the plain name). A name that is not
  in `list_registered_tools()` is fabricated or retired. A first cell whose backticked token
  is not a plain identifier (`tools/`) is skipped rather than reported, since it never claimed
  to be a tool name.
- `orphan_tool_names` (rename/omission detection) searches every surface's whole text for each
  registered tool name as `` `name` `` or `` `name( `` (tools are documented with their call
  form). A registered tool no surface names this way is an orphan: usually a rename whose old
  docs were fixed but whose new name was never written down anywhere.

Reach, stated plainly, two gaps: the fabrication check only reads Tools tables, so a fabricated
or retired name in running prose outside a table is invisible to it; and a table documenting
tools under another header is also invisible to it, since header text is how a table is
recognized as one at all. The phenology skill's piece inventory (headed "Piece", not "Tool") is
that second case: it names real tools (`build_plant_mapping`, `deliver_phenology_milestones`) alongside
internal module names in the same first column, and a fabricated or retired name there would go
unchecked. Matching by content instead of header (treating a table as a tool table once any data
row's first cell names a registered tool) would flag that table's module-name rows as
fabrications, rejecting valid prose rather than only catching a real problem, so it is not
used. Widening to the call/chain position (a token followed by `(`, or
named in a `->` chain) mostly catches legitimate non-tool identifiers (`train(ctx)`,
`grid_to_pixel`, `plant_id`, ...) rather than real fabrications, so it also stays out of scope;
the orphan check is what catches a rename missed everywhere, table or prose alike.

CLI: `python tools/verify_skill_tools.py`, exit 0 clean, 1 fabricated or orphaned names found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Registered tools deliberately undocumented anywhere, added here only after conscious review,
# never to silence a rename nobody wrote down. Empty: every registered tool is named somewhere.
ORPHAN_ALLOW: frozenset[str] = frozenset()

_TABLE_ROW = re.compile(r"^\|(.+)\|\s*$")
_BACKTICK_SPAN = re.compile(r"`([^`]*)`")
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def prose_surfaces() -> list[Path]:
    """Every agent-facing surface a tool name is documented (or fabricated) in.

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
    """The first cell of every data row in a table whose header's first column is "Tool"."""
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


def extract_tool_name(cell: str) -> str | None:
    """The plain identifier named by a Tools-table cell's first backticked token, or ``None``
    when that token is not a snake_case identifier (a call signature or a keyword note still
    yields the name; a path fragment like `tools/` yields nothing to check) or is a `tcip
    <command>` console invocation (never a claimed MCP tool name, so nothing to check either)."""
    m = _BACKTICK_SPAN.search(cell)
    if not m:
        return None
    token = m.group(1)
    if token.startswith("tcip "):
        return None
    for sep in ("(", " "):
        idx = token.find(sep)
        if idx != -1:
            token = token[:idx]
    return token if _IDENTIFIER.match(token) else None


def fabricated_tool_names(surfaces: list[Path] | None = None) -> dict[str, list[str]]:
    """Every Tools-table cell naming a tool the registry does not hold, by surface."""
    from tcip_mcp.server import list_registered_tools

    registered = set(list_registered_tools())
    result: dict[str, list[str]] = {}
    for surface in surfaces if surfaces is not None else prose_surfaces():
        names = [extract_tool_name(c) for c in tool_table_first_cells(surface.read_text(encoding="utf-8"))]
        bad = sorted({n for n in names if n and n not in registered})
        if bad:
            try:
                key = str(surface.relative_to(REPO_ROOT))
            except ValueError:
                key = str(surface)
            result[key] = bad
    return result


def orphan_tool_names(surfaces: list[Path] | None = None) -> list[str]:
    """Every registered tool no surface names as `` `name` `` or `` `name( ``."""
    from tcip_mcp.server import list_registered_tools

    registered = set(list_registered_tools())
    text = "\n".join(s.read_text(encoding="utf-8") for s in (surfaces if surfaces is not None else prose_surfaces()))
    mentioned = {n for n in registered if f"`{n}`" in text or f"`{n}(" in text}
    return sorted(registered - mentioned - ORPHAN_ALLOW)


def main() -> int:
    fabricated = fabricated_tool_names()
    orphans = orphan_tool_names()
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

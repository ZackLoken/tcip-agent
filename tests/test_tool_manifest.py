"""Guards against MCP tool-registry drift.

Two failure modes this catches:
  1. A tool that is decorated `@mcp.tool()` but never actually registers (import
     error, decorator mistake, duplicate name).
  2. Docs that hard-code a tool count which then goes stale (the original sin:
     README said 54, copilot-instructions said 57, reality was 57).
"""

from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "packages" / "tcip-mcp" / "src" / "tcip_mcp" / "tools"


def _decorated_tool_names() -> set[str]:
    """Function names decorated with `@mcp.tool(...)` across tools/*.py (via AST).

    FastMCP registers a tool under its function name by default, and every tool in
    this repo uses a bare `@mcp.tool()`, so the function name is the tool name.
    """
    names: set[str] = set()
    for py in TOOLS_DIR.glob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in node.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                if isinstance(target, ast.Attribute) and target.attr == "tool":
                    names.add(node.name)
    return names


def test_decorated_tools_exist():
    """Sanity: the AST actually finds the decorated tools."""
    assert len(_decorated_tool_names()) > 0


def test_registered_tool_names_unique():
    from tcip_mcp.server import list_registered_tools

    names = list_registered_tools()
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, f"duplicate MCP tool names registered: {dupes}"


def test_every_decorated_tool_registers():
    """Registry must match the decorated functions exactly.

    Skipped when torch is absent, since the torch-dependent tool modules are
    intentionally guarded and won't register in a torch-less environment.
    """
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed: torch-dependent tool modules won't register")

    from tcip_mcp.server import list_registered_tools

    registered = set(list_registered_tools())
    decorated = _decorated_tool_names()
    missing = decorated - registered
    extra = registered - decorated
    assert not missing and not extra, f"missing from registry={missing}, unexpected={extra}"


def test_docs_do_not_hardcode_tool_count():
    """Docs must point at scripts/list_tools.py, not cite a literal count.

    The regex allows an optional run of adjectives before "tool(s)" so the
    project's idiomatic phrasings are all caught — "57 MCP tools",
    "54 domain tools", "56 specialized tools", "57 total tools", bare "56 tools".
    It does not match prose like "all MCP tool calls" (no leading number).
    """
    pattern = re.compile(r"\b\d+\s+(?:[A-Za-z][A-Za-z-]*\s+){0,2}tools?\b", re.IGNORECASE)
    docs = [REPO_ROOT / "README.md", REPO_ROOT / "CLAUDE.md"]
    docs += sorted((REPO_ROOT / ".github").rglob("*.md"))
    offenders = {}
    for doc in docs:
        if not doc.exists():
            continue
        hits = pattern.findall(doc.read_text(encoding="utf-8"))
        if hits:
            offenders[str(doc.relative_to(REPO_ROOT))] = hits
    assert not offenders, (
        f"docs hard-code a tool count {offenders}; "
        "remove the number and reference `python scripts/list_tools.py` instead"
    )

"""SessionStart ritual hook: fast directive injection through the platform's own storage seam.

The hook stays quick and context-loading only: it injects an ``additionalContext`` directive
naming the project its ``--project`` names, and it spawns no subprocess and never imports the MCP
server's tool registration, which would cost tens of seconds at every session start.
"""

from __future__ import annotations

import ast
import io
import json
from pathlib import Path

from tcip_web import agent_session_start as hook

HOOK_SRC = Path(hook.__file__).read_text(encoding="utf-8")


def _imports(src: str) -> set[str]:
    """Every module a snippet imports, both ``import a.b`` and ``from a import b`` spellings
    resolved to the same dotted name, so a banned module cannot hide behind either form."""
    imported: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{a.name}" for a in node.names)
    return imported


def _imports_module(imported: set[str], banned: str) -> bool:
    return any(name == banned or name.startswith(f"{banned}.") for name in imported)


def _run(monkeypatch, capsys, stdin: str, argv: list[str]) -> str:
    monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    hook.main(argv)
    return capsys.readouterr().out


def _context(out: str) -> str:
    return json.loads(out)["hookSpecificOutput"]["additionalContext"]


def test_session_start_injects_ritual_directive_naming_the_project(project, monkeypatch, capsys):
    out = _run(monkeypatch, capsys, '{"source":"startup"}', ["--project", str(project)])
    ctx = _context(out)
    assert json.loads(out)["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "Test project" in ctx and str(project) in ctx
    for step in ("load_project_memory", "inspect_project", "tcip doctor"):
        assert step in ctx


def test_session_start_with_no_project_says_the_session_has_none(monkeypatch, capsys):
    ctx = _context(_run(monkeypatch, capsys, '{"source":"startup"}', []))
    assert "has no project" in ctx
    assert "initialize_project" in ctx


def test_session_start_names_why_a_project_record_does_not_read(tmp_path, monkeypatch, capsys):
    ctx = _context(_run(monkeypatch, capsys, '{"source":"startup"}', ["--project", str(tmp_path)]))
    assert "has no readable record" in ctx
    assert "report_friction" in ctx


def test_session_start_skips_on_compact(project, monkeypatch, capsys):
    out = _run(monkeypatch, capsys, '{"source":"compact"}', ["--project", str(project)])
    assert out.strip() == ""


def test_session_start_never_raises_on_garbage_stdin(monkeypatch, capsys):
    # Must degrade, never crash the session.
    _run(monkeypatch, capsys, "not json {{{", [])


def test_session_start_is_fast_no_subprocess_or_tool_registration(project, monkeypatch, capsys):
    """No subprocess, and never ``tcip_mcp.tools`` (the MCP server's full tool registration,
    several seconds by measurement). Parses the real imports via AST, so a docstring mention
    doesn't count."""
    imported = _imports(HOOK_SRC)
    assert not _imports_module(imported, "subprocess")
    assert not _imports_module(imported, "tcip_mcp.tools")
    out = _run(monkeypatch, capsys, '{"source":"startup"}', ["--project", str(project)])
    assert "SessionStart" in out


def test_import_scan_catches_the_from_import_spelling_of_a_banned_module():
    """``from tcip_mcp import tools`` is the same module access as ``import tcip_mcp.tools``;
    the guard above must catch both spellings, not only the dotted-import one."""
    assert _imports_module(_imports("from tcip_mcp import tools\n"), "tcip_mcp.tools")


def test_session_start_hook_runs_as_a_real_subprocess(project):
    """A fresh interpreter on the hook module itself, so the fresh-interpreter claim is measured
    against a real process rather than an in-process call sharing warm imports."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, hook.__file__, "--project", str(project)],
        input='{"source":"startup"}',
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    ctx = _context(result.stdout)
    assert "Test project" in ctx
    assert "has no readable record" not in ctx

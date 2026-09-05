"""SessionStart ritual hook: fast directive injection through the platform's own storage seam.

Locks the compliant design (Anthropic guidance: SessionStart must be quick, context-loading only):
it injects an ``additionalContext`` directive naming the active project, and it spawns no
subprocess and never imports the MCP server's tool registration (the standing check that guards
against the reverted 30s regression).
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


def _run(monkeypatch, capsys, stdin: str) -> str:
    monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    hook.main()
    return capsys.readouterr().out


def _workspace(tmp_path: Path, monkeypatch, *, reports=0, retros=0) -> str:
    """A workspace with a project active through the store, not a hand-written loose file."""
    ws = tmp_path / "ws"
    proj = ws / "currant_demo"
    (proj / ".tcip" / "reports").mkdir(parents=True)
    (proj / ".tcip" / "retrospectives").mkdir(parents=True)
    for i in range(reports):
        (proj / ".tcip" / "reports" / f"r{i}.json").write_text("{}\n")
    for i in range(retros):
        (proj / ".tcip" / "retrospectives" / f"s{i}.md").write_text("# retro\n")
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))

    from tcip_mcp import workspace

    workspace.activate_project("currant_demo")
    return str(proj)


def test_session_start_injects_ritual_directive_naming_the_active_project(
    tmp_path, monkeypatch, capsys
):
    _workspace(tmp_path, monkeypatch, reports=3, retros=2)
    out = _run(monkeypatch, capsys, '{"source":"startup"}')
    ctx = json.loads(out)["hookSpecificOutput"]["additionalContext"]
    assert json.loads(out)["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "currant_demo" in ctx
    for step in ("load_project_memory", "inspect_project", "doctor.py"):
        assert step in ctx


def test_session_start_no_active_project_covers_create_and_resume(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "empty_ws"))
    out = _run(monkeypatch, capsys, '{"source":"startup"}')
    ctx = json.loads(out)["hookSpecificOutput"]["additionalContext"]
    assert "No active project" in ctx
    # Both paths must be offered: creating a project is initialize_project, then activate_project.
    assert "initialize_project" in ctx and "activate_project" in ctx


def test_session_start_skips_on_compact(tmp_path, monkeypatch, capsys):
    _workspace(tmp_path, monkeypatch)
    assert _run(monkeypatch, capsys, '{"source":"compact"}').strip() == ""


def test_session_start_never_raises_on_garbage_stdin(monkeypatch, capsys):
    # Must degrade, never crash the session.
    _run(monkeypatch, capsys, "not json {{{")


def test_session_start_is_fast_no_subprocess_or_tool_registration(tmp_path, monkeypatch, capsys):
    """Standing check for the reverted 30s regression: no subprocess, and never
    ``tcip_mcp.tools`` (the MCP server's full tool registration, several seconds by
    measurement). ``tcip_mcp.workspace``/``tcip_store.binding`` are fine, milliseconds by
    measurement (see the module docstring), so only ``tcip_mcp.tools`` itself is banned.
    Parses the real imports via AST, so a docstring mention doesn't count.
    """
    imported = _imports(HOOK_SRC)
    assert not _imports_module(imported, "subprocess")
    assert not _imports_module(imported, "tcip_mcp.tools")
    # It emits its directive without touching either: a fresh interpreter would stay fast.
    _workspace(tmp_path, monkeypatch)
    out = _run(monkeypatch, capsys, '{"source":"startup"}')
    assert "SessionStart" in out


def test_import_scan_catches_the_from_import_spelling_of_a_banned_module():
    """``from tcip_mcp import tools`` is the same module access as ``import tcip_mcp.tools``;
    the guard above must catch both spellings, not only the dotted-import one."""
    assert _imports_module(_imports("from tcip_mcp import tools\n"), "tcip_mcp.tools")


def test_session_start_hook_runs_as_a_real_subprocess(tmp_path, monkeypatch):
    """The other tests call ``hook.main()`` in-process, sharing the pytest interpreter's
    already-warm imports; this one spawns a fresh interpreter on the hook module itself, so
    the fresh-interpreter claim above is measured against a real process, not inferred from
    an in-process call that never pays the cost a cold import would.
    """
    import subprocess
    import sys

    ws = tmp_path / "ws"
    proj = ws / "currant_demo"
    (proj / ".tcip").mkdir(parents=True)
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))

    from tcip_mcp import workspace

    workspace.activate_project("currant_demo")

    result = subprocess.run(
        [sys.executable, hook.__file__],
        input='{"source":"startup"}',
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    ctx = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "currant_demo" in ctx
    assert "No active project" not in ctx


def test_session_start_reads_a_marker_written_through_the_store(tmp_path, monkeypatch, capsys):
    """The marker is written through workspace.activate_project, not a hand-written file; a
    hook that assumed a loose file rather than reading through the seam would miss it.
    """
    _workspace(tmp_path, monkeypatch)
    out = _run(monkeypatch, capsys, '{"source":"startup"}')
    ctx = json.loads(out)["hookSpecificOutput"]["additionalContext"]
    assert "currant_demo" in ctx
    assert "No active project" not in ctx


def test_session_start_reads_the_loose_file_when_bound_to_the_file_backend(
    tmp_path, monkeypatch, capsys
):
    """Under TCIP_STORE_BACKEND=file the marker is genuinely the loose file, and the hook
    running under that same binding must read it and succeed, not refuse.
    """
    ws = tmp_path / "ws"
    proj = ws / "currant_demo"
    (proj / ".tcip").mkdir(parents=True)
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    monkeypatch.setenv("TCIP_STORE_BACKEND", "file")

    import tcip_store
    from tcip_mcp import workspace
    from tcip_store.file_backend import FileBackend

    tcip_store.bind(FileBackend())
    workspace.activate_project("currant_demo")

    out = _run(monkeypatch, capsys, '{"source":"startup"}')
    ctx = json.loads(out)["hookSpecificOutput"]["additionalContext"]
    assert "currant_demo" in ctx
    assert "could not be adopted" not in ctx


def test_session_start_names_the_store_refusal_for_a_loose_file_only_workspace(
    tmp_path, monkeypatch, capsys
):
    """A workspace holding only the loose file the file backend wrote, with no database, is
    refused by the default (sqlite) backend rather than silently read as if it were fine.
    """
    ws = tmp_path / "ws"
    proj = ws / "currant_demo"
    (proj / ".tcip").mkdir(parents=True)
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))

    import tcip_store
    from tcip_mcp import workspace
    from tcip_store.file_backend import FileBackend

    monkeypatch.setenv("TCIP_STORE_BACKEND", "file")
    file_backend = FileBackend()
    tcip_store.bind(file_backend)
    workspace.activate_project("currant_demo")
    file_backend.close()
    monkeypatch.delenv("TCIP_STORE_BACKEND", raising=False)

    out = _run(monkeypatch, capsys, '{"source":"startup"}')
    ctx = json.loads(out)["hookSpecificOutput"]["additionalContext"]
    assert "could not be adopted" in ctx
    assert "adopt_store" in ctx


def test_session_start_notes_which_processes_bind_from_the_marker(
    tmp_path, monkeypatch, capsys
):
    """The terminal's inherited platform-state root is a copy taken at spawn, so it can
    diverge from the marker until a process that binds from the marker converges: the web
    backend at its own startup, an MCP server launched here only at its next start."""
    proj = _workspace(tmp_path, monkeypatch)
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.setenv("TCIP_STATE_ROOT", str(other))

    out = _run(monkeypatch, capsys, '{"source":"startup"}')
    ctx = json.loads(out)["hookSpecificOutput"]["additionalContext"]
    assert str(other) in ctx
    assert proj in ctx
    assert "web backend" in ctx and "its own startup" in ctx
    assert "next start" in ctx
    assert "activate_project" in ctx


def test_session_start_reports_a_traversal_marker_through_the_shared_fold(
    tmp_path, monkeypatch, capsys
):
    """A marker naming an unsafe path (traversal) is not adoptable for a reason the hook must
    report through ``workspace.marker_problem``, the one place that fold lives, rather than a
    reason this hook re-derives on its own and gets wrong."""
    ws = tmp_path / "ws"
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))

    import tcip_store
    from tcip_mcp import workspace

    tcip_store.replace(workspace.active_project_key(), "../escapee")

    out = _run(monkeypatch, capsys, '{"source":"startup"}')
    ctx = json.loads(out)["hookSpecificOutput"]["additionalContext"]
    assert "could not be adopted" in ctx
    assert workspace.marker_problem(create=False) in ctx

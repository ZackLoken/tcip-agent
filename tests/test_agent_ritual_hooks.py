"""SessionStart ritual hook: fast, stdlib-only directive injection.

Locks the compliant design (Anthropic guidance: SessionStart must be quick, context-loading only):
it injects an ``additionalContext`` directive with live report/retro counts, and it spawns no
subprocess and imports nothing heavy (the standing check that guards against the reverted 30s
regression).
"""

from __future__ import annotations

import ast
import io
import json
from pathlib import Path

from tcip_web import agent_session_start as hook

HOOK_SRC = Path(hook.__file__).read_text(encoding="utf-8")


def _run(monkeypatch, capsys, stdin: str) -> str:
    monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    hook.main()
    return capsys.readouterr().out


def _workspace(tmp_path: Path, monkeypatch, *, reports=0, retros=0) -> str:
    ws = tmp_path / "ws"
    proj = ws / "hazelnut_demo"
    (proj / ".tcip" / "reports").mkdir(parents=True)
    (proj / ".tcip" / "retrospectives").mkdir(parents=True)
    for i in range(reports):
        (proj / ".tcip" / "reports" / f"r{i}.jsonl").write_text("{}\n")
    for i in range(retros):
        (proj / ".tcip" / "retrospectives" / f"s{i}.md").write_text("# retro\n")
    (ws / ".active").write_text("hazelnut_demo\n")
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    return str(proj)


def test_session_start_injects_ritual_directive_with_live_counts(tmp_path, monkeypatch, capsys):
    _workspace(tmp_path, monkeypatch, reports=3, retros=2)
    out = _run(monkeypatch, capsys, '{"source":"startup"}')
    ctx = json.loads(out)["hookSpecificOutput"]["additionalContext"]
    assert json.loads(out)["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "hazelnut_demo" in ctx
    assert "3 friction report(s)" in ctx and "2 retrospective(s)" in ctx
    for step in ("load_project_memory", "inspect_project", "doctor.py"):
        assert step in ctx


def test_session_start_no_active_project_covers_create_and_resume(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "empty_ws"))
    out = _run(monkeypatch, capsys, '{"source":"startup"}')
    ctx = json.loads(out)["hookSpecificOutput"]["additionalContext"]
    assert "No active project" in ctx
    # Both paths must be offered: creating a project is init_project, then set_active_project.
    assert "init_project" in ctx and "set_active_project" in ctx


def test_session_start_skips_on_compact(tmp_path, monkeypatch, capsys):
    _workspace(tmp_path, monkeypatch)
    assert _run(monkeypatch, capsys, '{"source":"compact"}').strip() == ""


def test_session_start_never_raises_on_garbage_stdin(monkeypatch, capsys):
    # Must degrade, never crash the session.
    _run(monkeypatch, capsys, "not json {{{")


def test_session_start_is_fast_stdlib_only_no_subprocess(tmp_path, monkeypatch, capsys):
    # Standing check for the reverted 30s regression: the hook must never spawn a subprocess or
    # import tcip_mcp at spawn time. Parse the real imports (via AST, so docstring mentions don't
    # count) so the fast property can't silently regress.
    imported: set[str] = set()
    for node in ast.walk(ast.parse(HOOK_SRC)):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "subprocess" not in imported
    assert "tcip_mcp" not in imported
    # And it emits its directive without touching either: a fresh interpreter would stay fast.
    _workspace(tmp_path, monkeypatch)
    out = _run(monkeypatch, capsys, '{"source":"startup"}')
    assert "SessionStart" in out

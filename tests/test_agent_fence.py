"""Tests for the in-app agent permission fence (AGENT_GOVERNANCE_PLAN Part 1)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tcip_web import terminal as pty_host

FENCE = Path(pty_host._FENCE_SETTINGS)
GUARD = FENCE.parent / "agent_bash_guard.py"


# ── the fence settings file ──────────────────────────────────────────────


def test_fence_settings_valid_json_denies_internals_allows_toolkit():
    data = json.loads(FENCE.read_text(encoding="utf-8"))
    deny = set(data["permissions"]["deny"])
    allow = set(data["permissions"]["allow"])
    # Platform internals are denied across the write tools.
    for rule in ("Edit(packages/**)", "Write(packages/**)", "Edit(tests/**)", "Edit(.github/**)"):
        assert rule in deny
    # Governance files are denied (proposals only, human applies).
    assert "Edit(CLAUDE.md)" in deny
    # The audited toolkit is the sanctioned mutation path — allowed.
    assert "mcp__tcip__*" in allow
    # A PreToolUse Bash guard is wired to the guard script.
    hook = data["hooks"]["PreToolUse"][0]
    assert hook["matcher"] == "Bash"
    assert "agent_bash_guard.py" in hook["hooks"][0]["command"]


# ── the spawn wiring ─────────────────────────────────────────────────────


def test_resolve_command_fences_the_real_cli(monkeypatch, tmp_path):
    monkeypatch.delenv("TCIP_TERMINAL_CMD", raising=False)
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setattr(pty_host.shutil, "which", lambda name: r"C:\fake\claude.exe")

    argv = pty_host.resolve_terminal_command()
    assert argv is not None
    assert argv[0] == r"C:\fake\claude.exe"
    assert "--settings" in argv and str(pty_host._FENCE_SETTINGS) in argv
    assert "--add-dir" in argv
    assert argv[argv.index("--permission-mode") + 1] == "default"


def test_override_is_not_fenced(monkeypatch):
    # The test double / power-user override must never inherit the fence flags.
    monkeypatch.setenv("TCIP_TERMINAL_CMD", "python fake.py")
    argv = pty_host.resolve_terminal_command()
    assert "--settings" not in argv


def test_repo_root_finds_the_marker():
    root = pty_host._repo_root()
    assert (root / ".mcp.json").is_file() or (root / "CLAUDE.md").is_file()


# ── the Bash guard hook ──────────────────────────────────────────────────


def _run_guard(command: str) -> subprocess.CompletedProcess:
    payload = json.dumps({"tool_input": {"command": command}})
    return subprocess.run(
        [sys.executable, str(GUARD)], input=payload, capture_output=True, text=True, timeout=30
    )


@pytest.mark.parametrize(
    "cmd",
    [
        "echo hacked > packages/tcip-mcp/x.py",
        "sed -i 's/a/b/' packages/tcip-web/src/tcip_web/app.py",
        'python -c "open(\'packages/x\',\'w\').write(\'x\')"',
        "cat CLAUDE.md > tests/y.txt",
        "node -e 'require(\"fs\").writeFileSync(\"packages/x\",\"y\")'",
    ],
)
def test_guard_denies_shell_writes_to_internals(cmd):
    r = _run_guard(cmd)
    assert r.returncode == 2, r.stdout
    assert "deny" in r.stdout


@pytest.mark.parametrize(
    "cmd",
    [
        "ls packages",
        "cat CLAUDE.md",
        "git status",
        "grep -r foo packages/tcip-mcp",
        "echo hello > /tmp/scratch.txt",  # a write, but not into repo internals
    ],
)
def test_guard_allows_reads_and_non_internal_writes(cmd):
    r = _run_guard(cmd)
    assert r.returncode == 0, r.stdout
    assert r.stdout.strip() == ""


def test_guard_fails_open_on_garbage_stdin():
    r = subprocess.run(
        [sys.executable, str(GUARD)], input="not json", capture_output=True, text=True, timeout=30
    )
    assert r.returncode == 0  # unparseable → fall through to normal permission flow

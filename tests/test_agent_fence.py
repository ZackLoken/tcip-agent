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
PS_GUARD = FENCE.parent / "agent_powershell_guard.py"


# ── the fence settings file ──────────────────────────────────────────────


def test_fence_settings_valid_json_denies_internals_allows_toolkit():
    data = json.loads(FENCE.read_text(encoding="utf-8"))
    deny = set(data["permissions"]["deny"])
    allow = set(data["permissions"]["allow"])
    # Platform internals are denied across the write tools.
    # Edit(X) covers Write/MultiEdit/NotebookEdit; standalone Write rules are deprecated no-ops.
    for rule in ("Edit(packages/**)", "Edit(tests/**)", "Edit(.github/**)", "Edit(CLAUDE.md)"):
        assert rule in deny
    # Governance files are denied (proposals only, human applies).
    assert "Edit(CLAUDE.md)" in deny
    # The audited toolkit is the sanctioned mutation path — allowed.
    assert "mcp__tcip__*" in allow
    # PreToolUse guards are wired for BOTH shells — PowerShell was the fence bypass.
    hooks = {h["matcher"]: h["hooks"][0]["command"] for h in data["hooks"]["PreToolUse"]}
    assert "agent_bash_guard.py" in hooks["Bash"]
    assert "agent_powershell_guard.py" in hooks["PowerShell"]


# the exact platform-internal deny set — pinned so the web-research grant can't widen the fence.
_EXPECTED_DENY = {
    "Edit(packages/**)",
    "Edit(tests/**)",
    "Edit(.github/**)",
    "Edit(scripts/**)",
    "Edit(.claude/**)",
    "Edit(CLAUDE.md)",
    "Edit(.mcp.json)",
    "Edit(pyproject.toml)",
    "Edit(package.json)",
    "Edit(tsconfig.json)",
    "Edit(.gitignore)",
    "Edit(README.md)",
    "Bash(rm:*)",
    "Bash(rmdir:*)",
    "Bash(git push:*)",
    "Bash(git commit:*)",
    "Bash(git reset:*)",
    "Bash(git checkout:*)",
    "Bash(git clean:*)",
}

# academic hosts WebFetch is scoped to (cv-research grant) — no open web.
_ACADEMIC_WEBFETCH = {
    "WebFetch(domain:arxiv.org)",
    "WebFetch(domain:www.arxiv.org)",
    "WebFetch(domain:semanticscholar.org)",
    "WebFetch(domain:www.semanticscholar.org)",
    "WebFetch(domain:openreview.net)",
    "WebFetch(domain:paperswithcode.com)",
    "WebFetch(domain:aclanthology.org)",
    "WebFetch(domain:biorxiv.org)",
    "WebFetch(domain:proceedings.mlr.press)",
    "WebFetch(domain:openaccess.thecvf.com)",
}


def test_fence_grants_academic_scoped_web_research_only():
    # The cv-research capability adds WebSearch + WebFetch, but WebFetch is scoped per-domain to
    # academic sources — never a blanket WebFetch(*) / open web.
    data = json.loads(FENCE.read_text(encoding="utf-8"))
    allow = set(data["permissions"]["allow"])
    assert "WebSearch" in allow
    assert _ACADEMIC_WEBFETCH <= allow, _ACADEMIC_WEBFETCH - allow
    # Every WebFetch grant is domain-scoped to an academic host; no blanket web access leaks in.
    webfetch = {r for r in allow if r.startswith("WebFetch")}
    assert webfetch == _ACADEMIC_WEBFETCH
    assert "WebFetch(*)" not in allow
    assert "WebFetch" not in allow  # bare (unscoped) WebFetch is not granted


def test_fence_web_research_grant_leaves_platform_deny_unchanged():
    # Adding web research must not touch the platform-internal deny fence.
    data = json.loads(FENCE.read_text(encoding="utf-8"))
    assert set(data["permissions"]["deny"]) == _EXPECTED_DENY


def test_fence_settings_wires_sessionstart_only_no_stop():
    # The fast session-start ritual injection is wired; the blocking Stop hook was reverted
    # (Anthropic guidance: Stop-block to force an action traps the user) and must stay absent.
    data = json.loads(FENCE.read_text(encoding="utf-8"))
    start = data["hooks"]["SessionStart"][0]["hooks"][0]["command"]
    assert "agent_session_start.py" in start
    assert "Stop" not in data["hooks"]


def test_bash_and_powershell_guards_protect_the_same_roots():
    # Drift guard: the two shells must fence the same platform paths, or one becomes a hole.
    # Compare the compiled _PROTECTED regexes by behaviour, not source text.
    from tcip_web import agent_bash_guard as bg
    from tcip_web import agent_powershell_guard as pg

    protected = [
        "packages/tcip-mcp/x.py",
        "tests/y.py",
        "scripts/z.py",
        ".github/skills/a.md",
        ".claude/settings.json",
        "CLAUDE.md",
        ".mcp.json",
        "pyproject.toml",
        "package.json",
        "tsconfig.json",
        ".gitignore",
    ]
    for path in protected:
        assert bg._PROTECTED.search(path), f"bash guard misses {path}"
        assert pg._PROTECTED.search(path), f"powershell guard misses {path}"
    # A breeder's workspace path is not platform-protected in either guard.
    workspace = "/c/Users/zack/tcip-projects/hazelnut/annotations/catkin/2026-02-11/detect/a.txt"
    assert not bg._PROTECTED.search(workspace)
    assert not pg._PROTECTED.search(workspace)


# ── the spawn wiring ─────────────────────────────────────────────────────


def test_resolve_command_fences_the_real_cli(monkeypatch, tmp_path):
    # resolve_terminal_command spawns claude DIRECTLY (no wrapping shell) with the fence flags;
    # --settings points at a materialized profile (valid JSON, still denies internals, guard hooks
    # absolute/cd-proof).
    monkeypatch.delenv("TCIP_TERMINAL_CMD", raising=False)
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setattr(pty_host.shutil, "which", lambda name: f"/fake/{name}")

    argv = pty_host.resolve_terminal_command()
    assert argv is not None
    assert argv[0] == "/fake/claude"  # claude directly, not a wrapping shell
    assert "--settings" in argv and "--add-dir" in argv
    assert argv[argv.index("--permission-mode") + 1] == "default"
    settings_path = Path(argv[argv.index("--settings") + 1])
    assert settings_path.is_file()
    cfg = json.loads(settings_path.read_text(encoding="utf-8"))
    assert "Edit(packages/**)" in cfg["permissions"]["deny"]
    guard_dir = pty_host._FENCE_SETTINGS.parent.as_posix()
    for group in cfg["hooks"]["PreToolUse"]:
        cmd = group["hooks"][0]["command"]
        assert "_guard.py" in cmd
        assert guard_dir in cmd  # absolute guard-directory path baked in
        assert cmd.startswith('"')  # quoted python exe, not a bare relative "python ..."


def test_prewarm_runs_without_raising(monkeypatch):
    # The daemon-thread kickoff returns immediately and never raises; the blocking worker (with
    # its subprocess stubbed out) swallows all errors and still imports the MCP tool graph.
    monkeypatch.setattr(pty_host.subprocess, "run", lambda *a, **k: None)
    pty_host.prewarm()
    pty_host._prewarm_blocking()
    import sys as _sys

    assert "tcip_mcp.server" in _sys.modules


def test_absolutize_guard_command_rewrites_relative_to_absolute():
    out = pty_host._absolutize_guard_command(
        "python packages/tcip-web/src/tcip_web/agent_bash_guard.py",
        "C:/py/python.exe",
        "C:/repo/packages/tcip-web/src/tcip_web",
    )
    assert out == '"C:/py/python.exe" "C:/repo/packages/tcip-web/src/tcip_web/agent_bash_guard.py"'
    # A non-guard command is left untouched.
    assert pty_host._absolutize_guard_command("ls -la", "C:/py/python.exe", "C:/x") == "ls -la"


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
        "echo x > agent_bash_guard.py",  # self-modify (even relative, after a cd)
        'bash -c "echo hi > packages/x"',  # nested shell
        "powershell -e U2V0",  # nested encoded shell from Bash
        # regression pins: a redirect/tee whose RESOLVED target is protected still blocks, even
        # when a protected token also appears as a mere exec argument (scripts/) or beside an
        # fd-dup (2>&1). These must stay DENY after the fence fix.
        "python scripts/foo.py > packages/out.txt",  # exec arg scripts/, real write to packages/
        "echo x > packages/y.py 2>&1",  # fd-dup present, but the > target is protected
        "python scripts/doctor.py /c/p 2>&1 | tee packages/x",  # tee target protected
    ],
)
def test_guard_denies_shell_writes_to_internals(cmd):
    r = _run_guard(cmd)
    assert r.returncode == 2, r.stdout
    assert "deny" in r.stdout


def test_both_guards_deny_a_write_to_the_same_protected_path():
    # Cross-shell parity: the same mutation intent (write into packages/) is blocked in both
    # shells — the "switch to the other shell" bypass class must not exist.
    assert _run_guard("echo x > packages/tcip-mcp/y.py").returncode == 2
    assert _run_ps_guard('Set-Content packages\\tcip-mcp\\y.py "x"').returncode == 2
    assert _run_guard("cp a.py packages/tcip-mcp/y.py").returncode == 2
    assert _run_ps_guard("cp a.py packages\\tcip-mcp\\y.py").returncode == 2


@pytest.mark.parametrize(
    "cmd",
    [
        "ls packages",
        "cat CLAUDE.md",
        "git status",
        "grep -r foo packages/tcip-mcp",
        "echo hello > /tmp/scratch.txt",  # a write, but not into repo internals
        # fd redirects / non-protected targets are not writes into internals — the mandated
        # diagnostics and their redirect variants must fall through (standing checks for the fix).
        "python scripts/doctor.py C:/Users/zack/tcip-projects/hazelnut 2>&1",  # mandated command
        "python scripts/doctor.py /c/proj 2>/dev/null",
        "python scripts/list_tools.py",  # no redirect at all
        "python scripts/list_tools.py > /tmp/tools.txt",  # real redirect, non-protected target
        "python scripts/doctor.py /c/proj 2>&1 | tee /tmp/doctor.log",
        "ls packages 2>&1",  # fd-dup while reading a protected dir
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


# ── the PowerShell guard hook (the closed bypass) ────────────────────────


def _run_ps_guard(command: str) -> subprocess.CompletedProcess:
    payload = json.dumps({"tool_input": {"command": command}})
    return subprocess.run(
        [sys.executable, str(PS_GUARD)], input=payload, capture_output=True, text=True, timeout=30
    )


@pytest.mark.parametrize(
    "cmd",
    [
        # writes into platform internals via full cmdlets
        'Set-Content packages/tcip-mcp/x.py "hacked"',
        "Out-File -FilePath packages\\tcip-web\\src\\tcip_web\\app.py",
        'Add-Content .github/skills/x.md "y"',
        '[IO.File]::WriteAllText("packages\\x.py", "y")',
        '[IO.File]::OpenWrite("packages\\tcip-mcp\\x.py")',  # .NET writer beyond WriteAllText
        "New-Item -Path tests/x.py -ItemType File",
        # writes via cmdlet ALIASES (the verified bypass — sc/ac/ni/cp/mv/clc/spi)
        'sc packages\\tcip-mcp\\server.py "hacked"',
        'ac packages\\tcip-mcp\\server.py "more"',
        "ni packages\\tcip-mcp\\evil.py -ItemType File",
        "cp CLAUDE.md packages\\tcip-mcp\\x.py",
        "mv foo.py packages\\tcip-mcp\\server.py",
        "clc packages\\tcip-mcp\\server.py",
        # self-modification of the fence itself (would permanently disable the guard)
        'Set-Content agent_powershell_guard.py "import sys; sys.exit(0)"',
        "ri agent_bash_guard.py",
        'Out-File tcip_agent_fence.settings.json "{}"',
        # deletions (blocked unconditionally, incl. the ``ri`` alias)
        "Remove-Item packages/tcip-mcp/server.py",
        "ri packages/tcip-mcp/server.py",
        "rm C:\\Users\\zack\\tcip-projects\\proj\\labels\\a.txt",
        # inline / nested / encoded execution
        'iex "malicious"',
        "Invoke-Expression $payload",
        'python -c "open(\'packages/x\',\'w\')"',
        "powershell -EncodedCommand U2V0LUNvbnRlbnQ=",
        "pwsh -e U2V0LUNvbnRlbnQ=",
        'cmd /c "del packages\\x"',
        # dangerous git
        "git push origin main",
        'git commit -m "x"',
        "git reset --hard HEAD~1",
        # regression pin: a redirect whose RESOLVED target is protected still blocks, even with
        # scripts/ present only as an exec argument. Must stay DENY after the fence fix.
        "python scripts/foo.py > packages\\out.txt",
    ],
)
def test_ps_guard_denies_mutations_and_dangerous(cmd):
    r = _run_ps_guard(cmd)
    assert r.returncode == 2, r.stdout
    assert "deny" in r.stdout


@pytest.mark.parametrize(
    "cmd",
    [
        "Get-ChildItem packages",
        'Select-String -Path packages\\tcip-mcp\\server.py -Pattern "def"',
        "Get-NetTCPConnection -State Listen -LocalPort 8765",
        "Test-Path packages/tcip-web",
        "Get-Content CLAUDE.md",
        # a delete/write ALIAS as a bareword in an ARGUMENT must not trip the guard (the
        # false-positive the anchoring fixes) — these are pure reads.
        "Get-Content elderberry-del.csv",
        "Get-Content .\\del.txt",
        "Get-ChildItem C:\\rm",
        "Select-String -Pattern sc -Path packages\\tcip-mcp\\server.py",
        "git status",
        "git log --oneline -5",
        # fd redirects / non-protected targets are not writes into internals (standing checks).
        "python scripts/doctor.py C:\\Users\\zack\\proj 2>&1",  # mandated command, redirect form
        "python scripts/list_tools.py 2>$null",
        "python scripts/list_tools.py > $env:TEMP\\tools.txt",  # real redirect, non-protected
        "Get-ChildItem packages 2>&1",  # fd-dup while reading a protected dir
    ],
)
def test_ps_guard_allows_reads(cmd):
    r = _run_ps_guard(cmd)
    assert r.returncode == 0, r.stdout
    assert r.stdout.strip() == ""


def test_deny_reason_points_to_claude_reports():
    # Requirement (c): a blocked protected-write self-documents — it tells the agent to file the
    # false-positive with claude_reports rather than route around the fence.
    r = _run_guard("echo x > packages/tcip-mcp/y.py")
    assert r.returncode == 2 and "claude_reports" in r.stdout
    r = _run_ps_guard('Set-Content packages\\tcip-mcp\\y.py "x"')
    assert r.returncode == 2 and "claude_reports" in r.stdout


def test_ps_guard_fails_open_on_garbage_stdin():
    r = subprocess.run(
        [sys.executable, str(PS_GUARD)], input="not json", capture_output=True, text=True, timeout=30
    )
    assert r.returncode == 0

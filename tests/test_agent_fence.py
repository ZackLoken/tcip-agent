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
    for rule in ("Edit(packages/**)", "Write(packages/**)", "Edit(tests/**)", "Edit(.github/**)"):
        assert rule in deny
    # Governance files are denied (proposals only, human applies).
    assert "Edit(CLAUDE.md)" in deny
    # The audited toolkit is the sanctioned mutation path — allowed.
    assert "mcp__tcip__*" in allow
    # PreToolUse guards are wired for BOTH shells — PowerShell was the fence bypass.
    hooks = {h["matcher"]: h["hooks"][0]["command"] for h in data["hooks"]["PreToolUse"]}
    assert "agent_bash_guard.py" in hooks["Bash"]
    assert "agent_powershell_guard.py" in hooks["PowerShell"]


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
    # A breeder's workspace path is NOT platform-protected in either guard.
    workspace = "/c/Users/zack/tcip-projects/hazelnut/annotations/catkin/2026-02-11/detect/a.txt"
    assert not bg._PROTECTED.search(workspace)
    assert not pg._PROTECTED.search(workspace)


# ── the spawn wiring ─────────────────────────────────────────────────────


def test_resolve_command_fences_the_real_cli(monkeypatch, tmp_path):
    monkeypatch.delenv("TCIP_TERMINAL_CMD", raising=False)
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setattr(pty_host.shutil, "which", lambda name: r"C:\fake\claude.exe")

    argv = pty_host.resolve_terminal_command()
    assert argv is not None
    assert argv[0] == r"C:\fake\claude.exe"
    assert "--settings" in argv
    assert "--add-dir" in argv
    assert argv[argv.index("--permission-mode") + 1] == "default"

    # --settings points at a MATERIALIZED profile: valid JSON, still denies internals, and its
    # guard hook commands are ABSOLUTE (cd-proof) rather than the repo-relative template form.
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
    ],
)
def test_ps_guard_allows_reads(cmd):
    r = _run_ps_guard(cmd)
    assert r.returncode == 0, r.stdout
    assert r.stdout.strip() == ""


def test_ps_guard_fails_open_on_garbage_stdin():
    r = subprocess.run(
        [sys.executable, str(PS_GUARD)], input="not json", capture_output=True, text=True, timeout=30
    )
    assert r.returncode == 0

"""Tests for the in-app agent permission fence."""

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
    # The audited toolkit is the sanctioned mutation path: allowed.
    assert "mcp__tcip__*" in allow
    # find has no legitimate-usage evidence anywhere in this repo (no transcript, skill, or doc
    # exercises it) and its own write actions (-fprint/-fprintf/-fls) aren't expressible as a
    # narrower prefix rule, so it is not blanket-allowed: an approval prompt gates it instead.
    assert "Bash(find:*)" not in allow
    # PreToolUse guards are wired for both shells: PowerShell was the fence bypass.
    hooks = {h["matcher"]: h["hooks"][0]["command"] for h in data["hooks"]["PreToolUse"]}
    assert "agent_bash_guard.py" in hooks["Bash"]
    assert "agent_powershell_guard.py" in hooks["PowerShell"]


# the exact platform-internal deny set, pinned so the web-research grant can't widen the fence.
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

# academic hosts WebFetch is scoped to (cv-research grant): no open web.
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
    # academic sources, never a blanket WebFetch(*) / open web.
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


def test_both_guards_read_the_protected_set_from_the_fence_declaration():
    # Both shells import one matcher built from the settings deny rules, so the paths they fence
    # are the paths declared there rather than a list either guard restates.
    from tcip_web import agent_bash_guard as bg
    from tcip_web import agent_fence_rules
    from tcip_web import agent_powershell_guard as pg

    assert bg.fence_rules is agent_fence_rules
    assert pg.fence_rules is agent_fence_rules
    pattern = agent_fence_rules.protected_pattern()
    deny = json.loads(FENCE.read_text(encoding="utf-8"))["permissions"]["deny"]
    declared = [r[len("Edit(") : -1] for r in deny if r.startswith("Edit(")]
    assert len(declared) >= 10, declared
    for target in declared:
        sample = target[: -len("/**")] + "/sample.txt" if target.endswith("/**") else target
        assert pattern.search(sample), f"the declared path {target} is not fenced"
    # A breeder's workspace path is not platform-protected: the rail admits their own data.
    workspace = "/c/Users/breeder/tcip-projects/hazelnut/annotations/catkin/2026-02-11/detect/a.txt"
    assert not pattern.search(workspace)


def test_a_protected_path_is_fenced_whatever_its_case():
    # The guards protect a filesystem, and on the platform's Windows host that filesystem is
    # case-insensitive, so PACKAGES/x.py and packages/x.py are one file and one denial.
    assert _run_guard("echo x > PACKAGES/tcip-mcp/x.py").returncode == 2
    assert _run_ps_guard('Set-Content PACKAGES\\tcip-mcp\\x.py "x"').returncode == 2
    assert _run_guard("echo '{}' > /c/proj/Labels/a.json").returncode == 2
    assert _run_ps_guard('Set-Content C:\\proj\\Labels\\a.json "{}"').returncode == 2


def test_relocating_a_whole_breeder_data_directory_is_denied_in_both_shells():
    # Moving annotations/ wholesale removes every label from where the platform tracks it, the
    # same harm as moving one file out, so a bare directory argument counts as a target too.
    assert _run_guard("mv /c/proj/annotations /tmp/exfil").returncode == 2
    assert _run_ps_guard("Move-Item C:\\proj\\annotations C:\\out").returncode == 2


# ── the spawn wiring ─────────────────────────────────────────────────────


def test_resolve_command_fences_the_real_cli(monkeypatch, tmp_path):
    # resolve_terminal_command spawns claude directly (no wrapping shell) with the fence flags;
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


def test_the_agent_runs_at_the_repo_root_the_fence_deny_paths_are_written_against(
    tmp_path, monkeypatch
):
    """The fence's deny paths (``packages/**``, ``tests/**``…) are repo-root-relative, so the
    spawned agent's cwd must be the repo root itself, not the package directory this module
    happens to live in, and must not depend on how the web process was started."""
    monkeypatch.delenv(pty_host.TERMINAL_CWD_ENV, raising=False)
    from_install_dir = Path(pty_host.terminal_cwd())
    monkeypatch.chdir(tmp_path)
    assert Path(pty_host.terminal_cwd()) == from_install_dir
    assert (from_install_dir / ".mcp.json").is_file()
    assert (from_install_dir / "packages").is_dir()
    assert FENCE.is_relative_to(from_install_dir)


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
        # a redirect/tee whose resolved target is protected still blocks, even
        # when a protected token also appears as a mere exec argument (scripts/) or beside an
        # fd-dup (2>&1); these must stay denied.
        "python scripts/foo.py > packages/out.txt",  # exec arg scripts/, real write to packages/
        "echo x > packages/y.py 2>&1",  # fd-dup present, but the > target is protected
        "python scripts/doctor.py /c/p 2>&1 | tee packages/x",  # tee target protected
        # find's own write actions carry no >/tee at all: caught by target, not by shape.
        "find . -maxdepth 1 -fprintf packages/tcip-mcp/evil.py '%p'",
        "find . -maxdepth 1 -fls packages/tcip-mcp/evil.txt",
    ],
)
def test_guard_denies_shell_writes_to_internals(cmd):
    r = _run_guard(cmd)
    assert r.returncode == 2, r.stdout
    assert "deny" in r.stdout


def test_both_guards_deny_a_write_to_the_same_protected_path():
    # Cross-shell parity: the same mutation intent (write into packages/) is blocked in both
    # shells: the "switch to the other shell" bypass class must not exist.
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
        # fd redirects / non-protected targets are not writes into internals: the mandated
        # diagnostics and their redirect variants must fall through.
        "python scripts/doctor.py C:/Users/breeder/tcip-projects/hazelnut 2>&1",  # mandated command
        "python scripts/doctor.py /c/proj 2>/dev/null",
        "python scripts/list_tools.py",  # no redirect at all
        "python scripts/list_tools.py > /tmp/tools.txt",  # real redirect, non-protected target
        "python scripts/doctor.py /c/proj 2>&1 | tee /tmp/doctor.log",
        "ls packages 2>&1",  # fd-dup while reading a protected dir
        "find . -printf '%p\\n'",  # -printf (no leading f) writes to stdout, not a file
    ],
)
def test_guard_allows_reads_and_non_internal_writes(cmd):
    r = _run_guard(cmd)
    assert r.returncode == 0, r.stdout
    assert r.stdout.strip() == ""


@pytest.mark.parametrize(
    "cmd",
    [
        "mv /c/proj/labels/a.json /tmp/out/a.json",
        "mv /tmp/out/a.json /c/proj/labels/a.json",
        "find /c/proj/annotations -exec mv {} /tmp/exfil \\;",
    ],
)
def test_guard_denies_moving_breeder_data(cmd):
    # mv relocates the tracked file exactly as rm would remove it from where it belongs, whether
    # the breeder-data path is named as mv's source, its destination, or (for the find form) the
    # search root the exec action walks; all three read the same way to this stateless check.
    r = _run_guard(cmd)
    assert r.returncode == 2, r.stdout
    assert "deny" in r.stdout


@pytest.mark.parametrize(
    "cmd",
    [
        # a bare "mv" as a substring of a path or grep pattern (not statement position) must not
        # trip the guard: the false-positive the _STMT anchoring on _MOVE_OP exists to prevent,
        # mirroring _DELETE_OP's own anchoring.
        "cat /c/proj/annotations/mv-notes.txt",
        "grep -rn mv /c/proj/annotations",
        "ls /c/proj/annotations/mv-backup",
        "find /c/proj/annotations -name '*mv*'",
    ],
)
def test_guard_allows_reads_that_merely_mention_move_words(cmd):
    r = _run_guard(cmd)
    assert r.returncode == 0, r.stdout
    assert r.stdout.strip() == ""


@pytest.mark.parametrize(
    "cmd",
    [
        "cp /c/proj/labels/a.json /tmp/backup/a.json",
        "cp /tmp/backup/a.json /c/proj/labels/a.json",
        "find /c/proj/annotations -exec cp {} /tmp/backup \\;",
    ],
)
def test_guard_allows_copying_breeder_data(cmd):
    # Deliberately not denied, mirroring the PowerShell guard's own Copy-Item exemption: this guard
    # is stateless and can't tell a two-argument command's source from its destination, so denying
    # cp here would also deny a legitimate backup/copy of a breeder file to elsewhere, not just a
    # copy into one. The rail must admit valid work, not only reject invalid work.
    r = _run_guard(cmd)
    assert r.returncode == 0, r.stdout


def test_guard_fails_open_on_garbage_stdin():
    r = subprocess.run(
        [sys.executable, str(GUARD)], input="not json", capture_output=True, text=True, timeout=30
    )
    assert r.returncode == 0  # unparseable → fall through to normal permission flow


# ── the Bash fence's delete/truncate parity with PowerShell ────────


@pytest.mark.parametrize(
    "cmd",
    [
        "rm C:/Users/breeder/tcip-projects/proj/labels/a.txt",
        "rmdir /c/Users/breeder/tcip-projects/proj/annotations/2026-01-01",
        "unlink /c/Users/breeder/tcip-projects/proj/labels/a.txt",
        "shred /c/Users/breeder/tcip-projects/proj/labels/a.txt",
        "truncate -s 0 /c/Users/breeder/tcip-projects/proj/annotations/a.json",
        # xargs, with and without a flag on the trailing verb
        "ls /c/proj/annotations/*.json | xargs rm",
        "ls /c/proj/annotations/*.json | xargs rm -f",
        # find, the sharpest hole: Bash(find:*) is allow-listed, so only the guard stops this
        "find /c/proj/annotations -name '*.json' -delete",
        "find /c/proj/annotations -name '*.json' -exec rm {} \\;",
        # every other verb _DELETE_OP treats as unconditionally destructive, reached the same way
        "find /c/proj/annotations -exec rmdir {} \\;",
        "find /c/proj/annotations -exec unlink {} \\;",
        "find /c/proj/annotations -exec shred {} \\;",
        "find /c/proj/annotations -exec truncate -s 0 {} \\;",
        # statement-position variants (after ; | & ( { } and at line start)
        "cd /c/proj && rm labels/a.txt",
        "echo hi; rm labels/a.txt",
    ],
)
def test_guard_denies_deletion_and_truncation(cmd):
    r = _run_guard(cmd)
    assert r.returncode == 2, r.stdout
    assert "deny" in r.stdout


@pytest.mark.parametrize(
    "cmd",
    [
        # a delete/truncate verb as a bareword in an argument (not statement position) must not trip
        # the guard: the false-positive the _STMT anchoring exists to prevent.
        "grep -r shred packages/tcip-mcp",
        "grep -rn truncate packages",
        "cat elderberry-rm-notes.txt",
        "ls /c/proj/rm-backup",
        "python scripts/doctor.py /c/proj",  # contains no delete verb at all
        "find /c/proj/annotations -name '*.json'",  # find without -delete/-exec rm is a plain read
        "find /c/proj -type f -name 'truncate_report.json'",
    ],
)
def test_guard_allows_reads_that_merely_mention_delete_words(cmd):
    r = _run_guard(cmd)
    assert r.returncode == 0, r.stdout
    assert r.stdout.strip() == ""


@pytest.mark.parametrize(
    "cmd",
    [
        "cat /c/proj/annotations/2026-01-01/a.json > /c/proj/annotations/2026-01-01/a.json.bak",
        "echo '{}' > /c/proj/labels/a.json",
        "echo '{}' > /c/proj/predictions/live/2026-01-01/a.json",
        "> /c/proj/annotations/a.json",
        "find /c/proj -maxdepth 1 -fprint /c/proj/labels/a.json",
    ],
)
def test_guard_denies_writing_into_breeder_data_paths(cmd):
    r = _run_guard(cmd)
    assert r.returncode == 2, r.stdout
    assert "deny" in r.stdout


def test_guard_allows_a_redirect_into_an_unrelated_path():
    # A redirect target that doesn't name annotations/labels/predictions/image_status.json is not
    # breeder data: the rail must admit valid work, not only reject invalid work.
    r = _run_guard("echo done > /tmp/scratch.txt")
    assert r.returncode == 0, r.stdout


def test_bash_and_powershell_guards_deny_the_same_breeder_data_operations():
    # Parity test: both shells must deny the same breeder-data delete/truncate/write
    # operations, the drift the platform-path-only comparison above
    # (test_bash_and_powershell_guards_protect_the_same_roots) cannot catch, since platform paths
    # and breeder data are orthogonal invariants. Covers all three harm classes (delete, truncate,
    # overwrite-via-redirect-or-cmdlet), not delete alone.
    pairs = [
        ("rm C:/Users/breeder/tcip-projects/proj/labels/a.txt",
         'rm C:\\Users\\breeder\\tcip-projects\\proj\\labels\\a.txt'),
        ("find /c/proj/annotations -name '*.json' -delete", None),  # Bash-only construct
        ("truncate -s 0 /c/proj/annotations/a.json",
         "Clear-Content C:\\proj\\annotations\\a.json"),
        ("echo '{}' > /c/proj/labels/a.json",
         'Set-Content C:\\proj\\labels\\a.json \'{}\''),
        ("> /c/proj/labels/a.json", "'{}' > C:\\proj\\labels\\a.json"),
    ]
    for bash_cmd, ps_cmd in pairs:
        assert _run_guard(bash_cmd).returncode == 2, bash_cmd
        if ps_cmd is not None:
            assert _run_ps_guard(ps_cmd).returncode == 2, ps_cmd


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
        # writes via cmdlet aliases (the verified bypass: sc/ac/ni/cp/mv/clc/spi)
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
        "rm C:\\Users\\breeder\\tcip-projects\\proj\\labels\\a.txt",
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
        # a redirect whose resolved target is protected still blocks, even with
        # scripts/ present only as an exec argument. Must stay denied.
        "python scripts/foo.py > packages\\out.txt",
    ],
)
def test_ps_guard_denies_mutations_and_dangerous(cmd):
    r = _run_ps_guard(cmd)
    assert r.returncode == 2, r.stdout
    assert "deny" in r.stdout


# ── the PowerShell fence's breeder-data truncate/write parity with Bash ──


@pytest.mark.parametrize(
    "cmd",
    [
        "Clear-Content C:\\Users\\breeder\\tcip-projects\\proj\\labels\\a.json",
        "clc C:\\Users\\breeder\\tcip-projects\\proj\\labels\\a.json",
        'Set-Content -Path C:\\proj\\annotations\\2026-01-01\\a.json -Value \'{}\'',
        "Out-File -FilePath C:\\proj\\predictions\\live\\2026-01-01\\a.json",
        'sc C:\\proj\\labels\\a.json "x"',
        "'{}' > C:\\proj\\labels\\a.json",
        "Get-Content x.json > C:\\proj\\annotations\\a.json.bak",
    ],
)
def test_ps_guard_denies_writing_into_breeder_data_paths(cmd):
    r = _run_ps_guard(cmd)
    assert r.returncode == 2, r.stdout
    assert "deny" in r.stdout


@pytest.mark.parametrize(
    "cmd",
    [
        # a write cmdlet/alias or "labels"/"annotations"/"predictions" as a bareword argument (not
        # a write target) must not trip the guard.
        "Get-Content elderberry-labels-notes.txt",
        "Select-String -Pattern predictions -Path packages\\tcip-mcp\\server.py",
        "Get-ChildItem C:\\proj\\annotations",  # a read, not a write
        "echo 'Set-Content is a cmdlet name'",
        "Get-Content C:\\proj\\annotations\\a.json > $env:TEMP\\scratch.json",  # write target is not breeder data
        # Copy-Item (and its aliases) can't be told source from destination by a stateless guard:
        # deliberately exempted from the breeder-data check in both directions, mirroring the Bash
        # guard's own exemption of cp, so a legitimate backup/duplicate of a label isn't blocked.
        "Copy-Item C:\\proj\\labels\\a.json -Destination C:\\out\\backup.json",
        "Copy-Item C:\\out\\x.json -Destination C:\\proj\\labels\\a.json",
        "cpi C:\\proj\\annotations\\a.json C:\\out\\x.json",
    ],
)
def test_ps_guard_allows_reads_that_merely_mention_breeder_data_words(cmd):
    r = _run_ps_guard(cmd)
    assert r.returncode == 0, r.stdout
    assert r.stdout.strip() == ""


@pytest.mark.parametrize(
    "cmd",
    [
        # Move/Rename do relocate the tracked file even when it's named as the "source", unlike
        # Copy-Item, these stay denied.
        "Move-Item C:\\proj\\labels\\a.json -Destination C:\\out\\a.json",
        "mv C:\\proj\\labels\\a.json C:\\out\\a.json",
        "Rename-Item C:\\proj\\labels\\a.json b.json",
        "rni C:\\proj\\annotations\\a.json b.json",
    ],
)
def test_ps_guard_still_denies_move_and_rename_of_breeder_data(cmd):
    # Regression pin for the Copy-Item exemption above: narrowing the write-op set for the
    # breeder-data check must not have accidentally exempted Move/Rename too.
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
        # a delete/write alias as a bareword in an argument must not trip the guard (the
        # false-positive the anchoring fixes): these are pure reads.
        "Get-Content elderberry-del.csv",
        "Get-Content .\\del.txt",
        "Get-ChildItem C:\\rm",
        "Select-String -Pattern sc -Path packages\\tcip-mcp\\server.py",
        "git status",
        "git log --oneline -5",
        # fd redirects / non-protected targets are not writes into internals (standing checks).
        "python scripts/doctor.py C:\\Users\\breeder\\proj 2>&1",  # mandated command, redirect form
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
    # Requirement (c): a blocked protected-write self-documents: it tells the agent to file the
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

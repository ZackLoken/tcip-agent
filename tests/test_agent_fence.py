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
    "Edit(.tcip/state/trait_operationalizations/**)",
    "Edit(.tcip/state/trait_spec_statements/**)",
    "Edit(.tcip/state/trait_specs/**)",
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


def test_both_guards_classify_targets_through_the_shared_fence_declaration():
    # Both shells import one classifier built from the settings deny rules, so the paths they fence
    # are the paths declared there rather than a list either guard restates.
    from tcip_web import agent_bash_guard as bg
    from tcip_web import agent_fence_rules
    from tcip_web import agent_powershell_guard as pg

    assert bg.fence_rules is agent_fence_rules
    assert pg.fence_rules is agent_fence_rules
    root = agent_fence_rules.repo_root()
    deny = json.loads(FENCE.read_text(encoding="utf-8"))["permissions"]["deny"]
    declared = [r[len("Edit(") : -1] for r in deny if r.startswith("Edit(")]
    assert len(declared) >= 10, declared
    for target in declared:
        sample = target[: -len("/**")] + "/sample.txt" if target.endswith("/**") else target
        kind = agent_fence_rules.classify(sample, root=root, mode="dev")
        assert kind in ("protected", "breeder"), f"the declared path {target} is not fenced"
    # A breeder's own project file sharing a repo-root basename is not platform code: anchoring
    # the single-file rules to the repo root admits it.
    own = "/c/Users/breeder/tcip-projects/currant/README.md"
    assert agent_fence_rules.classify(own, root=root, mode="dev") is None


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


# Shapes both shells express: a redirect, a copy, a move, a write cmdlet against an in-place
# edit. Each builder takes the target path and returns the command for its own shell.
_PARITY_SHAPES = {
    "redirect": (lambda t: f"echo x > {t}", lambda t: f"'{{}}' > {t}"),
    "copy": (lambda t: f"cp evil.py {t}", lambda t: f"Copy-Item evil.py -Destination {t}"),
    "move": (lambda t: f"mv evil.py {t}", lambda t: f"Move-Item evil.py -Destination {t}"),
    "write": (lambda t: f"sed -i 's/a/b/' {t}", lambda t: f"Set-Content {t} 'x'"),
}

# (bash target, PowerShell target, expected verdict) per path kind.
_PARITY_PATHS = {
    "protected": ("packages/tcip-mcp/server.py", "packages\\tcip-mcp\\server.py", "deny"),
    "breeder": ("/c/proj/labels/a.json", "C:\\proj\\labels\\a.json", "deny"),
    "scratch": ("/tmp/scratch.txt", "C:\\tmp\\scratch.txt", "allow"),
}


@pytest.mark.parametrize("path_kind", sorted(_PARITY_PATHS))
@pytest.mark.parametrize("shape", sorted(_PARITY_SHAPES))
def test_both_guards_agree_on_the_expected_verdict_per_shape_and_path(shape, path_kind):
    """Both guards feed one classifier (the same-object assertion above), so equal verdicts
    follow once each shell's extractor finds a token; asserting the expected verdict rather than
    equality alone means a pair that both fail to extract a target cannot pass. The extractors
    barely intersect (bash's patch/dd of=/ln/install/rsync/find -fprint and the line editors;
    PowerShell's .NET/stream-writer forms and the item-property cmdlets); those are
    outside parity by construction, each covered by its own shell's tests. Iterating the fence
    declaration's paths is not claimed as coverage here: the classifier is one object, checked
    elsewhere, and these three path kinds exercise the extractors, not the declaration.
    """
    bash_build, ps_build = _PARITY_SHAPES[shape]
    bash_target, ps_target, verdict = _PARITY_PATHS[path_kind]
    deny = verdict == "deny"
    assert (_run_guard(bash_build(bash_target)).returncode == 2) is deny
    assert (_run_ps_guard(ps_build(ps_target)).returncode == 2) is deny


@pytest.mark.parametrize("path_kind", sorted(_PARITY_PATHS))
def test_both_guards_deny_delete_on_every_path_kind(path_kind):
    """Delete is path-independent on both sides: the verb alone is denied, even against the
    scratch path every write shape above admits. Truncate and git push are not parity shapes:
    bash denies truncate unconditionally too, but PowerShell's Clear-Content is target-classified,
    and git push is settings-only in bash but guard-level in PowerShell; both are deliberate
    asymmetries, named here rather than asserted away.
    """
    bash_target, ps_target, _ = _PARITY_PATHS[path_kind]
    assert _run_guard(f"rm {bash_target}").returncode == 2
    assert _run_ps_guard(f"Remove-Item {ps_target}").returncode == 2


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
        "python scripts/doctor.py C:/Users/breeder/tcip-projects/currant 2>&1",  # mandated command
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
        # trip the guard, the same guarantee the delete-verb check gives its own leading token.
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
        "cp /c/proj/labels/a.json /tmp/backup/a.json",  # copy a breeder file OUT: destination is free
        "find /c/proj/annotations -exec cp {} /tmp/backup \\;",  # find-exec copy out, destination free
    ],
)
def test_guard_allows_copying_breeder_data_out(cmd):
    # Copying a breeder file to a free destination is a legitimate backup: the destination is
    # classified, so reading breeder data and writing elsewhere is admitted.
    r = _run_guard(cmd)
    assert r.returncode == 0, r.stdout


@pytest.mark.parametrize(
    "cmd",
    [
        "cp /tmp/x.json /c/proj/labels/a.json",  # copy INTO a breeder label overwrites it
        "cp evil.json /c/proj/annotations/2026-01-01/a.json",
    ],
)
def test_guard_denies_copying_into_breeder_data(cmd):
    # Overwriting a breeder's label via a shell copy is the harm the fence exists to stop;
    # destination-position parsing tells the copy-in from the admitted copy-out above.
    r = _run_guard(cmd)
    assert r.returncode == 2, r.stdout
    assert "deny" in r.stdout


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
        # every other verb the delete check treats as unconditionally destructive, reached the same way
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
        # Copy-Item OUT (breeder source, free destination) is a legitimate backup: the destination is
        # classified, so copying a label elsewhere is admitted while copying INTO one is not.
        "Copy-Item C:\\proj\\labels\\a.json -Destination C:\\out\\backup.json",
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
        "Copy-Item C:\\out\\x.json -Destination C:\\proj\\labels\\a.json",
        "cpi evil.json C:\\proj\\annotations\\2026-01-01\\a.json",
    ],
)
def test_ps_guard_denies_copying_into_breeder_data(cmd):
    # Overwriting a breeder's label via Copy-Item into a breeder destination is denied; the
    # destination is classified, so copying a label OUT stays admitted.
    r = _run_ps_guard(cmd)
    assert r.returncode == 2, r.stdout
    assert "deny" in r.stdout


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


def test_deny_reason_points_to_report_friction():
    # Requirement (c): a blocked protected-write self-documents: it tells the agent to file the
    # false-positive with report_friction rather than route around the fence.
    r = _run_guard("echo x > packages/tcip-mcp/y.py")
    assert r.returncode == 2 and "report_friction" in r.stdout
    r = _run_ps_guard('Set-Content packages\\tcip-mcp\\y.py "x"')
    assert r.returncode == 2 and "report_friction" in r.stdout


def test_ps_guard_fails_open_on_garbage_stdin():
    r = subprocess.run(
        [sys.executable, str(PS_GUARD)], input="not json", capture_output=True, text=True, timeout=30
    )
    assert r.returncode == 0


# ── redirect grammar and target normalization: bypasses closed, admit-valid-work preserved ──


@pytest.mark.parametrize(
    "cmd",
    [
        # A redirect grammar that recognises every file-writing form, into breeder data and into
        # platform code, so an allow-listed read prefix carrying one is stopped with no human loop.
        "echo x >| /c/proj/labels/a.json",  # noclobber override
        "ls -la >& /c/proj/labels/a.json",  # both streams to a file, riding allow-listed ls
        "echo x >| packages/tcip-mcp/x.py",
        "ls -la >& packages/tcip-mcp/x.py",
        "echo x &> packages/tcip-mcp/x.py",  # &> both streams
        "exec 3<> packages/tcip-mcp/server.py",  # read-write open, can create
        # A protected/breeder path reached past the separator character class or through ``..``.
        "dd of=packages/tcip-mcp/server.py",  # '=' embedded, no separator before packages
        "dd of=/c/proj/labels/a.json",
        "printf hacked > docs/../packages/tcip-mcp/server.py",  # .. collapses into packages/
        # Quote insertion: the shell strips the quotes and writes the real path.
        'echo x > /c/proj/la"bel"s/a.json',
        "echo x > pack'ages'/tcip-mcp/x.py",
        # Variable indirection, one hop resolved inside the command.
        "T=/c/proj/labels/a.json; echo '{}' > $T",
        "T=packages/tcip-mcp/x.py; echo x > $T",
        # Multi-target tee: every destination is a write, not just the first.
        "printf hacked | tee /tmp/a packages/tcip-mcp/server.py",
        # In-place writers whose destination is a protected/breeder path.
        "install -m 644 evil.py packages/tcip-mcp/server.py",
        "touch packages/tcip-mcp/evil.py",
        "ln -sf /dev/null packages/tcip-mcp/server.py",
        "rsync evil.py packages/tcip-mcp/server.py",
        "patch -p1 packages/tcip-mcp/server.py < evil.diff",
        "install -m 644 evil.py /c/proj/labels/a.json",  # install into breeder data
        # Inline execution with an interposed flag before -c.
        "python -X utf8 -c \"open('packages/x','w').write('x')\"",
    ],
)
def test_bash_guard_denies_redirect_and_in_place_writer_bypasses(cmd):
    r = _run_guard(cmd)
    assert r.returncode == 2, r.stdout
    assert "deny" in r.stdout


def test_bash_guard_fails_closed_on_an_opaque_target_off_an_allow_listed_prefix():
    # An allow-listed read prefix runs with no prompt, so a redirect into a target the guard cannot
    # resolve (an inherited environment variable) must fail closed rather than clobber unseen.
    assert _run_guard('cat evil.json > "$TCIP_TARGET"').returncode == 2
    assert _run_guard("grep -r x evil > $SECRET").returncode == 2
    # A concrete or literal-tailed target off the same prefix is seen and judged on its own path, so
    # it is not swept up by the fail-closed rule.
    assert _run_guard("cat evil > $HOME/scratch.txt").returncode == 0
    # A command that leads with assignments is not an allow-listed prefix, so Claude Code prompts on
    # it; the guard leaves that defense-in-depth path to the prompt rather than over-denying here.
    assert _run_guard("A=/tmp/x; echo hi > $A").returncode == 0


@pytest.mark.parametrize(
    "cmd",
    [
        # Reading a protected source and writing a free destination is legitimate: the destination
        # is what is classified, not any protected path merely named.
        "rsync -a packages/tcip-mcp/ /tmp/backup/",
        "install -m 644 packages/tcip-mcp/server.py /tmp/server.py",
        "ln -s packages/tcip-mcp/server.py /tmp/server.py",
        "cp packages/tcip-mcp/server.py /tmp/x.py",
        "cat packages/tcip-mcp/server.py > /tmp/out.txt",  # allow-listed prefix, but a concrete free target
        # A breeder editing their own project file that shares a repo-root basename is not
        # platform code: anchoring the single-file rules to the repo root admits it.
        "echo x > /c/Users/breeder/tcip-projects/currant/README.md",
        "echo x > /c/Users/breeder/tcip-projects/currant/pyproject.toml",
        # The interpreter option scan ends at -m / the script, so pytest's own -c is not inline exec.
        "python -m pytest -c pyproject.toml",
        # A redirect to a concrete non-protected target, even via a resolved local variable.
        "LOG=/tmp/x.log; echo hi > $LOG",
    ],
)
def test_bash_guard_admits_valid_work_past_the_redirect_grammar(cmd):
    r = _run_guard(cmd)
    assert r.returncode == 0, r.stdout
    assert r.stdout.strip() == ""


@pytest.mark.parametrize(
    "cmd",
    [
        "python -X utf8 -c \"open('packages/x','w')\"",  # interposed flag, PowerShell tool
        'Set-Content C:\\proj\\la"bel"s\\a.json "x"',  # quote insertion into breeder data
    ],
)
def test_ps_guard_denies_redirect_and_quote_insertion_bypasses(cmd):
    r = _run_ps_guard(cmd)
    assert r.returncode == 2, r.stdout
    assert "deny" in r.stdout


@pytest.mark.parametrize(
    "cmd",
    [
        # A breeder editing their own project file sharing a repo-root basename is not platform
        # code: anchoring the single-file rules to the repo root admits it.
        'Set-Content C:\\Users\\breeder\\tcip-projects\\currant\\README.md "x"',
        'Set-Content C:\\Users\\breeder\\tcip-projects\\currant\\pyproject.toml "x"',
        # Copying a protected source to a free destination is a read, not a platform mutation.
        "Copy-Item packages\\tcip-mcp\\server.py -Destination C:\\tmp\\x.py",
    ],
)
def test_ps_guard_admits_valid_work_past_the_redirect_grammar(cmd):
    r = _run_ps_guard(cmd)
    assert r.returncode == 0, r.stdout
    assert r.stdout.strip() == ""


def test_materialized_fence_is_written_to_a_private_directory_not_a_fixed_shared_path():
    # The live permission fence must not be a fixed, pre-createable name in the shared temp root;
    # each spawn materializes it into its own process-private directory instead.
    import tempfile

    p1 = pty_host._materialize_fence_settings()
    p2 = pty_host._materialize_fence_settings()
    assert p1 is not None and p2 is not None
    shared_root = Path(tempfile.gettempdir()).resolve()
    assert p1.parent.resolve() != shared_root, "fence still written directly into the shared temp root"
    assert "tcip_fence_" in p1.parent.name
    assert p1.parent != p2.parent, "each spawn must get its own private directory"
    assert p1.is_file()
    cfg = json.loads(p1.read_text(encoding="utf-8"))
    assert "Edit(packages/**)" in cfg["permissions"]["deny"]


def test_guards_resolve_variable_indirection_into_in_place_writers():
    # A destination named through a variable is resolved and classified, so the refactor keeps the
    # coverage the old protected-anywhere check gave cp/Set-Content indirection.
    assert _run_guard("DEST=packages/tcip-mcp/x.py; cp evil.py $DEST").returncode == 2
    assert _run_ps_guard("$t='packages\\tcip-mcp\\x.py'; Set-Content $t 'hacked'").returncode == 2


def test_classify_drops_repo_rules_in_production_mode_keeping_breeder_data():
    # In production the platform is installed with no repo tree, so the repo rules fall away (a
    # breeder's own README is theirs to edit) while breeder data and trait-state stay protected.
    from tcip_web import agent_fence_rules as fr

    root = fr.repo_root()
    assert fr.classify("packages/tcip-mcp/x.py", root=root, mode="prod") is None
    assert fr.classify("README.md", root=root, mode="prod") is None
    assert fr.classify("/c/proj/labels/a.json", root=root, mode="prod") == "breeder"
    assert fr.classify("/c/proj/.tcip/state/trait_specs/x.json", root=root, mode="prod") == "breeder"
    # Development mode keeps the repo rules on.
    assert fr.classify("packages/tcip-mcp/x.py", root=root, mode="dev") == "protected"


# ── the tokenizer: editors denied as a class, redirects and prefixes token-aware ─


@pytest.mark.parametrize(
    "cmd",
    [
        "ed packages/tcip-mcp/server.py",
        "ex -c 'w packages/tcip-mcp/server.py' scratch",
        "ex -R packages/tcip-mcp/server.py",
        "ex -sc 'w' packages/tcip-mcp/server.py",
    ],
)
def test_guard_denies_a_line_editor_regardless_of_its_arguments(cmd):
    # An editor names its write target inside its own command script, not as a plain argument, so
    # it is denied as a verb (the delete rule's own precedent) rather than judged by target.
    r = _run_guard(cmd)
    assert r.returncode == 2, r.stdout
    assert "deny" in r.stdout


@pytest.mark.parametrize(
    "cmd",
    [
        "cp '/tmp/a|b' packages/tcip-mcp/server.py",
        "cp '/tmp/a>b' packages/tcip-mcp/server.py",
    ],
)
def test_guard_denies_a_destination_reached_past_a_quoted_separator_or_redirect(cmd):
    # A quoted separator or redirect character inside the source argument stays inside its own
    # token, so the real destination that follows it is still read and classified.
    r = _run_guard(cmd)
    assert r.returncode == 2, r.stdout
    assert "deny" in r.stdout


@pytest.mark.parametrize(
    "cmd",
    [
        "env cp a packages/tcip-mcp/server.py",
        "busybox cp a packages/tcip-mcp/server.py",
        "nice cp a packages/tcip-mcp/server.py",
        "time cp a packages/tcip-mcp/server.py",
        "stdbuf -o0 cp a packages/tcip-mcp/server.py",
        "command cp a packages/tcip-mcp/server.py",
        "env FOO=1 BAR=2 cp a packages/tcip-mcp/server.py",
    ],
)
def test_guard_denies_a_writer_run_through_a_transparent_prefix(cmd):
    # busybox/command/env/nice/time/stdbuf run the real verb unchanged, so the verb underneath a
    # leading wrapper (env's own NAME=value assignments, stdbuf's own flags) is what gets classified.
    r = _run_guard(cmd)
    assert r.returncode == 2, r.stdout
    assert "deny" in r.stdout


def test_guard_admits_a_writer_run_through_a_transparent_prefix_to_a_free_destination():
    # The same wrapper stripping must not over-deny: a destination outside platform internals and
    # breeder data is still free, wrapper or not.
    r = _run_guard("env FOO=1 cp a /tmp/b")
    assert r.returncode == 0, r.stdout


@pytest.mark.parametrize(
    "cmd",
    [
        "cat evil.json > 'unclosed",
        "cat evil.json > /tmp/scratch.txt\\",
    ],
)
def test_guard_denies_a_command_the_tokenizer_cannot_parse(cmd):
    # An unclosed quote or a trailing backslash inside one leaves no reliable parse to judge, so
    # the guard denies by name rather than falling open on an uncaught tokenizer exception.
    r = _run_guard(cmd)
    assert r.returncode == 2, r.stdout
    assert "deny" in r.stdout


def test_guard_reads_a_multiline_commands_later_line_as_its_own_segment():
    # A newline is bash's own statement separator; a write on a later line must not hide behind
    # an unrelated first line the way it would if the tokenizer silently dropped the newline.
    r = _run_guard("echo hi\ntouch packages/tcip-mcp/x.py")
    assert r.returncode == 2, r.stdout
    assert "deny" in r.stdout


def test_guard_reads_a_subshells_verb_as_a_verb():
    # A subshell's parentheses are segment boundaries, so the verb inside one is read as a verb
    # rather than swallowed into an unrecognized leading token.
    r = _run_guard("(cp evil.py packages/tcip-mcp/x.py)")
    assert r.returncode == 2, r.stdout
    assert "deny" in r.stdout


def test_guard_reads_a_brace_groups_verb_as_a_verb():
    # A brace group's braces are segment boundaries too, the same as a subshell's parentheses.
    r = _run_guard("{ cp evil.py packages/tcip-mcp/x.py; }")
    assert r.returncode == 2, r.stdout
    assert "deny" in r.stdout


def test_guard_admits_a_brace_groups_writer_to_a_free_destination():
    r = _run_guard("{ cp evil.py /tmp/scratch.txt; }")
    assert r.returncode == 0, r.stdout


def test_guard_reads_a_destination_past_a_command_substitution():
    # $( ... ) stays inside the segment that encloses it rather than opening a fresh statement,
    # so the real destination that follows it is still read and classified.
    r = _run_guard("cp a $(echo ignored) packages/tcip-mcp/server.py")
    assert r.returncode == 2, r.stdout
    assert "deny" in r.stdout


def test_guard_admits_a_command_substitution_ahead_of_a_free_destination():
    r = _run_guard("cp a $(echo ignored) /tmp/scratch.txt")
    assert r.returncode == 0, r.stdout


def test_guard_denies_an_unbalanced_command_substitution():
    # No closing paren leaves no reliable parse to judge, so the guard denies by name rather
    # than guessing where the substitution would have ended.
    r = _run_guard("cp a $(echo unterminated packages/tcip-mcp/server.py")
    assert r.returncode == 2, r.stdout
    assert "deny" in r.stdout


def test_guard_treats_an_escaped_newline_as_a_continuation_not_a_break():
    # A backslash immediately before a newline joins the two lines into one statement, so a
    # write on the continued line is still read as part of the same segment as its verb.
    r = _run_guard("touch \\\npackages/tcip-mcp/x.py")
    assert r.returncode == 2, r.stdout
    assert "deny" in r.stdout


@pytest.mark.parametrize(
    "cmd",
    [
        "nice -n 5 cp a packages/tcip-mcp/server.py",
        "nice -5 cp a packages/tcip-mcp/server.py",
        "env -i cp a packages/tcip-mcp/server.py",
        "env -u PATH cp a packages/tcip-mcp/server.py",
        "command -p cp a packages/tcip-mcp/server.py",
        "command -v cp a packages/tcip-mcp/server.py",
        "time -p cp a packages/tcip-mcp/server.py",
    ],
)
def test_guard_denies_a_writer_run_through_a_wrappers_own_option(cmd):
    # Each wrapper's own option is consumed before the verb underneath is read, so a writer
    # cannot hide behind the wrapper's flag.
    r = _run_guard(cmd)
    assert r.returncode == 2, r.stdout
    assert "deny" in r.stdout


@pytest.mark.parametrize(
    "cmd",
    [
        "nice -n 5 cp a /tmp/scratch.txt",
        "env -i cp a /tmp/scratch.txt",
        "command -p cp a /tmp/scratch.txt",
        "time -p cp a /tmp/scratch.txt",
    ],
)
def test_guard_admits_a_writer_run_through_a_wrappers_own_option_to_a_free_destination(cmd):
    r = _run_guard(cmd)
    assert r.returncode == 0, r.stdout


def test_guard_denies_a_writer_wrapped_through_env_or_busybox_or_command_to_an_editor():
    # A wrapper hides the editor the same way it hides a writer, so both are read through the
    # same stripped token list rather than a raw-string regex.
    for cmd in ("env ed packages/tcip-mcp/server.py",
                "busybox ed packages/tcip-mcp/server.py",
                "command ed packages/tcip-mcp/server.py",
                "env rm -rf packages/tcip-mcp/server.py"):
        r = _run_guard(cmd)
        assert r.returncode == 2, (cmd, r.stdout)
        assert "deny" in r.stdout


def test_bash_guard_windows_separator_parity():
    # An unquoted backslash is not a path separator to Git Bash, which drops it, so the argument
    # names a flat file, not a path into packages/; a quoted one is preserved and does name it.
    assert _run_guard("cp a packages\\x.py").returncode == 0
    r = _run_guard("cp a 'packages\\x.py'")
    assert r.returncode == 2, r.stdout
    assert "deny" in r.stdout

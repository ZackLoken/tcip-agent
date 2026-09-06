"""Agreement between the three layers of the in-app agent fence.

The fence is one boundary declared in three places: the settings deny/allow lists, the Bash
guard, and the PowerShell guard. These tests derive their expectations from the settings file
and from the guards' real accept/reject decisions (each guard run as the hook runs it, over
stdin), so a rule that exists in one layer and not another is visible here rather than being
restated identically on both sides of an assertion.
"""

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

DENY = "deny"
ALLOW = "allow"


def _settings() -> dict:
    return json.loads(FENCE.read_text(encoding="utf-8"))


def _verdict(guard: Path, command: str) -> str:
    """The guard's real decision for one command: ``deny`` (exit 2 + a deny payload) or ``allow``.

    Anything else (a crash, a deny payload without the exit code, output on an allow) is an
    unusable verdict and raises, so a broken guard can never read as an allow.
    """
    payload = json.dumps({"tool_input": {"command": command}})
    r = subprocess.run(
        [sys.executable, str(guard)], input=payload, capture_output=True, text=True, timeout=60
    )
    if r.returncode == 2 and '"permissionDecision": "deny"' in r.stdout:
        return DENY
    if r.returncode == 0 and r.stdout.strip() == "":
        return ALLOW
    raise AssertionError(
        f"{guard.name} returned no usable verdict for {command!r}: "
        f"rc={r.returncode} stdout={r.stdout!r} stderr={r.stderr!r}"
    )


# -- the settings deny list against both guards' decisions ----------------


def _edit_deny_rules() -> list[str]:
    return [r for r in _settings()["permissions"]["deny"] if r.startswith("Edit(")]


def _sample_path(rule: str) -> str:
    """A concrete file path the ``Edit(...)`` rule covers, for feeding to the shell guards."""
    target = rule[len("Edit(") : -1]
    return target[: -len("/**")] + "/fence-sample.txt" if target.endswith("/**") else target


def test_every_edit_deny_rule_is_fenced_in_both_shells():
    """Whatever the settings deny for the Edit tool, both shell guards must refuse to write.

    Derived from the settings file rather than a hand-written path list, so a deny rule the
    guards do not know about is a failure here rather than an invisible hole.
    """
    rules = _edit_deny_rules()
    assert len(rules) >= 10, rules
    unfenced = set()
    for rule in rules:
        sample = _sample_path(rule)
        bash_cmd = f"echo x > {sample}"
        ps_cmd = 'Set-Content {} "x"'.format(sample.replace("/", "\\"))
        if _verdict(GUARD, bash_cmd) != DENY or _verdict(PS_GUARD, ps_cmd) != DENY:
            unfenced.add(rule)
    assert unfenced == set()


def _fence_copy(tmp_path: Path, deny_rules: list[str] | None) -> Path:
    """A runnable copy of the fence directory declaring ``deny_rules``, or (``None``) nothing."""
    dest = tmp_path / "fence"
    dest.mkdir()
    for script in FENCE.parent.glob("agent_*.py"):
        (dest / script.name).write_bytes(script.read_bytes())
    if deny_rules is not None:
        cfg = _settings()
        cfg["permissions"]["deny"] = deny_rules
        (dest / FENCE.name).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return dest


def test_a_path_added_to_the_declaration_is_fenced_in_both_shells_without_editing_a_guard(tmp_path):
    """The guards read the declaration, so a rule added to it lands in both shells at once.

    The equivalence between the two shells then holds by construction: neither guard carries a
    path list of its own for the other to drift from.
    """
    fence = _fence_copy(tmp_path, ["Edit(vineyard/**)", "Edit(FIELD-NOTES.txt)"])
    for sample in ("vineyard/plan.txt", "FIELD-NOTES.txt"):
        assert _verdict(fence / GUARD.name, f"echo x > {sample}") == DENY, sample
        ps_cmd = 'Set-Content {} "x"'.format(sample.replace("/", "\\"))
        assert _verdict(fence / PS_GUARD.name, ps_cmd) == DENY, sample
    # Only what the declaration names is fenced: a path it does not name still admits a write.
    assert _verdict(fence / GUARD.name, "echo x > orchard/plan.txt") == ALLOW
    assert _verdict(fence / PS_GUARD.name, 'Set-Content orchard\\plan.txt "x"') == ALLOW


def test_a_guard_that_cannot_read_the_declaration_refuses_instead_of_guessing(tmp_path):
    """With no declaration to read, a guard cannot tell platform code from a breeder's project.

    It refuses rather than falling through, so a lost or unparseable declaration is a visible,
    fixable stop instead of a fence that silently stopped protecting anything.
    """
    fence = _fence_copy(tmp_path, None)
    assert _verdict(fence / GUARD.name, "echo x > /tmp/scratch.txt") == DENY
    assert _verdict(fence / PS_GUARD.name, 'Set-Content $env:TEMP\\scratch.txt "x"') == DENY


# -- the two guards' decisions on the same intent -------------------------

# One intent, spelled for each shell, with the decision the fence owes it. Both spellings run
# through their real guard, so the assertion is about decisions, not about regex source text.
_INTENTS = [
    (
        "write into packages",
        "echo x > packages/tcip-mcp/y.py",
        "'x' > packages\\tcip-mcp\\y.py",
        DENY,
    ),
    (
        "write into the skills tree",
        "echo x > .github/skills/a.md",
        'Set-Content .github\\skills\\a.md "x"',
        DENY,
    ),
    ("append to the build config", "echo x >> pyproject.toml", 'Add-Content pyproject.toml "x"', DENY),
    ("delete a label file", "rm /c/proj/labels/a.json", "Remove-Item C:\\proj\\labels\\a.json", DENY),
    (
        "truncate an annotation",
        "truncate -s 0 /c/proj/annotations/a.json",
        "Clear-Content C:\\proj\\annotations\\a.json",
        DENY,
    ),
    (
        "relocate breeder data",
        "mv /c/proj/labels/a.json /tmp/x.json",
        "Move-Item C:\\proj\\labels\\a.json C:\\out\\x.json",
        DENY,
    ),
    (
        "overwrite a prediction",
        "echo x > /c/proj/predictions/a.json",
        "'x' > C:\\proj\\predictions\\a.json",
        DENY,
    ),
    (
        "overwrite the image status store",
        "echo x > /c/proj/.tcip/state/image_status.json",
        'Set-Content C:\\proj\\.tcip\\state\\image_status.json "x"',
        DENY,
    ),
    (
        "rewrite the fence settings",
        "echo x > agent_terminal.settings.json",
        'Set-Content agent_terminal.settings.json "x"',
        DENY,
    ),
    ("inline interpreter", 'python -c "print(1)"', 'python -c "print(1)"', DENY),
    ("nested cmd shell", 'cmd /c "del x"', 'cmd /c "del x"', DENY),
    ("encoded shell payload", "powershell -e U2V0", "powershell -EncodedCommand U2V0", DENY),
    (
        "list a protected directory",
        "ls packages/tcip-web",
        "Get-ChildItem packages\\tcip-web",
        ALLOW,
    ),
    (
        "search platform source",
        "grep -rn def packages/tcip-mcp",
        "Select-String -Pattern def -Path packages\\tcip-mcp\\server.py",
        ALLOW,
    ),
    (
        "send a diagnostic to a scratch file",
        "tcip doctor /c/proj > /tmp/fence-check.log",
        "tcip doctor C:\\proj > $env:TEMP\\fence-check.log",
        ALLOW,
    ),
    (
        "back a label file up elsewhere",
        "cp /c/proj/labels/a.json /tmp/backup-a.json",
        "Copy-Item C:\\proj\\labels\\a.json C:\\out\\backup-a.json",
        ALLOW,
    ),
    (
        "read a note named after a delete verb",
        "cat /c/proj/annotations/rm-queue-notes.txt",
        "Get-Content C:\\proj\\annotations\\rm-queue-notes.txt",
        ALLOW,
    ),
]


@pytest.mark.parametrize(
    ("bash_cmd", "ps_cmd", "expected"),
    [(b, p, e) for _, b, p, e in _INTENTS],
    ids=[intent for intent, _, _, _ in _INTENTS],
)
def test_both_shells_reach_the_same_verdict_for_one_intent(bash_cmd, ps_cmd, expected):
    """Switching shells must not change what the fence permits, in either direction.

    A denied form that only one guard denies is a bypass; an admitted form that only one guard
    admits is a rail refusing legitimate work in one shell.
    """
    bash_verdict = _verdict(GUARD, bash_cmd)
    ps_verdict = _verdict(PS_GUARD, ps_cmd)
    assert bash_verdict == expected, bash_cmd
    assert ps_verdict == expected, ps_cmd


def test_the_intent_table_covers_both_verdicts():
    """Neither half of the table is empty, so a guard that decided every command the same way
    would still be caught by the parametrized cases above."""
    verdicts = {expected for _, _, _, expected in _INTENTS}
    assert verdicts == {DENY, ALLOW}


# -- the git fence, which is split across two layers ----------------------

# The destructive subcommands the fence claims to stop plus the read-only ones the settings
# grant. Which of these is actually fenced is measured below, never assumed.
_GIT_SUBCOMMANDS = [
    "push",
    "commit",
    "reset",
    "checkout",
    "clean",
    "status",
    "log",
    "diff",
    "show",
    "blame",
]


def _git_verbs_denied_in_settings() -> set[str]:
    verbs = set()
    for rule in _settings()["permissions"]["deny"]:
        if rule.startswith("Bash(git ") and rule.endswith(":*)"):
            verbs.add(rule[len("Bash(git ") : -len(":*)")])
    return verbs


def test_powershell_guard_fences_exactly_the_git_verbs_the_settings_deny():
    """The two layers of the git fence must name the same verbs.

    Bash is fenced by the settings' deny prefixes and PowerShell by its own guard, so nothing
    compares them unless a test does. A verb dropped from either side is a shell where the
    other shell's denial does not apply.
    """
    from_settings = _git_verbs_denied_in_settings()
    denied_by_guard = {v for v in _GIT_SUBCOMMANDS if _verdict(PS_GUARD, f"git {v} x") == DENY}
    assert from_settings, "the settings declare no dangerous git verbs at all"
    assert denied_by_guard == from_settings


@pytest.mark.parametrize(
    "cmd",
    [
        'git log --grep="reset"',
        "git diff --stat HEAD",
        "git status --porcelain",
    ],
)
def test_read_only_git_stays_admitted_in_powershell(cmd):
    """A read-only git command keeps working even when it names a fenced verb as an argument."""
    assert _verdict(PS_GUARD, cmd) == ALLOW


# -- the network-egress side of the fence ---------------------------------

# Shell verbs that move bytes off the machine, a breeder's labels included. Neither guard
# fences them, so the absence of a pre-approval is what keeps them behind a human prompt.
_EGRESS_VERBS = {
    "curl",
    "wget",
    "scp",
    "sftp",
    "ssh",
    "nc",
    "Invoke-WebRequest",
    "iwr",
    "Invoke-RestMethod",
    "irm",
    "Start-BitsTransfer",
}


def _allowed_shell_verbs() -> set[str]:
    verbs = set()
    for rule in _settings()["permissions"]["allow"]:
        for tool in ("Bash(", "PowerShell("):
            if rule.startswith(tool):
                inner = rule[len(tool) : -1]
                verbs.add(inner.split(":")[0].split()[0])
    return verbs


def test_no_shell_egress_verb_is_preapproved_by_the_fence():
    """The web-research grant is domain-scoped to academic hosts through WebFetch alone.

    A shell fetch or upload is not domain-scoped by anything, so pre-approving one in the allow
    list would hand the agent unreviewed egress, breeder data included. The allow list stays
    free of those verbs and each one keeps reaching a human approval prompt.
    """
    allowed = _allowed_shell_verbs()
    assert allowed, "the fence grants no shell commands at all"
    assert allowed & _EGRESS_VERBS == set()

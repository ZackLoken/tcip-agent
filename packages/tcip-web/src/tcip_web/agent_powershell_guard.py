"""PreToolUse PowerShell guard for the FENCED in-app agent terminal.

The sibling ``agent_bash_guard.py`` guards the Bash tool, but on Windows the fenced agent
also has a **PowerShell** tool — and it was entirely ungoverned (no deny rule, no hook), so
an agent could sidestep the whole fence with ``Set-Content packages\\...`` or ``Remove-Item``.
A real in-app session did exactly that ("switch to PowerShell — no such hook"). This hook
closes that bypass by mirroring the Bash guard for PowerShell.

Honest scope (unchanged): a guardrail, not a sandbox. It closes the *direct* bypasses —
full cmdlets AND their aliases (``sc``/``ri``/…), inline/encoded execution, and writing the
fence's own files — but a determined agent can still evade a string matcher, most notably by
``cd``-ing into a protected directory and writing a *relative* path (the guard is stateless
and cwd-blind), or by assembling a path from fragments. Those residuals are accepted; real
isolation is a sandbox (the platform's stated next step). The point here is to stop
accidental/casual mutation and every *trivial* deliberate bypass.

Stdlib only. It only ever DENIES; anything it can't classify falls through to the normal
permission flow (a bug here fails open to prompting, never to a broken terminal).
"""

from __future__ import annotations

import json
import re
import sys

# Repo-internal paths the fenced agent must never write to. Mirrors the Bash guard's
# ``_PROTECTED`` (kept in sync by ``test_agent_fence.py``), accepting a leading backslash too
# since PowerShell paths use ``\``. The fence's OWN files are included by basename so that
# writing/deleting them is blocked even via a relative path after a ``cd`` — that is the
# self-modification chain that would otherwise permanently disable the guard. Case-insensitive.
_PROTECTED = re.compile(
    r"(?:^|[\s/\\'\"])(?:packages|tests|scripts|\.github|\.claude)\b"
    r"|CLAUDE\.md|\.mcp\.json|pyproject\.toml|package\.json|tsconfig\.json|\.gitignore"
    r"|agent_bash_guard\.py|agent_powershell_guard\.py"
    r"|agent_terminal\.settings\.json|tcip_agent_fence\.settings\.json",
    re.IGNORECASE,
)

# Start of a statement / pipeline segment — where an invoked verb (or its alias) appears.
# Anchoring the SHORT aliases here stops them false-matching a bareword buried in an argument
# (e.g. ``Get-Content del-notes.txt`` must not read as the ``del`` alias).
_STMT = r"(?:^|[\n;|&(){}]\s*)"

# Write / create cmdlets and .NET writers — matched anywhere (the full names are unambiguous).
# Only bites when paired with a protected path (see main()).
_WRITE_OP = re.compile(
    r">>?"  # > and >> redirection
    r"|\bTee-Object\b"
    r"|\b(?:Set|Add|Clear)-Content\b"
    r"|\bOut-File\b"
    r"|\b(?:New|Move|Copy|Rename)-Item\b"
    r"|\b(?:Set|New)-ItemProperty\b"
    r"|\[(?:System\.)?IO\.(?:File|Directory)\]::(?:Write|Create|Append|Move|Copy|Replace|Open)\w*"
    r"|\bStreamWriter\b",
    re.IGNORECASE,
)
# Write / create cmdlet ALIASES — anchored to statement position (see _STMT).
_WRITE_ALIAS = re.compile(
    _STMT + r"(?:sc|ac|clc|ni|cpi|cp|copy|mi|mv|move|rni|ren|sp|spi)\b",
    re.IGNORECASE,
)

# Deletes — blocked UNCONDITIONALLY, mirroring the Bash fence's blanket ``Bash(rm:*)`` /
# ``Bash(rmdir:*)`` deny: the agent mutates data through audited MCP tools, not raw shell
# deletion (of platform code OR of a breeder's labels).
_DELETE_OP = re.compile(
    r"\bRemove-Item\b|\bRemove-ItemProperty\b|\[(?:System\.)?IO\.(?:File|Directory)\]::Delete",
    re.IGNORECASE,
)
_DELETE_ALIAS = re.compile(_STMT + r"(?:ri|rm|rmdir|rd|del|erase)\b", re.IGNORECASE)

# Inline / arbitrary code execution — blocked unconditionally. Covers Invoke-Expression, a
# spawned interpreter (``python -c``…), AND a NESTED shell (``powershell -EncodedCommand``,
# ``pwsh -Command``, ``cmd /c``) whose payload the guard can't see through.
_INLINE_INTERP = re.compile(
    r"\bInvoke-Expression\b|\biex\b"
    r"|\b(?:python3?|node|perl|ruby|deno|bun)\b\s+-\w*[ce]"
    r"|\b(?:powershell|pwsh)\b[\s\S]*?-(?:e|ec|enc|encodedcommand|command|c)\b"
    r"|\bcmd\b[\s\S]*?/c\b",
    re.IGNORECASE,
)

# Dangerous git — blocked unconditionally, mirroring the Bash fence deny of
# ``git push/commit/reset/checkout/clean``. Anchored to the subcommand right after ``git`` so
# a benign ``git log --grep="reset"`` isn't caught (same scope as the Bash prefix deny).
_GIT_DANGER = re.compile(
    r"\bgit\s+(?:push|commit|reset|checkout|clean)\b",
    re.IGNORECASE,
)


def _deny(reason: str) -> None:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        )
    )
    # Exit 2 as well: some Claude Code versions gate on the exit code, others on the JSON
    # decision — emitting both makes the block robust across versions.
    sys.exit(2)


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # unparseable → fall through to normal permission flow
    cmd = (data.get("tool_input") or {}).get("command", "")
    if not isinstance(cmd, str) or not cmd.strip():
        sys.exit(0)

    if _INLINE_INTERP.search(cmd):
        _deny("Inline / nested / arbitrary code execution is blocked in the agent terminal — use the TCIP tools.")
    if _GIT_DANGER.search(cmd):
        _deny("Dangerous git (push/commit/reset/checkout/clean) is blocked in the agent terminal.")
    if _DELETE_OP.search(cmd) or _DELETE_ALIAS.search(cmd):
        _deny("File deletion via the shell is blocked — the agent mutates data through the audited TCIP tools.")
    if (_WRITE_OP.search(cmd) or _WRITE_ALIAS.search(cmd)) and _PROTECTED.search(cmd):
        _deny("Writing into platform internals via PowerShell is blocked — the agent edits projects, not platform code.")

    sys.exit(0)


if __name__ == "__main__":
    main()

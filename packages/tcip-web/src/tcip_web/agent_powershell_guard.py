"""PreToolUse PowerShell guard for the fenced in-app agent terminal.

The sibling ``agent_bash_guard.py`` guards the Bash tool, but on Windows the fenced agent
also has a PowerShell tool, and it was entirely ungoverned (no deny rule, no hook), so
an agent could sidestep the whole fence with ``Set-Content packages\\...`` or ``Remove-Item``.
A real in-app session did exactly that ("switch to PowerShell, no such hook"). This hook
closes that bypass by mirroring the Bash guard for PowerShell.

Honest scope (unchanged): a guardrail, not a sandbox. It closes the *direct* bypasses:
full cmdlets and their aliases (``sc``/``ri``/…), inline/encoded execution, and writing the
fence's own files, but a determined agent can still evade a string matcher, most notably by
``cd``-ing into a protected directory and writing a *relative* path (the guard is stateless
and cwd-blind), or by assembling a path from fragments. Those residuals are accepted; real
isolation is a sandbox (the platform's stated next step). The point here is to stop
accidental/casual mutation and every *trivial* deliberate bypass.

Which paths are protected, which hold a breeder's data, and what a refusal says come from
``agent_fence_rules``, shared with the Bash guard so the two shells fence one boundary.

Stdlib only. It only ever denies; anything it can't classify falls through to the normal
permission flow (a bug here fails open to prompting, never to a broken terminal).
"""

from __future__ import annotations

import json
import re
import sys

try:  # as a hook it runs as a bare script, with only its own directory importable
    from . import agent_fence_rules as fence_rules
except ImportError:
    import agent_fence_rules as fence_rules  # type: ignore[no-redef]

_STMT = fence_rules.STMT

# Write / create cmdlets and .NET writers, matched anywhere (the full names are unambiguous).
# Only bites when paired with a protected path (see main()).
_WRITE_OP = re.compile(
    r"\bTee-Object\b"
    r"|\b(?:Set|Add|Clear)-Content\b"
    r"|\bOut-File\b"
    r"|\b(?:New|Move|Copy|Rename)-Item\b"
    r"|\b(?:Set|New)-ItemProperty\b"
    r"|\[(?:System\.)?IO\.(?:File|Directory)\]::(?:Write|Create|Append|Move|Copy|Replace|Open)\w*"
    r"|\bStreamWriter\b",
    re.IGNORECASE,
)
# Write / create cmdlet aliases, anchored to statement position (see _STMT).
_WRITE_ALIAS = re.compile(
    _STMT + r"(?:sc|ac|clc|ni|cpi|cp|copy|mi|mv|move|rni|ren|sp|spi)\b",
    re.IGNORECASE,
)

# Deletes: blocked unconditionally, mirroring the Bash fence's blanket ``Bash(rm:*)`` /
# ``Bash(rmdir:*)`` deny: the agent mutates data through audited MCP tools, not raw shell
# deletion (of platform code OR of a breeder's labels).
_DELETE_OP = re.compile(
    r"\bRemove-Item\b|\bRemove-ItemProperty\b|\[(?:System\.)?IO\.(?:File|Directory)\]::Delete",
    re.IGNORECASE,
)
_DELETE_ALIAS = re.compile(_STMT + r"(?:ri|rm|rmdir|rd|del|erase)\b", re.IGNORECASE)

# The shared spawned-interpreter set plus this shell's own in-process evaluator, whose payload
# the guard can't see through. Blocked unconditionally.
_INLINE_INTERP = re.compile(
    r"\bInvoke-Expression\b|\biex\b|" + fence_rules.SPAWNED_INTERPRETER,
    re.IGNORECASE,
)

# Dangerous git: blocked unconditionally, mirroring the Bash fence deny of
# ``git push/commit/reset/checkout/clean``. Anchored to the subcommand right after ``git`` so
# a benign ``git log --grep="reset"`` isn't caught (same scope as the Bash prefix deny).
_GIT_DANGER = re.compile(
    r"\bgit\s+(?:push|commit|reset|checkout|clean)\b",
    re.IGNORECASE,
)


# A zero-byte label is not a negative without an explicit Complete, so truncating or overwriting
# one (``Clear-Content``, ``Set-Content``, a redirect) is the same harm as deleting it.
_BREEDER_DATA_TARGET = fence_rules.BREEDER_DATA_TARGET
# Write ops for the breeder-data check specifically: deliberately excludes Copy-Item/cpi/cp/copy
# and [IO.File]::Copy: this guard is stateless and can't tell a two-argument cmdlet's source from
# its destination, so "_WRITE_OP and _BREEDER_DATA_TARGET anywhere in the command" would deny a
# legitimate copy/backup of a breeder file to elsewhere, not just a copy into one. Move/Rename still
# relocate the tracked file even when named as the source, so they stay included. Mirrors the Bash
# guard's own choice to exempt cp from its breeder-data check entirely.
_BREEDER_DATA_WRITE_OP = re.compile(
    r"\bTee-Object\b"
    r"|\b(?:Set|Add|Clear)-Content\b"
    r"|\bOut-File\b"
    r"|\b(?:New|Move|Rename)-Item\b"
    r"|\b(?:Set|New)-ItemProperty\b"
    r"|\[(?:System\.)?IO\.(?:File|Directory)\]::(?:Write|Create|Append|Move|Replace|Open)\w*"
    r"|\bStreamWriter\b",
    re.IGNORECASE,
)
_BREEDER_DATA_WRITE_ALIAS = re.compile(
    _STMT + r"(?:sc|ac|clc|ni|mi|mv|move|rni|ren|sp|spi)\b",
    re.IGNORECASE,
)


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # unparseable → fall through to normal permission flow
    cmd = (data.get("tool_input") or {}).get("command", "")
    if not isinstance(cmd, str) or not cmd.strip():
        sys.exit(0)

    if _INLINE_INTERP.search(cmd):
        fence_rules.deny(fence_rules.INLINE_EXECUTION_MSG)
    if _GIT_DANGER.search(cmd):
        fence_rules.deny("Dangerous git (push/commit/reset/checkout/clean) is blocked in the agent terminal.")
    if _DELETE_OP.search(cmd) or _DELETE_ALIAS.search(cmd):
        fence_rules.deny(fence_rules.DELETE_MSG)
    if any(_BREEDER_DATA_TARGET.search(t) for t in fence_rules.redirect_targets(cmd)):
        fence_rules.deny(fence_rules.BREEDER_DATA_WRITE_MSG)
    if (_BREEDER_DATA_WRITE_OP.search(cmd) or _BREEDER_DATA_WRITE_ALIAS.search(cmd)) \
            and _BREEDER_DATA_TARGET.search(cmd):
        fence_rules.deny(fence_rules.BREEDER_DATA_WRITE_MSG)
    protected = fence_rules.protected_pattern()
    if any(protected.search(t) for t in fence_rules.redirect_targets(cmd)):
        fence_rules.deny(fence_rules.PROTECTED_WRITE_MSG)
    if (_WRITE_OP.search(cmd) or _WRITE_ALIAS.search(cmd)) and protected.search(cmd):
        fence_rules.deny(fence_rules.PROTECTED_WRITE_MSG)

    sys.exit(0)


if __name__ == "__main__":
    main()

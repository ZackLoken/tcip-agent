"""PreToolUse Bash guard for the fenced in-app agent terminal.

The Edit/Write deny-list can't see file writes performed *through the shell*
(`echo > packages/...`, `sed -i`, `python -c "open(...)"`). This PreToolUse hook reads
the tool call on stdin and denies Bash commands that write into platform internals, or
that run an inline interpreter at all.

Honest scope: a guardrail, not a sandbox. A determined user can obfuscate around these
patterns; the point is to stop *accidental / casual* repo edits via Bash. The primary
protection is still the Edit/Write deny-list plus `--permission-mode default` (which
surfaces un-allowed Bash to the human for approval).

Stdlib only, so it runs under whatever ``python`` the terminal inherits. It only ever
denies; anything it can't classify it lets fall through to the normal permission flow
(so a bug here fails open to prompting, never to a broken terminal).
"""

from __future__ import annotations

import json
import re
import sys

# Repo-internal paths the fenced agent must never write to (mirrors the settings deny). The
# fence's own files are included by basename so writing/deleting them is blocked even via a
# relative path after a ``cd``, the self-modification chain that would disable the guard.
# Kept in sync with the PowerShell guard by ``test_agent_fence.py``.
_PROTECTED = re.compile(
    r"(?:^|[\s/'\"])(?:packages|tests|scripts|\.github|\.claude)\b"
    r"|CLAUDE\.md|\.mcp\.json|pyproject\.toml|package\.json|tsconfig\.json|\.gitignore"
    r"|agent_bash_guard\.py|agent_powershell_guard\.py"
    r"|agent_terminal\.settings\.json|tcip_agent_fence\.settings\.json"
)
# A shell redirect that writes to a file: an optional leading fd number, ``>``/``>>``, then a
# target token. ``>&`` / ``2>&1`` duplicate a descriptor (no file) and are excluded by the
# ``(?!&)``, so ``2>&1`` / ``2>/dev/null`` on a read-only command no longer read as a write
# into whatever protected token the command line happens to name.
_REDIRECT = re.compile(r"\d*>>?(?!&)\s*(?P<target>[^\s;|&<>()]+)")
# ``tee [-a] FILE`` writes to FILE.
_TEE = re.compile(r"\btee\b\s+(?:-a\s+|--append\s+)?(?P<target>[^\s;|&<>()]+)")
# In-place / copy writers, matched coarsely (paired with a protected token in main()). These
# never appear in the read-only diagnostics the fence must let through.
_WRITE_OP = re.compile(r"\bsed\b\s+-i|\bcp\b|\bmv\b|\bdd\b")
# Start of a statement / pipeline segment: mirrors the PowerShell guard's ``_STMT`` so a delete
# verb is anchored to where a command actually starts, not to any substring (a path or grep
# pattern containing "unlink"/"shred"/"truncate" must not trip this).
_STMT = r"(?:^|[\n;|&(){}]\s*)"
# Deletes / truncates: blocked unconditionally, mirroring the PowerShell fence's unconditional
# _DELETE_OP/_DELETE_ALIAS: the agent mutates data through audited MCP tools, not raw shell
# deletion (of platform code OR of a breeder's labels).
_DELETE_OP = re.compile(_STMT + r"(?:rm|rmdir|unlink|shred|truncate)\b")
# ``... | xargs rm`` (with optional flags on either xargs or the trailing verb).
_XARGS_DELETE = re.compile(r"\bxargs\b\s+(?:-\S+\s+)*(?:rm|rmdir|unlink|shred)\b")
# ``find ... -delete`` / ``find ... -exec rm``, the sharpest hole of the three: ``Bash(find:*)``
# is allow-listed in agent_terminal.settings.json, so unlike the others this form runs with no
# human prompt at all if this guard doesn't deny it.
_FIND_DELETE = re.compile(r"\bfind\b[\s\S]*?\s-delete\b|\bfind\b[\s\S]*?-exec\s+rm\b")
# A redirect/tee target that names a breeder's annotation/label/prediction state, denied
# regardless of whether the file exists yet. An existence check would be filesystem state at
# hook time; the guard is stateless and cwd-blind, so it would silently stop firing after a
# ``cd`` (same honesty-about-scope the PowerShell guard already commits to).
_BREEDER_DATA_TARGET = re.compile(
    r"(?:^|[/\\])(?:annotations|labels|predictions)[/\\]|image_status\.json\b"
)
# Inline / nested / arbitrary code execution: a spawned interpreter (``python -c``…) or a
# nested shell (``bash -c``, ``sh -c``, ``powershell -EncodedCommand``, ``cmd /c``) whose
# payload the guard can't see through.
_INLINE_INTERP = re.compile(
    r"\b(?:python3?|node|perl|ruby|deno|bun)\b\s+-\w*[ce]"
    r"|\b(?:bash|sh|zsh)\b\s+-\w*c\b"
    r"|\b(?:powershell|pwsh)\b[\s\S]*?-(?:e|ec|enc|encodedcommand|command|c)\b"
    r"|\bcmd\b[\s\S]*?/c\b"
)


def _write_targets(cmd: str) -> list[str]:
    """Resolved file targets of redirects and ``tee``, only these gate the protected check.

    Executing a script under ``scripts/`` is passing an *argument*, not writing there; only a
    redirect/tee whose *target* lands under a protected dir is a mutation of platform code.
    """
    return [m.group("target") for m in _REDIRECT.finditer(cmd)] + [
        m.group("target") for m in _TEE.finditer(cmd)
    ]


_PROTECTED_WRITE_MSG = (
    "Writing into platform internals via the shell is blocked, the agent edits projects, not "
    "platform code. If this was a read-only diagnostic that got mis-flagged (e.g. "
    "`python scripts/doctor.py <root>`), that's a fence false-positive: file it with the "
    "claude_reports tool (category unexpected_behavior; include the exact command) so the fence "
    "can be fixed, do not route around it by editing platform files."
)
_DELETE_DENY_MSG = (
    "File deletion via the shell is blocked, the agent mutates data through the audited TCIP tools."
)
_BREEDER_DATA_WRITE_MSG = (
    "Writing or truncating a breeder's annotation, label, or prediction data via the shell is "
    "blocked, the agent mutates data through the audited TCIP tools."
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
    # Exit 2 as well: some Claude Code versions gate on the exit code, others on the
    # JSON decision, emitting both makes the block robust across versions.
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
        _deny("Inline interpreter execution is blocked in the agent terminal, use the TCIP tools.")
    if _DELETE_OP.search(cmd) or _XARGS_DELETE.search(cmd) or _FIND_DELETE.search(cmd):
        _deny(_DELETE_DENY_MSG)
    if any(_BREEDER_DATA_TARGET.search(t) for t in _write_targets(cmd)):
        _deny(_BREEDER_DATA_WRITE_MSG)
    if any(_PROTECTED.search(t) for t in _write_targets(cmd)):
        _deny(_PROTECTED_WRITE_MSG)
    if _WRITE_OP.search(cmd) and _PROTECTED.search(cmd):
        _deny(_PROTECTED_WRITE_MSG)

    sys.exit(0)


if __name__ == "__main__":
    main()

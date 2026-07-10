"""PreToolUse Bash guard for the FENCED in-app agent terminal.

The Edit/Write deny-list can't see file writes performed *through the shell*
(`echo > packages/...`, `sed -i`, `python -c "open(...)"`). This PreToolUse hook reads
the tool call on stdin and denies Bash commands that write into platform internals, or
that run an inline interpreter at all.

Honest scope: a guardrail, not a sandbox. A determined user can obfuscate around these
patterns; the point is to stop *accidental / casual* repo edits via Bash. The primary
protection is still the Edit/Write deny-list plus `--permission-mode default` (which
surfaces un-allowed Bash to the human for approval).

Stdlib only, so it runs under whatever ``python`` the terminal inherits. It only ever
DENIES; anything it can't classify it lets fall through to the normal permission flow
(so a bug here fails open to prompting, never to a broken terminal).
"""

from __future__ import annotations

import json
import re
import sys

# Repo-internal paths the fenced agent must never write to (mirrors the settings deny).
_PROTECTED = re.compile(
    r"(?:^|[\s/'\"])(?:packages|tests|scripts|\.github|\.claude)\b"
    r"|CLAUDE\.md|\.mcp\.json|pyproject\.toml|package\.json|tsconfig\.json|\.gitignore"
)
# Shell constructs that write to a file / run arbitrary code.
_WRITE_OP = re.compile(r">>?|\btee\b|\bsed\b\s+-i|\bcp\b|\bmv\b|\bdd\b")
_INLINE_INTERP = re.compile(r"\b(?:python3?|node|perl|ruby|deno|bun)\b\s+-\w*[ce]")


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
    # JSON decision — emitting both makes the block robust across versions.
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
        _deny("Inline interpreter execution is blocked in the agent terminal — use the TCIP tools.")
    if _WRITE_OP.search(cmd) and _PROTECTED.search(cmd):
        _deny("Writing into platform internals via the shell is blocked — the agent edits projects, not platform code.")

    sys.exit(0)


if __name__ == "__main__":
    main()

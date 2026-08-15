"""What the in-app agent fence protects, declared once for both shell guards.

``agent_bash_guard.py`` and ``agent_powershell_guard.py`` police the same boundary in two
syntaxes. The boundary itself, which repo paths are platform-internal, which paths hold a
breeder's data, and what a refusal tells the agent, lives here; each guard keeps only the
syntax that recognises a write in its own shell.

The platform-internal set is derived from ``agent_terminal.settings.json``'s ``Edit(...)`` deny
rules at hook time, so a path added there reaches both shells with nothing to keep in step by
hand, plus the fence's own files by basename. Those basenames are a shell-guard concern rather
than a settings one: the settings rules address the repo by path, and a relative write after a
``cd`` does not.

Path matching is case-insensitive in both shells. Case-insensitivity is a property of the
filesystem being protected, not of the shell doing the writing: on Windows, where the fenced
agent has a Bash tool and a PowerShell tool and the fence registers a guard for each,
``PACKAGES/x.py`` and ``packages/x.py`` are the same file.

``BREEDER_DATA_TARGET`` accepts a bare directory as well as a file under one: relocating or
emptying ``annotations/`` is the same harm as doing it to one file inside it.

The guards are stateless and cwd-blind, so every check here is a path-shape check, never an
existence check: an existence check would silently stop firing after a ``cd``.

Stdlib only, and importable both as ``tcip_web.agent_fence_rules`` and as a bare sibling
module, because the guards run as plain scripts under whatever ``python`` the terminal
inherits.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

FENCE_SETTINGS = Path(__file__).resolve().parent / "agent_terminal.settings.json"

# Writing or deleting one of these disables the fence itself, so they are protected by
# basename, reachable via a relative path after a ``cd``, not only by their repo path.
FENCE_OWN_FILES = (
    "agent_bash_guard.py",
    "agent_powershell_guard.py",
    "agent_fence_rules.py",
    "agent_terminal.settings.json",
    "tcip_agent_fence.settings.json",
)

# Start of a statement / pipeline segment, so a verb anchored here cannot false-match a
# bareword buried in an argument (a path or grep pattern naming "rm"/"del"/"mv").
STMT = r"(?:^|[\n;|&(){}]\s*)"

# A file-writing redirect, same syntax in both shells. ``(?!&)`` excludes ``2>&1`` / ``>&``,
# which duplicate a descriptor rather than writing a file.
REDIRECT = re.compile(r"\d*>>?(?!&)\s*(?P<target>[^\s;|&<>()]+)")

BREEDER_DATA_TARGET = re.compile(
    r"(?:^|[/\\])(?:annotations|labels|predictions)(?:[/\\]|[\s;|&)]|$)|image_status\.json\b",
    re.IGNORECASE,
)

# A spawned interpreter or nested shell whose payload no guard can see through; both shells can
# reach all of these, and each guard adds its own nested-shell forms on top.
SPAWNED_INTERPRETER = (
    r"\b(?:python3?|node|perl|ruby|deno|bun)\b\s+-\w*[ce]"
    r"|\b(?:powershell|pwsh)\b[\s\S]*?-(?:e|ec|enc|encodedcommand|command|c)\b"
    r"|\bcmd\b[\s\S]*?/c\b"
)

PROTECTED_WRITE_MSG = (
    "Writing into platform internals via the shell is blocked, the agent edits projects, not "
    "platform code. If this was a read-only diagnostic that got mis-flagged (e.g. "
    "`python scripts/doctor.py <root>`), that's a fence false-positive: file it with the "
    "claude_reports tool (category unexpected_behavior; include the exact command) so the fence "
    "can be fixed, do not route around it by editing platform files."
)
DELETE_MSG = (
    "File deletion via the shell is blocked, the agent mutates data through the audited TCIP tools."
)
BREEDER_DATA_WRITE_MSG = (
    "Writing or truncating a breeder's annotation, label, or prediction data via the shell is "
    "blocked, the agent mutates data through the audited TCIP tools."
)
INLINE_EXECUTION_MSG = (
    "Inline, nested, or encoded code execution is blocked in the agent terminal, use the TCIP tools."
)
UNREADABLE_DECLARATION_MSG = (
    "The agent terminal's permission declaration ({path}) could not be read, so the shell guard "
    "cannot tell which paths are platform-internal and refuses rather than guess. Restore the "
    "file to use the terminal."
)


def deny(reason: str) -> None:
    """Emit the PreToolUse deny decision for ``reason`` and exit."""
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
    # Some Claude Code versions gate on the exit code, others on the JSON decision.
    sys.exit(2)


def redirect_targets(cmd: str) -> list[str]:
    """Every file target a redirect in ``cmd`` writes to."""
    return [m.group("target") for m in REDIRECT.finditer(cmd)]


def _edit_deny_targets() -> list[str]:
    """The path targets of the fence settings' ``Edit(...)`` deny rules."""
    cfg = json.loads(FENCE_SETTINGS.read_text(encoding="utf-8"))
    rules = cfg["permissions"]["deny"]
    return [r[len("Edit(") : -1] for r in rules if r.startswith("Edit(") and r.endswith(")")]


def _path_alternative(target: str) -> str:
    """``target`` as a regex alternative that accepts either path separator."""
    return r"[/\\]".join(re.escape(segment) for segment in target.split("/"))


def protected_pattern() -> "re.Pattern[str]":
    """The matcher for platform-internal paths, derived from the fence settings.

    An ``Edit(<dir>/**)`` rule protects that directory wherever it appears as a path segment; an
    ``Edit(<file>)`` rule protects that name anywhere, since the guards are cwd-blind and a
    relative path after a ``cd`` names the same file. Denies the command outright when the
    declaration cannot be read: a guard that cannot see the boundary must refuse, not fall
    through to a matcher it invented.
    """
    try:
        targets = _edit_deny_targets()
    except (OSError, ValueError, KeyError, TypeError):
        targets = []
    if not targets:
        deny(UNREADABLE_DECLARATION_MSG.format(path=FENCE_SETTINGS))
    directories = [_path_alternative(t[: -len("/**")]) for t in targets if t.endswith("/**")]
    names = [_path_alternative(t) for t in targets if not t.endswith("/**")]
    alternatives = []
    if directories:
        alternatives.append(r"(?:^|[\s/\\'\"])(?:" + "|".join(directories) + r")\b")
    alternatives.extend(names)
    alternatives.extend(re.escape(name) for name in FENCE_OWN_FILES)
    return re.compile("|".join(alternatives), re.IGNORECASE)

"""PreToolUse Bash guard for the fenced in-app agent terminal.

The Edit/Write deny-list can't see file writes performed *through the shell*
(`echo > packages/...`, `sed -i`, `python -c "open(...)"`). This PreToolUse hook reads
the tool call on stdin and denies Bash commands that write into platform internals or a
breeder's data, or that run an inline interpreter at all.

Honest scope: a guardrail, not a sandbox. The one path where this guard is the sole gate is a
write that rides an allow-listed read prefix (`cat`, `ls`, `grep`, `git diff`) through a
redirect, which Claude Code runs with no prompt; the redirect grammar, target normalization, and
the fail-closed rule for an unresolvable target on such a prefix are airtight for that reason.
Every other write reaches a human approval prompt (its verb is not allow-listed), so the in-place
writer coverage here is defense-in-depth and deliberately not exhaustive. Real isolation from a
determined agent is the sandbox (the platform's stated next step); a `cd` into a protected
directory then a relative write, an inherited environment variable, and a hard link to a
protected file are accepted residuals of a command-only, cwd-blind guard.

What is protected, how a target is normalized, and what a refusal says come from
``agent_fence_rules``, shared with the PowerShell guard so the two shells fence one boundary.

Stdlib only, so it runs under whatever ``python`` the terminal inherits. It only ever
denies; anything it can't classify it lets fall through to the normal permission flow
(so a bug here fails open to prompting, never to a broken terminal).
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

# ``find``'s own write actions (``-fprint``/``-fprintf``/``-fls``) write to the FILE argument that
# follows, with no ``>``/``tee`` on the line, so they are matched by their find-specific flag name.
_FIND_WRITE_ACTION = re.compile(r"-f(?:print0?|printf|ls)\s+(?P<target>[^\s;|&<>()]+)")
# ``dd of=FILE`` names its output through a key=value argument, not a positional one.
_DD_OF = re.compile(r"\bdd\b[\s\S]*?\bof=(?P<target>[^\s;|&<>()]+)")

# Deletes / truncates: blocked unconditionally, mirroring the PowerShell fence. The agent mutates
# data through audited MCP tools, not raw shell deletion (of platform code OR a breeder's labels).
_DELETE_OP = re.compile(_STMT + r"(?:rm|rmdir|unlink|shred|truncate)\b")
_XARGS_DELETE = re.compile(r"\bxargs\b\s+(?:-\S+\s+)*(?:rm|rmdir|unlink|shred)\b")
_FIND_DELETE = re.compile(
    r"\bfind\b[\s\S]*?\s-delete\b"
    r"|\bfind\b[\s\S]*?-exec\s+(?:rm|rmdir|unlink|shred|truncate)\b"
)

_BREEDER_DATA_TARGET = fence_rules.BREEDER_DATA_TARGET
# ``mv`` relocates a breeder-data path (as source or destination) the way ``rm`` removes it, so it
# pairs with the breeder pattern anywhere; ``cp`` is excluded (copy, not relocation; dest-only below).
_MOVE_OP = re.compile(_STMT + r"mv\b")
_FIND_MOVE = re.compile(r"\bfind\b[\s\S]*?-exec\s+mv\b")

# The shared spawned-interpreter set plus this shell's own nested shells (``bash -c``…).
_NESTED_SHELL = re.compile(r"\b(?:bash|sh|zsh)\b\s+-\w*c\b")

# In-place / copy writers whose destination is a positional argument. cp/mv/install/rsync/ln take
# ``SRC... DEST`` (destination is the last non-flag token); touch/sed/patch name the file directly.
_SEG_SPLIT = re.compile(r"[;\n]|&&|\|\||[|&]")
_DEST_VERBS = ("cp", "mv", "install", "rsync", "ln")


def _segments(cmd: str) -> "list[str]":
    """Rough top-level statement/pipeline segments, so a leading verb can be read per segment."""
    return [s.strip() for s in _SEG_SPLIT.split(cmd) if s.strip()]


def _tee_targets(cmd: str) -> "list[str]":
    """Every file ``tee`` writes to (it accepts multiple output files)."""
    out = []
    for seg in _segments(cmd):
        tokens = re.split(r"[<>]", seg)[0].split()
        if tokens and tokens[0] == "tee":
            out.extend(t for t in tokens[1:] if not t.startswith("-"))
    return out


def _inplace_dests(cmd: str) -> "list[str]":
    """The file each in-place writer would write.

    For cp/mv/install/rsync/ln this is the destination (the last non-flag token), so a copy or move
    into a protected or breeder path is caught while reading a protected/breeder source and writing
    elsewhere is not. touch/sed/patch/dd name the file directly.
    """
    out: list[str] = []
    for seg in _segments(cmd):
        tokens = re.split(r"[<>]", seg)[0].split()
        if not tokens:
            continue
        verb, args = tokens[0], tokens[1:]
        non_flags = [t for t in args if not t.startswith("-")]
        if verb in _DEST_VERBS and non_flags:
            out.append(non_flags[-1])
        elif verb == "touch":
            out.extend(non_flags)
        elif verb == "sed" and any(t.startswith("-i") for t in args) and len(non_flags) >= 2:
            out.append(non_flags[-1])
        elif verb == "patch" and non_flags:
            out.append(non_flags[-1])
    for m in _DD_OF.finditer(cmd):
        out.append(m.group("target"))
    return out


def _write_targets(cmd: str) -> "list[str]":
    """Redirect, ``tee``, and ``find``-write targets: the airtight set, checked for both harms."""
    return (
        fence_rules.redirect_targets(cmd)
        + _tee_targets(cmd)
        + [m.group("target") for m in _FIND_WRITE_ACTION.finditer(cmd)]
    )


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # unparseable → fall through to normal permission flow
    cmd = (data.get("tool_input") or {}).get("command", "")
    if not isinstance(cmd, str) or not cmd.strip():
        sys.exit(0)

    root = fence_rules.repo_root()
    mode = fence_rules.fence_mode()

    if fence_rules.inline_exec(cmd) or _NESTED_SHELL.search(cmd):
        fence_rules.deny(fence_rules.INLINE_EXECUTION_MSG)
    if _DELETE_OP.search(cmd) or _XARGS_DELETE.search(cmd) or _FIND_DELETE.search(cmd):
        fence_rules.deny(fence_rules.DELETE_MSG)

    for target in _write_targets(cmd):
        kind = fence_rules.classify(fence_rules.resolve_token(target, cmd), root=root, mode=mode)
        if kind == "breeder":
            fence_rules.deny(fence_rules.BREEDER_DATA_WRITE_MSG)
        if kind == "protected":
            fence_rules.deny(fence_rules.PROTECTED_WRITE_MSG)

    for target in _inplace_dests(cmd):
        kind = fence_rules.classify(fence_rules.resolve_token(target, cmd), root=root, mode=mode)
        if kind == "protected":
            fence_rules.deny(fence_rules.PROTECTED_WRITE_MSG)
        if kind == "breeder":
            fence_rules.deny(fence_rules.BREEDER_DATA_WRITE_MSG)

    if (_MOVE_OP.search(cmd) or _FIND_MOVE.search(cmd)) and _BREEDER_DATA_TARGET.search(cmd):
        fence_rules.deny(fence_rules.BREEDER_DATA_WRITE_MSG)

    if fence_rules.leading_is_allow_listed(cmd, "Bash") and fence_rules.has_opaque_redirect_target(cmd):
        fence_rules.deny(fence_rules.DYNAMIC_TARGET_MSG)

    sys.exit(0)


if __name__ == "__main__":
    main()

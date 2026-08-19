"""PreToolUse PowerShell guard for the fenced in-app agent terminal.

The sibling ``agent_bash_guard.py`` guards the Bash tool, but on Windows the fenced agent
also has a PowerShell tool, and it was entirely ungoverned (no deny rule, no hook), so
an agent could sidestep the whole fence with ``Set-Content packages\\...`` or ``Remove-Item``.
This hook mirrors the Bash guard for PowerShell.

Honest scope (unchanged): a guardrail, not a sandbox. It closes the direct bypasses (full
cmdlets and their aliases, inline/encoded execution, writing the fence's own files), classifying
each write target through the shared ``agent_fence_rules`` so the two shells fence one boundary
and a breeder's own same-named file (their ``README.md``) is not caught by basename. A determined
agent can still evade a string matcher (a ``cd`` then a relative write, a path assembled from
fragments); those residuals are accepted, and real isolation is the sandbox.

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

# Inline / nested / encoded execution: the shared interpreters plus this shell's own evaluator.
_INLINE_PS = re.compile(r"\bInvoke-Expression\b|\biex\b", re.IGNORECASE)

# Deletes: blocked unconditionally, mirroring the Bash fence's blanket ``rm``/``rmdir`` deny.
_DELETE_OP = re.compile(
    r"\bRemove-Item\b|\bRemove-ItemProperty\b|\[(?:System\.)?IO\.(?:File|Directory)\]::Delete",
    re.IGNORECASE,
)
_DELETE_ALIAS = re.compile(_STMT + r"(?:ri|rm|rmdir|rd|del|erase)\b", re.IGNORECASE)

# Dangerous git, mirroring the Bash fence deny of push/commit/reset/checkout/clean. Anchored to the
# subcommand right after ``git`` so ``git log --grep="reset"`` is not caught.
_GIT_DANGER = re.compile(r"\bgit\s+(?:push|commit|reset|checkout|clean)\b", re.IGNORECASE)

_BREEDER_DATA_TARGET = fence_rules.BREEDER_DATA_TARGET
_SEG_SPLIT = re.compile(r"[;\n]|&&|\|\||[|&]")

# A named path parameter that gives a write cmdlet its target.
_PS_NAMED = re.compile(
    r"-(?:Path|FilePath|LiteralPath|Destination)\s+(?P<t>[^\s;|&<>()]+)", re.IGNORECASE
)
# .NET / StreamWriter writers name their target as the first quoted argument.
_PS_DOTNET = re.compile(
    r"\[(?:System\.)?IO\.(?:File|Directory)\]::\w+\s*\(\s*(['\"])(?P<t>[^'\"]+)\1", re.IGNORECASE
)
_PS_STREAMWRITER = re.compile(r"StreamWriter\s*\(\s*(['\"])(?P<t>[^'\"]+)\1", re.IGNORECASE)

# Write cmdlets/aliases whose target the guard extracts (positionally or by named parameter).
_WRITE_LEAD = re.compile(
    r"^(?:Set-Content|Add-Content|Clear-Content|Out-File|New-Item|Tee-Object"
    r"|Set-ItemProperty|New-ItemProperty|sc|ac|clc|ni|sp|spi)$",
    re.IGNORECASE,
)
# Copy/Move/Rename: destination-position, so a protected source read is not caught as a write.
_MOVE_LEAD = re.compile(r"^(?:Move-Item|Rename-Item|mi|mv|move|rni|ren)$", re.IGNORECASE)
_COPY_LEAD = re.compile(r"^(?:Copy-Item|cpi|cp|copy)$", re.IGNORECASE)
# A Move/Rename anywhere at statement position, for the source-or-destination breeder check.
_MOVE_STMT = re.compile(_STMT + r"(?:Move-Item|Rename-Item|mi|mv|move|rni|ren)\b", re.IGNORECASE)


def _first_positional(tokens: "list[str]") -> "str | None":
    """The first bare positional token, skipping flags and the value each named flag consumes."""
    prev_flag = False
    for tok in tokens:
        if tok.startswith("-"):
            prev_flag = True
            continue
        if prev_flag:
            prev_flag = False
            continue
        return tok
    return None


def _last_positional(tokens: "list[str]") -> "str | None":
    positionals = []
    prev_flag = False
    for tok in tokens:
        if tok.startswith("-"):
            prev_flag = True
            continue
        if prev_flag:
            prev_flag = False
            continue
        positionals.append(tok)
    return positionals[-1] if positionals else None


def _cmdlet_writes(cmd: str) -> "list[tuple[str, str]]":
    """``(kind, target)`` for each cmdlet/alias write, kind being ``write``/``move``/``copy``.

    ``copy`` is reported so the caller can exempt it from the breeder-data check, the same way the
    Bash guard exempts ``cp``: a stateless guard cannot tell a copy's source from its destination.
    """
    out: list[tuple[str, str]] = []
    for m in _PS_DOTNET.finditer(cmd):
        out.append(("write", m.group("t")))
    for m in _PS_STREAMWRITER.finditer(cmd):
        out.append(("write", m.group("t")))
    for seg in (s.strip() for s in _SEG_SPLIT.split(cmd) if s.strip()):
        tokens = seg.split()
        if not tokens:
            continue
        lead, rest = tokens[0], tokens[1:]
        named = [m.group("t") for m in _PS_NAMED.finditer(seg)]
        if _WRITE_LEAD.match(lead):
            targets = named or ([_first_positional(rest)] if _first_positional(rest) else [])
            out.extend(("write", t) for t in targets if t)
        elif _MOVE_LEAD.match(lead):
            dest = named[-1] if named else _last_positional(rest)
            if dest:
                out.append(("move", dest))
        elif _COPY_LEAD.match(lead):
            dest = named[-1] if named else _last_positional(rest)
            if dest:
                out.append(("copy", dest))
    return out


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

    if fence_rules.inline_exec(cmd) or _INLINE_PS.search(cmd):
        fence_rules.deny(fence_rules.INLINE_EXECUTION_MSG)
    if _GIT_DANGER.search(cmd):
        fence_rules.deny("Dangerous git (push/commit/reset/checkout/clean) is blocked in the agent terminal.")
    if _DELETE_OP.search(cmd) or _DELETE_ALIAS.search(cmd):
        fence_rules.deny(fence_rules.DELETE_MSG)

    for target in fence_rules.redirect_targets(cmd, ps=True):
        kind = fence_rules.classify(fence_rules.resolve_token(target, cmd, ps=True), root=root, mode=mode)
        if kind == "breeder":
            fence_rules.deny(fence_rules.BREEDER_DATA_WRITE_MSG)
        if kind == "protected":
            fence_rules.deny(fence_rules.PROTECTED_WRITE_MSG)

    for op, target in _cmdlet_writes(cmd):
        kind = fence_rules.classify(fence_rules.resolve_token(target, cmd, ps=True), root=root, mode=mode)
        if kind == "protected":
            fence_rules.deny(fence_rules.PROTECTED_WRITE_MSG)
        if kind == "breeder" and op != "copy":
            fence_rules.deny(fence_rules.BREEDER_DATA_WRITE_MSG)

    # Move/Rename relocates the tracked file whether the breeder path is its source or destination.
    if _MOVE_STMT.search(cmd) and _BREEDER_DATA_TARGET.search(cmd):
        fence_rules.deny(fence_rules.BREEDER_DATA_WRITE_MSG)

    if fence_rules.leading_is_allow_listed(cmd, "PowerShell") and fence_rules.has_opaque_redirect_target(cmd, ps=True):
        fence_rules.deny(fence_rules.DYNAMIC_TARGET_MSG)

    sys.exit(0)


if __name__ == "__main__":
    main()

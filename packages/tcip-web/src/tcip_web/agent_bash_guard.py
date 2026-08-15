"""PreToolUse Bash guard for the fenced in-app agent terminal.

The Edit/Write deny-list can't see file writes performed *through the shell*
(`echo > packages/...`, `sed -i`, `python -c "open(...)"`). This PreToolUse hook reads
the tool call on stdin and denies Bash commands that write into platform internals, or
that run an inline interpreter at all.

Honest scope: a guardrail, not a sandbox. A determined user can obfuscate around these
patterns; the point is to stop *accidental / casual* repo edits via Bash. The primary
protection is still the Edit/Write deny-list plus `--permission-mode default` (which
surfaces un-allowed Bash to the human for approval).

Which paths are protected, which hold a breeder's data, and what a refusal says come from
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

# ``tee [-a] FILE`` writes to FILE.
_TEE = re.compile(r"\btee\b\s+(?:-a\s+|--append\s+)?(?P<target>[^\s;|&<>()]+)")
# ``find``'s own file-writing actions (``-fprint``/``-fprint0``/``-fprintf``/``-fls``) write their
# output to the FILE argument that follows, bypassing every other write check here: no ``>``/``tee``
# ever appears on the command line, so a plain ``find ... -fprintf packages/x.py '%p'`` would reach
# `packages/` (or a breeder's ``labels/``) undetected without this check. Matched by flag name alone
# (these names are find-specific) so it fires whether or not ``find`` itself is in view.
_FIND_WRITE_ACTION = re.compile(r"-f(?:print0?|printf|ls)\s+(?P<target>[^\s;|&<>()]+)")
# In-place / copy writers, matched coarsely (paired with a protected token in main()). These
# never appear in the read-only diagnostics the fence must let through.
_WRITE_OP = re.compile(r"\bsed\b\s+-i|\bcp\b|\bmv\b|\bdd\b")
_STMT = fence_rules.STMT
# Deletes / truncates: blocked unconditionally, mirroring the PowerShell fence's unconditional
# _DELETE_OP/_DELETE_ALIAS: the agent mutates data through audited MCP tools, not raw shell
# deletion (of platform code OR of a breeder's labels).
_DELETE_OP = re.compile(_STMT + r"(?:rm|rmdir|unlink|shred|truncate)\b")
# ``... | xargs rm`` (with optional flags on either xargs or the trailing verb).
_XARGS_DELETE = re.compile(r"\bxargs\b\s+(?:-\S+\s+)*(?:rm|rmdir|unlink|shred)\b")
# ``find ... -delete`` / ``find ... -exec {rm,rmdir,unlink,shred,truncate}``: a ``find`` invocation
# the human has approved (it is not allow-listed in agent_terminal.settings.json, so it already
# reaches a permission prompt) can still carry a destructive action the human approving the prefix
# never sees, so every verb ``_DELETE_OP`` already treats as unconditionally destructive is denied
# here too when it appears as a ``-exec`` action, not just ``rm``. ``mv``/``cp`` under ``-exec`` are
# not deletion and stay out of this unconditional list; ``mv`` is instead caught below by the same
# breeder-data-relocation check that catches a bare ``mv`` (the find search root supplies the
# breeder-data match), and ``cp`` is deliberately never denied here, see the comment below.
_FIND_DELETE = re.compile(
    r"\bfind\b[\s\S]*?\s-delete\b"
    r"|\bfind\b[\s\S]*?-exec\s+(?:rm|rmdir|unlink|shred|truncate)\b"
)
_BREEDER_DATA_TARGET = fence_rules.BREEDER_DATA_TARGET
# ``mv`` relocates a breeder-data path exactly as ``rm`` would remove it from where it belongs
# (bare ``mv <path under annotations/labels/predictions> <dest>``, or ``find <that dir> -exec mv
# {} <dest> \;``, the find search root itself supplying the breeder-data match): paired with
# ``_BREEDER_DATA_TARGET`` anywhere in the command, the same way the PowerShell guard's
# ``_BREEDER_DATA_WRITE_OP``/``Move-Item``/``mv`` check already denies this. ``cp`` is deliberately
# excluded, mirroring the PowerShell guard's own documented reason: this guard is stateless and
# can't tell a two-argument command's source from its destination, so including ``cp`` would deny a
# legitimate backup/copy of a breeder file to elsewhere, not just a copy into one; ``mv`` stays
# included because relocating the tracked file is itself the harm regardless of which argument the
# breeder path was named in. Anchored with the same ``_STMT`` prefix ``_DELETE_OP`` uses so a
# read-only command that merely names "mv" in a path or grep pattern (``cat annotations/mv-notes.txt``,
# ``grep -rn mv annotations``) doesn't trip this.
_MOVE_OP = re.compile(_STMT + r"mv\b")
# ``find ... -exec mv``: the same find-approved-prefix-can-hide-a-relocation gap ``_FIND_DELETE``
# closes for the delete verbs, but for ``mv`` instead of removal.
_FIND_MOVE = re.compile(r"\bfind\b[\s\S]*?-exec\s+mv\b")
# The shared spawned-interpreter set plus this shell's own nested shells (``bash -c``…), whose
# payload the guard can't see through.
_INLINE_INTERP = re.compile(fence_rules.SPAWNED_INTERPRETER + r"|\b(?:bash|sh|zsh)\b\s+-\w*c\b")


def _write_targets(cmd: str) -> list[str]:
    """Resolved file targets of redirects, ``tee``, and ``find``'s own write actions.

    Executing a script under ``scripts/`` is passing an *argument*, not writing there; only a
    redirect/tee/``find -f*``, whose *target* lands under a protected dir, is a mutation of
    platform code.
    """
    return (
        fence_rules.redirect_targets(cmd)
        + [m.group("target") for m in _TEE.finditer(cmd)]
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

    if _INLINE_INTERP.search(cmd):
        fence_rules.deny(fence_rules.INLINE_EXECUTION_MSG)
    if _DELETE_OP.search(cmd) or _XARGS_DELETE.search(cmd) or _FIND_DELETE.search(cmd):
        fence_rules.deny(fence_rules.DELETE_MSG)
    if any(_BREEDER_DATA_TARGET.search(t) for t in _write_targets(cmd)):
        fence_rules.deny(fence_rules.BREEDER_DATA_WRITE_MSG)
    if (_MOVE_OP.search(cmd) or _FIND_MOVE.search(cmd)) and _BREEDER_DATA_TARGET.search(cmd):
        fence_rules.deny(fence_rules.BREEDER_DATA_WRITE_MSG)
    protected = fence_rules.protected_pattern()
    if any(protected.search(t) for t in _write_targets(cmd)):
        fence_rules.deny(fence_rules.PROTECTED_WRITE_MSG)
    if _WRITE_OP.search(cmd) and protected.search(cmd):
        fence_rules.deny(fence_rules.PROTECTED_WRITE_MSG)

    sys.exit(0)


if __name__ == "__main__":
    main()

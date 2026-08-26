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
protected file are accepted residuals of a command-only, cwd-blind guard. The verb enumeration
this guard reads is likewise never complete: an unlisted or exotic writer reaches the human
approval prompt instead of a denial, the same defense-in-depth as any other non-allow-listed verb.

`ed` and `ex` are denied outright, on any invocation, regardless of arguments: an editor names
the file it writes inside its own command script (an `-c 'w <path>'` argument, or an `e`/`w` pair
fed through a heredoc or a pipe), never as a plain positional argument, so no argument-reading
rule can see the target. A non-interactive agent terminal has no legitimate reason to open a
line editor at all, which is why the rule denies the verb rather than trying to classify a
target it cannot find. The delete and editor verbs are read from the same stripped, redirect-free
token list the writer checks use, so a wrapped invocation (`env rm -rf`, `busybox ed`, `command
ed`) is caught the same way a wrapped writer is; `xargs`/`find` still route their own verb
through a raw-string regex, since their verb sits inside `xargs`'s own flags or a `-exec`/
`-delete` clause rather than as a segment's leading token, which a segment-level check cannot
express.

What is protected, how a target is normalized, and what a refusal says come from
``agent_fence_rules``, shared with the PowerShell guard so the two shells fence one boundary.

Stdlib only, so it runs under whatever ``python`` the terminal inherits. It only ever
denies; anything it can't classify it lets fall through to the normal permission flow
(so a bug here fails open to prompting, never to a broken terminal).
"""

from __future__ import annotations

import json
import re
import shlex
import sys

try:  # as a hook it runs as a bare script, with only its own directory importable
    from . import agent_fence_rules as fence_rules
except ImportError:
    import agent_fence_rules as fence_rules  # type: ignore[no-redef]

_STMT = fence_rules.STMT

# ── tokenizing one command into statement segments ──────────────────────────

# bash's own separators, a subshell's ()/{}, and a literal newline (removed from whitespace
# below so it survives as its own statement-separator token instead of being dropped).
_PUNCTUATION = "(){};<>|&\n"

# A statement/pipeline boundary: any of these chars alone, or merged (shlex glues adjacent
# punctuation into one token, e.g. ``"&\n"``). A token built from ``<``/``>`` is a redirect instead.
_SEG_CHARS = set(";&|(){}\n")

_CONTINUATION = re.compile(r"\\\r?\n")


class TokenizeError(Exception):
    """Raised when a command cannot be tokenized (an unclosed quote, a trailing
    backslash inside quotes, an unbalanced ``$(``); the caller denies rather than guessing
    at a partial parse."""


def _mask_substitutions(cmd: str) -> str:
    """Replace each balanced ``$( ... )`` command substitution with one opaque word token.

    Without this, shlex reads a substitution's own parentheses as segment boundaries, splitting
    the statement that encloses it apart and hiding whatever token follows. Masking keeps the
    substitution as one harmless token in its enclosing segment, so a destination that follows it
    is still read and classified. Raises :class:`TokenizeError` naming a ``$(`` this scan cannot
    pair with a closing parenthesis, rather than guessing where it would have ended.
    """
    out: list[str] = []
    i = 0
    n = len(cmd)
    count = 0
    while i < n:
        if cmd[i] == "$" and i + 1 < n and cmd[i + 1] == "(":
            start = i
            depth = 1
            j = i + 2
            while j < n and depth:
                if cmd[j] == "(":
                    depth += 1
                elif cmd[j] == ")":
                    depth -= 1
                j += 1
            if depth:
                raise TokenizeError(f"an unbalanced $( at position {start} has no closing )")
            out.append(f"subst{count}token")
            count += 1
            i = j
        else:
            out.append(cmd[i])
            i += 1
    return "".join(out)


def tokenize(cmd: str) -> "list[str]":
    """``cmd`` as shlex tokens: a quoted separator or redirect character stays inside its
    own token instead of splitting the command apart, and an unquoted command tokenizes to
    the same words a plain whitespace split would produce. A backslash immediately before a
    newline is a line continuation, stripped before tokenizing so it is not read as a
    statement break, and a ``$( ... )`` substitution is masked to one opaque token so its
    internal parentheses stay inside the segment that encloses it. Raises
    :class:`TokenizeError` naming what shlex refused, or what this function could not pair,
    rather than letting a ``ValueError`` reach the caller unlabeled.
    """
    cmd = _CONTINUATION.sub("", cmd)
    cmd = _mask_substitutions(cmd)
    lexer = shlex.shlex(cmd, posix=True, punctuation_chars=_PUNCTUATION)
    lexer.whitespace_split = True
    lexer.commenters = ""
    lexer.whitespace = " \t\r"
    try:
        return list(lexer)
    except ValueError as exc:
        raise TokenizeError(str(exc)) from exc


def _is_boundary(tok: str) -> bool:
    return bool(tok) and set(tok) <= _SEG_CHARS


def _segments(cmd: str) -> "list[list[str]]":
    """Statement/pipeline segments as token lists, split on the operator tokens the
    tokenizer emits rather than on a raw character, so a quoted separator or redirect
    character never breaks a segment apart; a subshell's or brace group's delimiters are
    boundaries too, so the verb inside one is read as a verb rather than hidden behind it.
    """
    segments: list[list[str]] = []
    current: list[str] = []
    for tok in tokenize(cmd):
        if _is_boundary(tok):
            if current:
                segments.append(current)
            current = []
        else:
            current.append(tok)
    if current:
        segments.append(current)
    return segments


# A file-writing redirect operator, as the tokenizer emits it (never split character by
# character, so a quoted operator inside an argument is never mistaken for one of these).
_REDIRECT_OPS = {"<", ">", ">>", "<>", ">|", "&>", "&>>", ">&"}


def _before_redirect(tokens: "list[str]") -> "list[str]":
    """The tokens of one segment up to its first redirect operator, dropping a bare
    file-descriptor number immediately in front of one (the ``2`` in ``2>&1``)."""
    out: "list[str]" = []
    for tok in tokens:
        if tok in _REDIRECT_OPS:
            if out and out[-1].isdigit():
                out.pop()
            break
        out.append(tok)
    return out


# A leading wrapper the shell runs the real command through unchanged, and the flags each
# consumes first (env, nice, command, time, stdbuf); an unrecognised flag stops the strip there.
_TRANSPARENT_PREFIX = ("busybox", "command", "env", "nice", "time", "stdbuf")
_ASSIGNMENT = re.compile(r"[A-Za-z_]\w*=.*")
_NICE_ATTACHED = re.compile(r"^-n\d+$")
_NICE_BARE_ADJUSTMENT = re.compile(r"^-\d+$")


def _strip_transparent_prefix(argv: "list[str]") -> "list[str]":
    """Drop a leading transparent wrapper, and the options it consumes, so the verb
    underneath is what gets classified."""
    i = 0
    while i < len(argv) and argv[i] in _TRANSPARENT_PREFIX:
        wrapper = argv[i]
        i += 1
        if wrapper == "env":
            while i < len(argv):
                tok = argv[i]
                if _ASSIGNMENT.fullmatch(tok):
                    i += 1
                elif tok == "-i":
                    i += 1
                elif tok == "-u" and i + 1 < len(argv):
                    i += 2
                else:
                    break
        elif wrapper == "nice":
            tok = argv[i] if i < len(argv) else ""
            if tok == "-n" and i + 1 < len(argv) and re.fullmatch(r"-?\d+", argv[i + 1]):
                i += 2
            elif _NICE_ATTACHED.match(tok) or _NICE_BARE_ADJUSTMENT.match(tok):
                i += 1
        elif wrapper == "command":
            while i < len(argv) and argv[i] in ("-p", "-v"):
                i += 1
        elif wrapper == "time":
            while i < len(argv) and argv[i] == "-p":
                i += 1
        elif wrapper == "stdbuf":
            while i < len(argv) and argv[i].startswith("-"):
                i += 1
    return argv[i:]


# ``find``'s own write actions (``-fprint``/``-fprintf``/``-fls``) write to the FILE argument that
# follows, with no ``>``/``tee`` on the line, so they are matched by their find-specific flag name.
_FIND_WRITE_ACTION = re.compile(r"-f(?:print0?|printf|ls)\s+(?P<target>[^\s;|&<>()]+)")
# ``dd of=FILE`` names its output through a key=value argument, not a positional one.
_DD_OF = re.compile(r"\bdd\b[\s\S]*?\bof=(?P<target>[^\s;|&<>()]+)")

# Deletes/truncates and the line editors are blocked unconditionally (an editor's write target
# lives inside its own command script); read as a segment's leading token, not a raw-string regex.
_DELETE_VERBS = frozenset({"rm", "rmdir", "unlink", "shred", "truncate"})
_EDITOR_VERBS = frozenset({"ed", "ex"})
# xargs/find route their verb through their own flags or a -exec/-delete clause, not a leading
# token, so they stay matched by a raw-string regex instead of the segment-level check above.
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

# In-place / copy writers whose destination is a positional argument (cp/mv/install/rsync/ln take
# ``SRC... DEST``); ``chmod`` is declined, since a mode change writes no content to classify.
_DEST_VERBS = ("cp", "mv", "install", "rsync", "ln")


def _leading_verb(tokens: "list[str]") -> "str | None":
    """The verb one segment would run: its leading token once a redirect tail and a
    transparent wrapper (``env``, ``nice``, ...) are stripped away, the same path the writer
    checks read a destination through."""
    argv = _strip_transparent_prefix(_before_redirect(tokens))
    return argv[0] if argv else None


def _tee_targets(segments: "list[list[str]]") -> "list[str]":
    """Every file ``tee`` writes to (it accepts multiple output files)."""
    out: list[str] = []
    for tokens in segments:
        argv = _strip_transparent_prefix(_before_redirect(tokens))
        if argv and argv[0] == "tee":
            out.extend(t for t in argv[1:] if not t.startswith("-"))
    return out


def _inplace_dests(cmd: str, segments: "list[list[str]]") -> "list[str]":
    """The file each in-place writer would write.

    For cp/mv/install/rsync/ln this is the destination (the last non-flag token), so a copy or move
    into a protected or breeder path is caught while reading a protected/breeder source and writing
    elsewhere is not. touch/sed/patch/dd name the file directly.
    """
    out: list[str] = []
    for tokens in segments:
        argv = _strip_transparent_prefix(_before_redirect(tokens))
        if not argv:
            continue
        verb, args = argv[0], argv[1:]
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


def _write_targets(cmd: str, segments: "list[list[str]]") -> "list[str]":
    """Redirect, ``tee``, and ``find``-write targets: the airtight set, checked for both harms."""
    return (
        fence_rules.redirect_targets(cmd)
        + _tee_targets(segments)
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

    try:
        segments = _segments(cmd)
    except TokenizeError as exc:
        fence_rules.deny(
            f"This command could not be parsed ({exc}), so the guard denies rather than "
            "guessing at what it would write. Rephrase it, or use the TCIP tools."
        )
        return

    leading_verbs = {_leading_verb(seg) for seg in segments}
    if (
        leading_verbs & (_DELETE_VERBS | _EDITOR_VERBS)
        or _XARGS_DELETE.search(cmd)
        or _FIND_DELETE.search(cmd)
    ):
        fence_rules.deny(fence_rules.DELETE_MSG)

    write_targets = _write_targets(cmd, segments)
    inplace_targets = _inplace_dests(cmd, segments)

    for target in write_targets:
        kind = fence_rules.classify(fence_rules.resolve_token(target, cmd), root=root, mode=mode)
        if kind == "breeder":
            fence_rules.deny(fence_rules.BREEDER_DATA_WRITE_MSG)
        if kind == "protected":
            fence_rules.deny(fence_rules.PROTECTED_WRITE_MSG)

    for target in inplace_targets:
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

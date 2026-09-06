"""What the in-app agent fence protects, declared once for both shell guards.

``agent_bash_guard.py`` and ``agent_powershell_guard.py`` police the same boundary in two
syntaxes. The boundary itself, which repo paths are platform-internal, which paths hold a
breeder's data, how a write target is normalized before it is classified, and what a refusal
tells the agent, lives here; each guard keeps only the syntax that recognises a write in its
own shell and feeds normalized targets back to the shared classifier.

The fence is a guardrail, not a sandbox. It has exactly one path where it is the sole gate: a
write that rides an allow-listed read prefix (``cat``, ``ls``, ``grep``, ``git diff`` and their
PowerShell peers) through a redirect, which Claude Code runs with no human prompt. Along every
other path a non-allow-listed verb already reaches an approval prompt, so the guard is
defense-in-depth there. The redirect grammar and target normalization below are therefore
airtight; the in-place writer enumeration is deliberately not exhaustive, since a human prompt
backs it. Real isolation from a determined agent is the OS sandbox, the platform's stated next
step.

The redirect grammar (``REDIRECT`` below, and ``redirect_targets``) reads the raw command
string rather than a quote-aware token list, so a quoted ``>`` inside an earlier, unrelated
argument can be misread as a second redirect operator: ``cat 'text>packages' > scratch.txt``
denies falsely, reading the quoted fragment ``>packages`` as a second target. This is an
accepted residual (a false deny, never a missed one), since the source argument's content is
not something either shell guard needs to parse correctly to keep the redirect path airtight.

Two protected-set modes exist. In development the guard runs inside a source checkout and
protects the repo tree plus the breeder's project data. In production the platform is an
installed package with no repo tree and an OS sandbox, so only the breeder's project data
(annotations, labels, predictions, image status, and the trait-state records) is protected.
Mode defaults to development, the more protective choice, unless ``TCIP_FENCE_MODE=prod`` is set
explicitly, so an unknown deployment fails safe. The production settings filtering and the
enforced sandbox are designed but not built until an installed deployment exists to exercise
them; ``classify`` already answers correctly for either mode.

Path matching is case-insensitive: the protected filesystem is case-insensitive on the
platform's Windows host, so ``PACKAGES/x.py`` and ``packages/x.py`` are one file. The guards are
stateless and cwd-blind, so every check is a path-shape check, never an existence check, and a
relative write after a ``cd`` into a protected directory is an accepted residual of a
command-only guard (its real boundary there is the human prompt and, in production, the sandbox).

Stdlib only, and importable both as ``tcip_web.agent_fence_rules`` and as a bare sibling module,
because the guards run as plain scripts under whatever ``python`` the terminal inherits.
"""

from __future__ import annotations

import json
import os
import re
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

# Every file-writing redirect form (``>`` ``>>`` ``>|`` ``&>`` ``&>>`` ``<>`` ``>&FILE`` and their
# numbered ``N>`` forms); ``redirect_targets`` filters the ``>&`` target so fd dup/close drop out.
REDIRECT = re.compile(
    r"(?P<op>&>>?|\d*<>|\d*>&|\d*>>?\|?)\s*(?P<target>[^\s;|&<>()]+)"
)

BREEDER_DATA_TARGET = re.compile(
    r"(?:^|[/\\])(?:annotations|labels|predictions)(?:[/\\]|[\s;|&)]|$)|image_status\.json\b",
    re.IGNORECASE,
)

# A spawned interpreter or nested shell whose payload no guard can see through, both shells. The
# per-interpreter code-execution flag is matched anywhere in the argv by ``inline_exec`` below.
SPAWNED_INTERPRETER = (
    r"\b(?:powershell|pwsh)\b[\s\S]*?-(?:e|ec|enc|encodedcommand|command|c)\b"
    r"|\bcmd\b[\s\S]*?/c\b"
)
_INTERP_NAMES = ("python3", "python", "node", "perl", "ruby", "deno", "bun")

PROTECTED_WRITE_MSG = (
    "Writing into platform internals via the shell is blocked, the agent edits projects, not "
    "platform code. If this was a read-only diagnostic that got mis-flagged (e.g. "
    "`tcip doctor <root>`), that's a fence false-positive: file it with the "
    "report_friction tool (category unexpected_behavior; include the exact command) so the fence "
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
DYNAMIC_TARGET_MSG = (
    "An allow-listed command may not redirect into a target the fence cannot resolve (a variable "
    "set elsewhere, a command substitution). Name the file directly, or use the TCIP tools."
)
UNREADABLE_DECLARATION_MSG = (
    "The agent terminal's permission declaration ({path}) could not be read, so the shell guard "
    "cannot tell which paths are platform-internal and refuses rather than guess. Restore the "
    "file to use the terminal."
)


def deny(reason: str) -> None:
    """Emit the PreToolUse deny decision for ``reason`` and exit."""
    import sys

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


# ── deployment mode and the repo root the dev-mode rules anchor to ──────────


def repo_root() -> "Path | None":
    """The source-checkout root above this module, or ``None`` when installed.

    The nearest ancestor holding ``.mcp.json`` is the repo root (only the true root carries it),
    the same signal ``project_paths.repo_root_from_here`` uses. In an installed wheel no ancestor
    has it, which is the production signal. Used only to anchor the dev-mode repo rules, never to
    guess a working directory.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / ".mcp.json").is_file():
            return parent
    return None


def fence_mode() -> str:
    """``"prod"`` only when set explicitly, else ``"dev"`` (the more protective default)."""
    return "prod" if os.environ.get("TCIP_FENCE_MODE", "").strip().lower() == "prod" else "dev"


# ── the protected-set declaration, derived from the fence settings ──────────


def _edit_deny_targets() -> list[str]:
    """The path targets of the fence settings' ``Edit(...)`` deny rules."""
    cfg = json.loads(FENCE_SETTINGS.read_text(encoding="utf-8"))
    rules = cfg["permissions"]["deny"]
    return [r[len("Edit(") : -1] for r in rules if r.startswith("Edit(") and r.endswith(")")]


def _declared_targets() -> "tuple[list[str], list[str], list[str]]":
    """Split the settings deny rules into (repo dirs, repo single files, project-data segments).

    A single-segment ``Edit(<dir>/**)`` rule (``packages``) is repo code, anchored to the repo
    root and active in dev mode only. A multi-segment ``Edit(<a>/<b>/**)`` rule
    (``.tcip/state/trait_specs``) is the breeder's project data, matched as a path-segment
    subsequence wherever the project lives and active in both modes. A rule with no ``/**``
    (``README.md``) is a repo-root single file, anchored, dev mode only. Denies the command
    outright when the declaration cannot be read: a guard that cannot see the boundary refuses.
    """
    try:
        targets = _edit_deny_targets()
    except (OSError, ValueError, KeyError, TypeError):
        targets = []
    if not targets:
        deny(UNREADABLE_DECLARATION_MSG.format(path=FENCE_SETTINGS))
    repo_dirs, repo_files, project_segments = [], [], []
    for t in targets:
        if t.endswith("/**"):
            path = t[: -len("/**")]
            if "/" in path:
                project_segments.append(path.lower())
            else:
                repo_dirs.append(path.lower())
        else:
            repo_files.append(t.lower())
    return repo_dirs, repo_files, project_segments


# ── target normalization and classification ────────────────────────────────


def _strip_quotes(token: str) -> str:
    """Remove unescaped single and double quotes, mirroring shell quote removal.

    ``la"bel"s`` becomes ``labels``: the shell opens the dequoted path, so the classifier must
    see it too. Over-approximates quote removal in the safe direction (it can only make more
    targets match); a real filename carrying a literal quote is pathological.
    """
    return re.sub(r"(?<!\\)['\"]", "", token)


def normalize_target(token: str) -> str:
    """A redirect or path token as a lowercase, forward-slashed, lexically collapsed path.

    Strips quotes, unifies separators, drops a Windows drive-relative prefix (``C:foo`` resolves
    against the drive's current directory, the repo root at terminal start, so it is treated as
    the relative ``foo``), keeps an absolute drive or leading-slash root, and collapses ``.`` and
    ``..`` lexically without touching the filesystem (the guards are cwd-blind, so this is
    path-shape normalization, never resolution against a real cwd).
    """
    s = _strip_quotes(token).replace("\\", "/")
    root = ""
    drive = re.match(r"^([A-Za-z]):(/?)", s)
    if drive:
        if drive.group(2):  # C:/... absolute drive path
            root = drive.group(1).lower() + ":/"
            s = s[drive.end() :]
        else:  # C:foo drive-relative -> relative foo
            s = s[2:]
    elif s.startswith("//"):  # UNC or posix double slash
        root = "//"
        s = s[2:]
    elif s.startswith("/"):
        root = "/"
        s = s[1:]
    out: list[str] = []
    for seg in s.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if out:
                out.pop()
            continue
        out.append(seg.lower())
    return root + "/".join(out)


def _is_absolute(norm: str) -> bool:
    return norm.startswith("/") or bool(re.match(r"^[a-z]:/", norm))


def _repo_relative(norm: str, root: "Path | None") -> "str | None":
    """``norm`` expressed relative to the repo root, or ``None`` if it is not under it.

    An absolute target must sit under the repo root to be repo code; a relative target is treated
    as repo-root-relative (a bare ``packages/x`` after a ``cd`` back to the root is that
    directory), which is the conservative reading the dev-mode rules rely on.
    """
    if not _is_absolute(norm):
        return norm
    if root is None:
        return None
    root_norm = normalize_target(str(root))
    if norm == root_norm:
        return ""
    prefix = root_norm + "/"
    return norm[len(prefix) :] if norm.startswith(prefix) else None


def classify(token: str, *, root: "Path | None", mode: str) -> "str | None":
    """Classify a write target as ``"breeder"``, ``"protected"``, or ``None`` (free).

    ``breeder`` and the project-data trait-state records are protected in both modes; the repo
    directory and single-file rules only in ``dev`` mode. ``root`` anchors the dev-mode rules;
    it is unused in ``prod`` mode, where no repo tree exists.
    """
    norm = normalize_target(token)
    if not norm:
        return None
    # The fence's own files disable it if written, so they are protected by basename in both modes.
    if norm.split("/")[-1] in {f.lower() for f in FENCE_OWN_FILES}:
        return "protected"
    repo_dirs, repo_files, project_segments = _declared_targets()

    # Breeder data and the trait-state records: both modes, matched wherever the project lives.
    if BREEDER_DATA_TARGET.search("/" + norm):
        return "breeder"
    segs = norm.split("/")
    for seg_path in project_segments:
        needle = seg_path.split("/")
        for i in range(len(segs) - len(needle) + 1):
            if segs[i : i + len(needle)] == needle:
                return "breeder"

    if mode != "dev":
        return None

    rel = _repo_relative(norm, root)
    if rel is None:
        return None
    rel_segs = [s for s in rel.split("/") if s]
    # First-segment repo directory, or a repo-root single file (absolute ``<root>/README.md`` or a
    # bare basename); a breeder's own project file is absolute and not under the root, so excluded.
    if rel_segs and rel_segs[0] in repo_dirs:
        return "protected"
    if len(rel_segs) == 1 and rel_segs[0] in repo_files:
        return "protected"
    return None


# ── redirect targets, with one-hop variable resolution ─────────────────────


_ASSIGN = re.compile(r"(?:^|[\n;|&(){}]\s*)([A-Za-z_]\w*)=(?P<val>[^\s;|&<>()]*)")
_PS_ASSIGN = re.compile(r"\$(\w+)\s*=\s*(?P<val>(?:'[^']*'|\"[^\"]*\"|[^\s;|&<>()]+))")
_VAR_REF = re.compile(r"^\$\{?(\w+)\}?$|^\$env:(\w+)$")


def _assignments(cmd: str, ps: bool) -> "list[tuple[int, str, str]]":
    """Ordered ``(index, name, value)`` variable assignments in one command string."""
    pat = _PS_ASSIGN if ps else _ASSIGN
    out = []
    for m in pat.finditer(cmd):
        out.append((m.start(), m.group(1), _strip_quotes(m.group("val"))))
    return out


def _resolve_var(ref: str, at: int, assigns: "list[tuple[int, str, str]]") -> "str | None":
    """The value of variable reference ``ref`` from the last assignment before position ``at``."""
    m = _VAR_REF.match(ref)
    if not m:
        return None
    name = m.group(1) or m.group(2)
    value = None
    for idx, var, val in assigns:
        if var == name and idx < at:
            value = val
    return value


def redirect_targets(cmd: str, *, ps: bool = False) -> "list[str]":
    """Every file target a redirect in ``cmd`` writes to, one-hop variables resolved.

    A redirect target that is a bare variable reference is replaced by the value assigned to it
    earlier in the same command string. A target left dynamic after that (a variable assigned
    nowhere here, a command substitution) is returned unchanged so the caller can decide; see
    :func:`has_dynamic_redirect_target`.
    """
    assigns = _assignments(cmd, ps)
    out = []
    for m in REDIRECT.finditer(cmd):
        target = _strip_quotes(m.group("target"))
        # ``>&N`` / ``>&-`` duplicate or close a descriptor rather than write a file.
        if m.group("op").endswith("&") and re.fullmatch(r"\d+|-", target):
            continue
        resolved = _resolve_var(target, m.start(), assigns)
        out.append(resolved if resolved is not None else target)
    return out


# A target that is nothing but one variable reference or command substitution: the guard sees
# none of the path. One with a literal component has its tail classified normally instead.
_OPAQUE_TARGET = re.compile(r"^(?:\$\{?\w+\}?|\$env:\w+|\$\(.*\)|`.*`|%\w+%)$")


def resolve_token(token: str, cmd: str, *, ps: bool = False) -> str:
    """A bare variable-reference token resolved to its value from the command's assignments.

    Used at the classify call sites so an in-place writer or cmdlet naming its target through a
    variable (``DEST=packages/x; cp evil $DEST``) is judged on the resolved path. A token that is
    not a bare variable, or a variable assigned nowhere, is returned unchanged.
    """
    resolved = _resolve_var(_strip_quotes(token), len(cmd), _assignments(cmd, ps))
    return resolved if resolved is not None else _strip_quotes(token)


def has_opaque_redirect_target(cmd: str, *, ps: bool = False) -> bool:
    """True if a redirect writes to a wholly opaque target (a bare variable or substitution).

    Excludes ``$null`` (the PowerShell null sink) and any target carrying a literal path segment,
    so only the case where the guard can see nothing of the path (``> $T``, ``> $(getpath)``)
    trips it. The caller pairs this with an allow-listed leading verb (the no-prompt path) to fail
    closed rather than let an unresolvable target through unseen.
    """
    for target in redirect_targets(cmd, ps=ps):
        if target == "$null":
            continue
        if _OPAQUE_TARGET.match(target):
            return True
    return False


# ── allow-listed leading verbs (the sole-gate path) ────────────────────────


def _allow_prefixes(kind: str) -> "list[str]":
    """The command prefixes the settings allow-list grants for ``kind`` (``Bash``/``PowerShell``)."""
    try:
        cfg = json.loads(FENCE_SETTINGS.read_text(encoding="utf-8"))
        allow = cfg["permissions"]["allow"]
    except (OSError, ValueError, KeyError, TypeError):
        return []
    prefixes = []
    head = kind + "("
    for rule in allow:
        if not (rule.startswith(head) and rule.endswith(")")):
            continue
        inner = rule[len(head) : -1]
        prefixes.append(inner[: -len(":*")] if inner.endswith(":*") else inner)
    return prefixes


def leading_is_allow_listed(cmd: str, kind: str) -> bool:
    """True if ``cmd`` begins with an allow-listed prefix, so Claude Code would not prompt it."""
    head = cmd.strip()
    for prefix in _allow_prefixes(kind):
        if head == prefix or head.startswith(prefix + " "):
            return True
    return False


# ── inline interpreter execution, flag position independent ────────────────


def inline_exec(cmd: str) -> bool:
    """True if ``cmd`` runs inline/encoded code the guard cannot see through.

    Recognises the code-execution flag anywhere before a ``-m`` module or a script path, so an
    interposed flag (``python -X utf8 -c``) is caught while ``python -m pytest -c cfg`` and
    ``python tools/x.py`` stay free.
    """
    if re.search(SPAWNED_INTERPRETER, cmd, re.IGNORECASE):
        return True
    names = "|".join(_INTERP_NAMES)
    value_flags = {"-X", "-W", "--check-hash-based-pycs"}
    for m in re.finditer(rf"\b(?:{names})\b", cmd):
        rest = cmd[m.end() :]
        skip_next = False
        for tok in rest.split():
            if skip_next:
                skip_next = False
                continue
            if tok in (";", "|", "&", "&&", "||"):
                break
            if tok == "-m" or (not tok.startswith("-")):
                break  # module boundary or script path: option scan ends
            if tok in value_flags:
                skip_next = True  # this flag consumes the next token as its value
                continue
            if re.fullmatch(r"-\w*[ce]", tok):
                return True
    return False

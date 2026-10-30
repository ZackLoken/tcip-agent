"""Embedded agent terminal: an agent harness from :data:`PROVIDERS` spawned in a pseudo-terminal.

The pseudo-terminal is ConPTY via ``pywinpty`` on Windows and the stdlib ``pty`` on POSIX; its
raw bytes are streamed out and keystrokes streamed in. ``TCIP_TERMINAL_CMD`` overrides the spawn
command. Every row's launch registers one session hook (:func:`session_hook_argv`), through which
its harness reports its conversation and its agent is given its :func:`orientation`.
"""

from __future__ import annotations

import codecs
import io
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Buffer, Callable, Iterable, Mapping, Sequence
from contextlib import ExitStack, suppress
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from queue import Queue, ShutDown
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import psutil
from pydantic import TypeAdapter, ValidationError

from tcip_mcp.agent_identity import TERMINAL_SESSION_ENV
from tcip_mcp.project_paths import repo_root_from_here
from tcip_web.state import OpenProject, store

if TYPE_CHECKING:
    from tcip_mcp.tools.meta_tools import MemoryKind

logger = logging.getLogger(__name__)

TERMINAL_CMD_ENV = "TCIP_TERMINAL_CMD"

DEFAULT_ROWS = 30
DEFAULT_COLS = 100

# Seconds terminate() waits for the spawned process to exit: the Windows path bounds psutil's
# process wait after taskkill with it; the POSIX path bounds its wait() calls with it.
TERMINATE_WAIT_S = 5

MCP_CONFIG_ARG = "{mcp_config}"
"""A whole argument: the path of the launch's JSON MCP configuration."""
CODEX_CONFIG_ARG = "{codex_config}"
"""A whole argument: the launch's Codex configuration overrides, as several arguments."""
CLAUDE_SETTINGS_ARG = "{claude_settings}"
"""A whole argument: the path of the launch's Claude settings file."""

CLAUDE_SETTINGS = (
    Path(__file__).resolve().parent / "agent_terminal.settings.json"
)
"""Claude Code's row's shipped permission lists."""

PREPARATION_TIMEOUT_S = 60

ANTIGRAVITY_SETTINGS = (
    Path.home() / ".gemini" / "antigravity-cli" / "settings.json"
)
"""Antigravity's own settings file, whose ``permissions.allow`` list its tool approvals persist
to as ``mcp(<server>/<tool>)`` entries."""

ANTIGRAVITY_HOOKS = Path(".agents") / "hooks.json"
"""The workspace's Antigravity hooks file, relative to the workspace."""
ANTIGRAVITY_HOOK_ENTRY = "tcip-session-start"
"""The entry of :data:`ANTIGRAVITY_HOOKS` an Antigravity launch's session hook is set under."""
ANTIGRAVITY_RULE = ANTIGRAVITY_HOOKS.parent / "rules" / "tcip-orientation.md"
"""The always-on rule an Antigravity launch writes, relative to the workspace."""


class PreparationFailedError(Exception):
    """A row's launch preparation did not complete; the message names the step and why."""


PrepareFn = Callable[[str, OpenProject | None, str], list[str]]
"""A row's launch preparation: given the resolved executable, the launch's project (``None`` for
no project) and the id of the terminal session it launches for, it runs to completion and returns
one line per step it took, or raises :class:`PreparationFailedError`."""


@dataclass(frozen=True)
class Resume:
    """How a harness continues one of its own conversations, and for how long a launch does.

    ``argument`` is what goes before the conversation's id on its command line;
    ``cache_lifetime`` is the window after the conversation's last output in which a launch
    resumes it: the prompt-cache lifetime the row requests or its harness's provider documents,
    not a promise that the cache still holds the conversation.
    """

    argument: str
    cache_lifetime: timedelta


@dataclass(frozen=True)
class Provider:
    """One agent harness the terminal launches, its launch and what that launch confines.

    It carries the id a session names it by, its display name, the executable looked up on
    ``PATH``, the arguments after it, which may include the placeholders
    :data:`MCP_CONFIG_ARG`, :data:`CLAUDE_SETTINGS_ARG` and :data:`CODEX_CONFIG_ARG` as whole
    arguments and, for a harness whose conversations are resumed, its :class:`Resume` where
    the resumed conversation goes, the output sequence every composer frame the harness was
    recorded writing carries and none of its screens recorded before a composer carries, what
    the harness's own enforcement restricts under those arguments and leaves open, and the
    preparation the launch runs first, ``None`` for none.
    """

    id: str
    name: str
    executable: str
    args: tuple[str | Resume, ...]
    composer_ready: str
    confinement: str
    prepare: PrepareFn | None = None

    @property
    def resume(self) -> Resume | None:
        """The :class:`Resume` among the row's arguments, ``None`` for a row never resumed."""
        return next((a for a in self.args if isinstance(a, Resume)), None)

    @property
    def unavailable_reason(self) -> str:
        """Why this row cannot launch when its executable is not on ``PATH``."""
        return (
            f"{self.name} is not available: no `{self.executable}` executable is on PATH. "
            "Install it and sign in to enable the agent terminal."
        )


@dataclass(frozen=True)
class McpServer:
    """The stdio MCP server a launch hands its harness: a command and its arguments."""

    command: str
    args: tuple[str, ...]


def backend_interpreter() -> str:
    """Return this backend's Python interpreter, forward-slashed."""
    return Path(sys.executable).as_posix()


def mcp_server(project: OpenProject | None) -> McpServer:
    """Return :func:`backend_interpreter` running ``tcip_mcp`` for ``project`` or for none."""
    args = ("--project", project.root.as_posix()) if project else ()
    return McpServer(backend_interpreter(), ("-m", "tcip_mcp", *args))


def session_hook_argv() -> list[str]:
    """Return the session hook every row registers, naming nothing of any one launch.

    It is the ``tcip`` console's ``session-hook`` command under :func:`backend_interpreter`, the
    interpreter :func:`mcp_server` runs.
    """
    return [backend_interpreter(), "-m", "tcip_web.cli", "session-hook"]


SHELL_WORD_REFUSES = frozenset("|&;()<>'\"$`\\!*?[]{}~#@,‘’‚‛“”„")
"""The characters besides whitespace :func:`shell_line` refuses in a word, so the word is one
argument naming itself under Bash and PowerShell: what Bash reads outside quotes as an operator, a
quote, an expansion, an escape or a comment, ``| & ; ( ) < > ' " $ ` \\ ! * ? [ ] { } ~ #``; what
PowerShell reads there as one, ``| & ; ( ) { } < > ' " ` $ @ , [ ] #``; and the typographic single
and double quotes PowerShell reads as quotes. Every whitespace character (``str.isspace``) is
refused too, since PowerShell ends a word at Unicode whitespace."""


def shell_line(argv: Sequence[str]) -> str:
    """Return ``argv`` as one shell command line, each argument an unquoted word.

    Raises ``ValueError`` naming the argument when one is empty or holds whitespace or a
    character of :data:`SHELL_WORD_REFUSES`, naming those characters.
    """
    for word in argv:
        held = sorted(
            c for c in set(word) if c in SHELL_WORD_REFUSES or c.isspace()
        )
        if held or not word:
            raise ValueError(
                f"the argument {word!r} holds {held or 'nothing'!r}, which the "
                "session hook's shell line refuses"
            )
    return " ".join(argv)


type TomlValue = str | int | Sequence[TomlValue] | Mapping[str, TomlValue]
"""A value Codex's ``-c`` overrides carry: a string, an integer or boolean, or an array or table
of them."""


def session_start_hooks(
    handler: Mapping[str, TomlValue],
) -> Mapping[str, TomlValue]:
    """Return the ``hooks`` configuration running ``handler`` at every session start.

    It is one ``SessionStart`` matcher group with no matcher and the one handler, the shape
    Claude Code's settings and Codex's configuration share.
    """
    return {"SessionStart": [{"hooks": [handler]}]}


_TOML_BASIC_ESCAPES = {
    ord('"'): '\\"',
    ord("\\"): "\\\\",
    **{code: f"\\u{code:04X}" for code in (*range(0x20), 0x7F)},
}
"""What a TOML basic string must escape: the quotation mark, the backslash and every control
character (U+0000 to U+001F and U+007F); every other character is written as itself."""


def _toml_value(value: TomlValue) -> str:
    """Return ``value`` as one TOML inline value: a string, a number, an array or an inline table.

    A string is a TOML basic string, escaped by :data:`_TOML_BASIC_ESCAPES`; an integer or a
    boolean is written as JSON writes it, which TOML reads alike; a table's keys are written bare,
    so each must be a bare TOML key.
    """
    if isinstance(value, str):
        return f'"{value.translate(_TOML_BASIC_ESCAPES)}"'
    if isinstance(value, int):
        return json.dumps(value)
    if isinstance(value, Mapping):
        pairs = ", ".join(f"{k} = {_toml_value(v)}" for k, v in value.items())
        return f"{{{pairs}}}"
    return f"[{', '.join(_toml_value(item) for item in value)}]"


def codex_overrides(
    project: OpenProject | None, env_names: Iterable[str]
) -> list[str]:
    """Render a launch's Codex configuration for ``project`` as ``-c key=value`` TOML overrides.

    They are :func:`mcp_server` for ``project`` as Codex's ``tcip`` server, forwarding the
    variables ``env_names`` names to it, approving its tools without asking and required, and
    :func:`session_hook_argv` as :func:`shell_line` in :func:`session_start_hooks`, its whole
    ``additionalContext`` passed to the model. Raises what :func:`shell_line` raises.
    """
    server = mcp_server(project)
    hook: dict[str, TomlValue] = {
        "type": "command",
        "command": shell_line(session_hook_argv()),
        # Codex's hooks page: "0 to pass the handler's complete additional context directly".
        "additionalContextLimit": 0,
    }
    settings: dict[str, TomlValue] = {
        "mcp_servers.tcip.command": server.command,
        "mcp_servers.tcip.args": list(server.args),
        "mcp_servers.tcip.env_vars": sorted(env_names),
        "mcp_servers.tcip.default_tools_approval_mode": "approve",
        # Codex's configuration reference: "When true, fail startup/resume if this enabled MCP
        # server cannot initialize".
        "mcp_servers.tcip.required": True,
        "hooks": session_start_hooks(hook),
    }
    return [
        arg
        for key, value in settings.items()
        for arg in ("-c", f"{key}={_toml_value(value)}")
    ]


def run_to_completion(argv: list[str], step: str) -> str:
    """Run ``argv`` to completion, stdin closed and time bounded, and return it as one line.

    The line is ``argv`` as one command line. Raises :class:`PreparationFailedError` naming
    ``step`` when it cannot start, times out or exits non-zero, with the tail of what it
    printed.
    """
    try:
        done = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=PREPARATION_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PreparationFailedError(f"{step} did not run: {exc}") from exc
    if done.returncode != 0:
        detail = (done.stderr or done.stdout).strip()[-400:]
        raise PreparationFailedError(
            f"{step} exited {done.returncode}: {detail}"
        )
    return subprocess.list2cmdline(argv)


_JSON_OBJECT: TypeAdapter[dict[str, Any]] = TypeAdapter(dict[str, Any])


def _json_object(path: Path) -> dict[str, Any]:
    """Return the JSON object the file at ``path`` holds, an empty one when there is no file.

    Raises :class:`PreparationFailedError` when the file does not read, or when it does not hold
    a JSON object (:data:`_JSON_OBJECT` refusing it).
    """
    try:
        return (
            _JSON_OBJECT.validate_json(path.read_bytes())
            if path.is_file()
            else {}
        )
    except OSError as exc:
        raise PreparationFailedError(f"reading {path} failed: {exc}") from exc
    except ValidationError as exc:
        raise PreparationFailedError(
            f"{path} does not hold a JSON object: {exc}"
        ) from exc


def _write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` as UTF-8 with LF line ends, making its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _write_json(path: Path, body: dict) -> None:
    """Write ``body`` as indented JSON to ``path``, making its directory."""
    _write_text(path, json.dumps(body, indent=2) + "\n")


def allow_tcip_tools(settings: Path) -> str:
    """Allow every registered tool in the Antigravity settings ``settings``.

    It adds an ``mcp(tcip/<tool>)`` entry for every registered tool missing from the file's
    ``permissions.allow`` list, keeping everything else in the file, and returns one line saying
    how many were added; raises what :func:`_json_object` raises.
    """
    from tcip_web.cli import tool_listing

    body = _json_object(settings)
    allow = body.setdefault("permissions", {}).setdefault("allow", [])
    missing = [
        entry
        for entry in (f"mcp(tcip/{t.name})" for t in tool_listing())
        if entry not in allow
    ]
    if missing:
        allow += sorted(missing)
        _write_json(settings, body)
    return f"allowed {len(missing)} tcip entries in {settings}"


def prepare_antigravity(
    executable: str, project: OpenProject | None, session: str
) -> list[str]:
    """Prepare an Antigravity launch of ``executable`` for ``project`` and ``session``.

    It registers :func:`mcp_server` as ``tcip`` through ``agy mcp add``, runs
    :func:`allow_tcip_tools` on agy's settings file, sets the :data:`ANTIGRAVITY_HOOK_ENTRY`
    entry of the workspace's :data:`ANTIGRAVITY_HOOKS` to :func:`session_hook_argv` naming
    ``session``'s start route, as :func:`shell_line`, and writes the always-on rule
    :data:`ANTIGRAVITY_RULE` naming :func:`orientation_tools`; the entry and the rule replace an
    earlier launch's, leaving the rest of the hooks file and the rules directory.
    Returns one line per step.
    Raises :class:`PreparationFailedError` naming why, before anything is written, for a hook
    that cannot be spelled, a backend that recorded no port or a hooks file
    :func:`_json_object` refuses.
    """
    from tcip_web.cli.session_hook import session_start_url

    try:
        hook = shell_line(
            [
                *session_hook_argv(),
                session_start_url(store.workspace, session),
            ]
        )
    except (ValueError, LookupError) as exc:
        raise PreparationFailedError(
            f"the session hook was not registered: {exc}"
        ) from exc
    hooks = store.workspace / ANTIGRAVITY_HOOKS
    body = _json_object(hooks)
    body[ANTIGRAVITY_HOOK_ENTRY] = {
        "SessionStart": [{"type": "command", "command": hook}]
    }
    server = mcp_server(project)
    add = run_to_completion(
        [executable, "mcp", "add", "tcip", "--", server.command, *server.args],
        step="`agy mcp add`",
    )
    allowed = allow_tcip_tools(ANTIGRAVITY_SETTINGS)
    _write_json(hooks, body)
    rule = store.workspace / ANTIGRAVITY_RULE
    _write_text(
        rule,
        "---\ntrigger: always_on\n---\n"
        "This session's orientation is read from the tcip tools "
        f"{', '.join(tool.__name__ for tool in orientation_tools())}; when this "
        "conversation's earlier turns have been summarized, read it again through them.\n",
    )
    return [
        add,
        allowed,
        f"registered the session hook in {hooks}",
        f"wrote the orientation rule {rule}",
    ]


_CLAUDE_CACHE_TTLS = {"5m": timedelta(minutes=5), "1h": timedelta(hours=1)}
"""The two prompt-cache lifetimes Claude Code's ``promptCacheTtl`` setting takes, by its value
(code.claude.com/docs/en/prompt-caching, "Choose the TTL yourself")."""

CODEX_CACHE_LIFETIME = timedelta(minutes=30)
"""Provisional: the default prompt-cache lifetime OpenAI's prompt-caching guide documents for its
API, a read refreshing it (developers.openai.com/api/docs/guides/prompt-caching); Codex under a
ChatGPT login documents none."""

CLAUDE_CACHE_LIFETIME = _CLAUDE_CACHE_TTLS[
    json.loads(CLAUDE_SETTINGS.read_text(encoding="utf-8"))["promptCacheTtl"]
]
"""The prompt-cache lifetime :data:`CLAUDE_SETTINGS` requests through ``promptCacheTtl``."""

PROVIDERS: tuple[Provider, ...] = (
    Provider(
        id="claude",
        name="Claude Code",
        executable="claude",
        args=(
            "--settings",
            CLAUDE_SETTINGS_ARG,
            "--setting-sources",
            "project",
            "--add-dir",
            str(repo_root_from_here()),
            "--permission-mode",
            "default",
            "--mcp-config",
            MCP_CONFIG_ARG,
            "--strict-mcp-config",
            Resume("--resume", CLAUDE_CACHE_LIFETIME),
        ),
        composer_ready="\x1b[?1049h",
        confinement=(
            "Claude Code enforces the deny and allow lists of "
            "agent_terminal.settings.json, merged with the workspace's own project "
            "settings only, the user's own settings and instruction files not loaded; "
            "a tool call neither list decides asks first."
        ),
    ),
    Provider(
        id="antigravity",
        name="Antigravity",
        executable="agy",
        args=("--sandbox",),
        composer_ready="\x1b[0 q",
        confinement=(
            "agy's --sandbox turns on its terminal restrictions, whose extent agy "
            "defines; every tcip tool is allowed in agy's own settings file."
        ),
        prepare=prepare_antigravity,
    ),
    Provider(
        id="codex",
        name="Codex",
        executable="codex",
        args=(
            Resume("resume", CODEX_CACHE_LIFETIME),
            "--sandbox",
            "read-only",
            "--ask-for-approval",
            "never",
            "-c",
            "disable_paste_burst=true",
            CODEX_CONFIG_ARG,
        ),
        composer_ready="\x1b]0;",
        confinement=(
            "Codex's read-only sandbox holds the shell commands the model runs, which "
            "write nothing and never ask to leave it; the tcip server is not one of "
            "them, so its tools run, without a prompt, and write the project's records."
        ),
    ),
)


def _override_argv() -> list[str] | None:
    """Return ``TCIP_TERMINAL_CMD`` split into an argv, or ``None`` when unset or blank."""
    override = os.environ.get(TERMINAL_CMD_ENV, "").strip()
    if not override:
        return None
    if os.name == "nt":
        return [tok.strip('"') for tok in shlex.split(override, posix=False)]
    return shlex.split(override)


def resolve_terminal_command(
    provider: Provider,
) -> tuple[list[str | Resume], bool] | None:
    """Return the command that launches ``provider`` and whether it is the override.

    The override is ``TCIP_TERMINAL_CMD``'s argv when one is set; else the command is the row's
    executable resolved on ``PATH`` followed by its arguments with their placeholders unrendered.
    ``None`` when the executable is not on ``PATH``.
    """
    override = _override_argv()
    if override is not None:
        return [*override], True
    executable = shutil.which(provider.executable)
    if executable is None:
        return None
    return [executable, *provider.args], False


def _write_launch_file(name: str, body: dict) -> Path:
    """Write ``body`` as JSON to ``name`` in a new temporary directory and return its path."""
    dest = Path(tempfile.mkdtemp(prefix="tcip_launch_")) / name
    _write_json(dest, body)
    return dest


def write_mcp_config(project: OpenProject | None) -> Path:
    """Write :func:`mcp_server` for ``project`` as a JSON MCP configuration; return its path."""
    server = mcp_server(project)
    return _write_launch_file(
        "tcip.mcp.json",
        {
            "mcpServers": {
                "tcip": {"command": server.command, "args": list(server.args)}
            }
        },
    )


def write_claude_settings() -> Path:
    """Write :data:`CLAUDE_SETTINGS` with the session hook added; return the written file's path.

    The hook is :func:`session_hook_argv` in :func:`session_start_hooks`, in Claude Code's exec
    form: the interpreter as its ``command`` and the rest as its ``args``, spawned with no shell.
    """
    command, *args = session_hook_argv()
    body = json.loads(CLAUDE_SETTINGS.read_text(encoding="utf-8"))
    body["hooks"] = session_start_hooks(
        {"type": "command", "command": command, "args": args}
    )
    return _write_launch_file("settings.json", body)


def render_argv(
    argv: Sequence[str | Resume],
    project: OpenProject | None,
    env: dict[str, str],
    conversation: str | None,
) -> list[str]:
    """Return ``argv`` with each placeholder argument rendered for a launch for ``project``.

    Each :data:`MCP_CONFIG_ARG` is replaced by a configuration :func:`write_mcp_config` writes, each
    :data:`CLAUDE_SETTINGS_ARG` by the settings :func:`write_claude_settings` writes, and each
    :data:`CODEX_CONFIG_ARG` by :func:`codex_overrides` forwarding every variable of ``env``,
    the environment the launch spawns with; a :class:`Resume` is replaced by its argument and
    ``conversation``, or by nothing for a launch resuming none. Raises what
    :func:`codex_overrides` raises.
    """
    values = {
        MCP_CONFIG_ARG: lambda: [str(write_mcp_config(project))],
        CLAUDE_SETTINGS_ARG: lambda: [str(write_claude_settings())],
        CODEX_CONFIG_ARG: lambda: codex_overrides(project, env),
    }

    def rendered(arg: str | Resume) -> list[str]:
        if isinstance(arg, Resume):
            return [] if conversation is None else [arg.argument, conversation]
        return values[arg]() if arg in values else [arg]

    return [part for arg in argv for part in rendered(arg)]


def prepare_launch(
    executable: str,
    provider: Provider,
    project: OpenProject | None,
    session: str,
) -> list[str] | str:
    """Run ``provider.prepare`` with the resolved ``executable``, ``project`` and ``session``.

    Returns the steps it took (empty for a row with no preparation), or the reason it failed,
    prefixed with the row's name.
    """
    if provider.prepare is None:
        return []
    try:
        return provider.prepare(executable, project, session)
    except PreparationFailedError as exc:
        return f"{provider.name}'s launch preparation failed: {exc}"


_PRIVATE_MODE = re.compile(r"\x1b\[\?([0-9;]*)([hl])")
_BRACKETED_PASTE = "2004"


def bracketed_paste(output: str, enabled: bool) -> bool:
    """Return whether bracketed paste is enabled after processing ``output`` from ``enabled``."""
    for params, final in _PRIVATE_MODE.findall(output):
        if _BRACKETED_PASTE in params.split(";"):
            enabled = final == "h"
    return enabled


def paste(text: str) -> str:
    """Return ``text`` as the agent's input: a bracketed paste followed by Enter.

    Each line break inside it is sent as the carriage return a terminal sends for a pasted
    newline.
    """
    return (
        f"\x1b[200~{text.replace('\r\n', '\n').replace('\n', '\r')}\x1b[201~\r"
    )


SessionStartSource = Literal["startup", "resume", "clear", "compact", "fork"]
"""How a harness's session started, as Claude Code's and Codex's hooks pages each name the
``source`` of a ``SessionStart`` hook's input."""

ORIENTATION_CAP = 10_000
"""The most characters an orientation is cut to: the cap Claude Code's hooks page states for a
hook's plain stdout and its ``additionalContext``, which no setting raises
(code.claude.com/docs/en/hooks, "JSON output")."""


class _Part(NamedTuple):
    """One part of an orientation, cut to fit.

    It is its text holding its first ``k`` items, how many items it has, and the short fixed
    text that stands for it when not even its rendering of none fits.
    """

    render: Callable[[int], str]
    items: int
    omitted: str


def _whole(text: str, omitted: str) -> _Part:
    """Return the part of ``text`` alone, which has no items to cut."""
    return _Part(lambda _: text, 0, omitted)


def _fitted(parts: Sequence[_Part]) -> str:
    """Return ``parts`` in their order, joined by blank lines, out of :data:`ORIENTATION_CAP`.

    Every part stays whole while all of them whole fit the cap together. When they do not, the
    room is shared max-min fair: a part whose whole needs no more than an equal share of what
    remains keeps all of it, and the others each take an equal share of the rest, holding the
    most items that share fits, or standing as their ``omitted`` text when their rendering of
    none exceeds it. The whole stays within the cap while the separators fit it and each
    ``omitted`` text fits the share it stands in.
    """
    room = ORIENTATION_CAP - 2 * (len(parts) - 1)
    whole = [part.render(part.items) for part in parts]
    texts: dict[int, str] = {}
    waiting = sorted(range(len(parts)), key=lambda i: len(whole[i]))
    while waiting and len(whole[waiting[0]]) <= room // len(waiting):
        kept = waiting.pop(0)
        texts[kept] = whole[kept]
        room -= len(whole[kept])
    for i in waiting:
        share, part = room // len(waiting), parts[i]
        held = next(
            (k for k in range(part.items + 1) if len(part.render(k)) > share),
            part.items + 1,
        )
        texts[i] = part.render(held - 1) if held else part.omitted
    return "\n\n".join(texts[i] for i in range(len(parts)))


def orientation_tools() -> tuple[Callable[..., Any], ...]:
    """Return the tools an orientation is read from.

    They are the project's overview, its findings, its memory and the domain knowledge, in that
    order.
    """
    from tcip_mcp.tools.doctor_tools import diagnose_project
    from tcip_mcp.tools.knowledge_tools import serve_domain_knowledge
    from tcip_mcp.tools.meta_tools import load_project_memory
    from tcip_mcp.tools.project_tools import inspect_project

    return (
        inspect_project,
        diagnose_project,
        load_project_memory,
        serve_domain_knowledge,
    )


def orientation(
    project: OpenProject | None, source: SessionStartSource
) -> str | None:
    """Return what the agent of a session for ``project`` started by ``source`` is told.

    It is ``None`` for a ``resume``; otherwise the parts read from :func:`orientation_tools`
    that the start calls for, fit by :func:`_fitted`.
    """
    from tcip_mcp.knowledge import list_documents
    from tcip_mcp.tools.meta_tools import MemoryKind
    from tcip_mcp.tools.project_tools import initialize_project

    if source == "resume":
        return None
    overview, diagnose, memory, serve = orientation_tools()
    served_by = serve.__name__
    knowledge = _whole(
        "The platform's domain knowledge is the documents "
        f"{', '.join(d.name for d in list_documents())}, each served whole by "
        f"{served_by}(name) and, to a harness that reads skills, as the skill of its name.",
        f"The platform's domain knowledge documents are served by {served_by}, whose index "
        "names them.",
    )
    if project is None:
        no_project = (
            "This session was launched with no project open in the GUI. Its project tools "
            f"refuse until {initialize_project.__name__} creates a project, and act on that "
            "project from then on; that project's memory is read through "
            f"{memory.__name__}. An existing project is worked on by opening it "
            "in the GUI, then ending this agent from the rail and starting another."
        )
        return _fitted([_whole(no_project, no_project), knowledge])
    parts = [_memory_part(kind, project.root, memory) for kind in MemoryKind]
    if source != "compact":
        found = diagnose(project.root)["findings"]
        counted = (
            f"{diagnose.__name__} answers {len(found)} finding(s) over it"
        )
        parts[:0] = [
            _whole(
                f"This session's project, as {overview.__name__} answers for it: "
                f"{json.dumps(overview(project.root))}",
                "This session's project's overview is longer than this orientation holds; "
                f"{overview.__name__} answers it.",
            ),
            _Part(
                lambda k: (
                    f"{counted}; the first {k} of them, errors first: "
                    f"{json.dumps(found[:k])}"
                ),
                len(found),
                f"{counted}.",
            ),
        ]
    return _fitted([*parts, knowledge])


def _memory_part(
    kind: MemoryKind, project: Path, tool: Callable[..., Any]
) -> _Part:
    """Return the part of the newest entries of ``kind`` under ``project``.

    Its text holds them as the memory tool ``tool`` pages them, ``total_available`` among them.
    """
    from tcip_mcp.tools.meta_tools import memory_entries, memory_page

    entries = memory_entries(kind.record, str(project))
    call = f"{tool.__name__}(kind='{kind.value}')"

    def page(k: int) -> str:
        paged = memory_page(entries, k).model_dump(mode="json")
        return (
            f"Its newest {k} {kind.value}, paged as {call} pages them: "
            f"{json.dumps(paged)}"
        )

    return _Part(
        page,
        len(entries),
        f"Its {len(entries)} {kind.value} are read through {call}.",
    )


VERSION_PROBE_TIMEOUT_S = 15


def launched_program(argv: list[str], override: bool) -> dict:
    """Return ``{"argv", "version"}`` for ``argv``.

    That is the argv itself, and what its executable, the first element, declares to
    ``--version`` (probed at every call, stdin closed, time bounded), ``None`` for an
    ``override`` or an executable that does not answer cleanly.
    """
    executable = argv[0]
    if override:
        return {"argv": argv, "version": None}
    version: str | None = None
    try:
        probe = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=VERSION_PROBE_TIMEOUT_S,
            check=False,
        )
        first_line = probe.stdout.strip().splitlines()
        if probe.returncode == 0 and first_line:
            version = first_line[0].strip()
    except (OSError, subprocess.TimeoutExpired):
        logger.debug(
            "version probe of %s did not answer", executable, exc_info=True
        )
    return {"argv": argv, "version": version}


WITHHELD_ENV_PREFIXES = (
    "CLAUDE",
    "FORCE_PROMPT_CACHING_5M",
    "DISABLE_PROMPT_CACHING",
)
"""The prefixes of the variables no child inherits: every variable a Claude Code session sets,
among them ``CLAUDE_CODE_PROMPT_CACHE_TTL``, and the controls Claude Code's prompt-caching page
ranks above ``promptCacheTtl`` ("``FORCE_PROMPT_CACHING_5M=1``, which forces five minutes for
both buckets") or that turn its caching off (``DISABLE_PROMPT_CACHING`` and its ``_HAIKU``,
``_SONNET``, ``_OPUS`` and ``_FABLE`` variants), so the lifetime Claude's row requests is the one
its harness runs with (code.claude.com/docs/en/prompt-caching)."""


def spawn_env(session_id: str) -> dict[str, str]:
    """Return the child's environment: this process's own plus the terminal session id.

    Every variable named by one of :data:`WITHHELD_ENV_PREFIXES` is left out.
    """
    kept = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(WITHHELD_ENV_PREFIXES)
    }
    return {**kept, TERMINAL_SESSION_ENV: session_id}


def _prewarm_blocking() -> None:
    """Import the MCP server's tool graph."""
    try:
        import importlib

        importlib.import_module("tcip_mcp.server")
    except Exception:
        logger.debug("prewarm: tcip_mcp import failed", exc_info=True)


def prewarm() -> None:
    """Kick off best-effort cold-cache warming on a daemon thread; never blocks web startup.

    All failures are swallowed.
    """
    threading.Thread(
        target=_prewarm_blocking, name="terminal-prewarm", daemon=True
    ).start()


def launch_problem(provider: Provider) -> str | None:
    """Return why ``provider`` cannot launch here, or ``None`` when it can.

    The reasons are no PTY backend, or its executable is not on ``PATH``.
    """
    if os.name == "nt":
        try:
            import winpty  # noqa: F401
        except ImportError:
            return "pywinpty is not installed (pip install pywinpty)"
    return (
        provider.unavailable_reason
        if resolve_terminal_command(provider) is None
        else None
    )


# ── PTY backends ────────────────────────────────────────────────────────


class _WinPty:
    """ConPTY via pywinpty. ``read`` returns decoded text (pywinpty decodes)."""

    def __init__(
        self,
        argv: list[str],
        cwd: str,
        rows: int,
        cols: int,
        env: dict[str, str],
    ) -> None:
        from winpty import PtyProcess

        self._p = PtyProcess.spawn(
            argv, cwd=cwd, dimensions=(rows, cols), env=env
        )
        self.pid = self._p.pid

    def read(self) -> str:
        return self._p.read(4096)  # blocking; raises EOFError on process exit

    def write(self, data: bytes) -> int:
        """Hand all of ``data`` to ConPTY's input pipe and return the bytes it took, or raise.

        pywinpty 3.1.2's write is winpty-rs 1.0.8's: one overlapped ``WriteFile`` of the whole
        payload, waited on for up to ten seconds, its count the UTF-8 bytes the pipe took; a
        write still pending at that bound is cancelled and raised.
        """
        return self._p.write(data.decode("utf-8"))

    def resize(self, rows: int, cols: int) -> None:
        self._p.setwinsize(rows, cols)

    def isalive(self) -> bool:
        return self._p.isalive()

    def terminate(self) -> None:
        # Liveness guard: killing a long-dead pid risks tree-killing whatever unrelated
        # process now owns it (Windows recycles pids aggressively).
        if not self._p.isalive():
            return
        # /T tree-kill: the agent spawns children (its MCP servers) that must not orphan.
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(self.pid)],
            capture_output=True,
            check=False,
        )
        with suppress(psutil.NoSuchProcess, psutil.TimeoutExpired):
            psutil.Process(self.pid).wait(timeout=TERMINATE_WAIT_S)

    def close(self) -> None:
        """Perform no explicit cleanup."""

    def close_input(self) -> None:
        """Perform no explicit cleanup."""


class _PosixPty:
    """The stdlib ``pty`` plus a subprocess."""

    # __init__'s body is unreachable to a Windows-targeted mypy run (its sys.platform guard),
    # so these are declared here rather than left to its own inference.
    _master: int
    _input: int
    _proc: subprocess.Popen
    _decoder: codecs.IncrementalDecoder
    pid: int

    def __init__(
        self,
        argv: list[str],
        cwd: str,
        rows: int,
        cols: int,
        env: dict[str, str],
    ) -> None:
        # The guard tells mypy the POSIX-only stdlib members below are never checked on Windows.
        if sys.platform == "win32":
            raise AssertionError("_PosixPty is POSIX-only")
        import pty
        import termios

        # ``owned`` closes the master and the writer's descriptor unless the spawn succeeds.
        with ExitStack() as owned, ExitStack() as spent:
            self._master, slave = pty.openpty()
            spent.callback(os.close, slave)
            owned.callback(os.close, self._master)
            # The writer's own descriptor, so the reader's close never frees a number it writes on.
            self._input = os.dup(self._master)
            owned.callback(os.close, self._input)
            termios.tcsetwinsize(slave, (rows, cols))
            self._proc = subprocess.Popen(
                argv,
                cwd=cwd,
                env=env,
                stdin=slave,
                stdout=slave,
                stderr=slave,
                start_new_session=True,  # its own process group → killpg cleans the tree
                close_fds=True,
            )
            owned.pop_all()
        self.pid = self._proc.pid
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def read(self) -> str:
        chunk = os.read(self._master, 4096)
        if not chunk:
            raise EOFError
        return self._decoder.decode(chunk)

    def write(self, data: bytes) -> int:
        return os.write(self._input, data)

    def resize(self, rows: int, cols: int) -> None:
        if sys.platform == "win32":
            raise AssertionError("_PosixPty is POSIX-only")
        import termios

        termios.tcsetwinsize(self._master, (rows, cols))

    def isalive(self) -> bool:
        return self._proc.poll() is None

    def terminate(self) -> None:
        if sys.platform == "win32":
            raise AssertionError("_PosixPty is POSIX-only")
        import signal

        # Liveness guard: once poll() has reaped the child, its pid (and pgid) may have
        # been recycled; killpg would then hit an innocent process group.
        if self._proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(self.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self._proc.wait(timeout=TERMINATE_WAIT_S)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self._proc.wait(
                    timeout=TERMINATE_WAIT_S
                )  # reap: a permanent zombie pins the pid
            except subprocess.TimeoutExpired:  # pragma: no cover
                pass
        # The master fd is closed by the reader thread (see close()) after it drains to
        # EOF; closing it here would let a concurrent openpty() reuse the fd number
        # while the old reader is still blocked in os.read on it.

    def close(self) -> None:
        """Close the master file descriptor, ignoring ``OSError``."""
        try:
            os.close(self._master)
        except OSError:
            pass

    def close_input(self) -> None:
        """Close the descriptor ``write`` writes on."""
        os.close(self._input)


type Pty = _WinPty | _PosixPty
"""A platform PTY: ConPTY through pywinpty on Windows, the stdlib ``pty`` elsewhere.

Its ``write`` takes UTF-8 bytes, writes a leading part of them and returns that part's length,
raising when the PTY refuses them; ``close_input`` releases what ``write`` writes on.
"""


def spawn_pty(
    argv: list[str], cwd: str, rows: int, cols: int, env: dict[str, str]
) -> Pty:
    """Spawn ``argv`` attached to a platform PTY under ``env``.

    Raises ``OSError`` on failure.
    """
    if os.name == "nt":
        return _WinPty(argv, cwd, rows, cols, env)
    return _PosixPty(argv, cwd, rows, cols, env)


# ── The reader and writer pumps ─────────────────────────────────────────


def start_reader(
    pty: Pty,
    on_output: Callable[[str], None],
    on_exit: Callable[[], object],
    name: str,
) -> threading.Thread:
    """Pump PTY output on a daemon thread until the process exits, then call ``on_exit``.

    Any read failure ends the pump. The reader closes what it reads from (``pty.close()``) as it
    ends, so an fd can never be recycled while a read is still blocked on it.
    """

    def _loop() -> None:
        try:
            while True:
                data = pty.read()
                if data:
                    on_output(data)
        except Exception:
            logger.debug("terminal reader ended", exc_info=True)
        finally:
            try:
                pty.close()
            except Exception:  # pragma: no cover - close is best-effort
                pass
            on_exit()

    t = threading.Thread(target=_loop, name=name, daemon=True)
    t.start()
    return t


type Input = tuple[str, Callable[[], object]]
"""Text for the agent's input and what to call once it is written whole."""


class _PtyInput(io.RawIOBase):
    """The PTY's input as a raw stream, refusing every write once ``inputs`` is shut down."""

    def __init__(self, pty: Pty, inputs: Queue[Input]) -> None:
        self._pty = pty
        self._inputs = inputs

    def writable(self) -> bool:
        return True

    def write(self, b: Buffer) -> int:
        """Write a leading part of ``b`` to the PTY and return its length.

        Raises ``queue.ShutDown``, writing nothing, once ``inputs`` is shut down.
        """
        if self._inputs.is_shutdown:
            raise ShutDown
        return self._pty.write(bytes(b))

    def close(self) -> None:
        self._pty.close_input()
        super().close()


def start_writer(pty: Pty, inputs: Queue[Input], name: str) -> None:
    """Write each input taken from ``inputs`` to ``pty`` on a daemon thread, in order.

    Each is written as UTF-8, and its callback is called once it is written whole. The writer
    stops at the first PTY write it finds ``inputs`` shut down before, the rest of an input it
    did not finish dropped and its callback never called, or once a write or callback fails,
    shutting ``inputs`` down so no later input is accepted; then it closes the PTY's input.
    """

    def _loop() -> None:
        stream = io.BufferedWriter(_PtyInput(pty, inputs))
        try:
            while True:
                text, written = inputs.get()
                stream.write(text.encode("utf-8"))
                stream.flush()
                written()
        except Exception:
            logger.debug("terminal writer ended", exc_info=True)
        finally:
            inputs.shutdown(immediate=True)
            # Closed here, as the finalizer would report the refused flush of an unfinished input.
            with suppress(ShutDown):
                stream.close()

    threading.Thread(target=_loop, name=name, daemon=True).start()

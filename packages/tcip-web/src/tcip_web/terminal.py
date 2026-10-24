"""Embedded agent terminal: an agent harness from :data:`PROVIDERS` spawned in a pseudo-terminal.

The pseudo-terminal is ConPTY via ``pywinpty`` on Windows and the stdlib ``pty`` on POSIX; its
raw bytes are streamed out and keystrokes streamed in. ``TCIP_TERMINAL_CMD`` overrides the spawn
command.
"""

from __future__ import annotations

import codecs
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
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path, PurePath

from tcip_mcp.agent_identity import TERMINAL_SESSION_ENV
from tcip_mcp.project_paths import repo_root_from_here
from tcip_web.state import OpenProject

logger = logging.getLogger(__name__)

TERMINAL_CMD_ENV = "TCIP_TERMINAL_CMD"

DEFAULT_ROWS = 30
DEFAULT_COLS = 100

# Seconds terminate() waits for the spawned process to actually exit: the Windows path polls
# isalive() after taskkill returns; the POSIX path bounds its wait() calls with it.
TERMINATE_WAIT_S = 5

MCP_CONFIG_ARG = "{mcp_config}"
"""A whole argument: the path of the launch's JSON MCP configuration."""
CODEX_MCP_ARG = "{codex_mcp}"
"""A whole argument: the launch's Codex ``tcip`` server overrides, as several arguments."""
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


class PreparationFailedError(Exception):
    """A row's launch preparation did not complete; the message names the step and why."""


@dataclass(frozen=True)
class ProjectLaunch:
    """A launch for an open project: the project and its ritual's doctor step.

    The doctor step is the line or the reason it cannot be spelled, exactly one of the two set.
    """

    project: OpenProject
    doctor_line: str | None
    doctor_refused: str | None


PrepareFn = Callable[[str, ProjectLaunch | None], list[str]]
"""A row's launch preparation: given the resolved executable and the launch's
:class:`ProjectLaunch` (``None`` for no project), it runs to completion and returns one line per
step it took, or raises :class:`PreparationFailedError`."""


@dataclass(frozen=True)
class Provider:
    """One agent harness the terminal launches, its launch and what that launch confines.

    It carries the id a session names it by, its display name, the executable looked up on
    ``PATH``, the arguments after it, which may include the placeholders
    :data:`MCP_CONFIG_ARG`, :data:`CLAUDE_SETTINGS_ARG` and :data:`CODEX_MCP_ARG` as whole
    arguments, the output sequence one installed version of the harness was recorded writing
    when its composer appeared, what the harness's own enforcement restricts under those
    arguments and leaves open, and the preparation the launch runs first, ``None`` for none.
    """

    id: str
    name: str
    executable: str
    args: tuple[str, ...]
    composer_ready: str
    confinement: str
    prepare: PrepareFn | None = None

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


def mcp_server(launch: ProjectLaunch | None) -> McpServer:
    """Return :func:`backend_interpreter` running ``tcip_mcp`` for ``launch``'s project.

    The server runs for that project's directory, or for no project when ``launch`` is ``None``.
    """
    project = ("--project", launch.project.root.as_posix()) if launch else ()
    return McpServer(backend_interpreter(), ("-m", "tcip_mcp", *project))


def codex_mcp_overrides(
    launch: ProjectLaunch | None, env_names: Iterable[str]
) -> list[str]:
    """Render :func:`mcp_server` for ``launch`` as Codex's ``tcip`` server in ``-c`` overrides.

    The overrides are ``-c key=value``, each value TOML (a JSON string or array of strings is
    valid TOML), forwarding the variables ``env_names`` names to the server and approving its
    tools without asking.
    """
    server = mcp_server(launch)
    settings = {
        "command": server.command,
        "args": list(server.args),
        "env_vars": sorted(env_names),
        "default_tools_approval_mode": "approve",
    }
    return [
        arg
        for key, value in settings.items()
        for arg in ("-c", f"mcp_servers.tcip.{key}={json.dumps(value)}")
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


DOCTOR_QUOTED_REFUSES = frozenset('"$`\\!\r\n“”„')
"""The characters the doctor line refuses in its double-quoted project path, so the path is one
argument on one line under Bash and PowerShell: the double quote and the typographic double
quotes PowerShell reads as one, which close the quoted segment; the dollar, which both shells
expand inside double quotes; the backtick, a command substitution under Bash and the escape
character under PowerShell; the backslash, an escape under Bash before some of the characters
that may follow it; Bash's history expansion mark, expanded in an interactive shell with history
expansion on; and a line break, which the ritual's one-line contract excludes. A typographic
single quote is literal inside double quotes under both shells and is admitted."""

DOCTOR_TOKEN_REFUSES = frozenset("|&;()<>'\"$`\\!*?[]{}~#@,‘’‚‛“”„")
"""The characters besides whitespace the doctor line refuses in its unquoted interpreter, so it
is one word naming one command under Bash and PowerShell: what Bash reads outside quotes as an
operator, a quote, an expansion, an escape or a comment, ``| & ; ( ) < > ' " $ ` \\ ! * ? [ ] {
} ~ #``; what PowerShell reads there as one, ``| & ; ( ) { } < > ' " ` $ @ , [ ] #``; and the
typographic single and double quotes PowerShell reads as quotes. Every whitespace character
(``str.isspace``) is refused there too, since PowerShell ends a word at Unicode whitespace."""

_DOCTOR_LINE = '{} -m tcip_web.cli doctor "{}"'
"""The doctor line with a ``{}`` for each of :data:`_DOCTOR_ARGUMENTS`."""

_DOCTOR_ARGUMENTS = (
    ("interpreter", DOCTOR_TOKEN_REFUSES, True),
    ("project path", DOCTOR_QUOTED_REFUSES, False),
)
"""Each argument of :data:`_DOCTOR_LINE`, in order, with the characters it refuses and whether
it refuses every whitespace character as well."""

_REGEX_LITERAL_ESCAPES = frozenset(".^$*+?()[]{}|\\")


def _regex_literal(text: str) -> str:
    """Return ``text`` as a regular expression matching exactly it.

    A backslash goes before each of the metacharacters RE2, JavaScript and Python share.
    """
    return "".join(
        f"\\{c}" if c in _REGEX_LITERAL_ESCAPES else c for c in text
    )


def doctor_command(interpreter: str, project: PurePath) -> str:
    """Return the one spelling of the ritual's doctor step for ``project``.

    It is :data:`_DOCTOR_LINE` of ``interpreter``, unquoted, and the project's path,
    forward-slashed, in double quotes. Raises
    ``ValueError`` naming the argument when one is empty or holds a character
    :data:`_DOCTOR_ARGUMENTS` refuses it, naming those characters.
    """
    values = (interpreter, project.as_posix())
    for (name, refused, blank), value in zip(_DOCTOR_ARGUMENTS, values):
        held = sorted(
            c for c in set(value) if c in refused or (blank and c.isspace())
        )
        if held or not value:
            raise ValueError(
                f"the {name} {value!r} holds {held or 'nothing'!r}, which the "
                "doctor line refuses"
            )
    return _DOCTOR_LINE.format(*values)


def project_launch(project: OpenProject | None) -> ProjectLaunch | None:
    """Return the launch for ``project``, ``None`` for no project, its doctor step spelled once.

    The step is :func:`doctor_command` of :func:`backend_interpreter`, or the refusal it raises.
    """
    if project is None:
        return None
    try:
        return ProjectLaunch(
            project, doctor_command(backend_interpreter(), project.root), None
        )
    except ValueError as exc:
        return ProjectLaunch(project, None, str(exc))


def _command_entry(line: str) -> str:
    """Return Agy's allow entry for exactly the command ``line``.

    The entry is in the regex form agy's settings spell a command with arguments: the line
    between anchors as :func:`_regex_literal`.
    """
    return f"command(regex:^{_regex_literal(line)}$)"


def _entry_argument(refused: frozenset[str], blank: bool) -> str:
    r"""Return a regular expression matching one argument in its backslash-escaped form.

    The argument is one or more of: a character outside ``refused``, not one of the escaped
    metacharacters and, when ``blank``, not whitespace (``\s``, the characters ``str.isspace``
    admits); or a backslash before a metacharacter ``refused`` does not hold.
    """
    bare = re.escape("".join(sorted(refused | _REGEX_LITERAL_ESCAPES))) + (
        "\\s" if blank else ""
    )
    escaped = re.escape("".join(sorted(_REGEX_LITERAL_ESCAPES - refused)))
    return f"(?:[^{bare}]|\\\\[{escaped}])+"


_ENTRY_PARTS = _command_entry(
    _DOCTOR_LINE.format(*("\0" for _ in _DOCTOR_ARGUMENTS))
).split("\0")
_DOCTOR_ENTRY = re.compile(
    re.escape(_ENTRY_PARTS[0])
    + "".join(
        _entry_argument(refused, blank) + re.escape(part)
        for (_, refused, blank), part in zip(
            _DOCTOR_ARGUMENTS, _ENTRY_PARTS[1:]
        )
    )
)
"""A pattern matching the :func:`_command_entry` of :data:`_DOCTOR_LINE` with any arguments
:data:`_DOCTOR_ARGUMENTS` admits."""


def _unsandboxed_entry(line: str) -> str:
    """Return Agy's entry allowing exactly the command ``line`` to run outside its sandbox."""
    return f"unsandboxed({line})"


_LINE_PARTS = _DOCTOR_LINE.format(*("\0" for _ in _DOCTOR_ARGUMENTS)).split(
    "\0"
)
_UNSANDBOXED_DOCTOR_ENTRY = re.compile(
    re.escape(_unsandboxed_entry("\0")).replace(
        "\0",
        re.escape(_LINE_PARTS[0])
        + "".join(
            "[^"
            + re.escape("".join(sorted(refused)))
            + ("\\s" if blank else "")
            + "]+"
            + re.escape(part)
            for (_, refused, blank), part in zip(
                _DOCTOR_ARGUMENTS, _LINE_PARTS[1:]
            )
        ),
    )
)
"""A pattern matching the :func:`_unsandboxed_entry` of :data:`_DOCTOR_LINE` with any arguments
:data:`_DOCTOR_ARGUMENTS` admits."""


def allow_tcip_tools(settings: Path, doctor_line: str | None) -> str:
    """Allow every registered tool and the doctor line in the Antigravity settings ``settings``.

    It adds an ``mcp(tcip/<tool>)`` entry for every registered tool and, when given, the
    :func:`_command_entry` and the :func:`_unsandboxed_entry` of ``doctor_line`` to the file's
    ``permissions.allow`` list, dropping every other doctor entry of either kind and keeping
    everything else in it, and returns one line saying how many were added and dropped; raises
    :class:`PreparationFailedError` when the file does not parse as a JSON object.
    """
    from tcip_mcp.server import list_registered_tools

    try:
        body = (
            json.loads(settings.read_text(encoding="utf-8"))
            if settings.is_file()
            else {}
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationFailedError(
            f"reading {settings} failed: {exc}"
        ) from exc
    if not isinstance(body, dict):
        raise PreparationFailedError(f"{settings} does not hold a JSON object")
    allow = body.setdefault("permissions", {}).setdefault("allow", [])
    wanted = [f"mcp(tcip/{name})" for name in list_registered_tools()]
    if doctor_line is not None:
        wanted.append(_command_entry(doctor_line))
        wanted.append(_unsandboxed_entry(doctor_line))
    stale = [
        entry
        for entry in allow
        if entry not in wanted
        and (
            _DOCTOR_ENTRY.fullmatch(entry)
            or _UNSANDBOXED_DOCTOR_ENTRY.fullmatch(entry)
        )
    ]
    missing = [entry for entry in wanted if entry not in allow]
    if missing or stale:
        allow[:] = [entry for entry in allow if entry not in stale] + missing
        settings.parent.mkdir(parents=True, exist_ok=True)
        settings.write_text(
            json.dumps(body, indent=2) + "\n", encoding="utf-8"
        )
    return (
        f"allowed {len(missing)} tcip entries in {settings}, dropped {len(stale)} other "
        "doctor entries"
    )


def prepare_antigravity(
    executable: str, launch: ProjectLaunch | None
) -> list[str]:
    """Prepare an Antigravity launch: register the tcip server and allow its tools and doctor line.

    The server is :func:`mcp_server` for ``launch``, registered as ``tcip`` through ``agy mcp
    add``; every tcip tool and the launch's doctor line, as a command and as an unsandboxed
    run, are allowed in its settings file. A doctor step that could not be spelled fails the
    preparation naming why.
    """
    server = mcp_server(launch)
    add = run_to_completion(
        [executable, "mcp", "add", "tcip", "--", server.command, *server.args],
        step="`agy mcp add`",
    )
    if launch is not None and launch.doctor_refused is not None:
        raise PreparationFailedError(
            f"the doctor allowance was not written: {launch.doctor_refused}"
        )
    return [
        add,
        allow_tcip_tools(
            ANTIGRAVITY_SETTINGS, launch.doctor_line if launch else None
        ),
    ]


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
        ),
        composer_ready="\x1b[?1049h",
        confinement=(
            "Claude Code enforces the deny and allow lists of "
            "agent_terminal.settings.json, with the read-only tcip console commands "
            "allowed and, when the launch spelled it, the ritual's own doctor line, "
            "merged with the workspace's own project settings only, the user's own "
            "settings and instruction files not loaded; a tool call neither list "
            "decides asks first."
        ),
    ),
    Provider(
        id="antigravity",
        name="Antigravity",
        executable="agy",
        args=("--sandbox",),
        composer_ready="\x1b[0 q\r\x1b[J",
        confinement=(
            "agy's --sandbox turns on its terminal restrictions, whose extent agy "
            "defines; every tcip tool and, when the launch spelled it, the ritual's own "
            "read-only doctor line are allowed in agy's own settings file."
        ),
        prepare=prepare_antigravity,
    ),
    Provider(
        id="codex",
        name="Codex",
        executable="codex",
        args=(
            "--sandbox",
            "read-only",
            "--ask-for-approval",
            "never",
            "-c",
            "disable_paste_burst=true",
            CODEX_MCP_ARG,
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
) -> tuple[list[str], bool] | None:
    """Return the command that launches ``provider`` and whether it is the override.

    The override is ``TCIP_TERMINAL_CMD``'s argv when one is set; else the command is the row's
    executable resolved on ``PATH`` followed by its arguments with their placeholders unrendered.
    ``None`` when the executable is not on ``PATH``.
    """
    override = _override_argv()
    if override is not None:
        return override, True
    executable = shutil.which(provider.executable)
    if executable is None:
        return None
    return [executable, *provider.args], False


def _write_launch_file(name: str, body: dict) -> Path:
    """Write ``body`` as JSON to ``name`` in a new temporary directory and return its path."""
    dest = Path(tempfile.mkdtemp(prefix="tcip_launch_")) / name
    dest.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return dest


def write_mcp_config(launch: ProjectLaunch | None) -> Path:
    """Write :func:`mcp_server` for ``launch`` as a JSON MCP configuration and return its path."""
    server = mcp_server(launch)
    return _write_launch_file(
        "tcip.mcp.json",
        {
            "mcpServers": {
                "tcip": {"command": server.command, "args": list(server.args)}
            }
        },
    )


READ_ONLY_CONSOLE_COMMANDS = (
    "scan-dataset",
    "inspect-compute-resources",
    "render-failure-cases",
)
"""The ``tcip`` console commands Claude's row allows under any arguments, each read-only."""

_CONSOLE_INVOCATIONS = ("python -m tcip_web.cli", "tcip")
"""The two invocations of the ``tcip`` console command: its module and its script."""


def write_claude_settings(launch: ProjectLaunch | None) -> Path:
    """Write :data:`CLAUDE_SETTINGS` with the console and doctor entries added; return its path.

    A ``Bash(<line>)`` and a ``PowerShell(<line>)`` entry are added to its allow list for each
    of :data:`READ_ONLY_CONSOLE_COMMANDS` under each of :data:`_CONSOLE_INVOCATIONS`, with any
    arguments, and for the launch's doctor line when it spelled one.
    """
    body = json.loads(CLAUDE_SETTINGS.read_text(encoding="utf-8"))
    lines = [
        f"{invocation} {command}:*"
        for invocation in _CONSOLE_INVOCATIONS
        for command in READ_ONLY_CONSOLE_COMMANDS
    ]
    if launch is not None and launch.doctor_line is not None:
        lines.append(launch.doctor_line)
    body["permissions"]["allow"] += [
        f"{shell}({line})"
        for shell in ("Bash", "PowerShell")
        for line in lines
    ]
    return _write_launch_file("settings.json", body)


def render_argv(
    argv: list[str], launch: ProjectLaunch | None, env: dict[str, str]
) -> list[str]:
    """Return ``argv`` with each placeholder argument rendered for ``launch``.

    Each :data:`MCP_CONFIG_ARG` is replaced by a configuration :func:`write_mcp_config` writes, each
    :data:`CLAUDE_SETTINGS_ARG` by the settings :func:`write_claude_settings` writes, and each
    :data:`CODEX_MCP_ARG` by :func:`codex_mcp_overrides` forwarding every variable of ``env``,
    the environment the launch spawns with, each for ``launch``.
    """
    values = {
        MCP_CONFIG_ARG: lambda: [str(write_mcp_config(launch))],
        CLAUDE_SETTINGS_ARG: lambda: [str(write_claude_settings(launch))],
        CODEX_MCP_ARG: lambda: codex_mcp_overrides(launch, env),
    }
    return [
        rendered
        for arg in argv
        for rendered in (values[arg]() if arg in values else [arg])
    ]


def prepare_launch(
    executable: str, provider: Provider, launch: ProjectLaunch | None
) -> list[str] | str:
    """Run ``provider.prepare`` with the resolved ``executable`` and ``launch``.

    Returns the steps it took (empty for a row with no preparation), or the reason it failed,
    prefixed with the row's name.
    """
    if provider.prepare is None:
        return []
    try:
        return provider.prepare(executable, launch)
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


_RITUAL_HEADER = "[TCIP session-start ritual] "
_RITUAL_FRICTION = (
    "If any mandated action is blocked or errors, that itself is a "
    "report_friction, never a silent skip."
)


def session_ritual(launch: ProjectLaunch | None) -> str:
    """Return the session-start directive for an agent of ``launch``, as one line.

    It names the display name its project's record holds now and the ritual to run first, its
    doctor step the launch's doctor line or the refusal naming why it cannot be spelled, or what
    a session with no project can do when ``launch`` is ``None``. Raises what
    :func:`tcip_mcp.project_record.existing_project` raises for a record that will not read.
    """
    from tcip_mcp.project_record import existing_project

    if launch is None:
        return (
            f"{_RITUAL_HEADER}This session has no project: the GUI had none open when the "
            "terminal started, so every tool that acts on a project refuses until one is "
            "created. A project initialize_project creates is this session's from then on; "
            "an existing one is worked on by opening it in the GUI, then ending this agent "
            f"from the rail and starting a new one. {_RITUAL_FRICTION}"
        )
    _, record = existing_project(launch.project.root)
    if launch.doctor_line is None:
        return (
            f"{_RITUAL_HEADER}Project: {record['display_name']}. Its doctor step cannot be "
            f"spelled on a shell line: {launch.doctor_refused}. Run load_project_memory "
            "(kind='reports' and kind='retrospectives') and inspect_project, then file this "
            f"with report_friction before any project work. {_RITUAL_FRICTION}"
        )
    return (
        f"{_RITUAL_HEADER}Project: {record['display_name']}. Run the ritual first: "
        "load_project_memory (kind='reports' and kind='retrospectives'), inspect_project, "
        f"then {launch.doctor_line}. {_RITUAL_FRICTION}"
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


def spawn_env(session_id: str) -> dict[str, str]:
    """Return the child's environment: this process's own plus the terminal session id.

    Every variable a Claude Code session set (the ``CLAUDE`` prefix) is left out.
    """
    kept = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE")}
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

    def write(self, data: str) -> None:
        self._p.write(data)

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
        # taskkill returning is not the process exiting: bounded-poll isalive() (the POSIX
        # path's own wait discipline), so a caller reading it right after sees the real state.
        deadline = time.monotonic() + TERMINATE_WAIT_S
        while self._p.isalive() and time.monotonic() < deadline:
            time.sleep(0.1)

    def close(self) -> None:
        """Perform no explicit cleanup."""


def _set_winsize(fd: int, rows: int, cols: int) -> None:
    """Set the POSIX terminal on ``fd`` to ``rows`` by ``cols``."""
    if sys.platform == "win32":
        raise AssertionError("_set_winsize is POSIX-only")
    import fcntl
    import struct
    import termios

    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


class _PosixPty:
    """The stdlib ``pty`` plus a subprocess."""

    # __init__'s body is unreachable to a Windows-targeted mypy run (its sys.platform guard),
    # so these are declared here rather than left to its own inference.
    _master: int
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

        self._master, slave = pty.openpty()
        _set_winsize(slave, rows, cols)
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
        os.close(slave)
        self.pid = self._proc.pid
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def read(self) -> str:
        chunk = os.read(self._master, 4096)
        if not chunk:
            raise EOFError
        return self._decoder.decode(chunk)

    def write(self, data: str) -> None:
        os.write(self._master, data.encode("utf-8"))

    def resize(self, rows: int, cols: int) -> None:
        _set_winsize(self._master, rows, cols)

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


type Pty = _WinPty | _PosixPty
"""A platform PTY: ConPTY through pywinpty on Windows, the stdlib ``pty`` elsewhere."""


def spawn_pty(
    argv: list[str], cwd: str, rows: int, cols: int, env: dict[str, str]
) -> Pty:
    """Spawn ``argv`` attached to a platform PTY under ``env``.

    Raises ``OSError`` on failure.
    """
    if os.name == "nt":
        return _WinPty(argv, cwd, rows, cols, env)
    return _PosixPty(argv, cwd, rows, cols, env)


# ── The reader pump ─────────────────────────────────────────────────────


def start_reader(
    pty: Pty,
    on_output: Callable[[str], None],
    on_exit: Callable[[], object],
    name: str,
) -> threading.Thread:
    """Pump PTY output on a daemon thread until the process exits, then call ``on_exit``.

    Any read failure ends the pump. The reader owns closing the PTY's OS resources
    (``pty.close()``) so an fd can never be recycled while a read is still blocked on it.
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

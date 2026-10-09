"""Embedded agent terminal: an agent harness from :data:`PROVIDERS` spawned directly in a
pseudo-terminal (ConPTY via ``pywinpty`` on Windows, the stdlib ``pty`` on POSIX), its raw bytes
streamed out and keystrokes streamed in. ``TCIP_TERMINAL_CMD`` overrides the spawn command.
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
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Callable, Iterable, Optional

from tcip_mcp.agent_identity import TERMINAL_SESSION_ENV
from tcip_mcp.project_paths import repo_root_from_here
from tcip_web.state import OpenProject, store

logger = logging.getLogger(__name__)

TERMINAL_CMD_ENV = "TCIP_TERMINAL_CMD"
TERMINAL_CWD_ENV = "TCIP_TERMINAL_CWD"

DEFAULT_ROWS = 30
DEFAULT_COLS = 100

# Seconds terminate() waits for the spawned process to actually exit: the Windows path polls
# isalive() after taskkill returns; the POSIX path bounds its wait() calls with it.
TERMINATE_WAIT_S = 5

WORKSPACE_ARG = "{workspace}"
"""A whole argument: the backend's workspace."""
MCP_CONFIG_ARG = "{mcp_config}"
"""A whole argument: the path of the JSON MCP configuration :func:`write_mcp_config` writes."""
CODEX_MCP_ARG = "{codex_mcp}"
"""A whole argument: the Codex configuration overrides :func:`codex_mcp_overrides` renders, as
several arguments."""
CLAUDE_SETTINGS_ARG = "{claude_settings}"
"""A whole argument: the path of the Claude settings :func:`write_claude_settings` writes."""

CLAUDE_SETTINGS = Path(__file__).resolve().parent / "agent_terminal.settings.json"
"""Claude Code's row's shipped permission lists, which :func:`write_claude_settings` renders."""

PREPARATION_TIMEOUT_S = 60

ANTIGRAVITY_SETTINGS = Path.home() / ".gemini" / "antigravity-cli" / "settings.json"
"""Antigravity's own settings file, whose ``permissions.allow`` list its tool approvals persist
to as ``mcp(<server>/<tool>)`` entries."""


class PreparationFailedError(Exception):
    """A row's launch preparation did not complete; the message names the step and why."""


@dataclass(frozen=True)
class ProjectLaunch:
    """A launch for an open project: the project, and its ritual's doctor step as
    :func:`project_launch` spelled it once, the line or the reason it cannot be spelled, exactly
    one of the two set."""

    project: OpenProject
    doctor_line: Optional[str]
    doctor_refused: Optional[str]


PrepareFn = Callable[[str, Optional[ProjectLaunch]], list[str]]
"""A row's launch preparation: given the resolved executable and the launch's
:class:`ProjectLaunch` (``None`` for no project), it runs to completion and returns one line per
step it took, or raises :class:`PreparationFailedError`."""


@dataclass(frozen=True)
class Provider:
    """One agent harness the terminal launches: the id a session names it by, its display name,
    the executable looked up on ``PATH``, the arguments after it, in which an argument equal to
    :data:`WORKSPACE_ARG`, :data:`MCP_CONFIG_ARG`, :data:`CLAUDE_SETTINGS_ARG` or
    :data:`CODEX_MCP_ARG` is rendered by :func:`render_argv`, the output sequence the harness
    version ``composer_ready_version`` names (its ``--version`` line) was recorded writing when
    its composer appeared, what the harness's own enforcement restricts under those arguments
    and leaves open, and the preparation the launch runs first, ``None`` for none."""

    id: str
    name: str
    executable: str
    args: tuple[str, ...]
    composer_ready: str
    composer_ready_version: str
    confinement: str
    prepare: Optional[PrepareFn] = None

    @property
    def unavailable_reason(self) -> str:
        """Why this row cannot launch when its executable is not on ``PATH``."""
        return (f"{self.name} is not available: no `{self.executable}` executable is on PATH. "
                "Install it and sign in to enable the agent terminal.")

    def delivery_unverified(self, version: Optional[str]) -> Optional[str]:
        """Why delivery to a launch declaring ``version`` rests on an unrecorded sequence, naming
        both versions, or ``None`` when ``version`` is the one ``composer_ready`` was recorded
        on."""
        if version == self.composer_ready_version:
            return None
        return (f"{self.name}'s composer sequence was recorded on {self.composer_ready_version}, "
                f"and this launch runs {version or 'a version it does not declare'}: the ritual "
                "and requests may never reach it, or reach its loading screen and be lost. If "
                "nothing arrives, type the request in the terminal yourself.")


@dataclass(frozen=True)
class McpServer:
    """The stdio MCP server a launch hands its harness: a command and its arguments."""

    command: str
    args: tuple[str, ...]


def backend_interpreter() -> str:
    """This backend's Python interpreter, forward-slashed."""
    return Path(sys.executable).as_posix()


def mcp_server(launch: Optional[ProjectLaunch]) -> McpServer:
    """:func:`backend_interpreter` running ``tcip_mcp`` for the directory of ``launch``'s
    project, for no project when ``launch`` is ``None``."""
    project = ("--project", launch.project.root.as_posix()) if launch else ()
    return McpServer(backend_interpreter(), ("-m", "tcip_mcp", *project))


def codex_mcp_overrides(launch: Optional[ProjectLaunch], env_names: Iterable[str]) -> list[str]:
    """:func:`mcp_server` for ``launch`` as Codex's ``tcip`` server in ``-c key=value``
    overrides, each value TOML (a JSON string or array of strings is valid TOML), forwarding the
    variables ``env_names`` names to the server and approving its tools without asking."""
    server = mcp_server(launch)
    settings = {"command": server.command, "args": list(server.args),
                "env_vars": sorted(env_names), "default_tools_approval_mode": "approve"}
    return [arg for key, value in settings.items()
            for arg in ("-c", f"mcp_servers.tcip.{key}={json.dumps(value)}")]


def run_to_completion(argv: list[str], step: str) -> str:
    """Run ``argv`` to completion, stdin closed and time bounded, and return it as one command
    line; raises :class:`PreparationFailedError` naming ``step`` when it cannot start, times
    out or exits non-zero, with the tail of what it printed."""
    try:
        done = subprocess.run(argv, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                              timeout=PREPARATION_TIMEOUT_S, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PreparationFailedError(f"{step} did not run: {exc}") from exc
    if done.returncode != 0:
        detail = (done.stderr or done.stdout).strip()[-400:]
        raise PreparationFailedError(f"{step} exited {done.returncode}: {detail}")
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

_DOCTOR_ARGUMENTS = (("interpreter", DOCTOR_TOKEN_REFUSES, True),
                     ("project path", DOCTOR_QUOTED_REFUSES, False))
"""Each argument of :data:`_DOCTOR_LINE`, in order, with the characters it refuses and whether
it refuses every whitespace character as well."""

_REGEX_LITERAL_ESCAPES = frozenset(".^$*+?()[]{}|\\")


def _regex_literal(text: str) -> str:
    """``text`` as a regular expression matching exactly it, a backslash before each of the
    metacharacters RE2, JavaScript and Python share."""
    return "".join(f"\\{c}" if c in _REGEX_LITERAL_ESCAPES else c for c in text)


def doctor_command(interpreter: str, project: PurePath) -> str:
    """The one spelling of the ritual's doctor step for ``project``: :data:`_DOCTOR_LINE` of
    ``interpreter``, unquoted, and the project's path, forward-slashed, in double quotes. Raises
    ``ValueError`` naming the argument when one is empty or holds a character
    :data:`_DOCTOR_ARGUMENTS` refuses it, naming those characters."""
    values = (interpreter, project.as_posix())
    for (name, refused, blank), value in zip(_DOCTOR_ARGUMENTS, values):
        held = sorted(c for c in set(value) if c in refused or (blank and c.isspace()))
        if held or not value:
            raise ValueError(f"the {name} {value!r} holds {held or 'nothing'!r}, which the "
                             "doctor line refuses")
    return _DOCTOR_LINE.format(*values)


def project_launch(project: Optional[OpenProject]) -> Optional[ProjectLaunch]:
    """The launch for ``project``, ``None`` for no project, its doctor step spelled once:
    :func:`doctor_command` of :func:`backend_interpreter`, or the refusal it raises."""
    if project is None:
        return None
    try:
        return ProjectLaunch(project, doctor_command(backend_interpreter(), project.root), None)
    except ValueError as exc:
        return ProjectLaunch(project, None, str(exc))


def _command_entry(line: str) -> str:
    """agy's allow entry for exactly the command ``line``, in the regex form agy's settings
    spell a command with arguments: the line between anchors as :func:`_regex_literal`."""
    return f"command(regex:^{_regex_literal(line)}$)"


def _entry_argument(refused: frozenset[str], blank: bool) -> str:
    """A regular expression matching, as :func:`_regex_literal` writes it, an argument of one or
    more characters outside ``refused`` and, when ``blank``, not whitespace (``\\s``, the
    characters ``str.isspace`` admits)."""
    bare = re.escape("".join(sorted(refused | _REGEX_LITERAL_ESCAPES))) + ("\\s" if blank else "")
    escaped = re.escape("".join(sorted(_REGEX_LITERAL_ESCAPES - refused)))
    return f"(?:[^{bare}]|\\\\[{escaped}])+"


_ENTRY_PARTS = _command_entry(_DOCTOR_LINE.format(*("\0" for _ in _DOCTOR_ARGUMENTS))).split("\0")
_DOCTOR_ENTRY = re.compile(re.escape(_ENTRY_PARTS[0]) + "".join(
    _entry_argument(refused, blank) + re.escape(part)
    for (_, refused, blank), part in zip(_DOCTOR_ARGUMENTS, _ENTRY_PARTS[1:])))
"""Every :func:`_command_entry` of a line :func:`doctor_command` renders, whatever its
arguments: :data:`_DOCTOR_LINE` with each argument admitted by :data:`_DOCTOR_ARGUMENTS`."""


def allow_tcip_tools(settings: Path, doctor_line: Optional[str]) -> str:
    """Add an ``mcp(tcip/<tool>)`` entry for every registered tool and, when given, the
    :func:`_command_entry` of ``doctor_line`` to the ``permissions.allow`` list of the
    Antigravity settings file ``settings``, dropping every other doctor entry and keeping
    everything else in it, and return one line saying how many were added and dropped; raises
    :class:`PreparationFailedError` when the file does not parse as a JSON object."""
    from tcip_mcp.server import list_registered_tools

    try:
        body = json.loads(settings.read_text(encoding="utf-8")) if settings.is_file() else {}
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationFailedError(f"reading {settings} failed: {exc}") from exc
    if not isinstance(body, dict):
        raise PreparationFailedError(f"{settings} does not hold a JSON object")
    allow = body.setdefault("permissions", {}).setdefault("allow", [])
    wanted = [f"mcp(tcip/{name})" for name in list_registered_tools()]
    if doctor_line is not None:
        wanted.append(_command_entry(doctor_line))
    stale = [entry for entry in allow if _DOCTOR_ENTRY.fullmatch(entry) and entry not in wanted]
    missing = [entry for entry in wanted if entry not in allow]
    if missing or stale:
        allow[:] = [entry for entry in allow if entry not in stale] + missing
        settings.parent.mkdir(parents=True, exist_ok=True)
        settings.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return (f"allowed {len(missing)} tcip entries in {settings}, dropped {len(stale)} other "
            "doctor entries")


def prepare_antigravity(executable: str, launch: Optional[ProjectLaunch]) -> list[str]:
    """Antigravity's launch preparation: register :func:`mcp_server` for ``launch`` as its
    ``tcip`` server through ``agy mcp add``, and allow every tcip tool and the launch's doctor
    line in its settings file; a doctor step that could not be spelled fails the preparation
    naming why."""
    server = mcp_server(launch)
    add = run_to_completion([executable, "mcp", "add", "tcip", "--", server.command, *server.args],
                            step="`agy mcp add`")
    if launch is not None and launch.doctor_refused is not None:
        raise PreparationFailedError(f"the doctor allowance was not written: "
                                     f"{launch.doctor_refused}")
    return [add, allow_tcip_tools(ANTIGRAVITY_SETTINGS, launch.doctor_line if launch else None)]


PROVIDERS: tuple[Provider, ...] = (
    Provider(
        id="claude",
        name="Claude Code",
        executable="claude",
        args=(
            "--settings", CLAUDE_SETTINGS_ARG,
            "--add-dir", WORKSPACE_ARG,
            "--permission-mode", "default",
            "--mcp-config", MCP_CONFIG_ARG, "--strict-mcp-config",
        ),
        composer_ready="\x1b[?1049h",
        composer_ready_version="2.1.292 (Claude Code)",
        confinement=("Claude Code enforces the deny and allow lists of "
                     "agent_terminal.settings.json, with the read-only tcip console commands "
                     "allowed and, when the launch spelled it, the ritual's own doctor line, "
                     "merged with the user's own Claude Code settings; a tool call neither list "
                     "decides asks first."),
    ),
    Provider(
        id="antigravity",
        name="Antigravity",
        executable="agy",
        args=("--add-dir", WORKSPACE_ARG, "--sandbox"),
        composer_ready="\x1b[0 q\r\x1b[J",
        composer_ready_version="1.3.1",
        confinement=("agy's --sandbox turns on its terminal restrictions, whose extent agy "
                     "defines; every tcip tool and, when the launch spelled it, the ritual's own "
                     "read-only doctor line are allowed in agy's own settings file."),
        prepare=prepare_antigravity,
    ),
    Provider(
        id="codex",
        name="Codex",
        executable="codex",
        args=("--sandbox", "read-only", "--ask-for-approval", "never",
              "-c", "disable_paste_burst=true", CODEX_MCP_ARG),
        composer_ready="\x1b]0;",
        composer_ready_version="codex-cli 0.160.1",
        confinement=("Codex's read-only sandbox holds the shell commands the model runs, which "
                     "write nothing and never ask to leave it; the tcip server is not one of "
                     "them, so its tools run, without a prompt, and write the project's records."),
    ),
)


def _override_argv() -> Optional[list[str]]:
    """``TCIP_TERMINAL_CMD`` split into an argv, or ``None`` when it is unset or blank."""
    override = os.environ.get(TERMINAL_CMD_ENV, "").strip()
    if not override:
        return None
    if os.name == "nt":
        return [tok.strip('"') for tok in shlex.split(override, posix=False)]
    return shlex.split(override)


def resolve_terminal_command(provider: Provider) -> Optional[tuple[list[str], bool]]:
    """The command that launches ``provider`` and whether it is the ``TCIP_TERMINAL_CMD``
    override: the override's argv when one is set, else the row's executable resolved on ``PATH``
    followed by its arguments with their placeholders unrendered (see :func:`render_argv`).
    ``None`` when the executable is not on ``PATH``."""
    override = _override_argv()
    if override is not None:
        return override, True
    executable = shutil.which(provider.executable)
    if executable is None:
        return None
    return [executable, *provider.args], False


def _write_launch_file(name: str, body: dict) -> Path:
    """``body`` written as JSON to ``name`` in a new temporary directory; its path."""
    dest = Path(tempfile.mkdtemp(prefix="tcip_launch_")) / name
    dest.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return dest


def write_mcp_config(launch: Optional[ProjectLaunch]) -> Path:
    """Write :func:`mcp_server` for ``launch`` as a JSON MCP configuration and return its
    path."""
    server = mcp_server(launch)
    return _write_launch_file("tcip.mcp.json", {
        "mcpServers": {"tcip": {"command": server.command, "args": list(server.args)}}})


READ_ONLY_CONSOLE_COMMANDS = ("scan-dataset", "inspect-compute-resources", "render-failure-cases")
"""The ``tcip`` console commands Claude's row allows under any arguments, each read-only."""

_CONSOLE_INVOCATIONS = ("python -m tcip_web.cli", "tcip")
"""The two invocations of the ``tcip`` console command: its module and its script."""


def write_claude_settings(launch: Optional[ProjectLaunch]) -> Path:
    """Write :data:`CLAUDE_SETTINGS` with a ``Bash(<line>)`` and a ``PowerShell(<line>)`` entry
    added to its allow list for each of :data:`READ_ONLY_CONSOLE_COMMANDS` under each of
    :data:`_CONSOLE_INVOCATIONS`, with any arguments, and for the launch's doctor line when it
    spelled one, and return its path."""
    body = json.loads(CLAUDE_SETTINGS.read_text(encoding="utf-8"))
    lines = [f"{invocation} {command}:*" for invocation in _CONSOLE_INVOCATIONS
             for command in READ_ONLY_CONSOLE_COMMANDS]
    if launch is not None and launch.doctor_line is not None:
        lines.append(launch.doctor_line)
    body["permissions"]["allow"] += [f"{shell}({line})" for shell in ("Bash", "PowerShell")
                                     for line in lines]
    return _write_launch_file("settings.json", body)


def render_argv(argv: list[str], launch: Optional[ProjectLaunch],
                env: dict[str, str]) -> list[str]:
    """``argv`` with each :data:`WORKSPACE_ARG` replaced by the backend's workspace, each
    :data:`MCP_CONFIG_ARG` by a configuration :func:`write_mcp_config` writes, each
    :data:`CLAUDE_SETTINGS_ARG` by the settings :func:`write_claude_settings` writes, and each
    :data:`CODEX_MCP_ARG` by :func:`codex_mcp_overrides` forwarding every variable of ``env``,
    the environment the launch spawns with, each for ``launch``."""
    values = {WORKSPACE_ARG: lambda: [str(store.workspace)],
              MCP_CONFIG_ARG: lambda: [str(write_mcp_config(launch))],
              CLAUDE_SETTINGS_ARG: lambda: [str(write_claude_settings(launch))],
              CODEX_MCP_ARG: lambda: codex_mcp_overrides(launch, env)}
    return [rendered for arg in argv for rendered in (values[arg]() if arg in values else [arg])]


def prepare_launch(executable: str, provider: Provider,
                   launch: Optional[ProjectLaunch]) -> tuple[list[str], Optional[str]]:
    """Run ``provider.prepare`` with the resolved ``executable`` and ``launch``: the steps it
    took (empty for a row with no preparation) and the reason it failed, ``None`` when it
    succeeded, prefixed with the row's name."""
    if provider.prepare is None:
        return [], None
    try:
        return provider.prepare(executable, launch), None
    except PreparationFailedError as exc:
        return [], f"{provider.name}'s launch preparation failed: {exc}"


_PRIVATE_MODE = re.compile(r"\x1b\[\?([0-9;]*)([hl])")
_BRACKETED_PASTE = "2004"


def bracketed_paste(output: str, enabled: bool) -> bool:
    """Whether the agent has bracketed paste on after writing ``output``, from ``enabled``."""
    for params, final in _PRIVATE_MODE.findall(output):
        if _BRACKETED_PASTE in params.split(";"):
            enabled = final == "h"
    return enabled


def paste(text: str) -> str:
    """``text`` as the agent's input: a bracketed paste followed by Enter, each line break inside
    it sent as the carriage return a terminal sends for a pasted newline."""
    return f"\x1b[200~{text.replace('\r\n', '\n').replace('\n', '\r')}\x1b[201~\r"


_RITUAL_HEADER = "[TCIP session-start ritual] "
_RITUAL_FRICTION = ("If any mandated action is blocked or errors, that itself is a "
                    "report_friction, never a silent skip.")


def session_ritual(launch: Optional[ProjectLaunch]) -> str:
    """The session-start directive for an agent of ``launch``, as one line: the display name its
    project's record holds now and the ritual to run first, its doctor step the launch's doctor
    line or the refusal naming why it cannot be spelled, or what a session with no project can do
    when ``launch`` is ``None``. Raises what :func:`tcip_mcp.project_record.existing_project`
    raises for a record that will not read."""
    from tcip_mcp.project_record import existing_project

    if launch is None:
        return (f"{_RITUAL_HEADER}This session has no project: the GUI had none open when the "
                "terminal started, so every tool that acts on a project refuses. Create one with "
                "initialize_project, or open one in the GUI, then restart the terminal to work on "
                f"it. {_RITUAL_FRICTION}")
    _, record = existing_project(launch.project.root)
    if launch.doctor_line is None:
        return (f"{_RITUAL_HEADER}Project: {record['display_name']}. Its doctor step cannot be "
                f"spelled on a shell line: {launch.doctor_refused}. Run load_project_memory "
                "(kind='reports' and kind='retrospectives') and inspect_project, then file this "
                f"with report_friction before any project work. {_RITUAL_FRICTION}")
    return (f"{_RITUAL_HEADER}Project: {record['display_name']}. Run the ritual first: "
            "load_project_memory (kind='reports' and kind='retrospectives'), inspect_project, "
            f"then {launch.doctor_line}. {_RITUAL_FRICTION}")


VERSION_PROBE_TIMEOUT_S = 15


def launched_program(argv: list[str], override: bool) -> dict:
    """``{"argv", "version"}`` for ``argv``: the argv itself, and what its executable, the first
    element, declares to ``--version`` (probed at every call, so a harness updated in place is
    seen at its next launch; stdin closed, time bounded), ``None`` for an ``override`` or an
    executable that does not answer cleanly."""
    executable = argv[0]
    if override:
        return {"argv": argv, "version": None}
    version: Optional[str] = None
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
        logger.debug("version probe of %s did not answer", executable, exc_info=True)
    return {"argv": argv, "version": version}


def spawn_env(session_id: str) -> dict[str, str]:
    """The child's environment: this process's own plus the terminal session id."""
    return {**os.environ, TERMINAL_SESSION_ENV: session_id}


def _prewarm_blocking() -> None:
    """Import the MCP server's tool graph so the first spawn's server starts from a warm cache."""
    try:
        import importlib

        importlib.import_module("tcip_mcp.server")
    except Exception:
        logger.debug("prewarm: tcip_mcp import failed", exc_info=True)


def prewarm() -> None:
    """Kick off best-effort cold-cache warming on a daemon thread; never blocks web startup. All
    failures are swallowed.
    """
    threading.Thread(target=_prewarm_blocking, name="terminal-prewarm", daemon=True).start()


def launch_problem(provider: Provider) -> Optional[str]:
    """Why ``provider`` cannot launch here (no PTY backend, or its executable is not on
    ``PATH``), or ``None`` when it can."""
    if os.name == "nt":
        try:
            import winpty  # noqa: F401
        except ImportError:
            return "pywinpty is not installed (pip install pywinpty)"
    return provider.unavailable_reason if resolve_terminal_command(provider) is None else None


def terminal_cwd() -> str:
    """Where the agent runs: ``TCIP_TERMINAL_CWD``, else the repository root."""
    return os.environ.get(TERMINAL_CWD_ENV, str(repo_root_from_here()))


# ── PTY backends ────────────────────────────────────────────────────────


class _WinPty:
    """ConPTY via pywinpty. ``read`` returns decoded text (pywinpty decodes)."""

    def __init__(self, argv: list[str], cwd: str, rows: int, cols: int, env: dict[str, str]):
        from winpty import PtyProcess

        self._p = PtyProcess.spawn(argv, cwd=cwd, dimensions=(rows, cols), env=env)
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
            ["taskkill", "/F", "/T", "/PID", str(self.pid)], capture_output=True, check=False
        )
        # taskkill returning is not the process exiting: bounded-poll isalive() (the POSIX
        # path's own wait discipline), so a caller reading it right after sees the real state.
        deadline = time.monotonic() + TERMINATE_WAIT_S
        while self._p.isalive() and time.monotonic() < deadline:
            time.sleep(0.1)

    def close(self) -> None:
        """Reader-owned cleanup after EOF (winpty frees its handles internally)."""


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

    def __init__(self, argv: list[str], cwd: str, rows: int, cols: int, env: dict[str, str]):
        # _open_pty dispatches this class off os.name == "nt", so it never runs on Windows;
        # the guard also tells mypy the POSIX-only stdlib members below are never checked there.
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
                self._proc.wait(timeout=TERMINATE_WAIT_S)  # reap: a permanent zombie pins the pid
            except subprocess.TimeoutExpired:  # pragma: no cover
                pass
        # The master fd is closed by the reader thread (see close()) after it drains to
        # EOF; closing it here would let a concurrent openpty() reuse the fd number
        # while the old reader is still blocked in os.read on it.

    def close(self) -> None:
        """Reader-owned: close the master fd after EOF."""
        try:
            os.close(self._master)
        except OSError:
            pass


def spawn_pty(argv: list[str], cwd: str, rows: int, cols: int, env: dict[str, str]):
    """Spawn ``argv`` attached to a platform PTY under ``env`` (see :func:`spawn_env`). Raises
    ``OSError`` on failure."""
    if os.name == "nt":
        return _WinPty(argv, cwd, rows, cols, env)
    return _PosixPty(argv, cwd, rows, cols, env)


# ── The reader pump ─────────────────────────────────────────────────────


def start_reader(
    pty, on_output: Callable[[str], None], on_exit: Callable[[], None], name: str
) -> threading.Thread:
    """Pump PTY output on a daemon thread until the process exits.

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

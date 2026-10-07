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
from pathlib import Path
from typing import Callable, Iterable, Optional

from tcip_mcp.agent_identity import TERMINAL_SESSION_ENV
from tcip_mcp.project_paths import repo_root_from_here

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

CLAUDE_SETTINGS = Path(__file__).resolve().parent / "agent_terminal.settings.json"
"""The settings file Claude Code's row passes: its permission lists."""

PREPARATION_TIMEOUT_S = 60

ANTIGRAVITY_SETTINGS = Path.home() / ".gemini" / "antigravity-cli" / "settings.json"
"""Antigravity's own settings file, whose ``permissions.allow`` list its tool approvals persist
to as ``mcp(<server>/<tool>)`` entries."""


class PreparationFailedError(Exception):
    """A row's launch preparation did not complete; the message names the step and why."""


PrepareFn = Callable[[str, Optional[Path]], list[str]]
"""A row's launch preparation: given the resolved executable and the session's project, it
runs to completion and returns one line per step it took, or raises
:class:`PreparationFailedError`."""


@dataclass(frozen=True)
class Provider:
    """One agent harness the terminal launches: the id a session names it by, its display name,
    the executable looked up on ``PATH``, the arguments after it, in which an argument equal to
    :data:`WORKSPACE_ARG`, :data:`MCP_CONFIG_ARG` or :data:`CODEX_MCP_ARG` is rendered by
    :func:`render_argv`, the output sequence the harness version ``composer_ready_version``
    names (its ``--version`` line) was recorded writing when its composer appeared, which no
    harness promises, what the harness's own enforcement restricts under those arguments and what
    it leaves open, recorded on every launch, and the preparation the launch
    runs first when the harness takes its MCP server or its tool approvals only through its own
    configuration, ``None`` for a harness that takes them on the command line.

    A row is listed only for a harness that turns bracketed paste on, since the session-start
    ritual and staged requests reach the agent only as a :func:`paste`, written once the agent
    has bracketed paste on and has written ``composer_ready`` since it started: a harness can turn
    bracketed paste on while a loading screen still discards input."""

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


def mcp_server(project: Optional[Path]) -> McpServer:
    """This interpreter running ``tcip_mcp`` for ``project``, for no project when ``None``."""
    args = ("-m", "tcip_mcp", *(("--project", project.as_posix()) if project else ()))
    return McpServer(Path(sys.executable).as_posix(), args)


def codex_mcp_overrides(project: Optional[Path], env_names: Iterable[str]) -> list[str]:
    """:func:`mcp_server` for ``project`` as Codex's ``tcip`` server in ``-c key=value``
    overrides, each value TOML (a JSON string or array of strings is valid TOML), forwarding the
    variables ``env_names`` names to the server and approving its tools without asking."""
    server = mcp_server(project)
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


def allow_tcip_tools(settings: Path) -> str:
    """Add an ``mcp(tcip/<tool>)`` entry for every registered tool to the ``permissions.allow``
    list of the Antigravity settings file ``settings``, keeping everything else in it, and
    return one line saying how many were added; raises :class:`PreparationFailedError` when the
    file does not parse as a JSON object."""
    from tcip_mcp.server import list_registered_tools

    try:
        body = json.loads(settings.read_text(encoding="utf-8")) if settings.is_file() else {}
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationFailedError(f"reading {settings} failed: {exc}") from exc
    if not isinstance(body, dict):
        raise PreparationFailedError(f"{settings} does not hold a JSON object")
    allow = body.setdefault("permissions", {}).setdefault("allow", [])
    missing = [entry for entry in (f"mcp(tcip/{name})" for name in list_registered_tools())
               if entry not in allow]
    if missing:
        allow.extend(missing)
        settings.parent.mkdir(parents=True, exist_ok=True)
        settings.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return f"allowed {len(missing)} tcip tools in {settings}"


def prepare_antigravity(executable: str, project: Optional[Path]) -> list[str]:
    """Antigravity's launch preparation: register :func:`mcp_server` for ``project`` as its
    ``tcip`` server through ``agy mcp add``, and allow every tcip tool in its settings file."""
    server = mcp_server(project)
    add = run_to_completion([executable, "mcp", "add", "tcip", "--", server.command, *server.args],
                            step="`agy mcp add`")
    return [add, allow_tcip_tools(ANTIGRAVITY_SETTINGS)]


PROVIDERS: tuple[Provider, ...] = (
    Provider(
        id="claude",
        name="Claude Code",
        executable="claude",
        args=(
            "--settings", str(CLAUDE_SETTINGS),
            "--add-dir", WORKSPACE_ARG,
            "--permission-mode", "default",
            "--mcp-config", MCP_CONFIG_ARG, "--strict-mcp-config",
        ),
        composer_ready="\x1b[?1049h",
        composer_ready_version="2.1.292 (Claude Code)",
        confinement=("Claude Code enforces the deny and allow lists of "
                     "agent_terminal.settings.json, merged with the user's own Claude Code "
                     "settings; a tool call neither list decides asks first."),
    ),
    Provider(
        id="antigravity",
        name="Antigravity",
        executable="agy",
        args=("--add-dir", WORKSPACE_ARG, "--sandbox"),
        composer_ready="\x1b[?1049l",
        composer_ready_version="1.3.0",
        confinement=("agy's --sandbox turns on its terminal restrictions, whose extent agy "
                     "defines; every tcip tool is allowed in agy's own settings file."),
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


def write_mcp_config(project: Optional[Path]) -> Path:
    """Write :func:`mcp_server` for ``project`` as a JSON MCP configuration and return its
    path."""
    server = mcp_server(project)
    config = {"mcpServers": {"tcip": {"command": server.command, "args": list(server.args)}}}
    dest = Path(tempfile.mkdtemp(prefix="tcip_mcp_")) / "tcip.mcp.json"
    dest.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return dest


def render_argv(argv: list[str], project: Optional[Path], env: dict[str, str]) -> list[str]:
    """``argv`` with each :data:`WORKSPACE_ARG` replaced by the backend's workspace, each
    :data:`MCP_CONFIG_ARG` by a configuration :func:`write_mcp_config` writes for ``project``,
    and each :data:`CODEX_MCP_ARG` by :func:`codex_mcp_overrides` for ``project`` forwarding
    every variable of ``env``, the environment the launch spawns with."""
    from tcip_web.state import store

    values = {WORKSPACE_ARG: lambda: [str(store.workspace)],
              MCP_CONFIG_ARG: lambda: [str(write_mcp_config(project))],
              CODEX_MCP_ARG: lambda: codex_mcp_overrides(project, env)}
    return [rendered for arg in argv for rendered in (values[arg]() if arg in values else [arg])]


def prepare_launch(executable: str, provider: Provider,
                   project: Optional[Path]) -> tuple[list[str], Optional[str]]:
    """Run ``provider.prepare`` with the resolved ``executable`` for ``project``: the steps it
    took (empty for a row with no preparation) and the reason it failed, ``None`` when it
    succeeded, prefixed with the row's name."""
    if provider.prepare is None:
        return [], None
    try:
        return provider.prepare(executable, project), None
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


def session_ritual(project: Optional[Path]) -> str:
    """The session-start directive for an agent launched for ``project``, as one line: the
    project's display name and the ritual to run first when its record reads, the reason when it
    does not, and what a session with no project can do when ``project`` is ``None``."""
    from tcip_mcp.project_record import record_fields

    if project is None:
        return (f"{_RITUAL_HEADER}This session has no project: the GUI had none open when the "
                "terminal started, so every tool that acts on a project refuses. Create one with "
                "initialize_project, or open one in the GUI, then restart the terminal to work on "
                f"it. {_RITUAL_FRICTION}")
    record = record_fields(project)
    if record["display_name"] is None:
        return (f"{_RITUAL_HEADER}This session's project ({project}) has no readable record: "
                f"{record['record_problem']} File this with report_friction before any project "
                f"work. {_RITUAL_FRICTION}")
    return (f"{_RITUAL_HEADER}Project: {record['display_name']} ({project}). Run the ritual "
            "first: load_project_memory (kind='reports' and kind='retrospectives'), "
            f"inspect_project, then tcip doctor {project}. {_RITUAL_FRICTION}")


VERSION_PROBE_TIMEOUT_S = 15


def launched_program(argv: list[str], override: bool) -> dict:
    """``{"executable", "version"}`` for ``argv``: its first element, and what that executable
    declares to ``--version`` (probed at every call, so a harness updated in place is seen at its
    next launch; stdin closed, time bounded), ``None`` for an ``override`` or an executable that
    does not answer cleanly."""
    executable = argv[0]
    if override:
        return {"executable": executable, "version": None}
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
    return {"executable": executable, "version": version}


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

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
from typing import Callable, Optional

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
MCP_CONFIG_ARG = "{mcp_config}"

CLAUDE_SETTINGS = Path(__file__).resolve().parent / "agent_terminal.settings.json"
"""The settings file Claude Code's row passes: its permission lists."""


@dataclass(frozen=True)
class Provider:
    """One agent harness the terminal launches: the id a session names it by, its display name,
    the executable looked up on ``PATH``, and the arguments after it, in which an argument equal
    to :data:`WORKSPACE_ARG` or :data:`MCP_CONFIG_ARG` stands for the backend's workspace or the
    path of the MCP configuration :func:`write_mcp_config` writes for the session's project.

    A row is listed only for a harness that turns bracketed paste on, since the session-start
    ritual and staged requests reach the agent only as a :func:`paste` once it has."""

    id: str
    name: str
    executable: str
    args: tuple[str, ...]

    @property
    def unavailable_reason(self) -> str:
        """Why this row cannot launch when its executable is not on ``PATH``."""
        return (f"{self.name} is not available: no `{self.executable}` executable is on PATH. "
                "Install it and sign in to enable the agent terminal.")


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
    """Write a spawn-time MCP configuration that starts this interpreter's ``tcip_mcp`` for
    ``project`` (for no project when ``None``) and return its path."""
    args = ["-m", "tcip_mcp", *(["--project", project.as_posix()] if project else [])]
    config = {"mcpServers": {"tcip": {"command": Path(sys.executable).as_posix(), "args": args}}}
    dest = Path(tempfile.mkdtemp(prefix="tcip_mcp_")) / "tcip.mcp.json"
    dest.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return dest


def render_argv(argv: list[str], project: Optional[Path]) -> list[str]:
    """``argv`` with each :data:`WORKSPACE_ARG` replaced by the backend's workspace and each
    :data:`MCP_CONFIG_ARG` by a configuration :func:`write_mcp_config` writes for ``project``."""
    from tcip_web.state import store

    values = {WORKSPACE_ARG: lambda: str(store.workspace),
              MCP_CONFIG_ARG: lambda: str(write_mcp_config(project))}
    return [values[arg]() if arg in values else arg for arg in argv]


_PRIVATE_MODE = re.compile(r"\x1b\[\?([0-9;]*)([hl])")
_BRACKETED_PASTE = "2004"


def bracketed_paste(output: str, enabled: bool) -> bool:
    """Whether the agent has bracketed paste on after writing ``output``, from ``enabled``."""
    for params, final in _PRIVATE_MODE.findall(output):
        if _BRACKETED_PASTE in params.split(";"):
            enabled = final == "h"
    return enabled


def paste(text: str) -> str:
    """``text`` as the agent's input: a bracketed paste followed by Enter."""
    return f"\x1b[200~{text}\x1b[201~\r"


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


_CLI_VERSIONS: dict[str, Optional[str]] = {}
VERSION_PROBE_TIMEOUT_S = 15


def launched_program(argv: list[str], override: bool) -> dict:
    """``{"executable", "version"}`` for ``argv``: its first element, and what that executable
    declares to ``--version`` (probed once per executable per process, stdin closed, time
    bounded), ``None`` for an ``override`` or an executable that does not answer cleanly."""
    executable = argv[0]
    if override:
        return {"executable": executable, "version": None}
    if executable not in _CLI_VERSIONS:
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
        _CLI_VERSIONS[executable] = version
    return {"executable": executable, "version": _CLI_VERSIONS[executable]}


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

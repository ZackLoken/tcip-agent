"""Embedded agent terminal: run the real Claude Code CLI in a PTY.

The in-app agent surface is the *actual* ``claude`` interactive TUI, not a chat
re-implementation: we spawn the fenced CLI inside an interactive shell (PowerShell on
Windows, bash on POSIX) in a pseudo-terminal (ConPTY via ``pywinpty`` on Windows, the
stdlib ``pty`` on POSIX) with cwd = the repo root, so it loads ``CLAUDE.md`` /
``.claude/skills/`` / ``.mcp.json`` and inherits the machine's existing Claude Code auth
exactly like a terminal session. The shell (not claude) is the PTY's top process, so an
in-TUI ``/exit`` drops back to a live prompt where the ``claude --resume <id>`` hint stays
usable. Raw PTY bytes stream to xterm.js in the browser over a WebSocket; keystrokes
stream back. No translation layer: fidelity is the point, and a translation layer is
exactly where silent failures hide.

This module owns the process/PTY concerns; the HTTP/WS surface is
``routes/terminal.py``. The spawn command is injectable via ``TCIP_TERMINAL_CMD`` so
tests drive a scripted fake program at the process boundary, and CI (no ``claude``
installed) cleanly reports unavailable.

Trust boundary: same as every GUI surface (``tcip_web.trust_boundary``: local arrivals
served, network arrivals refused until the operator opts in, an Origin check the middleware
applies to this and every other WebSocket connect and to every state-changing request). The
terminal gives keyboard access to Claude Code, equivalent power to the terminal the operator
already has on this machine; exposing it is what the opt-in's disclosure names.
"""

from __future__ import annotations

import codecs
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Callable, Optional

from tcip_mcp.agent_identity import TERMINAL_SESSION_ENV
from tcip_mcp.project_paths import repo_root_from_here

logger = logging.getLogger(__name__)

TERMINAL_CMD_ENV = "TCIP_TERMINAL_CMD"
TERMINAL_CLI_ENV = "TCIP_TERMINAL_CLI"
TERMINAL_CWD_ENV = "TCIP_TERMINAL_CWD"
DEFAULT_CLI = "claude"

DEFAULT_ROWS = 30
DEFAULT_COLS = 100

# The committed permission fence for the in-app (breeder-lane) agent. Passed via --settings, which merges its allow/deny lists
# (union) with the repo's and the user's own settings; a `claude` session with none (this one) is unaffected by the fence file.
_FENCE_SETTINGS = Path(__file__).resolve().parent / "agent_terminal.settings.json"

_UNAVAILABLE_REASON = (
    "Claude Code is not available. Install the `claude` CLI and sign in "
    "(subscription or ANTHROPIC_API_KEY) to enable the in-app agent terminal."
)


def _absolutize_guard_command(command: str, python: str, guard_dir: str) -> str:
    """Rewrite ``python <path>/agent_*.py`` → ``"<python>" "<guard_dir>/<script>"``.

    Matches on basename alone (an ``agent_`` prefix, a ``.py`` suffix), the shape of the
    PreToolUse guards, the SessionEnd learning-capture hook, and the SessionStart
    ritual-injection hook; a match is rewritten to ``guard_dir/<basename>`` regardless of the
    directory the original token pointed at. Anything that doesn't match is returned unchanged.
    Quoted, forward-slashed paths parse under both cmd.exe and POSIX sh.
    """
    for tok in command.split():
        name = tok.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if name.startswith("agent_") and name.endswith(".py"):
            return f'"{python}" "{guard_dir}/{name}"'
    return command


def _materialize_fence_settings() -> Optional[Path]:
    """Write a spawn-time copy of the fence settings with absolute hook commands.

    The committed template stores each PreToolUse guard command repo-relative
    (``python packages/tcip-web/src/tcip_web/agent_bash_guard.py``) for readability, but a
    PreToolUse hook runs from an unpredictable cwd: the Bash/PowerShell tool's persistent
    ``cd`` moves it, so a relative path fails to even *locate* the script: the interpreter
    exits 2, which Claude Code reads as a *block*, denying every command after a ``cd`` (the
    reported "guard blocks all listings after cd"). We rewrite each guard command to an
    absolute ``"<python>" "<guard_dir>/agent_*_guard.py"`` (this process's ``sys.executable``
    + the guard directory) and hand that file to ``--settings``. Python, not the shell,
    resolves the path, so there is no cwd dependency and no ``$VAR`` cross-platform hazard.

    Returns the materialized file, or ``None`` if the template is missing/unreadable (the
    caller then falls back to the committed file, so the fence is never silently dropped).
    """
    if not _FENCE_SETTINGS.is_file():
        return None
    try:
        cfg = json.loads(_FENCE_SETTINGS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    guard_dir = _FENCE_SETTINGS.parent.as_posix()
    python = Path(sys.executable).as_posix()
    # Absolutize every agent_*.py hook across all events (PreToolUse guards, SessionEnd capture,
    # SessionStart ritual injection): none should depend on cwd, same as the guards themselves.
    for event_groups in cfg.get("hooks", {}).values():
        for group in event_groups:
            for hook in group.get("hooks", []):
                hook["command"] = _absolutize_guard_command(hook.get("command", ""), python, guard_dir)
    # A process-private directory (mkdtemp, mode 0700 on POSIX), not a fixed shared-temp name, so the
    # live fence cannot be pre-created or race-written by another local process before the CLI reads it.
    try:
        dest = Path(tempfile.mkdtemp(prefix="tcip_fence_")) / "tcip_agent_fence.settings.json"
        dest.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except OSError:
        return None
    return dest


def _resolve_fence() -> tuple[Optional[str], Optional[str]]:
    """The fenced flags' values: ``(settings_path, workspace_dir)``, or ``(None, None)`` if no fence.

    ``--settings`` applies the committed breeder-lane permission profile, materialized with
    absolute hook paths so the Bash/PowerShell guards survive a ``cd``; ``--add-dir`` grants the
    out-of-repo workspace. The wrapper below turns these into the fenced ``claude`` invocation.
    """
    if not _FENCE_SETTINGS.is_file():
        return None, None
    from tcip_mcp.workspace import workspace_root

    materialized = _materialize_fence_settings()
    if materialized is None:
        logger.warning(
            "Could not materialize absolute fence hook paths; falling back to the committed "
            "template. Its relative hook paths over-deny after a shell cd (fail-safe, but "
            "friction), check temp-dir writability."
        )
    settings_path = materialized or _FENCE_SETTINGS
    return str(settings_path), str(workspace_root())


def resolve_terminal_command() -> Optional[list[str]]:
    """Argv for the agent terminal, or ``None`` when unavailable.

    Order: an explicit ``TCIP_TERMINAL_CMD`` override (tests / power users), then the ``claude``
    CLI on PATH. The real CLI is spawned fenced and directly (no wrapping shell), so
    ``/exit`` ends the process cleanly: ``--settings`` applies the breeder-lane profile (absolute
    hook paths), ``--add-dir`` grants the workspace, ``--permission-mode default`` surfaces
    un-allowed actions for approval.
    """
    override = os.environ.get(TERMINAL_CMD_ENV, "").strip()
    if override:
        if os.name == "nt":
            return [tok.strip('"') for tok in shlex.split(override, posix=False)]
        return shlex.split(override)
    cli = os.environ.get(TERMINAL_CLI_ENV, DEFAULT_CLI)
    exe = shutil.which(cli)
    if exe is None:
        return None
    settings_path, ws = _resolve_fence()
    if settings_path:
        return [exe, "--settings", settings_path, "--add-dir", ws or "", "--permission-mode", "default"]
    return [exe]


_CLI_VERSIONS: dict[str, Optional[str]] = {}
VERSION_PROBE_TIMEOUT_S = 15


def launched_program(argv: list[str]) -> dict:
    """What the terminal launches: ``{"executable", "version"}``, the executable being ``argv[0]``
    and the version what that executable declares to ``--version``.

    Only the resolved CLI (no ``TCIP_TERMINAL_CMD`` override in force) is probed: an override is
    any argv an operator or a test chose, whose meaning for ``--version`` is unknown, so it is
    recorded as launched with no version rather than run a second time. The bare executable is
    probed, not the fenced argv, because the fenced ``claude`` invocation refuses a missing
    settings file before it answers. Probed once per executable per process with stdin closed and
    a time bound, ``None`` when it does not answer cleanly. Which agent harness the launched
    program turns out to be is recorded by the MCP server from that harness's own handshake, not
    inferred here.
    """
    executable = argv[0]
    if os.environ.get(TERMINAL_CMD_ENV, "").strip():
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
    """The child's environment: this process's own (Claude Code auth included) plus the terminal
    session id, which the MCP server the agent launches records as a correlation."""
    return {**os.environ, TERMINAL_SESSION_ENV: session_id}


def _prewarm_blocking() -> None:
    """Warm the controllable part of the cold first-spawn into the OS cache (see ``prewarm``)."""
    try:
        import importlib

        importlib.import_module("tcip_mcp.server")  # the tool graph the MCP server loads at launch
    except Exception:
        logger.debug("prewarm: tcip_mcp import failed", exc_info=True)
    if os.name == "nt":
        try:
            shell = shutil.which("powershell.exe") or "powershell.exe"
            # Throwaway PowerShell to warm PowerShell + .NET; output discarded, bounded, best-effort.
            subprocess.run(
                [shell, "-NoProfile", "-Command", "$null"],
                capture_output=True,
                timeout=30,
                check=False,
            )
        except Exception:
            logger.debug("prewarm: powershell warm failed", exc_info=True)


def prewarm() -> None:
    """Kick off best-effort cold-cache warming on a daemon thread; never blocks web startup.

    The first ``claude`` spawn is ~4-5s vs ~1s on restart, cold cache, not structure: cold
    ``tcip_mcp`` import (the MCP server loads it at launch) plus, on Windows, cold PowerShell +
    .NET. Warm both here so the operator's first terminal open pays only claude/node's own cold
    cost. All failures are swallowed; the terminal still works if warming does nothing.
    """
    threading.Thread(target=_prewarm_blocking, name="terminal-prewarm", daemon=True).start()


def _pty_backend_available() -> tuple[bool, str]:
    if os.name == "nt":
        try:
            import winpty  # noqa: F401

            return True, ""
        except ImportError:
            return False, "pywinpty is not installed (pip install pywinpty)"
    return True, ""


def terminal_status() -> dict:
    """Preflight for ``GET /api/terminal/status``: ``{available, reason?}``."""
    ok, why = _pty_backend_available()
    if not ok:
        return {"available": False, "reason": why}
    if resolve_terminal_command() is None:
        return {"available": False, "reason": _UNAVAILABLE_REASON}
    return {"available": True}


def terminal_cwd() -> str:
    """Where the agent runs: the repo root (so .mcp.json / CLAUDE.md / skills load and the
    fence's cwd-relative deny paths bind correctly). Overridable via env.

    ``repo_root_from_here`` finds ``.mcp.json`` across every ancestor before falling back to
    ``CLAUDE.md``, so it climbs past the ``CLAUDE.md`` this file's own package carries instead
    of stopping there, and resolves from the module rather than the launch cwd.
    """
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
        # /T tree-kill: claude spawns children (its MCP servers) that must not orphan.
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(self.pid)], capture_output=True, check=False
        )

    def close(self) -> None:
        """Reader-owned cleanup after EOF (winpty frees its handles internally)."""


class _PosixPty:
    """stdlib pty + subprocess (used on Linux CI and any POSIX deployment)."""

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
        import fcntl
        import pty
        import struct
        import termios

        self._master, slave = pty.openpty()
        winsz = struct.pack("HHHH", rows, cols, 0, 0)
        fcntl.ioctl(slave, termios.TIOCSWINSZ, winsz)
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
        if sys.platform == "win32":
            raise AssertionError("_PosixPty is POSIX-only")
        import fcntl
        import struct
        import termios

        winsz = struct.pack("HHHH", rows, cols, 0, 0)
        fcntl.ioctl(self._master, termios.TIOCSWINSZ, winsz)

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
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self._proc.wait(timeout=5)  # reap: a permanent zombie pins the pid
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


def start_reader(pty, on_output: Callable[[str], None], on_exit: Callable[[], None], name: str) -> threading.Thread:
    """Pump PTY output on a daemon thread until the process exits.

    Catches broadly: pywinpty raises its own exception types (``WinptyError``,
    ``EOFError``) rather than ``OSError``, and any read failure means the same thing:
    this PTY is done. The reader owns closing the PTY's OS resources (``pty.close()``)
    so an fd can never be recycled while a read is still blocked on it.
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

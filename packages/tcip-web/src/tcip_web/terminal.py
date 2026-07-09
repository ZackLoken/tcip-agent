"""Embedded agent terminal — run the real Claude Code CLI in a PTY.

The in-app agent surface is the *actual* ``claude`` interactive TUI, not a chat
re-implementation: we spawn the CLI in a pseudo-terminal (ConPTY via ``pywinpty`` on
Windows, the stdlib ``pty`` on POSIX) with cwd = the repo root, so it loads
``CLAUDE.md`` / ``.github/skills/`` / ``.mcp.json`` and inherits the machine's existing
Claude Code auth exactly like a terminal session. Raw PTY bytes stream to xterm.js in
the browser over a WebSocket; keystrokes stream back. No translation layer — fidelity
is the point, and the translation layer is where the old chat's silent failures lived.

This module owns the process/PTY concerns; the HTTP/WS surface is
``routes/terminal.py``. The spawn command is injectable via ``TCIP_TERMINAL_CMD`` so
tests drive a scripted fake program at the process boundary, and CI (no ``claude``
installed) cleanly reports unavailable.

Trust boundary: same as every GUI surface (loopback bind + TrustedHost + WS Origin
check). The terminal gives keyboard access to Claude Code — equivalent power to the
terminal the operator already has on this machine; it must never be exposed beyond the
loopback trust boundary without adding auth.
"""

from __future__ import annotations

import codecs
import logging
import os
import shlex
import shutil
import subprocess
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)

TERMINAL_CMD_ENV = "TCIP_TERMINAL_CMD"
TERMINAL_CLI_ENV = "TCIP_TERMINAL_CLI"
TERMINAL_CWD_ENV = "TCIP_TERMINAL_CWD"
DEFAULT_CLI = "claude"

DEFAULT_ROWS = 30
DEFAULT_COLS = 100

_UNAVAILABLE_REASON = (
    "Claude Code is not available. Install the `claude` CLI and sign in "
    "(subscription or ANTHROPIC_API_KEY) to enable the in-app agent terminal."
)


def resolve_terminal_command() -> Optional[list[str]]:
    """Argv for the agent terminal, or ``None`` when unavailable.

    Order: an explicit ``TCIP_TERMINAL_CMD`` override (tests / power users), then the
    ``claude`` CLI on PATH — spawned with **no arguments**: the interactive TUI is the
    product here.
    """
    override = os.environ.get(TERMINAL_CMD_ENV, "").strip()
    if override:
        if os.name == "nt":
            return [tok.strip('"') for tok in shlex.split(override, posix=False)]
        return shlex.split(override)
    cli = os.environ.get(TERMINAL_CLI_ENV, DEFAULT_CLI)
    exe = shutil.which(cli)
    if exe:
        return [exe]
    return None


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
    """Where the agent runs: the repo root tcip-web was launched from (overridable)."""
    return os.environ.get(TERMINAL_CWD_ENV, os.getcwd())


# ── PTY backends ────────────────────────────────────────────────────────


class _WinPty:
    """ConPTY via pywinpty. ``read`` returns decoded text (pywinpty decodes)."""

    def __init__(self, argv: list[str], cwd: str, rows: int, cols: int):
        from winpty import PtyProcess

        # env=None inherits this process's environment (Claude Code auth included).
        self._p = PtyProcess.spawn(argv, cwd=cwd, dimensions=(rows, cols))
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

    def __init__(self, argv: list[str], cwd: str, rows: int, cols: int):
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
        import fcntl
        import struct
        import termios

        winsz = struct.pack("HHHH", rows, cols, 0, 0)
        fcntl.ioctl(self._master, termios.TIOCSWINSZ, winsz)

    def isalive(self) -> bool:
        return self._proc.poll() is None

    def terminate(self) -> None:
        import signal

        # Liveness guard: once poll() has reaped the child, its pid (and pgid) may have
        # been recycled — killpg would then hit an innocent process group.
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
                self._proc.wait(timeout=5)  # reap — a permanent zombie pins the pid
            except subprocess.TimeoutExpired:  # pragma: no cover
                pass
        # The master fd is closed by the reader thread (see close()) after it drains to
        # EOF — closing it here would let a concurrent openpty() reuse the fd number
        # while the old reader is still blocked in os.read on it.

    def close(self) -> None:
        """Reader-owned: close the master fd after EOF."""
        try:
            os.close(self._master)
        except OSError:
            pass


def spawn_pty(argv: list[str], cwd: str, rows: int, cols: int):
    """Spawn ``argv`` attached to a platform PTY. Raises ``OSError`` on failure."""
    if os.name == "nt":
        return _WinPty(argv, cwd, rows, cols)
    return _PosixPty(argv, cwd, rows, cols)


# ── The reader pump ─────────────────────────────────────────────────────


def start_reader(pty, on_output: Callable[[str], None], on_exit: Callable[[], None], name: str) -> threading.Thread:
    """Pump PTY output on a daemon thread until the process exits.

    Catches broadly: pywinpty raises its own exception types (``WinptyError``,
    ``EOFError``) rather than ``OSError``, and any read failure means the same thing —
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

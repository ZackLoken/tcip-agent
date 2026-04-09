"""Process manager — spawns and monitors the Rust agent (which manages MCP internally)."""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from .bridge import AgentBridge

log = logging.getLogger(__name__)


def _find_agent_binary() -> str | None:
    """Locate the tcip-agent binary."""
    # Check in workspace's target/debug or target/release
    workspace = Path(__file__).resolve().parents[4]  # tcip-agent repo root
    for profile in ("release", "debug"):
        for name in ("tcip-cli", "tcip-agent"):
            candidate = workspace / "tcip-agent" / "target" / profile / (name + ".exe")
            if candidate.is_file():
                return str(candidate)
            candidate = candidate.with_suffix("")
            if candidate.is_file():
                return str(candidate)
    # Check PATH
    return shutil.which("tcip-cli") or shutil.which("tcip-agent")


class ProcessManager(QObject):
    """Manages the agent process lifecycle."""

    agent_ready = pyqtSignal()
    agent_crashed = pyqtSignal(str)  # error message
    status_changed = pyqtSignal(str)  # status text

    def __init__(self, workspace: str | Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._workspace = Path(workspace)
        self._bridge: AgentBridge | None = None
        self._restart_timer = QTimer(self)
        self._restart_timer.setSingleShot(True)
        self._restart_timer.setInterval(2000)
        self._restart_timer.timeout.connect(self._do_restart)
        self._auto_restart = True
        self._restart_count = 0
        self._max_restarts = 3

    @property
    def bridge(self) -> AgentBridge | None:
        return self._bridge

    def start(self) -> AgentBridge | None:
        """Start the agent process. Returns the bridge or None on failure."""
        agent_bin = _find_agent_binary()
        if agent_bin is None:
            msg = "Could not find tcip-agent binary. Build it with 'cargo build' first."
            log.error(msg)
            self.agent_crashed.emit(msg)
            return None

        args = [
            "--workspace", str(self._workspace),
            "--jsonrpc",
        ]

        self._bridge = AgentBridge(agent_bin, args, self)
        self._bridge.agent_started.connect(self._on_started)
        self._bridge.agent_stopped.connect(self._on_stopped)
        self._bridge.start()
        return self._bridge

    def stop(self) -> None:
        """Gracefully stop the agent."""
        self._auto_restart = False
        if self._bridge:
            self._bridge.stop()
            self._bridge = None
        self.status_changed.emit("Disconnected")

    def restart(self) -> None:
        """Restart the agent process."""
        self.status_changed.emit("Restarting agent...")
        if self._bridge:
            self._bridge.stop()
            self._bridge = None
        self._restart_timer.start()

    def _do_restart(self) -> None:
        self._restart_count += 1
        self.start()

    def _on_started(self) -> None:
        # Don't reset restart count here — the process may crash immediately.
        # Count is reset only after the first successful user interaction
        # (i.e., when a full message is received from the agent).
        self.status_changed.emit("Agent connected")
        self.agent_ready.emit()

    def clear_restart_count(self) -> None:
        """Reset restart counter. Call after agent proves it's healthy."""
        self._restart_count = 0

    def _on_stopped(self, exit_code: int, status: str) -> None:
        if status == "crashed" or exit_code != 0:
            msg = f"Agent crashed (exit code {exit_code})"
            log.error(msg)
            self.agent_crashed.emit(msg)
            self.status_changed.emit("Agent crashed")
            if self._auto_restart and self._restart_count < self._max_restarts:
                log.info("Auto-restarting agent in 2 seconds... (attempt %d/%d)",
                         self._restart_count + 1, self._max_restarts)
                self._restart_timer.start()
            elif self._restart_count >= self._max_restarts:
                msg = (f"Agent failed {self._max_restarts} times. "
                       "Check ANTHROPIC_API_KEY and agent logs.")
                log.error(msg)
                self.agent_crashed.emit(msg)
                self.status_changed.emit("Agent stopped — check configuration")
        else:
            self.status_changed.emit("Agent stopped")

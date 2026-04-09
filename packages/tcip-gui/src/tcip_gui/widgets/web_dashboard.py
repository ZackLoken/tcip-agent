"""Embeddable web-dashboard widgets for TensorBoard and Ray Tune.

Each widget manages a subprocess (tensorboard / ray dashboard) and embeds its
UI inside a QWebEngineView.  A placeholder is shown while the server starts or
when the required package is missing.
"""

from __future__ import annotations

import logging
import shutil
import socket
import subprocess
import sys
from typing import Any

from PyQt6.QtCore import QTimer, QUrl, Qt
from PyQt6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView

    _HAS_WEBENGINE = True
except ImportError:
    QWebEngineView = None  # type: ignore[assignment,misc]
    _HAS_WEBENGINE = False

log = logging.getLogger(__name__)


def _free_port() -> int:
    """Find an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _WebDashboard(QWidget):
    """Base class: manages a subprocess + QWebEngineView."""

    def __init__(self, placeholder_text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._proc: subprocess.Popen[str] | None = None
        self._port: int | None = None
        self._placeholder_text = placeholder_text

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if _HAS_WEBENGINE and QWebEngineView is not None:
            self._view = QWebEngineView()
            self._view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self._placeholder = QLabel(placeholder_text)
            self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._placeholder.setStyleSheet("color: #888; font-size: 13px;")
            layout.addWidget(self._placeholder)
            layout.addWidget(self._view)
            self._view.hide()
        else:
            self._view = None
            self._placeholder = QLabel("PyQt6-WebEngine is required for this panel.\n"
                                       "Install with: pip install PyQt6-WebEngine")
            self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._placeholder.setStyleSheet("color: #f44336; font-size: 13px;")
            layout.addWidget(self._placeholder)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(800)
        self._poll_timer.timeout.connect(self._check_server)
        self._poll_attempts = 0
        self._max_poll_attempts = 30  # ~24 seconds

    # ── Subclass interface ─────────────────────────────────────────────

    def _build_command(self, port: int) -> list[str] | None:
        """Return the command to launch the server, or None if unavailable."""
        raise NotImplementedError

    def _server_url(self, port: int) -> str:
        return f"http://127.0.0.1:{port}"

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self, **kwargs: Any) -> None:
        """Start the dashboard server subprocess."""
        self.stop()

        port = _free_port()
        cmd = self._build_command(port, **kwargs)
        if cmd is None:
            self._placeholder.setText(self._placeholder_text + "\n(required package not found)")
            return

        log.info("Starting dashboard: %s", " ".join(cmd))
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except FileNotFoundError:
            self._placeholder.setText(self._placeholder_text + "\n(failed to launch server)")
            return

        self._port = port
        self._poll_attempts = 0
        self._placeholder.setText(self._placeholder_text + "\nStarting server...")
        self._placeholder.show()
        self._poll_timer.start()

    def stop(self) -> None:
        """Terminate the running server subprocess."""
        self._poll_timer.stop()
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
        self._port = None
        if self._view is not None:
            self._view.setUrl(QUrl("about:blank"))
            self._view.hide()
        self._placeholder.setText(self._placeholder_text)
        self._placeholder.show()

    def _check_server(self) -> None:
        """Poll until the server is accepting connections."""
        if self._port is None:
            self._poll_timer.stop()
            return

        self._poll_attempts += 1
        try:
            with socket.create_connection(("127.0.0.1", self._port), timeout=0.3):
                pass
            # Server is ready
            self._poll_timer.stop()
            self._show_web_view()
        except OSError:
            if self._poll_attempts >= self._max_poll_attempts:
                self._poll_timer.stop()
                self._placeholder.setText(self._placeholder_text + "\n(server did not start in time)")

    def _show_web_view(self) -> None:
        if self._view is None or self._port is None:
            return
        url = self._server_url(self._port)
        log.info("Loading dashboard at %s", url)
        self._placeholder.hide()
        self._view.show()
        self._view.setUrl(QUrl(url))

    def closeEvent(self, event: Any) -> None:
        self.stop()
        super().closeEvent(event)


# ── TensorBoard ───────────────────────────────────────────────────────


class TensorBoardWidget(_WebDashboard):
    """Embeds a TensorBoard instance for a given logdir."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("TensorBoard", parent)
        self._logdir: str | None = None

    def _build_command(self, port: int, *, logdir: str = "") -> list[str] | None:  # type: ignore[override]
        tb = shutil.which("tensorboard")
        if tb is None:
            return None
        logdir = logdir or self._logdir or "."
        return [tb, "--logdir", logdir, "--port", str(port),
                "--host", "127.0.0.1", "--reload_interval", "5"]

    def set_logdir(self, logdir: str) -> None:
        """Set the logdir and (re)start TensorBoard."""
        self._logdir = logdir
        self.start(logdir=logdir)


# ── Ray Tune ──────────────────────────────────────────────────────────


class RayTuneDashboardWidget(_WebDashboard):
    """Embeds the Ray Dashboard for monitoring HPO trials."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Ray Tune Dashboard", parent)
        self._experiment_dir: str | None = None

    def _build_command(self, port: int, *, experiment_dir: str = "") -> list[str] | None:  # type: ignore[override]
        ray = shutil.which("ray")
        if ray is None:
            return None
        # `ray dashboard` is not a standalone command; we launch the full
        # dashboard via `ray start --head` with a specific dashboard port.
        # For an already-running Ray cluster the dashboard port is fixed at
        # init time, so we use the Ray dashboard API endpoint directly.
        # Fallback: if no cluster is running, start a head node.
        return [ray, "start", "--head",
                "--dashboard-port", str(port),
                "--include-dashboard", "true",
                "--disable-usage-stats"]

    def set_experiment_dir(self, experiment_dir: str) -> None:
        """Set experiment directory and start the dashboard."""
        self._experiment_dir = experiment_dir
        self.start(experiment_dir=experiment_dir)

    def _server_url(self, port: int) -> str:
        # Ray dashboard serves on this path
        return f"http://127.0.0.1:{port}"

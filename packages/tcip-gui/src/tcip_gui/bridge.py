"""Agent bridge — communicates with the Rust agent via JSON-RPC over stdio."""

from __future__ import annotations

import json
import logging
from typing import Any

from PyQt6.QtCore import QObject, QProcess, QThread, pyqtSignal

from .protocol import (
    ControlCancel,
    ControlShutdown,
    JsonRpcMessage,
    PermissionResponse,
    UserMessage,
    parse_agent_message,
)

log = logging.getLogger(__name__)


class AgentReaderThread(QThread):
    """Reads newline-delimited JSON-RPC from the agent's stdout in a background thread."""

    message_received = pyqtSignal(object)  # parsed protocol message
    error_occurred = pyqtSignal(str)

    def __init__(self, process: QProcess, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._process = process
        self._running = True

    def run(self) -> None:
        while self._running:
            if self._process.state() == QProcess.ProcessState.NotRunning:
                break
            if self._process.waitForReadyRead(200):
                while self._process.canReadLine():
                    raw = self._process.readLine().data().decode("utf-8", errors="replace").strip()
                    if not raw:
                        continue
                    try:
                        msg = JsonRpcMessage.from_line(raw)
                        parsed = parse_agent_message(msg)
                        self.message_received.emit(parsed)
                    except (json.JSONDecodeError, KeyError) as exc:
                        log.warning("Malformed message from agent: %s", exc)

    def stop(self) -> None:
        self._running = False
        self.wait(2000)


class AgentBridge(QObject):
    """Manages the agent process and JSON-RPC communication."""

    # Signals emitted when agent sends messages
    text_delta = pyqtSignal(str)
    text_done = pyqtSignal(str)
    tool_call_start = pyqtSignal(str, str, dict)  # id, name, input
    tool_call_result = pyqtSignal(str, str, bool)  # id, output, is_error
    permission_request = pyqtSignal(str, str, dict, str, str)  # id, tool, input, desc, level
    usage_update = pyqtSignal(int, int, float)  # input_tokens, output_tokens, cost
    turn_complete = pyqtSignal(int)  # turn_number
    agent_error = pyqtSignal(int, str)  # code, message

    # Canvas signals (agent → GUI)
    canvas_load_image = pyqtSignal(str, object)  # path, annotations (list|None)
    canvas_show_predictions = pyqtSignal(str)  # predictions_path
    canvas_clear = pyqtSignal()
    canvas_highlight = pyqtSignal(list)  # annotation_ids

    # Training signals (agent → GUI)
    training_started = pyqtSignal(str, str, int)  # run_name, metrics_path, total_epochs
    training_metrics_update = pyqtSignal(object)  # TrainingMetricsUpdate dataclass
    training_complete = pyqtSignal(str, int, float, object)  # run_name, best_epoch, best_metric, metrics
    results_show = pyqtSignal(object)  # ResultsShow dataclass

    # Process lifecycle signals
    agent_started = pyqtSignal()
    agent_stopped = pyqtSignal(int, str)  # exit_code, exit_status

    def __init__(self, agent_command: str, agent_args: list[str], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._command = agent_command
        self._args = agent_args
        self._process: QProcess | None = None
        self._reader: AgentReaderThread | None = None

    def start(self) -> None:
        """Spawn the agent process and start reading."""
        if self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning:
            log.warning("Agent already running")
            return

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self._process.finished.connect(self._on_process_finished)
        self._process.errorOccurred.connect(self._on_process_error)
        self._process.readyReadStandardError.connect(self._on_stderr)

        self._process.start(self._command, self._args)
        if not self._process.waitForStarted(5000):
            log.error("Failed to start agent process")
            return

        self._reader = AgentReaderThread(self._process, self)
        self._reader.message_received.connect(self._dispatch_message)
        self._reader.start()

        self.agent_started.emit()
        log.info("Agent process started: %s %s", self._command, self._args)

    def stop(self) -> None:
        """Gracefully shut down the agent."""
        if self._reader:
            self._reader.stop()
            self._reader = None

        if self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            self._send(ControlShutdown().to_rpc())
            if not self._process.waitForFinished(5000):
                log.warning("Agent did not exit gracefully, killing")
                self._process.kill()
                self._process.waitForFinished(2000)

    def send_message(self, text: str) -> None:
        """Send a user message to the agent."""
        self._send(UserMessage(text=text).to_rpc())

    def send_permission_response(self, request_id: str, allowed: bool, reason: str | None = None) -> None:
        """Send a permission response to the agent."""
        self._send(PermissionResponse(request_id=request_id, allowed=allowed, reason=reason).to_rpc())

    def send_cancel(self) -> None:
        """Cancel the current operation."""
        self._send(ControlCancel().to_rpc())

    def is_running(self) -> bool:
        return self._process is not None and self._process.state() == QProcess.ProcessState.Running

    def _send(self, msg: JsonRpcMessage) -> None:
        if self._process is None or self._process.state() == QProcess.ProcessState.NotRunning:
            log.warning("Cannot send — agent not running")
            return
        payload = msg.to_line() + "\n"
        self._process.write(payload.encode("utf-8"))

    def _dispatch_message(self, parsed: Any) -> None:
        """Route parsed protocol message to the appropriate signal."""
        from .protocol import (
            AgentError,
            CanvasClear,
            CanvasHighlight,
            CanvasLoadImage,
            CanvasShowPredictions,
            PermissionRequest,
            ResultsShow,
            TextDelta,
            TextDone,
            ToolCallResult,
            ToolCallStart,
            TrainingComplete,
            TrainingMetricsUpdate,
            TrainingStarted,
            TurnComplete,
            UsageStatus,
        )

        if isinstance(parsed, TextDelta):
            self.text_delta.emit(parsed.text)
        elif isinstance(parsed, TextDone):
            self.text_done.emit(parsed.text)
        elif isinstance(parsed, ToolCallStart):
            self.tool_call_start.emit(parsed.tool_id, parsed.name, parsed.input)
        elif isinstance(parsed, ToolCallResult):
            self.tool_call_result.emit(parsed.tool_id, parsed.output, parsed.is_error)
        elif isinstance(parsed, PermissionRequest):
            self.permission_request.emit(
                parsed.request_id, parsed.tool, parsed.input,
                parsed.description, parsed.level,
            )
        elif isinstance(parsed, UsageStatus):
            self.usage_update.emit(parsed.input_tokens, parsed.output_tokens, parsed.cost)
        elif isinstance(parsed, TurnComplete):
            self.turn_complete.emit(parsed.turn_number)
        elif isinstance(parsed, AgentError):
            self.agent_error.emit(parsed.code, parsed.message)
        elif isinstance(parsed, CanvasLoadImage):
            self.canvas_load_image.emit(parsed.path, parsed.annotations)
        elif isinstance(parsed, CanvasShowPredictions):
            self.canvas_show_predictions.emit(parsed.predictions_path)
        elif isinstance(parsed, CanvasClear):
            self.canvas_clear.emit()
        elif isinstance(parsed, CanvasHighlight):
            self.canvas_highlight.emit(parsed.annotation_ids)
        elif isinstance(parsed, TrainingStarted):
            self.training_started.emit(parsed.run_name, parsed.metrics_path, parsed.total_epochs)
        elif isinstance(parsed, TrainingMetricsUpdate):
            self.training_metrics_update.emit(parsed)
        elif isinstance(parsed, TrainingComplete):
            self.training_complete.emit(parsed.run_name, parsed.best_epoch, parsed.best_metric, parsed.metrics)
        elif isinstance(parsed, ResultsShow):
            self.results_show.emit(parsed)

    def _on_process_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        status_str = "normal" if exit_status == QProcess.ExitStatus.NormalExit else "crashed"
        log.info("Agent exited: code=%d status=%s", exit_code, status_str)
        if self._reader:
            self._reader.stop()
            self._reader = None
        self.agent_stopped.emit(exit_code, status_str)

    def _on_process_error(self, error: QProcess.ProcessError) -> None:
        log.error("Agent process error: %s", error)

    def _on_stderr(self) -> None:
        if self._process:
            stderr = self._process.readAllStandardError().data().decode("utf-8", errors="replace")
            for line in stderr.strip().splitlines():
                log.debug("[agent stderr] %s", line)

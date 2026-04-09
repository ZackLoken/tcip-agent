"""Main application — QApplication, MainWindow, tab-based layout with chat overlay."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QResizeEvent
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QWidget,
)

from .bridge import AgentBridge
from .panels.annotation_panel import AnnotationPanel
from .panels.chat_overlay import ChatOverlay
from .panels.dataset_browser import DatasetBrowser
from .panels.results_panel import ResultsPanel
from .panels.review_panel import ReviewPanel
from .panels.status_bar import AgentStatusBar
from .panels.training_dashboard import TrainingDashboard
from .process_manager import ProcessManager
from .protocol import CanvasAnnotationSaved

log = logging.getLogger(__name__)

# Tab indices
_TAB_ANNOTATE = 0
_TAB_REVIEW = 1
_TAB_TRAINING = 2
_TAB_RESULTS = 3


class MainWindow(QMainWindow):
    """Primary application window — full-bleed tabs with floating chat overlay."""

    def __init__(self, workspace: str | Path) -> None:
        super().__init__()
        self.setWindowTitle("TCIP Agent")
        self.resize(1400, 900)
        self.setMinimumSize(QSize(800, 600))

        self._workspace = Path(workspace)

        # ── Central tab widget (full-bleed) ──
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)  # cleaner tab bar

        self._annotation_panel = AnnotationPanel()
        self._review_panel = ReviewPanel()
        self._training_dashboard = TrainingDashboard()
        self._results_panel = ResultsPanel()

        self._tabs.addTab(self._annotation_panel, "  Annotate  ")
        self._tabs.addTab(self._review_panel, "  Review  ")
        self._tabs.addTab(self._training_dashboard, "  Training  ")
        self._tabs.addTab(self._results_panel, "  Results  ")
        self.setCentralWidget(self._tabs)

        # ── Status bar (compact 32px) ──
        self._status_bar = AgentStatusBar(self)
        self.setStatusBar(self._status_bar)

        # ── Chat overlay (floating panel over right edge) ──
        central = self.centralWidget()
        assert central is not None
        self._chat_overlay = ChatOverlay(central)
        self._chat = self._chat_overlay.chat  # convenience alias

        # Chat toggle button (floating, bottom-right)
        self._chat_toggle = QPushButton("💬")
        self._chat_toggle.setObjectName("chat-toggle-btn")
        self._chat_toggle.setParent(central)
        self._chat_toggle.setToolTip("Toggle Agent Chat  (Ctrl+/)")
        self._chat_toggle.clicked.connect(self._chat_overlay.toggle)
        self._chat_toggle.raise_()

        # Dataset browser not shown as dock — embedded in annotation panel sidebar later
        self._dataset_browser = DatasetBrowser()

        # ── Process manager & bridge ──
        self._pm = ProcessManager(self._workspace, self)
        self._pm.status_changed.connect(self._status_bar.set_status)
        self._pm.agent_crashed.connect(self._on_agent_crash)

        self._bridge: AgentBridge | None = None

        # Connect chat overlay signals
        self._chat_overlay.message_submitted.connect(self._on_message)
        self._chat_overlay.permission_responded.connect(self._on_permission_response)

        # Connect dataset browser
        self._dataset_browser.image_selected.connect(self._on_image_selected)

        # Connect annotation save
        self._annotation_panel.annotation_saved.connect(self._on_annotation_saved)

    def start_agent(self) -> None:
        """Start the agent process and wire up the bridge."""
        bridge = self._pm.start()
        if bridge is None:
            return
        self._bridge = bridge

        # Wire bridge signals to chat panel
        bridge.text_delta.connect(self._chat.on_text_delta)
        bridge.text_done.connect(self._chat.on_text_done)
        bridge.tool_call_start.connect(self._chat.on_tool_call_start)
        bridge.tool_call_result.connect(self._chat.on_tool_call_result)
        bridge.permission_request.connect(self._chat.on_permission_request)
        bridge.usage_update.connect(self._status_bar.update_usage)
        bridge.turn_complete.connect(lambda _: None)
        bridge.agent_error.connect(self._chat.on_agent_error)
        bridge.agent_started.connect(self._status_bar.set_connected)
        bridge.agent_stopped.connect(self._on_agent_stopped)

        # Wire canvas signals from agent
        bridge.canvas_load_image.connect(self._on_canvas_load_image)
        bridge.canvas_show_predictions.connect(self._on_canvas_show_predictions)
        bridge.canvas_clear.connect(self._annotation_panel.canvas.clear_predictions)
        bridge.canvas_highlight.connect(self._on_canvas_highlight)

        # Wire training signals from agent
        bridge.training_started.connect(self._on_training_started)
        bridge.training_metrics_update.connect(self._on_training_metrics_update)
        bridge.training_complete.connect(self._on_training_complete)
        bridge.results_show.connect(self._on_results_show)

        # Wire training dashboard control signals
        self._training_dashboard.pause_requested.connect(self._on_training_pause)
        self._training_dashboard.stop_requested.connect(self._on_training_stop)

        # Wire results panel action signals
        self._results_panel.accept_model.connect(self._on_result_accept)
        self._results_panel.retrain_requested.connect(self._on_result_retrain)
        self._results_panel.retrain_hpo_requested.connect(self._on_result_retrain_hpo)

    def show_annotation_mode(self) -> None:
        self._tabs.setCurrentIndex(_TAB_ANNOTATE)

    def show_review_mode(self) -> None:
        self._tabs.setCurrentIndex(_TAB_REVIEW)

    def show_training_mode(self) -> None:
        self._tabs.setCurrentIndex(_TAB_TRAINING)

    def show_results_mode(self) -> None:
        self._tabs.setCurrentIndex(_TAB_RESULTS)

    def show_welcome(self) -> None:
        self._tabs.setCurrentIndex(_TAB_ANNOTATE)

    def load_image_on_canvas(self, path: str) -> None:
        """Load an image onto the annotation canvas and switch to annotation mode."""
        if self._annotation_panel.load_image(path):
            self.show_annotation_mode()

    def _on_message(self, text: str) -> None:
        if self._bridge and self._bridge.is_running():
            self._bridge.send_message(text)
        else:
            self._chat.on_agent_error(-1, "Agent is not connected. Restarting...")
            self.start_agent()

    def _on_permission_response(self, request_id: str, allowed: bool, reason: str) -> None:
        if self._bridge and self._bridge.is_running():
            self._bridge.send_permission_response(request_id, allowed, reason or None)

    def _on_image_selected(self, path: str) -> None:
        self.load_image_on_canvas(path)

    def _on_annotation_saved(self, path: str, count: int) -> None:
        log.info("Saved %d annotations for %s", count, path)
        # Notify agent that annotations were saved
        if self._bridge and self._bridge.is_running():
            msg = CanvasAnnotationSaved(image_path=path, count=count).to_rpc()
            self._bridge._send(msg)

    def _on_canvas_load_image(self, path: str, annotations: object) -> None:
        """Handle agent requesting an image to be loaded on canvas."""
        self.load_image_on_canvas(path)

    def _on_canvas_show_predictions(self, predictions_path: str) -> None:
        """Handle agent requesting predictions overlay."""
        from tcip_annotation.label_io import parse_detect_predictions
        canvas = self._annotation_panel.canvas
        w, h = canvas.image_size
        if w <= 0 or h <= 0:
            return
        pred_boxes, _ = parse_detect_predictions(predictions_path, w, h)
        canvas.clear_predictions()
        for pb in pred_boxes:
            canvas.add_pred_box(pb.x1, pb.y1, pb.x2 - pb.x1, pb.y2 - pb.y1,
                                class_id=pb.class_id, confidence=pb.confidence)
        self.show_annotation_mode()

    def _on_canvas_highlight(self, annotation_ids: list) -> None:
        """Handle agent requesting specific annotations be highlighted."""
        indices = [int(i) for i in annotation_ids if str(i).isdigit()]
        self._annotation_panel.canvas.highlight_items(indices)

    def _on_training_started(self, run_name: str, metrics_path: str, total_epochs: int) -> None:
        """Handle agent starting a training run — switch to dashboard."""
        self._training_dashboard.reset()
        self._training_dashboard.set_run_name(run_name)
        self._training_dashboard.set_metrics_path(metrics_path)
        self._training_dashboard.update_progress(0, total_epochs)
        self.show_training_mode()

    def _on_training_metrics_update(self, update: object) -> None:
        """Handle training metrics from agent."""
        from .protocol import TrainingMetricsUpdate

        if not isinstance(update, TrainingMetricsUpdate):
            return
        self._training_dashboard.add_metrics(
            epoch=update.epoch,
            train_loss=update.train_loss,
            val_loss=update.val_loss,
            map50=update.map50,
        )
        if update.lr is not None or update.stage is not None or update.eta is not None:
            self._training_dashboard.update_progress(
                epoch=int(update.epoch),
                total_epochs=self._training_dashboard._total_epochs,
                stage=update.stage,
                lr=update.lr,
                eta=update.eta,
            )

    def _on_training_complete(self, run_name: str, best_epoch: int, best_metric: float, metrics: object) -> None:
        """Handle training completion."""
        self._training_dashboard.set_complete(best_metric)

    def _on_results_show(self, results: object) -> None:
        """Handle agent requesting results display."""
        from .protocol import ResultsShow

        if not isinstance(results, ResultsShow):
            return
        self._results_panel.reset()
        self._results_panel.set_run_name(results.run_name)
        self._results_panel.set_overall_metrics(
            map50=results.overall.get("map50", 0),
            map50_95=results.overall.get("map50_95"),
            precision=results.overall.get("precision"),
            recall=results.overall.get("recall"),
        )
        self._results_panel.set_per_class_metrics(results.per_class)
        if results.worst_images:
            self._results_panel.set_worst_predictions(results.worst_images)
        if results.csv_path:
            self._results_panel.load_csv_preview(results.csv_path)
        self.show_results_mode()

    def _on_training_pause(self) -> None:
        """Handle user clicking Pause on the training dashboard."""
        if self._bridge and self._bridge.is_running():
            from .protocol import TrainingPauseRequested
            self._bridge._send(TrainingPauseRequested().to_rpc())

    def _on_training_stop(self) -> None:
        """Handle user clicking Stop on the training dashboard."""
        if self._bridge and self._bridge.is_running():
            from .protocol import TrainingStopRequested
            self._bridge._send(TrainingStopRequested().to_rpc())

    def _on_result_accept(self) -> None:
        """Handle user accepting the model in results panel."""
        if self._bridge and self._bridge.is_running():
            from .protocol import ResultAction
            self._bridge._send(ResultAction(action="accept").to_rpc())

    def _on_result_retrain(self) -> None:
        """Handle user requesting retrain."""
        if self._bridge and self._bridge.is_running():
            from .protocol import ResultAction
            self._bridge._send(ResultAction(action="retrain").to_rpc())

    def _on_result_retrain_hpo(self) -> None:
        """Handle user requesting retrain with HPO."""
        if self._bridge and self._bridge.is_running():
            from .protocol import ResultAction
            self._bridge._send(ResultAction(action="retrain_hpo").to_rpc())

    def _on_agent_crash(self, message: str) -> None:
        self._chat.on_agent_crash(message)
        self._status_bar.set_crashed()

    def _on_agent_stopped(self, exit_code: int, status: str) -> None:
        if status == "crashed":
            self._status_bar.set_crashed()
        else:
            self._status_bar.set_disconnected()

    def resizeEvent(self, event: QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._chat_overlay.reposition()
        self._position_chat_toggle()

    def _position_chat_toggle(self) -> None:
        """Place the chat toggle button at bottom-right of the central widget."""
        cw = self.centralWidget()
        if cw is None:
            return
        btn = self._chat_toggle
        margin = 16
        btn.move(cw.width() - btn.width() - margin, cw.height() - btn.height() - margin)
        btn.raise_()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._pm.stop()
        super().closeEvent(event)


def _load_stylesheet() -> str:
    qss_path = Path(__file__).parent / "resources" / "style.qss"
    if qss_path.is_file():
        return qss_path.read_text(encoding="utf-8")
    return ""


def main() -> None:
    """Entry point for the GUI application."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Determine workspace
    workspace = "."
    if len(sys.argv) > 1:
        workspace = sys.argv[1]
    workspace_path = Path(workspace).resolve()

    app = QApplication(sys.argv)
    app.setApplicationName("TCIP Agent")
    app.setOrganizationName("TCIP")

    # Force dark window frame on Windows
    from PyQt6.QtGui import QPalette, QColor

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#1E1E1E"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#E0E0E0"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#2A2A2A"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#252525"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#E0E0E0"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#2A2A2A"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#E0E0E0"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#507754"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    app.setPalette(palette)

    stylesheet = _load_stylesheet()
    if stylesheet:
        app.setStyleSheet(stylesheet)

    window = MainWindow(workspace_path)
    window.show()
    window.start_agent()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

"""Training dashboard panel — real-time training metrics and HPO trial display.

Left pane:  embedded TensorBoard (loss, mAP, lr schedules — all auto-discovered).
Right pane: embedded Ray Tune dashboard (HPO trials, parallel coords, importance).
Falls back to simple text labels when PyQt6-WebEngine or the servers are unavailable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..widgets.web_dashboard import TensorBoardWidget, RayTuneDashboardWidget
from ..widgets.trial_table import TrialTable


class TrainingDashboard(QWidget):
    """Live training dashboard with loss/mAP charts, progress, and HPO trials."""

    pause_requested = pyqtSignal()
    stop_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # ── Header ──
        header_row = QHBoxLayout()
        self._title = QLabel("Training")
        self._title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        header_row.addWidget(self._title)
        header_row.addStretch()

        self._pause_btn = QPushButton("⏸ Pause")
        self._pause_btn.clicked.connect(self.pause_requested.emit)
        header_row.addWidget(self._pause_btn)

        self._stop_btn = QPushButton("⏹ Stop")
        self._stop_btn.setStyleSheet("color: #f44336;")
        self._stop_btn.clicked.connect(self.stop_requested.emit)
        header_row.addWidget(self._stop_btn)
        layout.addLayout(header_row)

        # ── Progress ──
        self._stage_label = QLabel("Waiting for training to start...")
        layout.addWidget(self._stage_label)

        progress_row = QHBoxLayout()
        self._epoch_label = QLabel("Epoch: —")
        progress_row.addWidget(self._epoch_label)
        self._lr_label = QLabel("LR: —")
        progress_row.addWidget(self._lr_label)
        self._eta_label = QLabel("ETA: —")
        progress_row.addWidget(self._eta_label)
        progress_row.addStretch()
        layout.addLayout(progress_row)

        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(100)
        self._progress_bar.setValue(0)
        layout.addWidget(self._progress_bar)

        # ── Dashboards (TensorBoard + Ray Tune) ──
        dashboard_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._tensorboard = TensorBoardWidget()
        self._ray_dashboard = RayTuneDashboardWidget()
        dashboard_splitter.addWidget(self._tensorboard)
        dashboard_splitter.addWidget(self._ray_dashboard)
        dashboard_splitter.setSizes([600, 400])
        layout.addWidget(dashboard_splitter, stretch=3)

        # ── Best checkpoint info ──
        self._best_label = QLabel("Best checkpoint: —")
        self._best_label.setStyleSheet("color: #4caf50; font-weight: bold;")
        layout.addWidget(self._best_label)

        # ── HPO Trials (table fallback when Ray dashboard is unavailable) ──
        self._hpo_label = QLabel("HPO Trials")
        self._hpo_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self._hpo_label.setVisible(False)
        layout.addWidget(self._hpo_label)

        self._trial_table = TrialTable()
        self._trial_table.setVisible(False)
        layout.addWidget(self._trial_table, stretch=1)

        # ── Metrics file polling ──
        self._metrics_path: str | None = None
        self._last_line_count = 0
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_metrics)

        # ── State ──
        self._total_epochs = 0
        self._best_metric = 0.0
        self._best_epoch = 0

    def set_run_name(self, name: str) -> None:
        self._title.setText(f"Training: {name}")

    def set_metrics_path(self, path: str) -> None:
        """Set the path to the metrics JSONL file and start TensorBoard.

        TensorBoard will auto-discover event files under the parent directory
        of the metrics path (the run directory).
        """
        self._metrics_path = path
        self._last_line_count = 0
        # Start TensorBoard pointed at the run's log directory
        logdir = str(Path(path).parent)
        self._tensorboard.set_logdir(logdir)
        self._poll_timer.start(2000)  # poll JSONL for progress bar updates

    def stop_polling(self) -> None:
        self._poll_timer.stop()
        # TensorBoard stays running so user can still explore after training

    def update_progress(
        self,
        epoch: int,
        total_epochs: int,
        stage: str | None = None,
        lr: float | None = None,
        eta: str | None = None,
    ) -> None:
        """Update progress display."""
        self._total_epochs = total_epochs
        pct = int((epoch / total_epochs) * 100) if total_epochs > 0 else 0
        self._progress_bar.setValue(pct)
        self._epoch_label.setText(f"Epoch {epoch}/{total_epochs}")
        if stage:
            self._stage_label.setText(stage)
        if lr is not None:
            self._lr_label.setText(f"LR: {lr:.6f}")
        if eta:
            self._eta_label.setText(f"ETA: {eta}")

    def add_metrics(
        self,
        epoch: float,
        train_loss: float | None = None,
        val_loss: float | None = None,
        map50: float | None = None,
    ) -> None:
        """Track best checkpoint from incoming metrics.

        Chart rendering is handled by TensorBoard; this method only updates the
        best-checkpoint label from the JSONL polling data.
        """
        if map50 is not None:
            if map50 > self._best_metric:
                self._best_metric = map50
                self._best_epoch = int(epoch)
                self._best_label.setText(
                    f"Best checkpoint: epoch {self._best_epoch}  mAP@50: {self._best_metric:.4f}"
                )

    def set_hpo_trials(self, trials: list[dict[str, Any]]) -> None:
        """Show HPO trial table and start Ray dashboard."""
        self._hpo_label.setVisible(True)
        self._trial_table.setVisible(True)
        self._trial_table.set_trials(trials)

        # Start Ray dashboard if an experiment dir is available
        if trials and "experiment_dir" in trials[0]:
            self._ray_dashboard.set_experiment_dir(trials[0]["experiment_dir"])

    def update_hpo_trial(self, trial_id: int, metric: float | None = None, status: str | None = None) -> None:
        self._trial_table.update_trial(trial_id, metric, status)

    def set_complete(self, final_metric: float | None = None) -> None:
        """Mark training as complete."""
        self._poll_timer.stop()
        self._progress_bar.setValue(100)
        self._stage_label.setText("Training complete!")
        self._eta_label.setText("")
        if final_metric is not None:
            self._best_label.setText(
                f"Final: epoch {self._best_epoch}  mAP@50: {self._best_metric:.4f}"
            )

    def reset(self) -> None:
        """Reset the dashboard for a new training run."""
        self._poll_timer.stop()
        self._metrics_path = None
        self._last_line_count = 0
        self._progress_bar.setValue(0)
        self._stage_label.setText("Waiting for training to start...")
        self._epoch_label.setText("Epoch: —")
        self._lr_label.setText("LR: —")
        self._eta_label.setText("ETA: —")
        self._best_label.setText("Best checkpoint: —")
        self._best_metric = 0.0
        self._best_epoch = 0
        self._total_epochs = 0
        self._tensorboard.stop()
        self._ray_dashboard.stop()
        self._trial_table.clear()
        self._hpo_label.setVisible(False)
        self._trial_table.setVisible(False)
        self._trial_table.clear()
        self._hpo_label.setVisible(False)
        self._trial_table.setVisible(False)

    def _poll_metrics(self) -> None:
        """Read new lines from the metrics JSONL file."""
        if not self._metrics_path or not os.path.isfile(self._metrics_path):
            return

        try:
            with open(self._metrics_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            return

        new_lines = lines[self._last_line_count :]
        self._last_line_count = len(lines)

        for line in new_lines:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            epoch = data.get("epoch", 0)
            self.add_metrics(
                epoch=epoch,
                train_loss=data.get("train_loss"),
                val_loss=data.get("val_loss"),
                map50=data.get("map50"),
            )

            if "total_epochs" in data:
                lr = data.get("lr")
                stage = data.get("stage")
                eta = data.get("eta")
                self.update_progress(
                    epoch=int(epoch),
                    total_epochs=data["total_epochs"],
                    stage=stage,
                    lr=lr,
                    eta=eta,
                )

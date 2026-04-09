"""HPO trial table widget — displays hyperparameter optimization trial status."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)


class TrialTable(QWidget):
    """Table displaying HPO trial status and results."""

    # Column indices
    COL_TRIAL = 0
    COL_PARAMS = 1
    COL_METRIC = 2
    COL_STATUS = 3

    STATUS_ICONS = {
        "completed": "✓",
        "running": "●",
        "pruned": "✗",
        "pending": "…",
        "failed": "✗",
    }

    STATUS_COLORS = {
        "completed": "#4caf50",
        "running": "#ff9800",
        "pruned": "#9e9e9e",
        "pending": "#42a5f5",
        "failed": "#f44336",
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from PyQt6.QtWidgets import QVBoxLayout

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Trial", "Params", "Metric", "Status"])
        header = self._table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(self.COL_PARAMS, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(self.COL_METRIC, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(self.COL_STATUS, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        layout.addWidget(self._table)

        self._trials: list[dict[str, Any]] = []

    def set_trials(self, trials: list[dict[str, Any]]) -> None:
        """Set all trial data. Each dict: {trial_id, params, metric, status}."""
        self._trials = trials
        self._table.setRowCount(len(trials))

        for row, trial in enumerate(trials):
            # Trial ID
            id_item = QTableWidgetItem(str(trial.get("trial_id", row + 1)))
            id_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, self.COL_TRIAL, id_item)

            # Params summary
            params = trial.get("params", {})
            params_str = ", ".join(f"{k}={v}" for k, v in params.items())
            self._table.setItem(row, self.COL_PARAMS, QTableWidgetItem(params_str))

            # Metric value
            metric = trial.get("metric")
            metric_str = f"{metric:.4f}" if isinstance(metric, (int, float)) else str(metric or "—")
            metric_item = QTableWidgetItem(metric_str)
            metric_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, self.COL_METRIC, metric_item)

            # Status with icon + color
            status = trial.get("status", "pending")
            icon = self.STATUS_ICONS.get(status, "?")
            status_item = QTableWidgetItem(f"{icon} {status}")
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            color_hex = self.STATUS_COLORS.get(status, "#ffffff")
            status_item.setForeground(QColor(color_hex))
            self._table.setItem(row, self.COL_STATUS, status_item)

    def update_trial(self, trial_id: int, metric: float | None = None, status: str | None = None) -> None:
        """Update a single trial's metric and/or status."""
        for row, trial in enumerate(self._trials):
            if trial.get("trial_id") == trial_id:
                if metric is not None:
                    trial["metric"] = metric
                    metric_str = f"{metric:.4f}"
                    item = QTableWidgetItem(metric_str)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self._table.setItem(row, self.COL_METRIC, item)
                if status is not None:
                    trial["status"] = status
                    icon = self.STATUS_ICONS.get(status, "?")
                    item = QTableWidgetItem(f"{icon} {status}")
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    item.setForeground(QColor(self.STATUS_COLORS.get(status, "#ffffff")))
                    self._table.setItem(row, self.COL_STATUS, item)
                break

    def clear(self) -> None:
        self._trials.clear()
        self._table.setRowCount(0)

    @property
    def trial_count(self) -> int:
        return len(self._trials)

"""Results panel — post-training evaluation display with per-class metrics and CSV export."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..widgets.csv_preview import CsvPreview


class ResultsPanel(QWidget):
    """Post-training results display: overall metrics, per-class table, worst predictions, CSV preview."""

    accept_model = pyqtSignal()  # user accepts model
    retrain_requested = pyqtSignal()  # user wants to retrain
    retrain_hpo_requested = pyqtSignal()  # user wants retrain with HPO

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # ── Header ──
        self._title = QLabel("Results")
        self._title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        layout.addWidget(self._title)

        # ── Overall metrics ──
        self._overall_label = QLabel("No results loaded")
        self._overall_label.setWordWrap(True)
        self._overall_label.setStyleSheet("font-size: 11pt;")
        layout.addWidget(self._overall_label)

        # ── Per-class table ──
        per_class_label = QLabel("Per-class metrics:")
        per_class_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        layout.addWidget(per_class_label)

        self._class_table = QTableWidget(0, 4)
        self._class_table.setHorizontalHeaderLabels(["Class", "AP@50", "Precision", "Recall"])
        header = self._class_table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            for col in range(1, 4):
                header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self._class_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._class_table.setMaximumHeight(200)
        layout.addWidget(self._class_table)

        # ── Worst predictions ──
        self._worst_label = QLabel("Worst predictions:")
        self._worst_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        layout.addWidget(self._worst_label)

        self._worst_images_row = QHBoxLayout()
        self._worst_buttons: list[QPushButton] = []
        layout.addLayout(self._worst_images_row)

        # ── CSV Preview ──
        csv_label = QLabel("Export Preview:")
        csv_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        layout.addWidget(csv_label)

        self._csv_preview = CsvPreview()
        layout.addWidget(self._csv_preview, stretch=1)

        # ── Action buttons ──
        action_row = QHBoxLayout()
        self._accept_btn = QPushButton("Accept & Deploy")
        self._accept_btn.setStyleSheet("background: #4caf50; color: white; font-weight: bold; padding: 8px 16px;")
        self._accept_btn.clicked.connect(self.accept_model.emit)
        action_row.addWidget(self._accept_btn)

        self._retrain_btn = QPushButton("Retrain")
        self._retrain_btn.setStyleSheet("background: #ff9800; color: white; padding: 8px 16px;")
        self._retrain_btn.clicked.connect(self.retrain_requested.emit)
        action_row.addWidget(self._retrain_btn)

        self._retrain_hpo_btn = QPushButton("Retrain + HPO")
        self._retrain_hpo_btn.setStyleSheet("background: #42a5f5; color: white; padding: 8px 16px;")
        self._retrain_hpo_btn.clicked.connect(self.retrain_hpo_requested.emit)
        action_row.addWidget(self._retrain_hpo_btn)

        action_row.addStretch()
        layout.addLayout(action_row)

        self._worst_image_paths: list[str] = []

    def set_run_name(self, name: str) -> None:
        self._title.setText(f"Results: {name}")

    def set_overall_metrics(
        self,
        map50: float,
        map50_95: float | None = None,
        precision: float | None = None,
        recall: float | None = None,
    ) -> None:
        parts = [f"mAP@50: {map50:.4f}"]
        if map50_95 is not None:
            parts.append(f"mAP@50-95: {map50_95:.4f}")
        if precision is not None:
            parts.append(f"Precision: {precision:.4f}")
        if recall is not None:
            parts.append(f"Recall: {recall:.4f}")
        self._overall_label.setText("  |  ".join(parts))

    def set_per_class_metrics(self, classes: list[dict[str, Any]]) -> None:
        """Set per-class metrics. Each dict: {name, ap50, precision, recall}."""
        self._class_table.setRowCount(len(classes))
        for row, cls in enumerate(classes):
            self._class_table.setItem(row, 0, QTableWidgetItem(str(cls.get("name", ""))))
            self._class_table.setItem(
                row, 1, QTableWidgetItem(f"{cls.get('ap50', 0):.4f}")
            )
            self._class_table.setItem(
                row, 2, QTableWidgetItem(f"{cls.get('precision', 0):.4f}")
            )
            self._class_table.setItem(
                row, 3, QTableWidgetItem(f"{cls.get('recall', 0):.4f}")
            )

    def set_worst_predictions(self, image_paths: list[str]) -> None:
        """Show clickable image names for worst predictions."""
        # Clear old
        for btn in self._worst_buttons:
            self._worst_images_row.removeWidget(btn)
            btn.deleteLater()
        self._worst_buttons.clear()
        self._worst_image_paths = image_paths

        from pathlib import Path

        for path in image_paths[:8]:  # show max 8
            name = Path(path).name
            btn = QPushButton(f"📷 {name}")
            btn.setToolTip(path)
            btn.setStyleSheet("padding: 4px 8px;")
            self._worst_buttons.append(btn)
            self._worst_images_row.addWidget(btn)

    def load_csv_preview(self, path: str) -> bool:
        return self._csv_preview.load_file(path)

    def set_csv_data(self, headers: list[str], rows: list[list[str]]) -> None:
        self._csv_preview.load_data(headers, rows)

    def reset(self) -> None:
        self._title.setText("Results")
        self._overall_label.setText("No results loaded")
        self._class_table.setRowCount(0)
        for btn in self._worst_buttons:
            self._worst_images_row.removeWidget(btn)
            btn.deleteLater()
        self._worst_buttons.clear()
        self._worst_image_paths.clear()
        self._csv_preview.clear()

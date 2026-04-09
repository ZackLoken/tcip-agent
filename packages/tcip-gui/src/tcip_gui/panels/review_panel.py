"""Review panel — TP/FP/FN review mode with color-coded annotations."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..widgets.confidence_slider import ConfidenceSlider


class ReviewPanel(QWidget):
    """Review mode panel for cycling through TP/FP/FN detections."""

    review_action = pyqtSignal(int, str)  # detection_index, action ("accept"/"edit"/"reject")
    iou_changed = pyqtSignal(float)
    confidence_changed = pyqtSignal(float)
    filter_changed = pyqtSignal(str)  # "all", "tp", "fp", "fn"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # Header
        self._title = QLabel("Review Mode")
        self._title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        layout.addWidget(self._title)

        # Stats row
        stats_row = QHBoxLayout()
        self._tp_label = QLabel("TP: 0")
        self._tp_label.setStyleSheet("color: #4caf50; font-weight: bold;")
        stats_row.addWidget(self._tp_label)
        self._fp_label = QLabel("FP: 0")
        self._fp_label.setStyleSheet("color: #f44336; font-weight: bold;")
        stats_row.addWidget(self._fp_label)
        self._fn_label = QLabel("FN: 0")
        self._fn_label.setStyleSheet("color: #2196f3; font-weight: bold;")
        stats_row.addWidget(self._fn_label)
        layout.addLayout(stats_row)

        # Current detection info
        self._det_info = QLabel("No detections loaded")
        self._det_info.setWordWrap(True)
        layout.addWidget(self._det_info)

        # Navigation
        nav_row = QHBoxLayout()
        self._prev_btn = QPushButton("◀ Prev")
        self._prev_btn.clicked.connect(self._on_prev)
        nav_row.addWidget(self._prev_btn)

        self._accept_btn = QPushButton("Accept")
        self._accept_btn.setStyleSheet("background: #4caf50; color: white;")
        self._accept_btn.clicked.connect(lambda: self._on_action("accept"))
        nav_row.addWidget(self._accept_btn)

        self._edit_btn = QPushButton("Edit")
        self._edit_btn.setStyleSheet("background: #ff9800; color: white;")
        self._edit_btn.clicked.connect(lambda: self._on_action("edit"))
        nav_row.addWidget(self._edit_btn)

        self._reject_btn = QPushButton("Reject")
        self._reject_btn.setStyleSheet("background: #f44336; color: white;")
        self._reject_btn.clicked.connect(lambda: self._on_action("reject"))
        nav_row.addWidget(self._reject_btn)

        self._next_btn = QPushButton("Next ▶")
        self._next_btn.clicked.connect(self._on_next)
        nav_row.addWidget(self._next_btn)
        layout.addLayout(nav_row)

        layout.addSpacing(12)

        # Filters
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["All", "TP", "FP", "FN"])
        self._filter_combo.currentTextChanged.connect(
            lambda t: self.filter_changed.emit(t.lower())
        )
        filter_row.addWidget(self._filter_combo)
        layout.addLayout(filter_row)

        # IoU threshold
        iou_row = QHBoxLayout()
        iou_row.addWidget(QLabel("IoU threshold:"))
        self._iou_combo = QComboBox()
        for v in ["0.50", "0.55", "0.60", "0.65", "0.70", "0.75"]:
            self._iou_combo.addItem(v)
        self._iou_combo.currentTextChanged.connect(
            lambda t: self.iou_changed.emit(float(t))
        )
        iou_row.addWidget(self._iou_combo)
        layout.addLayout(iou_row)

        # Confidence slider
        self._conf_slider = ConfidenceSlider("Confidence:", initial=0.25)
        self._conf_slider.value_changed.connect(self.confidence_changed.emit)
        layout.addWidget(self._conf_slider)

        layout.addStretch()

        # State
        self._detections: list[dict[str, Any]] = []
        self._current_idx: int = -1

    def set_detections(self, detections: list[dict[str, Any]]) -> None:
        """Load detections for review. Supports both boxes and segmentation masks."""
        self._detections = detections
        self._current_idx = 0 if detections else -1

        tp = sum(1 for d in detections if d.get("det_type") == "tp")
        fp = sum(1 for d in detections if d.get("det_type") == "fp")
        fn = sum(1 for d in detections if d.get("det_type") == "fn")
        self._tp_label.setText(f"TP: {tp}")
        self._fp_label.setText(f"FP: {fp}")
        self._fn_label.setText(f"FN: {fn}")

        self._update_info()

    def _update_info(self) -> None:
        if not self._detections or self._current_idx < 0:
            self._det_info.setText("No detections loaded")
            return
        total = len(self._detections)
        d = self._detections[self._current_idx]
        dtype = d.get("det_type", "?").upper()
        conf = d.get("conf", 0)
        iou = d.get("iou", 0)
        cls = d.get("class_name", d.get("class_id", "?"))
        # Show annotation type (bbox vs mask)
        ann_type = "mask" if d.get("mask") is not None else "bbox"
        self._det_info.setText(
            f"Detection {self._current_idx + 1} of {total}\n"
            f"Type: {dtype}  Class: {cls}  Conf: {conf:.2f}  IoU: {iou:.2f}  [{ann_type}]"
        )

    def _on_prev(self) -> None:
        if self._detections and self._current_idx > 0:
            self._current_idx -= 1
            self._update_info()

    def _on_next(self) -> None:
        if self._detections and self._current_idx < len(self._detections) - 1:
            self._current_idx += 1
            self._update_info()

    def _on_action(self, action: str) -> None:
        if self._current_idx >= 0:
            self.review_action.emit(self._current_idx, action)
            self._on_next()

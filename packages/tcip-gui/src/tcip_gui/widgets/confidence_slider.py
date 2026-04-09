"""Confidence threshold slider widget."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSlider, QWidget
from PyQt6.QtCore import Qt


class ConfidenceSlider(QWidget):
    """Slider for confidence threshold with value display."""

    value_changed = pyqtSignal(float)  # 0.0–1.0

    def __init__(self, label: str = "Confidence", initial: float = 0.25, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel(label))

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 100)
        self._slider.setValue(int(initial * 100))
        self._slider.valueChanged.connect(self._on_changed)
        layout.addWidget(self._slider, stretch=1)

        self._value_label = QLabel(f"{initial:.2f}")
        self._value_label.setFixedWidth(40)
        layout.addWidget(self._value_label)

    def value(self) -> float:
        return self._slider.value() / 100.0

    def _on_changed(self, v: int) -> None:
        f = v / 100.0
        self._value_label.setText(f"{f:.2f}")
        self.value_changed.emit(f)

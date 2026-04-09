"""Tool card — collapsible display of a tool call and its result."""

from __future__ import annotations

import json
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ToolCard(QFrame):
    """Collapsible card showing a tool call's name, input, and result."""

    def __init__(
        self, tool_id: str, name: str, input_data: dict[str, Any], parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("tool-card")

        self._tool_id = tool_id
        self._expanded = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        # Header row
        header = QHBoxLayout()
        self._status_icon = QLabel("⏳")
        header.addWidget(self._status_icon)

        self._title = QLabel(f"Tool: {name}")
        self._title.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        header.addWidget(self._title, stretch=1)

        self._toggle_btn = QPushButton("▶")
        self._toggle_btn.setFixedSize(24, 24)
        self._toggle_btn.setFlat(True)
        self._toggle_btn.clicked.connect(self._toggle)
        header.addWidget(self._toggle_btn)
        layout.addLayout(header)

        # Collapsible detail area
        self._detail = QWidget()
        detail_layout = QVBoxLayout(self._detail)
        detail_layout.setContentsMargins(0, 0, 0, 0)

        input_label = QLabel("Input:")
        input_label.setFont(QFont("Segoe UI", 8))
        detail_layout.addWidget(input_label)

        input_text = QTextEdit()
        input_text.setReadOnly(True)
        input_text.setPlainText(json.dumps(input_data, indent=2))
        input_text.setMaximumHeight(80)
        input_text.setFont(QFont("Consolas", 9))
        detail_layout.addWidget(input_text)

        self._output_label = QLabel("Output:")
        self._output_label.setFont(QFont("Segoe UI", 8))
        self._output_label.hide()
        detail_layout.addWidget(self._output_label)

        self._output_text = QTextEdit()
        self._output_text.setReadOnly(True)
        self._output_text.setMaximumHeight(120)
        self._output_text.setFont(QFont("Consolas", 9))
        self._output_text.hide()
        detail_layout.addWidget(self._output_text)

        self._detail.hide()
        layout.addWidget(self._detail)

    def set_result(self, output: str, is_error: bool) -> None:
        self._status_icon.setText("❌" if is_error else "✅")
        self._output_label.show()
        self._output_text.setPlainText(output)
        self._output_text.show()
        if is_error:
            self.setStyleSheet("background: #fff0f0;")
            self._output_text.setMaximumHeight(200)
            # Auto-expand on error so the detail is immediately visible
            if not self._expanded:
                self._toggle()

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self._detail.setVisible(self._expanded)
        self._toggle_btn.setText("▼" if self._expanded else "▶")

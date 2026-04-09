"""Permission card — HITL checkpoint approval dialog inline in chat."""

from __future__ import annotations

import json
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class PermissionCard(QFrame):
    """Inline permission approval card with Approve/Deny buttons."""

    responded = pyqtSignal(str, bool, str)  # request_id, allowed, reason

    def __init__(
        self,
        request_id: str,
        tool: str,
        input_data: dict[str, Any],
        description: str,
        level: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("permission-card")
        self.setStyleSheet("background: #fff8e1; border: 1px solid #ffcc80;")

        self._request_id = request_id

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Header
        header = QLabel(f"⚠ Checkpoint: {tool}")
        header.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        layout.addWidget(header)

        if description:
            desc = QLabel(description)
            desc.setWordWrap(True)
            layout.addWidget(desc)

        # Config display
        config = QTextEdit()
        config.setReadOnly(True)
        config.setPlainText(json.dumps(input_data, indent=2))
        config.setMaximumHeight(100)
        config.setFont(QFont("Consolas", 9))
        layout.addWidget(config)

        # Level indicator
        level_label = QLabel(f"Permission level: {level}")
        level_label.setStyleSheet("color: #666;")
        layout.addWidget(level_label)

        # Buttons
        btn_row = QHBoxLayout()
        self._approve_btn = QPushButton("Approve")
        self._approve_btn.setStyleSheet("background: #4caf50; color: white; padding: 4px 16px;")
        self._approve_btn.clicked.connect(self._approve)
        btn_row.addWidget(self._approve_btn)

        self._deny_btn = QPushButton("Deny")
        self._deny_btn.setStyleSheet("background: #f44336; color: white; padding: 4px 16px;")
        self._deny_btn.clicked.connect(self._deny)
        btn_row.addWidget(self._deny_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

    def _approve(self) -> None:
        self._approve_btn.setEnabled(False)
        self._deny_btn.setEnabled(False)
        self.setStyleSheet("background: #e8f5e9; border: 1px solid #a5d6a7;")
        self.responded.emit(self._request_id, True, "")

    def _deny(self) -> None:
        self._approve_btn.setEnabled(False)
        self._deny_btn.setEnabled(False)
        self.setStyleSheet("background: #ffebee; border: 1px solid #ef9a9a;")
        self.responded.emit(self._request_id, False, "User denied")

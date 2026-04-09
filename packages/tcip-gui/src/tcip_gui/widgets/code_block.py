"""Code block widget — syntax-highlighted code display."""

from __future__ import annotations

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QTextEdit, QWidget


class CodeBlock(QTextEdit):
    """Read-only code display with monospace font."""

    def __init__(self, code: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 10))
        self.setStyleSheet(
            "background: #1e1e1e; color: #d4d4d4; "
            "border: 1px solid #333; border-radius: 4px; padding: 8px;"
        )
        if code:
            self.setPlainText(code)

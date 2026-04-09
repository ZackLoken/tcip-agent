"""Placeholder panel for the center area (Phase 3)."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


class PlaceholderPanel(QWidget):
    """Placeholder for the center area — replaced by annotation canvas in Phase 4."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("TCIP Agent")
        title.setFont(QFont("Segoe UI", 24, QFont.Weight.Light))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel(
            "Tree Crop Image Phenotyping Platform\n\n"
            "Use the chat panel on the left to interact with the agent.\n"
            "The annotation canvas and training dashboard will appear here."
        )
        subtitle.setFont(QFont("Segoe UI", 11))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #888;")
        layout.addWidget(subtitle)

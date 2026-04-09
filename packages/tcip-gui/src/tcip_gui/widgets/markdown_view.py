"""Markdown rendering widget."""

from __future__ import annotations

from PyQt6.QtWidgets import QTextBrowser, QWidget


class MarkdownView(QTextBrowser):
    """Simple markdown rendering using QTextBrowser's built-in HTML support."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setOpenExternalLinks(True)
        self.setReadOnly(True)

    def set_markdown(self, md: str) -> None:
        self.setMarkdown(md)

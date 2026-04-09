"""Chat panel — displays conversation history with tool cards and permission prompts."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QKeyEvent
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..widgets.tool_card import ToolCard
from ..widgets.permission_card import PermissionCard


class ChatInput(QTextEdit):
    """Multi-line text input that sends on Enter (Shift+Enter for newline)."""

    submitted = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setPlaceholderText("Type a message...")
        self.setMaximumHeight(100)
        self.setAcceptRichText(False)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (
            event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            text = self.toPlainText().strip()
            if text:
                self.submitted.emit(text)
                self.clear()
            return
        super().keyPressEvent(event)


class MessageBubble(QFrame):
    """A single message bubble in the chat history."""

    def __init__(self, role: str, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName(f"bubble-{role}")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)

        header = QLabel(role.capitalize())
        header.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        header.setObjectName("bubble-header")
        layout.addWidget(header)

        self._body = QTextBrowser()
        self._body.setOpenExternalLinks(True)
        self._body.setFrameShape(QFrame.Shape.NoFrame)
        self._body.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._body.setStyleSheet("background: transparent; border: none; padding: 0;")
        self._body.setMarkdown(text)
        doc = self._body.document()
        if doc is not None:
            doc.setDocumentMargin(0)
        layout.addWidget(self._body)

    def append_text(self, text: str) -> None:
        current = self._body.toPlainText()
        self._body.setMarkdown(current + text)


class ChatPanel(QWidget):
    """Full chat panel with scrollable history, tool cards, and input box."""

    message_submitted = pyqtSignal(str)
    permission_responded = pyqtSignal(str, bool, str)  # request_id, allowed, reason

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Scrollable message area
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._history = QWidget()
        self._history_layout = QVBoxLayout(self._history)
        self._history_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._history_layout.setSpacing(8)
        self._history_layout.setContentsMargins(8, 8, 8, 8)

        self._scroll.setWidget(self._history)
        layout.addWidget(self._scroll, stretch=1)

        # Input area
        input_bar = QWidget()
        input_layout = QHBoxLayout(input_bar)
        input_layout.setContentsMargins(8, 4, 8, 8)

        self._input = ChatInput()
        input_layout.addWidget(self._input, stretch=1)

        send_btn = QPushButton("▶")
        send_btn.setFixedSize(36, 36)
        send_btn.clicked.connect(self._on_send)
        input_layout.addWidget(send_btn)

        layout.addWidget(input_bar)

        # Connect input
        self._input.submitted.connect(self._on_submitted)

        # Track current streaming bubble
        self._current_bubble: MessageBubble | None = None
        self._tool_cards: dict[str, ToolCard] = {}

    # ── Public API (connected to bridge signals) ──

    def add_user_message(self, text: str) -> None:
        bubble = MessageBubble("you", text)
        self._history_layout.addWidget(bubble)
        self._scroll_to_bottom()

    def on_text_delta(self, text: str) -> None:
        if self._current_bubble is None:
            self._current_bubble = MessageBubble("agent", "")
            self._history_layout.addWidget(self._current_bubble)
        self._current_bubble.append_text(text)
        self._scroll_to_bottom()

    def on_text_done(self, text: str) -> None:
        if self._current_bubble is None:
            self._current_bubble = MessageBubble("agent", text)
            self._history_layout.addWidget(self._current_bubble)
        self._current_bubble = None
        self._scroll_to_bottom()

    def on_tool_call_start(self, tool_id: str, name: str, input_data: dict) -> None:
        card = ToolCard(tool_id, name, input_data)
        self._tool_cards[tool_id] = card
        self._history_layout.addWidget(card)
        self._scroll_to_bottom()

    def on_tool_call_result(self, tool_id: str, output: str, is_error: bool) -> None:
        card = self._tool_cards.get(tool_id)
        if card:
            card.set_result(output, is_error)
        self._scroll_to_bottom()

    def on_permission_request(
        self, request_id: str, tool: str, input_data: dict, description: str, level: str,
    ) -> None:
        card = PermissionCard(request_id, tool, input_data, description, level)
        card.responded.connect(self._on_permission_response)
        self._history_layout.addWidget(card)
        self._scroll_to_bottom()

    def on_agent_error(self, code: int, message: str) -> None:
        bubble = MessageBubble("error", f"[{code}] {message}")
        self._history_layout.addWidget(bubble)
        self._scroll_to_bottom()

    def on_agent_crash(self, message: str) -> None:
        bubble = MessageBubble("system", f"⚠ {message}")
        self._history_layout.addWidget(bubble)
        self._scroll_to_bottom()

    def set_input_enabled(self, enabled: bool) -> None:
        self._input.setEnabled(enabled)

    # ── Private ──

    def _on_send(self) -> None:
        text = self._input.toPlainText().strip()
        if text:
            self._input.clear()
            self._on_submitted(text)

    def _on_submitted(self, text: str) -> None:
        self.add_user_message(text)
        self.message_submitted.emit(text)

    def _on_permission_response(self, request_id: str, allowed: bool, reason: str) -> None:
        self.permission_responded.emit(request_id, allowed, reason)

    def _scroll_to_bottom(self) -> None:
        sb = self._scroll.verticalScrollBar()
        if sb is not None:
            sb.setValue(sb.maximum())

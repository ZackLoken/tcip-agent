"""Chat overlay — floating panel that slides in/out from the right edge.

Supports three visual states:
  - hidden  (fully off-screen, invisible)
  - open    (full-height panel pinned to the right edge)
  - minimized (header-only strip pinned at the bottom-right corner)
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QRect, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .chat_panel import ChatPanel

_OVERLAY_WIDTH = 380
_HEADER_HEIGHT = 36


class ChatOverlay(QFrame):
    """Floating chat panel that can be toggled over the main content."""

    message_submitted = pyqtSignal(str)
    permission_responded = pyqtSignal(str, bool, str)

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("chat-overlay")
        self.setFrameShape(QFrame.Shape.NoFrame)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        # Header bar
        self._header = QFrame()
        self._header.setObjectName("chat-overlay-header")
        self._header.setFixedHeight(_HEADER_HEIGHT)
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(12, 0, 8, 0)

        title = QLabel("Agent Chat")
        title.setStyleSheet("font-size: 13px; font-weight: 600; color: #E0E0E0;")
        header_layout.addWidget(title)
        header_layout.addStretch()

        _btn_style = (
            "QPushButton { background: transparent; color: #AAAAAA; border: none; "
            "font-size: 14px; border-radius: 14px; } "
            "QPushButton:hover { background: #3A3A3A; color: #E0E0E0; }"
        )

        self._minimize_btn = QPushButton("▾")
        self._minimize_btn.setFixedSize(28, 28)
        self._minimize_btn.setToolTip("Minimize")
        self._minimize_btn.setStyleSheet(_btn_style)
        self._minimize_btn.clicked.connect(self.minimize)
        header_layout.addWidget(self._minimize_btn)

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(28, 28)
        close_btn.setToolTip("Close")
        close_btn.setStyleSheet(_btn_style)
        close_btn.clicked.connect(self.slide_out)
        header_layout.addWidget(close_btn)

        self._layout.addWidget(self._header)

        # Embedded chat panel
        self._chat = ChatPanel()
        self._chat.setMinimumWidth(0)  # Override ChatPanel's min-width
        self._layout.addWidget(self._chat, stretch=1)

        # Forward signals
        self._chat.message_submitted.connect(self.message_submitted)
        self._chat.permission_responded.connect(self.permission_responded)

        # Animation
        self._animation = QPropertyAnimation(self, b"geometry")
        self._animation.setDuration(250)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._visible = False
        self._minimized = False
        self.hide()

    @property
    def chat(self) -> ChatPanel:
        return self._chat

    @property
    def is_open(self) -> bool:
        return self._visible and not self._minimized

    @property
    def is_minimized(self) -> bool:
        return self._visible and self._minimized

    def slide_in(self) -> None:
        """Slide the overlay in from the right (full height)."""
        if self._visible and not self._minimized:
            return
        was_minimized = self._minimized
        self._visible = True
        self._minimized = False
        self._chat.setVisible(True)
        self._minimize_btn.setText("▾")
        self._minimize_btn.setToolTip("Minimize")

        parent = self.parentWidget()
        if parent is None:
            return
        h = parent.height()
        w = _OVERLAY_WIDTH

        self.setFixedWidth(w)
        self.setFixedHeight(h)
        self.show()
        self.raise_()

        if was_minimized:
            # Animate from minimized position (bottom-right) to full height
            start = QRect(parent.width() - w, h - _HEADER_HEIGHT, w, _HEADER_HEIGHT)
        else:
            # Animate from off-screen right
            start = QRect(parent.width(), 0, w, h)
        end = QRect(parent.width() - w, 0, w, h)

        self._animation.setStartValue(start)
        self._animation.setEndValue(end)
        self._animation.start()

    def slide_out(self) -> None:
        """Slide the overlay out to the right (hide completely)."""
        if not self._visible:
            return
        self._visible = False
        self._minimized = False
        parent = self.parentWidget()
        if parent is None:
            self.hide()
            return
        h = parent.height()
        w = _OVERLAY_WIDTH

        start = self.geometry()
        end = QRect(parent.width(), 0, w, h)

        self._animation.setStartValue(start)
        self._animation.setEndValue(end)
        self._animation.finished.connect(self._on_slide_out_done)
        self._animation.start()

    def minimize(self) -> None:
        """Collapse to header-only strip at bottom-right, or restore if already minimized."""
        if self._minimized:
            self.slide_in()
            return

        if not self._visible:
            return

        self._minimized = True
        self._minimize_btn.setText("▴")
        self._minimize_btn.setToolTip("Restore")

        parent = self.parentWidget()
        if parent is None:
            return
        h = parent.height()
        w = _OVERLAY_WIDTH

        start = self.geometry()
        end = QRect(parent.width() - w, h - _HEADER_HEIGHT, w, _HEADER_HEIGHT)

        self._chat.setVisible(False)
        self.setMinimumHeight(0)
        self.setMaximumHeight(16777215)

        self._animation.setStartValue(start)
        self._animation.setEndValue(end)
        self._animation.start()

    def toggle(self) -> None:
        if self._minimized:
            self.slide_in()
        elif self._visible:
            self.slide_out()
        else:
            self.slide_in()

    def reposition(self) -> None:
        """Reposition the overlay when parent is resized."""
        if not self._visible:
            return
        parent = self.parentWidget()
        if parent is None:
            return
        w = _OVERLAY_WIDTH
        if self._minimized:
            self.setGeometry(parent.width() - w, parent.height() - _HEADER_HEIGHT, w, _HEADER_HEIGHT)
        else:
            self.setGeometry(parent.width() - w, 0, w, parent.height())

    def _on_slide_out_done(self) -> None:
        self._animation.finished.disconnect(self._on_slide_out_done)
        self.hide()

"""Status bar — compact 32px bar with agent status, workspace, tokens, and cost."""

from __future__ import annotations

from PyQt6.QtWidgets import QStatusBar, QLabel, QWidget


class AgentStatusBar(QStatusBar):
    """Compact status bar matching yolo-annotator style (32px height)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(32)

        # Connection indicator (left)
        self._connection_label = QLabel("⚪ Disconnected")
        self._connection_label.setStyleSheet("color: #AAAAAA; padding: 0 8px;")
        self.addWidget(self._connection_label, stretch=0)

        # Workspace path (center, stretch)
        self._workspace_label = QLabel("")
        self._workspace_label.setStyleSheet("color: #666666; padding: 0 8px;")
        self.addWidget(self._workspace_label, stretch=1)

        # Token usage (right)
        self._tokens_label = QLabel("")
        self._tokens_label.setStyleSheet("color: #AAAAAA; padding: 0 6px;")
        self.addPermanentWidget(self._tokens_label)

        # Cost (far right)
        self._cost_label = QLabel("")
        self._cost_label.setStyleSheet("color: #AAAAAA; padding: 0 8px;")
        self.addPermanentWidget(self._cost_label)

    def set_workspace(self, path: str) -> None:
        self._workspace_label.setText(path)

    def set_connected(self) -> None:
        self._connection_label.setText("🟢 Agent connected")
        self._connection_label.setStyleSheet("color: #4CAF50; padding: 0 8px;")

    def set_disconnected(self) -> None:
        self._connection_label.setText("⚪ Disconnected")
        self._connection_label.setStyleSheet("color: #AAAAAA; padding: 0 8px;")

    def set_crashed(self) -> None:
        self._connection_label.setText("🔴 Agent crashed")
        self._connection_label.setStyleSheet("color: #EF5350; padding: 0 8px;")

    def set_status(self, text: str) -> None:
        self._connection_label.setText(text)

    def update_usage(self, input_tokens: int, output_tokens: int, cost: float) -> None:
        self._tokens_label.setText(f"↑{input_tokens:,}  ↓{output_tokens:,}")
        self._cost_label.setText(f"${cost:.4f}")

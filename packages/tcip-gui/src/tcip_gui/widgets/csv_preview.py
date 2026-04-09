"""CSV preview widget — displays CSV data in a table view."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class CsvPreview(QWidget):
    """Table widget for previewing CSV file contents."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._info_label = QLabel("No CSV loaded")
        layout.addWidget(self._info_label)

        self._table = QTableWidget()
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        layout.addWidget(self._table, stretch=1)

        self._rows: list[list[str]] = []
        self._headers: list[str] = []

    def load_file(self, path: str, max_rows: int = 200) -> bool:
        """Load a CSV file into the table. Returns True on success."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
        except (OSError, csv.Error):
            self._info_label.setText(f"Error loading: {path}")
            return False

        if not rows:
            self._info_label.setText("Empty CSV file")
            return False

        self._headers = rows[0]
        self._rows = rows[1 : max_rows + 1]
        total_rows = len(rows) - 1  # exclude header

        self._info_label.setText(
            f"{Path(path).name}  |  {total_rows} rows × {len(self._headers)} columns"
            + (f"  (showing first {max_rows})" if total_rows > max_rows else "")
        )
        self._populate_table()
        return True

    def load_data(self, headers: list[str], rows: list[list[str]]) -> None:
        """Load CSV data directly (without file)."""
        self._headers = headers
        self._rows = rows
        self._info_label.setText(f"{len(rows)} rows × {len(headers)} columns")
        self._populate_table()

    def _populate_table(self) -> None:
        self._table.clear()
        self._table.setColumnCount(len(self._headers))
        self._table.setRowCount(len(self._rows))
        self._table.setHorizontalHeaderLabels(self._headers)

        header = self._table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
            header.setStretchLastSection(True)

        for row_idx, row_data in enumerate(self._rows):
            for col_idx, cell in enumerate(row_data):
                if col_idx >= len(self._headers):
                    break
                item = QTableWidgetItem(cell)
                item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                self._table.setItem(row_idx, col_idx, item)

    def clear(self) -> None:
        self._rows.clear()
        self._headers.clear()
        self._table.clear()
        self._table.setRowCount(0)
        self._table.setColumnCount(0)
        self._info_label.setText("No CSV loaded")

    @property
    def row_count(self) -> int:
        return len(self._rows)

    @property
    def column_count(self) -> int:
        return len(self._headers)

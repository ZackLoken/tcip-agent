"""Class selector — combobox for annotation class label with color indicator."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor, QPixmap, QIcon
from PyQt6.QtWidgets import QComboBox, QWidget

from .box_item import CLASS_COLORS


class ClassSelector(QComboBox):
    """Dropdown to pick the active annotation class."""

    class_changed = pyqtSignal(int)  # class_id

    def __init__(self, classes: list[str] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._classes: list[str] = []
        if classes:
            self.set_classes(classes)
        self.currentIndexChanged.connect(lambda idx: self.class_changed.emit(idx))

    def set_classes(self, classes: list[str]) -> None:
        self._classes = classes
        self.blockSignals(True)
        self.clear()
        for i, name in enumerate(classes):
            pixmap = QPixmap(16, 16)
            pixmap.fill(QColor(CLASS_COLORS[i % len(CLASS_COLORS)]))
            self.addItem(QIcon(pixmap), name)
        self.blockSignals(False)

    def current_class_id(self) -> int:
        return max(0, self.currentIndex())

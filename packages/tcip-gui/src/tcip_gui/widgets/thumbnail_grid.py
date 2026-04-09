"""Thumbnail grid widget for the dataset browser."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import QListWidget, QListWidgetItem, QWidget

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


class ThumbnailGrid(QListWidget):
    """Grid of image thumbnails with status indicators."""

    image_selected = pyqtSignal(str)  # image path

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setIconSize(QSize(96, 96))
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setSpacing(4)
        self.setWrapping(True)
        self.setUniformItemSizes(True)
        self.itemClicked.connect(self._on_click)
        self._image_paths: dict[str, Path] = {}

    def load_directory(self, directory: str | Path, label_dir: str | Path | None = None) -> int:
        """Load images from a directory. Returns count of images loaded."""
        d = Path(directory)
        if not d.is_dir():
            self.clear()
            self._image_paths.clear()
            return 0

        label_d = Path(label_dir) if label_dir else None
        annotated_stems: set[str] = set()
        if label_d and label_d.is_dir():
            annotated_stems = {
                f.stem for f in label_d.iterdir()
                if f.suffix == ".txt" and f.stat().st_size > 0
            }

        paths = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        return self.load_paths(paths, annotated_stems)

    def load_paths(self, paths: list[Path], annotated_stems: set[str] | None = None) -> int:
        """Load a pre-filtered list of image paths. Returns count loaded."""
        self.clear()
        self._image_paths.clear()
        if annotated_stems is None:
            annotated_stems = set()

        count = 0
        for p in paths:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, str(p))
            item.setToolTip(str(p))

            has_label = p.stem in annotated_stems
            item.setText(f"\u2705 {p.stem}" if has_label else p.stem)

            pixmap = QPixmap(str(p))
            if not pixmap.isNull():
                scaled = pixmap.scaled(QSize(96, 96), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                item.setIcon(QIcon(scaled))

            self.addItem(item)
            self._image_paths[str(p)] = p
            count += 1
        return count

    def _on_click(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.image_selected.emit(path)

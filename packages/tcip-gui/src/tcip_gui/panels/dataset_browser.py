"""Dataset browser — thumbnail grid with statistics and filtering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from ..widgets.thumbnail_grid import ThumbnailGrid


class DatasetBrowser(QWidget):
    """Side panel with dataset stats, thumbnail grid, and filters."""

    image_selected = pyqtSignal(str)  # image path

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(200)
        layout = QVBoxLayout(self)

        # Dataset header
        self._name_label = QLabel("No dataset loaded")
        self._name_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        layout.addWidget(self._name_label)

        self._stats_label = QLabel("")
        self._stats_label.setWordWrap(True)
        layout.addWidget(self._stats_label)

        # Search
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search images...")
        self._search.textChanged.connect(self._apply_filters)
        layout.addWidget(self._search)

        # Filter — annotation status
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Status:"))
        self._filter = QComboBox()
        self._filter.addItems(["All", "Annotated", "Unannotated"])
        self._filter.currentTextChanged.connect(lambda _: self._apply_filters())
        filter_row.addWidget(self._filter)
        layout.addLayout(filter_row)

        # Filter — crop
        crop_row = QHBoxLayout()
        crop_row.addWidget(QLabel("Crop:"))
        self._crop_filter = QComboBox()
        self._crop_filter.addItem("All")
        self._crop_filter.currentTextChanged.connect(lambda _: self._apply_filters())
        crop_row.addWidget(self._crop_filter)
        layout.addLayout(crop_row)

        # Filter — date
        date_row = QHBoxLayout()
        date_row.addWidget(QLabel("Date:"))
        self._date_filter = QComboBox()
        self._date_filter.addItem("All")
        self._date_filter.currentTextChanged.connect(lambda _: self._apply_filters())
        date_row.addWidget(self._date_filter)
        layout.addLayout(date_row)

        # Sort
        sort_row = QHBoxLayout()
        sort_row.addWidget(QLabel("Sort:"))
        self._sort = QComboBox()
        self._sort.addItems(["Name", "Modified", "Annotated first", "Unannotated first"])
        self._sort.currentTextChanged.connect(lambda _: self._apply_filters())
        sort_row.addWidget(self._sort)
        layout.addLayout(sort_row)

        # Thumbnail grid
        self._grid = ThumbnailGrid()
        self._grid.image_selected.connect(self.image_selected.emit)
        layout.addWidget(self._grid, stretch=1)

        self._image_dir: Path | None = None
        self._label_dir: Path | None = None
        self._all_images: list[Path] = []
        self._annotated_stems: set[str] = set()
        self._image_metadata: dict[str, dict[str, str]] = {}  # stem → {crop, date}

    def load_dataset(self, image_dir: str | Path, label_dir: str | Path | None = None) -> None:
        """Load a dataset directory into the browser."""
        self._image_dir = Path(image_dir)
        self._label_dir = Path(label_dir) if label_dir else None

        self._name_label.setText(self._image_dir.name)

        # Collect all images
        from ..widgets.thumbnail_grid import IMAGE_EXTS
        if self._image_dir.is_dir():
            self._all_images = sorted(
                p for p in self._image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
            )
        else:
            self._all_images = []

        # Collect annotated stems
        self._annotated_stems.clear()
        if self._label_dir and self._label_dir.is_dir():
            self._annotated_stems = {
                f.stem for f in self._label_dir.iterdir()
                if f.suffix == ".txt" and f.stat().st_size > 0
            }

        # Extract metadata from filenames for crop/date filters
        self._image_metadata.clear()
        crops: set[str] = set()
        dates: set[str] = set()
        for p in self._all_images:
            meta = _parse_image_metadata(p.stem)
            self._image_metadata[p.stem] = meta
            if meta.get("crop"):
                crops.add(meta["crop"])
            if meta.get("date"):
                dates.add(meta["date"])

        # Populate filter dropdowns
        self._crop_filter.blockSignals(True)
        self._crop_filter.clear()
        self._crop_filter.addItem("All")
        for c in sorted(crops):
            self._crop_filter.addItem(c)
        self._crop_filter.blockSignals(False)

        self._date_filter.blockSignals(True)
        self._date_filter.clear()
        self._date_filter.addItem("All")
        for d in sorted(dates):
            self._date_filter.addItem(d)
        self._date_filter.blockSignals(False)

        # Update stats and render
        annotated = len(self._annotated_stems)
        total = len(self._all_images)
        self._stats_label.setText(
            f"Images: {total}\nAnnotated: {annotated} ({annotated * 100 // max(total, 1)}%)"
        )

        self._apply_filters()

    def _apply_filters(self) -> None:
        """Apply all active filters and sort, then refresh the grid."""
        if not self._all_images:
            return

        filtered = list(self._all_images)

        # Text search
        search_text = self._search.text().strip().lower()
        if search_text:
            filtered = [p for p in filtered if search_text in p.stem.lower()]

        # Annotation status filter
        status = self._filter.currentText()
        if status == "Annotated":
            filtered = [p for p in filtered if p.stem in self._annotated_stems]
        elif status == "Unannotated":
            filtered = [p for p in filtered if p.stem not in self._annotated_stems]

        # Crop filter
        crop = self._crop_filter.currentText()
        if crop != "All":
            filtered = [
                p for p in filtered
                if self._image_metadata.get(p.stem, {}).get("crop") == crop
            ]

        # Date filter
        date = self._date_filter.currentText()
        if date != "All":
            filtered = [
                p for p in filtered
                if self._image_metadata.get(p.stem, {}).get("date") == date
            ]

        # Sort
        sort_key = self._sort.currentText()
        if sort_key == "Modified":
            filtered.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        elif sort_key == "Annotated first":
            filtered.sort(key=lambda p: (0 if p.stem in self._annotated_stems else 1, p.name))
        elif sort_key == "Unannotated first":
            filtered.sort(key=lambda p: (1 if p.stem in self._annotated_stems else 0, p.name))
        # else: Name (default sorted already)

        self._grid.load_paths(filtered, self._annotated_stems)


def _parse_image_metadata(stem: str) -> dict[str, str]:
    """Extract crop and date hints from image filenames.

    Supports common naming patterns:
    - ``crop_date_id`` (e.g., ``hazelnut_20260315_001``)
    - ``IMG_NNNN`` (no metadata)
    - ``crop-date-id`` with hyphens
    """
    import re

    meta: dict[str, str] = {}
    # Try pattern: word_YYYYMMDD or word-YYYYMMDD
    m = re.match(r"^([a-zA-Z_]+)[_-](\d{8})", stem)
    if m:
        meta["crop"] = m.group(1).lower().replace("_", " ")
        raw_date = m.group(2)
        meta["date"] = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
        return meta

    # Try pattern: word_YYYY-MM-DD
    m = re.match(r"^([a-zA-Z_]+)[_-](\d{4}-\d{2}-\d{2})", stem)
    if m:
        meta["crop"] = m.group(1).lower().replace("_", " ")
        meta["date"] = m.group(2)
        return meta

    return meta

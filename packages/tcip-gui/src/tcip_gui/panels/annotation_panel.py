"""Annotation panel — toolbar and canvas for image annotation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from tcip_annotation.label_io import write_detect_labels, write_segment_labels

from ..widgets.canvas import AnnotationCanvas
from ..widgets.class_selector import ClassSelector
from ..widgets.confidence_slider import ConfidenceSlider


class AnnotationPanel(QWidget):
    """Annotation mode panel with toolbar and canvas."""

    annotation_saved = pyqtSignal(str, int)  # image_path, box_count

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = QToolBar()
        toolbar.setMovable(False)

        self._select_btn = QPushButton("Select")
        self._select_btn.setCheckable(True)
        self._select_btn.setChecked(True)
        self._select_btn.clicked.connect(lambda: self._set_mode("select"))
        toolbar.addWidget(self._select_btn)

        self._box_btn = QPushButton("Box")
        self._box_btn.setCheckable(True)
        self._box_btn.clicked.connect(lambda: self._set_mode("box"))
        toolbar.addWidget(self._box_btn)

        self._poly_btn = QPushButton("Polygon")
        self._poly_btn.setCheckable(True)
        self._poly_btn.clicked.connect(lambda: self._set_mode("polygon"))
        toolbar.addWidget(self._poly_btn)

        toolbar.addSeparator()

        self._class_selector = ClassSelector()
        self._class_selector.class_changed.connect(self._on_class_changed)
        toolbar.addWidget(QLabel(" Class: "))
        toolbar.addWidget(self._class_selector)

        toolbar.addSeparator()

        undo_btn = QPushButton("Undo")
        undo_btn.clicked.connect(lambda: self._canvas.undo())
        toolbar.addWidget(undo_btn)

        redo_btn = QPushButton("Redo")
        redo_btn.clicked.connect(lambda: self._canvas.redo())
        toolbar.addWidget(redo_btn)

        toolbar.addSeparator()

        save_btn = QPushButton("Save")
        save_btn.setStyleSheet("font-weight: bold;")
        save_btn.clicked.connect(self._on_save)
        toolbar.addWidget(save_btn)

        toolbar.addSeparator()

        # Confidence slider — filters prediction overlays
        self._confidence_slider = ConfidenceSlider(label="Conf ≥", initial=0.25)
        self._confidence_slider.value_changed.connect(self._on_confidence_changed)
        toolbar.addWidget(self._confidence_slider)

        layout.addWidget(toolbar)

        # Canvas
        self._canvas = AnnotationCanvas()
        layout.addWidget(self._canvas, stretch=1)

        self._mode_buttons = [self._select_btn, self._box_btn, self._poly_btn]

    @property
    def canvas(self) -> AnnotationCanvas:
        return self._canvas

    def set_classes(self, classes: list[str]) -> None:
        self._class_selector.set_classes(classes)

    def load_image(self, path: str) -> bool:
        return self._canvas.load_image(path)

    def _set_mode(self, mode: str) -> None:
        self._canvas.mode = mode
        for btn in self._mode_buttons:
            btn.setChecked(btn.text().lower() == mode)

    def _on_class_changed(self, class_id: int) -> None:
        self._canvas.active_class = class_id

    def _on_confidence_changed(self, threshold: float) -> None:
        """Hide prediction overlays below the threshold."""
        for item in self._canvas._pred_box_items:
            conf = item.confidence if item.confidence is not None else 1.0
            item.setVisible(conf >= threshold)
        for item in self._canvas._pred_polygon_items:
            conf = item.confidence if item.confidence is not None else 1.0
            item.setVisible(conf >= threshold)

    @property
    def confidence_threshold(self) -> float:
        return self._confidence_slider.value()

    def _on_save(self) -> None:
        if not self._canvas.image_path:
            return
        path = Path(self._canvas.image_path)
        label_dir = path.parent.parent / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)

        state = self._canvas.state
        w, h = self._canvas.image_size

        # Use tcip_annotation label_io for YOLO format writing
        detect_path = str(label_dir / (path.stem + ".txt"))
        segment_path = str(label_dir / (path.stem + "_seg.txt"))

        write_detect_labels(detect_path, state.boxes, w, h)
        write_segment_labels(segment_path, state.polygons, w, h)

        total = len(state.boxes) + len(state.polygons)
        self.annotation_saved.emit(str(path), total)

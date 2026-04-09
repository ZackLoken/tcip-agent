"""Annotation canvas — QGraphicsView-based image annotation with box/polygon drawing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QCursor,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QMenu,
    QWidget,
)

from tcip_annotation.state import AnnotationState, BBox, Polygon
from tcip_annotation.engine import AnnotationEngine

from .box_item import BoxItem, class_color
from .polygon_item import PolygonItem


class AnnotationCanvas(QGraphicsView):
    """QGraphicsView-based canvas for image annotation (boxes and polygons)."""

    annotation_changed = pyqtSignal()  # emitted when annotations are modified
    image_loaded = pyqtSignal(str)  # path
    prev_image = pyqtSignal()  # arrow key left
    next_image = pyqtSignal()  # arrow key right

    # Drawing modes
    MODE_SELECT = "select"
    MODE_BOX = "box"
    MODE_POLYGON = "polygon"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # Rendering
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

        # Image
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._image_path: str | None = None
        self._img_width: int = 0
        self._img_height: int = 0

        # Shared state + engine (from tcip_annotation)
        self._state = AnnotationState()
        self._engine = AnnotationEngine(self._state)

        # Drawing state
        self._mode: str = self.MODE_SELECT
        self._active_class: int = 0
        self._drawing = False
        self._draw_start: QPointF | None = None
        self._temp_rect: BoxItem | None = None

        # Polygon drawing state
        self._poly_points: list[QPointF] = []
        self._poly_lines: list[QGraphicsLineItem] = []

        # Visual items (scene representations of state data)
        self._box_items: list[BoxItem] = []
        self._polygon_items: list[PolygonItem] = []
        self._pred_box_items: list[BoxItem] = []
        self._pred_polygon_items: list[PolygonItem] = []

    # ── Properties ──

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, m: str) -> None:
        self._cancel_drawing()
        self._mode = m
        if m == self.MODE_SELECT:
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)

    @property
    def active_class(self) -> int:
        return self._active_class

    @active_class.setter
    def active_class(self, c: int) -> None:
        self._active_class = c
        self._state.active_class = c

    @property
    def state(self) -> AnnotationState:
        """Access the shared AnnotationState."""
        return self._state

    @property
    def engine(self) -> AnnotationEngine:
        """Access the shared AnnotationEngine."""
        return self._engine

    @property
    def image_path(self) -> str | None:
        return self._image_path

    # ── Image loading ──

    def load_image(self, path: str) -> bool:
        """Load an image file onto the canvas. Returns True on success."""
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return False

        self.clear_all()
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._pixmap_item)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self._image_path = path
        self._img_width = pixmap.width()
        self._img_height = pixmap.height()

        # Sync shared state
        self._state.image_path = path
        self._state.img_width = self._img_width
        self._state.img_height = self._img_height

        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.image_loaded.emit(path)
        return True

    # ── Annotation management ──

    def add_box(self, x: float, y: float, w: float, h: float, class_id: int = 0) -> BoxItem:
        """Add a ground-truth bounding box."""
        bbox = BBox(x, y, x + w, y + h, class_id)
        self._engine.add_box(bbox)
        item = BoxItem(x, y, w, h, class_id=class_id)
        self._scene.addItem(item)
        self._box_items.append(item)
        self.annotation_changed.emit()
        return item

    def add_polygon(self, points: list[tuple[float, float]], class_id: int = 0) -> PolygonItem:
        """Add a ground-truth polygon."""
        polygon = Polygon(points, class_id)
        self._engine.add_polygon(polygon)
        item = PolygonItem(points, class_id=class_id)
        self._scene.addItem(item)
        self._polygon_items.append(item)
        self.annotation_changed.emit()
        return item

    def add_pred_box(
        self, x: float, y: float, w: float, h: float,
        class_id: int = 0, confidence: float = 0.0, match_type: str | None = None,
    ) -> BoxItem:
        """Add a prediction bounding box overlay."""
        item = BoxItem(x, y, w, h, class_id=class_id, is_prediction=True, confidence=confidence, match_type=match_type)
        self._scene.addItem(item)
        self._pred_box_items.append(item)
        return item

    def add_pred_polygon(
        self, points: list[tuple[float, float]],
        class_id: int = 0, confidence: float = 0.0, match_type: str | None = None,
    ) -> PolygonItem:
        """Add a prediction polygon overlay."""
        item = PolygonItem(points, class_id=class_id, is_prediction=True, confidence=confidence, match_type=match_type)
        self._scene.addItem(item)
        self._pred_polygon_items.append(item)
        return item

    def clear_predictions(self) -> None:
        """Remove all prediction overlays."""
        for item in self._pred_box_items:
            self._scene.removeItem(item)
        for item in self._pred_polygon_items:
            self._scene.removeItem(item)
        self._pred_box_items.clear()
        self._pred_polygon_items.clear()

    def clear_annotations(self) -> None:
        """Remove all GT annotations."""
        self._engine.clear()
        for item in self._box_items:
            self._scene.removeItem(item)
        for item in self._polygon_items:
            self._scene.removeItem(item)
        self._box_items.clear()
        self._polygon_items.clear()
        self.annotation_changed.emit()

    def clear_all(self) -> None:
        """Clear everything including the image."""
        self._scene.clear()
        self._pixmap_item = None
        self._box_items.clear()
        self._polygon_items.clear()
        self._pred_box_items.clear()
        self._pred_polygon_items.clear()
        # Reset shared state
        self._state.boxes.clear()
        self._state.polygons.clear()
        self._state.pred_boxes.clear()
        self._state.pred_polygons.clear()
        self._state._undo_stack.clear()
        self._state._redo_stack.clear()

    def remove_selected(self) -> None:
        """Delete selected annotations."""
        selected = self._scene.selectedItems()
        if not selected:
            return
        self._engine.push_undo()
        for item in selected:
            if isinstance(item, BoxItem) and item in self._box_items:
                idx = self._box_items.index(item)
                self._box_items.remove(item)
                self._scene.removeItem(item)
                if 0 <= idx < len(self._state.boxes):
                    self._state.boxes.pop(idx)
            elif isinstance(item, PolygonItem) and item in self._polygon_items:
                idx = self._polygon_items.index(item)
                self._polygon_items.remove(item)
                self._scene.removeItem(item)
                if 0 <= idx < len(self._state.polygons):
                    self._state.polygons.pop(idx)
        self.annotation_changed.emit()

    def highlight_items(self, indices: list[int], item_type: str = "box") -> None:
        """Flash/highlight specific annotations by index."""
        items = self._box_items if item_type == "box" else self._polygon_items
        for i, item in enumerate(items):
            if i in indices:
                pen = item.pen()
                pen.setWidth(4)
                item.setPen(pen)

    # ── Undo / Redo (delegates to engine, then rebuilds scene items) ──

    def undo(self) -> None:
        if self._engine.undo():
            self._rebuild_items_from_state()
            self.annotation_changed.emit()

    def redo(self) -> None:
        if self._engine.redo():
            self._rebuild_items_from_state()
            self.annotation_changed.emit()

    def _rebuild_items_from_state(self) -> None:
        """Recreate scene items from the shared AnnotationState."""
        for item in self._box_items:
            self._scene.removeItem(item)
        for item in self._polygon_items:
            self._scene.removeItem(item)
        self._box_items.clear()
        self._polygon_items.clear()

        for b in self._state.boxes:
            item = BoxItem(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1, class_id=b.class_id)
            self._scene.addItem(item)
            self._box_items.append(item)
        for p in self._state.polygons:
            item = PolygonItem(p.points, class_id=p.class_id)
            self._scene.addItem(item)
            self._polygon_items.append(item)

    # ── Data export ──

    def get_boxes(self) -> list[dict]:
        """Get all GT boxes as dicts: {x1, y1, x2, y2, class_id}."""
        return [
            {"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2, "class_id": b.class_id}
            for b in self._state.boxes
        ]

    def get_polygons(self) -> list[dict]:
        """Get all GT polygons as dicts: {points, class_id}."""
        return [{"points": p.points, "class_id": p.class_id} for p in self._state.polygons]

    @property
    def image_size(self) -> tuple[int, int]:
        return (self._img_width, self._img_height)

    # ── Mouse events ──

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        scene_pos = self.mapToScene(event.pos())

        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            fake_event = QMouseEvent(
                event.type(), QPointF(event.pos()), Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton, event.modifiers(),
            )
            super().mousePressEvent(fake_event)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if self._mode == self.MODE_BOX:
                self._drawing = True
                self._draw_start = scene_pos
                return
            elif self._mode == self.MODE_POLYGON:
                self._poly_points.append(scene_pos)
                if len(self._poly_points) > 1:
                    line = QGraphicsLineItem(
                        self._poly_points[-2].x(), self._poly_points[-2].y(),
                        self._poly_points[-1].x(), self._poly_points[-1].y(),
                    )
                    line.setPen(QPen(class_color(self._active_class), 2, Qt.PenStyle.DashLine))
                    self._scene.addItem(line)
                    self._poly_lines.append(line)
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._drawing and self._draw_start is not None and self._mode == self.MODE_BOX:
            scene_pos = self.mapToScene(event.pos())
            x = min(self._draw_start.x(), scene_pos.x())
            y = min(self._draw_start.y(), scene_pos.y())
            w = abs(scene_pos.x() - self._draw_start.x())
            h = abs(scene_pos.y() - self._draw_start.y())

            if self._temp_rect:
                self._scene.removeItem(self._temp_rect)
            self._temp_rect = BoxItem(x, y, w, h, class_id=self._active_class)
            self._temp_rect.setPen(QPen(class_color(self._active_class), 2, Qt.PenStyle.DashLine))
            self._scene.addItem(self._temp_rect)
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.NoDrag if self._mode != self.MODE_SELECT else QGraphicsView.DragMode.RubberBandDrag)
            return

        if event.button() == Qt.MouseButton.LeftButton and self._drawing and self._draw_start is not None:
            scene_pos = self.mapToScene(event.pos())
            x = min(self._draw_start.x(), scene_pos.x())
            y = min(self._draw_start.y(), scene_pos.y())
            w = abs(scene_pos.x() - self._draw_start.x())
            h = abs(scene_pos.y() - self._draw_start.y())

            if self._temp_rect:
                self._scene.removeItem(self._temp_rect)
                self._temp_rect = None

            if w > 3 and h > 3:  # minimum size
                self.add_box(x, y, w, h, class_id=self._active_class)

            self._drawing = False
            self._draw_start = None
            return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._mode == self.MODE_POLYGON and len(self._poly_points) >= 3:
            points = [(p.x(), p.y()) for p in self._poly_points]
            self._clear_poly_preview()
            self.add_polygon(points, class_id=self._active_class)
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Escape:
            self._cancel_drawing()
            self._scene.clearSelection()
        elif event.key() == Qt.Key.Key_Delete:
            self.remove_selected()
        elif event.key() == Qt.Key.Key_Z and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.undo()
        elif event.key() == Qt.Key.Key_Y and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.redo()
        elif event.key() == Qt.Key.Key_C and not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self._change_selected_class()
        elif event.key() == Qt.Key.Key_Left:
            self.prev_image.emit()
        elif event.key() == Qt.Key.Key_Right:
            self.next_image.emit()
        else:
            super().keyPressEvent(event)

    def contextMenuEvent(self, event: Any) -> None:  # type: ignore[override]
        """Right-click context menu."""
        menu = QMenu(self)
        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(self.remove_selected)
        menu.addAction(delete_action)

        change_class_action = QAction("Change Class...", self)
        change_class_action.triggered.connect(self._change_selected_class)
        menu.addAction(change_class_action)

        menu.exec(QCursor.pos())

    def _change_selected_class(self) -> None:
        """Cycle class of selected items to the next class."""
        selected = self._scene.selectedItems()
        if not selected:
            return
        self._engine.push_undo()
        for item in selected:
            if isinstance(item, BoxItem):
                new_cls = (item.class_id + 1) % 20
                item.set_class_id(new_cls)
                idx = self._box_items.index(item) if item in self._box_items else -1
                if 0 <= idx < len(self._state.boxes):
                    self._state.boxes[idx].class_id = new_cls
            elif isinstance(item, PolygonItem):
                new_cls = (item.class_id + 1) % 20
                item.set_class_id(new_cls)
                idx = self._polygon_items.index(item) if item in self._polygon_items else -1
                if 0 <= idx < len(self._state.polygons):
                    self._state.polygons[idx].class_id = new_cls
        self.annotation_changed.emit()

    # ── Internal ──

    def _cancel_drawing(self) -> None:
        self._drawing = False
        self._draw_start = None
        if self._temp_rect:
            self._scene.removeItem(self._temp_rect)
            self._temp_rect = None
        self._clear_poly_preview()

    def _clear_poly_preview(self) -> None:
        for line in self._poly_lines:
            self._scene.removeItem(line)
        self._poly_lines.clear()
        self._poly_points.clear()

"""Draggable polygon annotation item for QGraphicsScene with vertex editing."""

from __future__ import annotations

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QBrush, QColor, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPolygonItem,
    QGraphicsSceneMouseEvent,
)

from .box_item import class_color

_VERTEX_RADIUS = 5.0


class PolygonItem(QGraphicsPolygonItem):
    """Interactive polygon annotation on the canvas with vertex editing."""

    def __init__(
        self,
        points: list[tuple[float, float]],
        class_id: int = 0,
        is_prediction: bool = False,
        confidence: float | None = None,
        match_type: str | None = None,
    ) -> None:
        poly = QPolygonF([QPointF(x, y) for x, y in points])
        super().__init__(poly)
        self.class_id = class_id
        self.is_prediction = is_prediction
        self.confidence = confidence
        self.match_type = match_type

        # Vertex editing state
        self._vertex_handles: list[QGraphicsEllipseItem] = []
        self._dragging_vertex: int | None = None
        self._handles_visible = False

        self._setup_appearance()

        if not is_prediction:
            self.setFlags(
                QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
                | QGraphicsItem.GraphicsItemFlag.ItemIsMovable
                | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
            )

    def _setup_appearance(self) -> None:
        if self.match_type == "tp":
            color = QColor("#4caf50")
        elif self.match_type == "fp":
            color = QColor("#f44336")
        elif self.match_type == "fn":
            color = QColor("#2196f3")
        else:
            color = class_color(self.class_id)

        pen = QPen(color, 2)
        if self.is_prediction:
            pen.setStyle(Qt.PenStyle.DashLine)
        self.setPen(pen)

        fill = QColor(color)
        fill.setAlpha(30)
        self.setBrush(QBrush(fill))

    def set_class_id(self, class_id: int) -> None:
        self.class_id = class_id
        self._setup_appearance()

    def get_points(self) -> list[tuple[float, float]]:
        poly = self.polygon()
        pos = self.pos()
        return [(poly[i].x() + pos.x(), poly[i].y() + pos.y()) for i in range(poly.count())]

    # ── Vertex handles ──

    def show_vertex_handles(self) -> None:
        """Show draggable vertex handles."""
        if self._handles_visible:
            return
        self._handles_visible = True
        poly = self.polygon()
        color = QColor("#ffffff")
        for i in range(poly.count()):
            pt = poly[i]
            handle = QGraphicsEllipseItem(
                pt.x() - _VERTEX_RADIUS, pt.y() - _VERTEX_RADIUS,
                _VERTEX_RADIUS * 2, _VERTEX_RADIUS * 2, self,
            )
            handle.setPen(QPen(Qt.GlobalColor.black, 1))
            handle.setBrush(QBrush(color))
            handle.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
            self._vertex_handles.append(handle)

    def hide_vertex_handles(self) -> None:
        """Remove vertex handles from the scene."""
        for h in self._vertex_handles:
            scene = self.scene()
            if scene is not None:
                scene.removeItem(h)
        self._vertex_handles.clear()
        self._handles_visible = False
        self._dragging_vertex = None

    def _find_vertex_near(self, pos: QPointF, threshold: float = 10.0) -> int | None:
        """Return the index of the vertex near pos, or None."""
        poly = self.polygon()
        for i in range(poly.count()):
            pt = poly[i]
            dx = pos.x() - pt.x()
            dy = pos.y() - pt.y()
            if dx * dx + dy * dy <= threshold * threshold:
                return i
        return None

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # type: ignore[override]
        if self._handles_visible and event.button() == Qt.MouseButton.LeftButton:
            idx = self._find_vertex_near(event.pos())
            if idx is not None:
                self._dragging_vertex = idx
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # type: ignore[override]
        if self._dragging_vertex is not None:
            poly = self.polygon()
            poly[self._dragging_vertex] = event.pos()
            self.setPolygon(poly)
            # Update handle position
            if self._dragging_vertex < len(self._vertex_handles):
                h = self._vertex_handles[self._dragging_vertex]
                h.setPos(0, 0)  # reset
                h.setRect(
                    event.pos().x() - _VERTEX_RADIUS, event.pos().y() - _VERTEX_RADIUS,
                    _VERTEX_RADIUS * 2, _VERTEX_RADIUS * 2,
                )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # type: ignore[override]
        if self._dragging_vertex is not None:
            self._dragging_vertex = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value: object) -> object:
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedChange:
            if value:
                self.show_vertex_handles()
            else:
                self.hide_vertex_handles()
        return super().itemChange(change, value)

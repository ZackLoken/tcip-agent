"""Draggable bounding box annotation item for QGraphicsScene."""

from __future__ import annotations

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QPen
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsSceneMouseEvent,
    QStyleOptionGraphicsItem,
    QWidget,
)


# Class colors (first 20 classes)
CLASS_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]


def class_color(class_id: int, alpha: int = 180) -> QColor:
    hex_color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
    c = QColor(hex_color)
    c.setAlpha(alpha)
    return c


class BoxItem(QGraphicsRectItem):
    """Interactive bounding box on the annotation canvas."""

    def __init__(
        self,
        x: float, y: float, w: float, h: float,
        class_id: int = 0,
        is_prediction: bool = False,
        confidence: float | None = None,
        match_type: str | None = None,
    ) -> None:
        super().__init__(x, y, w, h)
        self.class_id = class_id
        self.is_prediction = is_prediction
        self.confidence = confidence
        self.match_type = match_type  # "tp", "fp", "fn", None

        self._setup_appearance()

        if not is_prediction:
            self.setFlags(
                QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
                | QGraphicsItem.GraphicsItemFlag.ItemIsMovable
                | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
            )

    def _setup_appearance(self) -> None:
        if self.match_type == "tp":
            color = QColor("#4caf50")  # green
        elif self.match_type == "fp":
            color = QColor("#f44336")  # red
        elif self.match_type == "fn":
            color = QColor("#2196f3")  # blue
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

    def to_bbox_tuple(self) -> tuple[float, float, float, float]:
        r = self.rect()
        pos = self.pos()
        return (pos.x() + r.x(), pos.y() + r.y(), r.width(), r.height())

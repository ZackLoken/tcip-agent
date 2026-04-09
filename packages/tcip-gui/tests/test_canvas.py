"""Tests for the annotation canvas widget."""

import pytest
from PyQt6.QtCore import QPointF

from tcip_gui.widgets.canvas import AnnotationCanvas
from tcip_gui.widgets.box_item import BoxItem, class_color, CLASS_COLORS
from tcip_gui.widgets.polygon_item import PolygonItem
from tcip_annotation.state import BBox, Polygon


class TestAnnotationCanvas:
    def test_canvas_creation(self):
        canvas = AnnotationCanvas()
        assert canvas.mode == AnnotationCanvas.MODE_SELECT
        assert canvas.active_class == 0
        assert canvas.image_path is None

    def test_mode_switching(self):
        canvas = AnnotationCanvas()
        canvas.mode = AnnotationCanvas.MODE_BOX
        assert canvas.mode == "box"
        canvas.mode = AnnotationCanvas.MODE_POLYGON
        assert canvas.mode == "polygon"
        canvas.mode = AnnotationCanvas.MODE_SELECT
        assert canvas.mode == "select"

    def test_add_box(self):
        canvas = AnnotationCanvas()
        item = canvas.add_box(10, 20, 100, 50, class_id=1)
        assert isinstance(item, BoxItem)
        boxes = canvas.get_boxes()
        assert len(boxes) == 1
        assert boxes[0]["class_id"] == 1

    def test_add_polygon(self):
        canvas = AnnotationCanvas()
        points = [(10, 10), (100, 10), (100, 100), (10, 100)]
        item = canvas.add_polygon(points, class_id=2)
        assert isinstance(item, PolygonItem)
        polys = canvas.get_polygons()
        assert len(polys) == 1
        assert polys[0]["class_id"] == 2

    def test_add_pred_box(self):
        canvas = AnnotationCanvas()
        item = canvas.add_pred_box(10, 20, 100, 50, class_id=0, confidence=0.9, match_type="tp")
        assert item.is_prediction
        assert item.confidence == 0.9
        assert item.match_type == "tp"

    def test_clear_predictions(self):
        canvas = AnnotationCanvas()
        canvas.add_pred_box(10, 20, 100, 50)
        canvas.add_pred_box(20, 30, 80, 40)
        assert len(canvas._pred_box_items) == 2
        canvas.clear_predictions()
        assert len(canvas._pred_box_items) == 0

    def test_clear_annotations(self):
        canvas = AnnotationCanvas()
        canvas.add_box(10, 20, 100, 50)
        canvas.add_polygon([(0, 0), (10, 0), (10, 10)])
        assert len(canvas._box_items) == 1
        assert len(canvas._polygon_items) == 1
        canvas.clear_annotations()
        assert len(canvas._box_items) == 0
        assert len(canvas._polygon_items) == 0

    def test_undo_redo(self):
        canvas = AnnotationCanvas()
        canvas.add_box(10, 20, 100, 50)
        assert len(canvas.get_boxes()) == 1

        canvas.add_box(20, 30, 80, 40)
        assert len(canvas.get_boxes()) == 2

        canvas.undo()
        assert len(canvas.get_boxes()) == 1

        canvas.redo()
        assert len(canvas.get_boxes()) == 2

    def test_undo_empty_does_nothing(self):
        canvas = AnnotationCanvas()
        canvas.undo()  # Should not crash
        canvas.redo()  # Should not crash

    def test_clear_all(self):
        canvas = AnnotationCanvas()
        canvas.add_box(10, 20, 100, 50)
        canvas.add_pred_box(10, 20, 100, 50)
        canvas.clear_all()
        assert len(canvas._box_items) == 0
        assert len(canvas._pred_box_items) == 0

    def test_image_size_default(self):
        canvas = AnnotationCanvas()
        assert canvas.image_size == (0, 0)


class TestBoxItem:
    def test_creation(self):
        item = BoxItem(10, 20, 100, 50, class_id=1)
        assert item.class_id == 1
        assert not item.is_prediction

    def test_prediction_item(self):
        item = BoxItem(10, 20, 100, 50, class_id=0, is_prediction=True, confidence=0.8)
        assert item.is_prediction
        assert item.confidence == 0.8

    def test_match_type_colors(self):
        tp = BoxItem(0, 0, 10, 10, match_type="tp")
        fp = BoxItem(0, 0, 10, 10, match_type="fp")
        fn = BoxItem(0, 0, 10, 10, match_type="fn")
        # Each should have different colors (green, red, blue)
        assert tp.pen().color() != fp.pen().color()
        assert fp.pen().color() != fn.pen().color()

    def test_class_color_wraps(self):
        # Should not crash for class_id > len(CLASS_COLORS)
        c = class_color(100)
        assert c.alpha() == 180

    def test_to_bbox_tuple(self):
        item = BoxItem(10, 20, 100, 50)
        t = item.to_bbox_tuple()
        assert t == (10.0, 20.0, 100.0, 50.0)

    def test_set_class_id(self):
        item = BoxItem(0, 0, 10, 10, class_id=0)
        item.set_class_id(5)
        assert item.class_id == 5


class TestPolygonItem:
    def test_creation(self):
        points = [(0, 0), (10, 0), (10, 10), (0, 10)]
        item = PolygonItem(points, class_id=3)
        assert item.class_id == 3
        assert not item.is_prediction

    def test_get_points(self):
        points = [(0, 0), (10, 0), (10, 10), (0, 10)]
        item = PolygonItem(points)
        retrieved = item.get_points()
        assert len(retrieved) == 4
        # Points should match (allowing float conversion)
        for (x1, y1), (x2, y2) in zip(points, retrieved):
            assert abs(x1 - x2) < 0.01
            assert abs(y1 - y2) < 0.01


class TestAnnotationStateIntegration:
    """Tests that canvas correctly syncs with tcip_annotation.AnnotationState."""

    def test_add_box_syncs_state(self):
        canvas = AnnotationCanvas()
        canvas.add_box(10, 20, 100, 50, class_id=1)
        assert len(canvas.state.boxes) == 1
        b = canvas.state.boxes[0]
        assert b.x1 == 10 and b.y1 == 20
        assert b.x2 == 110 and b.y2 == 70
        assert b.class_id == 1

    def test_add_polygon_syncs_state(self):
        canvas = AnnotationCanvas()
        pts = [(0, 0), (10, 0), (10, 10)]
        canvas.add_polygon(pts, class_id=2)
        assert len(canvas.state.polygons) == 1
        assert canvas.state.polygons[0].class_id == 2
        assert canvas.state.polygons[0].points == pts

    def test_engine_undo_redo(self):
        canvas = AnnotationCanvas()
        canvas.add_box(10, 20, 100, 50)
        canvas.add_box(20, 30, 80, 40)
        assert len(canvas.state.boxes) == 2
        canvas.undo()
        assert len(canvas.state.boxes) == 1
        canvas.redo()
        assert len(canvas.state.boxes) == 2

    def test_clear_annotations_clears_state(self):
        canvas = AnnotationCanvas()
        canvas.add_box(10, 20, 100, 50)
        canvas.add_polygon([(0, 0), (5, 0), (5, 5)])
        canvas.clear_annotations()
        assert len(canvas.state.boxes) == 0
        assert len(canvas.state.polygons) == 0

    def test_clear_all_resets_state(self):
        canvas = AnnotationCanvas()
        canvas.add_box(10, 20, 100, 50)
        canvas.clear_all()
        assert len(canvas.state.boxes) == 0
        assert len(canvas.state._undo_stack) == 0


class TestPolygonVertexEditing:
    """Tests for polygon vertex editing feature."""

    def test_vertex_handles_initially_hidden(self):
        points = [(0, 0), (10, 0), (10, 10), (0, 10)]
        item = PolygonItem(points)
        assert item._handles_visible is False
        assert len(item._vertex_handles) == 0

    def test_show_vertex_handles(self):
        points = [(0, 0), (10, 0), (10, 10), (0, 10)]
        item = PolygonItem(points)
        # Need a scene to display handles
        from PyQt6.QtWidgets import QGraphicsScene
        scene = QGraphicsScene()
        scene.addItem(item)
        item.show_vertex_handles()
        assert item._handles_visible is True
        assert len(item._vertex_handles) == 4

    def test_hide_vertex_handles(self):
        points = [(0, 0), (10, 0), (10, 10), (0, 10)]
        item = PolygonItem(points)
        from PyQt6.QtWidgets import QGraphicsScene
        scene = QGraphicsScene()
        scene.addItem(item)
        item.show_vertex_handles()
        item.hide_vertex_handles()
        assert item._handles_visible is False
        assert len(item._vertex_handles) == 0

    def test_find_vertex_near(self):
        points = [(0, 0), (100, 0), (100, 100), (0, 100)]
        item = PolygonItem(points)
        assert item._find_vertex_near(QPointF(1, 1), threshold=10) == 0
        assert item._find_vertex_near(QPointF(99, 1), threshold=10) == 1
        assert item._find_vertex_near(QPointF(50, 50), threshold=10) is None

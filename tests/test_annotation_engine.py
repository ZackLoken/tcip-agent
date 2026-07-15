"""Unit tests for tcip_annotation.annotation_engine.AnnotationEngine."""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_annotation import AnnotationEngine, AnnotationState, BBox, Polygon


@pytest.fixture
def state() -> AnnotationState:
    return AnnotationState(image_path="/tmp/img.jpg", img_width=1000, img_height=800)


def _box(cid: int = 0, offset: float = 0.0) -> BBox:
    return BBox(
        x1=10.0 + offset, y1=20.0 + offset, x2=110.0 + offset, y2=120.0 + offset, class_id=cid
    )


def test_add_box(state: AnnotationState) -> None:
    eng = AnnotationEngine(state, current_user="alice")
    idx = eng.add_box(_box())
    assert idx == 0
    assert len(state.boxes) == 1
    assert eng.box_authors == ["alice"]


def test_update_box(state: AnnotationState) -> None:
    eng = AnnotationEngine(state)
    eng.add_box(_box())
    eng.update_box(0, _box(offset=5.0))
    assert state.boxes[0].x1 == 15.0


def test_delete_box(state: AnnotationState) -> None:
    eng = AnnotationEngine(state, current_user="alice")
    eng.add_box(_box())
    eng.add_box(_box(offset=50.0))
    eng.delete_box(0)
    assert len(state.boxes) == 1
    assert state.boxes[0].x1 == 60.0
    assert eng.box_authors == ["alice"]


def test_box_index_out_of_range_raises(state: AnnotationState) -> None:
    eng = AnnotationEngine(state)
    with pytest.raises(IndexError):
        eng.update_box(0, _box())
    with pytest.raises(IndexError):
        eng.delete_box(5)


def test_undo_redo_box(state: AnnotationState) -> None:
    eng = AnnotationEngine(state)
    eng.add_box(_box())
    eng.add_box(_box(offset=50.0))
    assert len(state.boxes) == 2
    assert eng.undo_snapshot() is True
    assert len(state.boxes) == 1  # undoes second add
    assert eng.redo_snapshot() is True
    assert len(state.boxes) == 2


def test_undo_stack_empty_returns_false(state: AnnotationState) -> None:
    eng = AnnotationEngine(state)
    assert eng.undo_snapshot() is False
    assert eng.redo_snapshot() is False


def test_undo_stack_bounded(state: AnnotationState) -> None:
    eng = AnnotationEngine(state)
    for _ in range(50):
        eng.add_box(_box())
    assert len(state._undo_stack) == 30


def test_close_polygon_happy_path(state: AnnotationState) -> None:
    state.current_polygon = [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)]
    state.active_class = 3
    eng = AnnotationEngine(state, current_user="alice")
    assert eng.close_current_polygon() is True
    assert len(state.polygons) == 1
    assert state.polygons[0].class_id == 3
    assert state.current_polygon == []
    assert eng.polygon_authors == ["alice"]


def test_close_polygon_rejects_less_than_3_vertices(state: AnnotationState) -> None:
    state.current_polygon = [(10.0, 10.0), (20.0, 20.0)]
    eng = AnnotationEngine(state)
    assert eng.close_current_polygon() is False
    assert state.polygons == []
    assert state.current_polygon == []


def test_close_polygon_clamps_to_image_bounds(state: AnnotationState) -> None:
    state.img_width = 100
    state.img_height = 100
    state.current_polygon = [(-10.0, -10.0), (200.0, 50.0), (50.0, 200.0)]
    eng = AnnotationEngine(state)
    assert eng.close_current_polygon() is True
    pts = state.polygons[0].points
    assert pts == [(0.0, 0.0), (100.0, 50.0), (50.0, 100.0)]


def test_delete_polygon_updates_selection(state: AnnotationState) -> None:
    eng = AnnotationEngine(state)
    eng.add_polygon(Polygon([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], class_id=0))
    eng.add_polygon(Polygon([(10.0, 10.0), (11.0, 10.0), (10.0, 11.0)], class_id=0))
    state.selected_polygon_idx = 1
    eng.delete_polygon(0)
    assert state.selected_polygon_idx == 0


def test_delete_polygon_clears_selection_when_same(state: AnnotationState) -> None:
    eng = AnnotationEngine(state)
    eng.add_polygon(Polygon([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], class_id=0))
    state.selected_polygon_idx = 0
    eng.delete_polygon(0)
    assert state.selected_polygon_idx is None


def test_save_writes_json_files(state: AnnotationState, tmp_path: Path) -> None:
    from tcip_annotation.json_io import read_detect, read_segment

    state.img_width = 1000
    state.img_height = 500
    state.boxes = [BBox(x1=100.0, y1=50.0, x2=200.0, y2=150.0, class_id=0)]
    state.polygons = [Polygon(points=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)], class_id=1)]

    eng = AnnotationEngine(state)
    det_path = tmp_path / "detect" / "img.json"
    seg_path = tmp_path / "segment" / "img.json"
    assert eng.save(detect_path=str(det_path), segment_path=str(seg_path)) is True

    # Detect: one box, pixel xyxy preserved, class in category_id.
    det_boxes, _ = read_detect(str(det_path))
    assert len(det_boxes) == 1
    b = det_boxes[0]
    assert b.class_id == 0
    assert (b.x1, b.y1, b.x2, b.y2) == (100.0, 50.0, 200.0, 150.0)

    # Segment: one polygon, class 1.
    seg_polys, _ = read_segment(str(seg_path))
    assert len(seg_polys) == 1
    assert seg_polys[0].class_id == 1


def test_save_rejects_invalid_image_dimensions(state: AnnotationState, tmp_path: Path) -> None:
    state.img_width = 0
    state.img_height = 0
    eng = AnnotationEngine(state)
    assert eng.save(detect_path=str(tmp_path / "out.json")) is False


def test_ensure_poly_bboxes_caches_bounds(state: AnnotationState) -> None:
    eng = AnnotationEngine(state)
    eng.add_polygon(Polygon([(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)], class_id=0))
    eng.ensure_poly_bboxes()
    assert state._poly_bboxes == [(0.0, 0.0, 100.0, 100.0)]
    assert state._poly_bboxes_dirty is False
    # modification should invalidate the cache
    eng.add_polygon(Polygon([(200.0, 200.0), (300.0, 200.0), (300.0, 300.0)], class_id=0))
    assert state._poly_bboxes_dirty is True

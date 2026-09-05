"""Unit tests for tcip_annotation.annotation_engine.AnnotationEngine."""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_annotation import Annotation, AnnotationEngine, AnnotationState, BBox, Polygon


@pytest.fixture
def state() -> AnnotationState:
    return AnnotationState(image_path="/tmp/img.jpg", img_width=1000, img_height=800)


def _box(offset: float = 0.0) -> BBox:
    return BBox(x1=10.0 + offset, y1=20.0 + offset, x2=110.0 + offset, y2=120.0 + offset)


def test_add_box(state: AnnotationState) -> None:
    eng = AnnotationEngine(state, current_user="alice")
    idx = eng.add_box(_box(), subject="bud")
    assert idx == 0
    assert len(state.annotations) == 1
    assert eng.authors == ["alice"]


def test_update_box(state: AnnotationState) -> None:
    eng = AnnotationEngine(state)
    eng.add_box(_box(), subject="bud")
    eng.update_annotation(0, Annotation(subject="bud", geometry=_box(offset=5.0)))
    assert state.annotations[0].geometry.x1 == 15.0


def test_delete_box(state: AnnotationState) -> None:
    eng = AnnotationEngine(state, current_user="alice")
    eng.add_box(_box(), subject="bud")
    eng.add_box(_box(offset=50.0), subject="bud")
    eng.delete_annotation(0)
    assert len(state.annotations) == 1
    assert state.annotations[0].geometry.x1 == 60.0
    assert eng.authors == ["alice"]


def test_box_index_out_of_range_raises(state: AnnotationState) -> None:
    eng = AnnotationEngine(state)
    with pytest.raises(IndexError):
        eng.update_annotation(0, Annotation(subject="bud", geometry=_box()))
    with pytest.raises(IndexError):
        eng.delete_annotation(5)


def test_undo_redo_box(state: AnnotationState) -> None:
    eng = AnnotationEngine(state)
    eng.add_box(_box(), subject="bud")
    eng.add_box(_box(offset=50.0), subject="bud")
    assert len(state.annotations) == 2
    assert eng.undo_snapshot() is True
    assert len(state.annotations) == 1  # undoes second add
    assert eng.redo_snapshot() is True
    assert len(state.annotations) == 2


def test_undo_stack_empty_returns_false(state: AnnotationState) -> None:
    eng = AnnotationEngine(state)
    assert eng.undo_snapshot() is False
    assert eng.redo_snapshot() is False


def test_undo_stack_bounded(state: AnnotationState) -> None:
    eng = AnnotationEngine(state)
    for _ in range(50):
        eng.add_box(_box(), subject="bud")
    assert len(state._undo_stack) == 30


def test_close_polygon_happy_path(state: AnnotationState) -> None:
    state.current_polygon = [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)]
    state.active_subject = "bud"
    eng = AnnotationEngine(state, current_user="alice")
    assert eng.close_current_polygon() is True
    assert len(state.annotations) == 1
    assert state.annotations[0].subject == "bud"  # authored under the active subject
    assert isinstance(state.annotations[0].geometry, Polygon)
    assert state.current_polygon == []
    assert eng.authors == ["alice"]


def test_close_polygon_rejects_less_than_3_vertices(state: AnnotationState) -> None:
    state.current_polygon = [(10.0, 10.0), (20.0, 20.0)]
    eng = AnnotationEngine(state)
    assert eng.close_current_polygon() is False
    assert state.annotations == []
    assert state.current_polygon == []


def test_close_polygon_clamps_to_image_bounds(state: AnnotationState) -> None:
    state.img_width = 100
    state.img_height = 100
    state.current_polygon = [(-10.0, -10.0), (200.0, 50.0), (50.0, 200.0)]
    eng = AnnotationEngine(state)
    assert eng.close_current_polygon() is True
    assert state.annotations[0].geometry.rings == [[(0.0, 0.0), (100.0, 50.0), (50.0, 100.0)]]


def test_delete_polygon_updates_selection(state: AnnotationState) -> None:
    eng = AnnotationEngine(state)
    eng.add_polygon(Polygon([[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]]), subject="bud")
    eng.add_polygon(Polygon([[(10.0, 10.0), (11.0, 10.0), (10.0, 11.0)]]), subject="bud")
    state.selected_polygon_idx = 1
    eng.delete_annotation(0)
    assert state.selected_polygon_idx == 0


def test_delete_polygon_clears_selection_when_same(state: AnnotationState) -> None:
    eng = AnnotationEngine(state)
    eng.add_polygon(Polygon([[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]]), subject="bud")
    state.selected_polygon_idx = 0
    eng.delete_annotation(0)
    assert state.selected_polygon_idx is None


def test_save_writes_json_files(state: AnnotationState, tmp_path: Path) -> None:
    from tcip_annotation.json_io import read_annotations

    state.img_width = 1000
    state.img_height = 500
    state.annotations = [
        Annotation(subject="bud", geometry=BBox(x1=100.0, y1=50.0, x2=200.0, y2=150.0)),
        Annotation(subject="leaf", geometry=Polygon(rings=[[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]])),
    ]

    eng = AnnotationEngine(state)
    path = tmp_path / "labels" / "img.json"
    assert eng.save(path=str(path)) is True

    # One merged per-image file holds every subject; box (pixel xyxy) and polygon both survive.
    read_back = read_annotations(str(path))
    assert len(read_back) == 2
    box_ann = next(a for a in read_back if isinstance(a.geometry, BBox))
    assert box_ann.subject == "bud"
    assert (box_ann.geometry.x1, box_ann.geometry.y1, box_ann.geometry.x2, box_ann.geometry.y2) == (
        100.0, 50.0, 200.0, 150.0)
    poly_ann = next(a for a in read_back if isinstance(a.geometry, Polygon))
    assert poly_ann.subject == "leaf"


def test_save_rejects_invalid_image_dimensions(state: AnnotationState, tmp_path: Path) -> None:
    state.img_width = 0
    state.img_height = 0
    eng = AnnotationEngine(state)
    assert eng.save(path=str(tmp_path / "out.json")) is False


def test_ensure_poly_bboxes_caches_bounds(state: AnnotationState) -> None:
    eng = AnnotationEngine(state)
    eng.add_polygon(Polygon([[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]]), subject="bud")
    eng.ensure_poly_bboxes()
    assert state._poly_bboxes == [(0.0, 0.0, 100.0, 100.0)]
    assert state._poly_bboxes_dirty is False
    # modification should invalidate the cache
    eng.add_polygon(Polygon([[(200.0, 200.0), (300.0, 200.0), (300.0, 300.0)]]), subject="bud")
    assert state._poly_bboxes_dirty is True


def test_ensure_poly_bboxes_spans_every_ring(state: AnnotationState) -> None:
    # The spatial-index box of an occlusion-split instance covers all of its lobes; a box over only
    # the first ring would make the rest of the object unhittable.
    eng = AnnotationEngine(state)
    eng.add_polygon(
        Polygon([[(0.0, 0.0), (20.0, 0.0), (20.0, 40.0), (0.0, 40.0)],
                 [(70.0, 10.0), (90.0, 10.0), (90.0, 60.0), (70.0, 60.0)]]),
        subject="bud")
    eng.ensure_poly_bboxes()
    assert state._poly_bboxes == [(0.0, 0.0, 90.0, 60.0)]

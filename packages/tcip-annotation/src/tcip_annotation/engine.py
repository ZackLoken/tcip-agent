"""AnnotationEngine — Annotation CRUD, spatial index, undo/redo.

GUI-free.  Operates on an AnnotationState instance.
"""

from __future__ import annotations

import os

from tcip_annotation.state import AnnotationState, BBox, Polygon
from tcip_annotation.label_io import write_detect_labels, write_segment_labels


_MAX_UNDO = 30


class AnnotationEngine:
    """Annotation logic that operates on AnnotationState without any GUI dependency."""

    def __init__(self, state: AnnotationState):
        self.state = state

    # ── Spatial index (polygon bounding-box cache) ────────────────────────

    def invalidate_poly_bboxes(self) -> None:
        self.state._poly_bboxes_dirty = True

    def ensure_poly_bboxes(self) -> None:
        s = self.state
        if not s._poly_bboxes_dirty:
            return
        s._poly_bboxes = []
        for poly in s.polygons:
            if poly.points:
                xs = [p[0] for p in poly.points]
                ys = [p[1] for p in poly.points]
                s._poly_bboxes.append((min(xs), min(ys), max(xs), max(ys)))
            else:
                s._poly_bboxes.append((0.0, 0.0, 0.0, 0.0))
        s._poly_bboxes_dirty = False

    # ── Undo / redo ───────────────────────────────────────────────────────

    def push_undo(self) -> None:
        """Snapshot current annotation state before a mutation."""
        s = self.state
        snapshot = (list(s.boxes), list(s.polygons), s.selected_polygon_idx)
        s._undo_stack.append(snapshot)
        s._redo_stack.clear()
        if len(s._undo_stack) > _MAX_UNDO:
            s._undo_stack.pop(0)

    def undo(self) -> bool:
        """Restore previous state.  Returns True if restored."""
        s = self.state
        if not s._undo_stack:
            return False
        redo_snap = (list(s.boxes), list(s.polygons), s.selected_polygon_idx)
        s._redo_stack.append(redo_snap)
        snap = s._undo_stack.pop()
        s.boxes, s.polygons, s.selected_polygon_idx = snap
        self.invalidate_poly_bboxes()
        return True

    def redo(self) -> bool:
        """Restore state from redo stack.  Returns True if restored."""
        s = self.state
        if not s._redo_stack:
            return False
        undo_snap = (list(s.boxes), list(s.polygons), s.selected_polygon_idx)
        s._undo_stack.append(undo_snap)
        snap = s._redo_stack.pop()
        s.boxes, s.polygons, s.selected_polygon_idx = snap
        self.invalidate_poly_bboxes()
        return True

    # ── CRUD ──────────────────────────────────────────────────────────────

    def add_box(self, box: BBox) -> None:
        self.push_undo()
        self.state.boxes.append(box)

    def remove_box(self, idx: int) -> BBox | None:
        s = self.state
        if 0 <= idx < len(s.boxes):
            self.push_undo()
            return s.boxes.pop(idx)
        return None

    def add_polygon(self, polygon: Polygon) -> None:
        self.push_undo()
        self.state.polygons.append(polygon)
        self.invalidate_poly_bboxes()

    def remove_polygon(self, idx: int) -> Polygon | None:
        s = self.state
        if 0 <= idx < len(s.polygons):
            self.push_undo()
            removed = s.polygons.pop(idx)
            if s.selected_polygon_idx == idx:
                s.selected_polygon_idx = None
            elif s.selected_polygon_idx is not None and s.selected_polygon_idx > idx:
                s.selected_polygon_idx -= 1
            self.invalidate_poly_bboxes()
            return removed
        return None

    def close_current_polygon(self) -> bool:
        """Finalise the in-progress polygon.  Returns True if added."""
        s = self.state
        if len(s.current_polygon) < 3:
            s.current_polygon = []
            return False
        clamped = [
            (max(0.0, min(float(s.img_width), x)), max(0.0, min(float(s.img_height), y)))
            for x, y in s.current_polygon
        ]
        self.add_polygon(Polygon(clamped, s.active_class))
        s.current_polygon = []
        return True

    def clear(self) -> None:
        """Remove all annotations for the current image."""
        self.push_undo()
        self.state.boxes.clear()
        self.state.polygons.clear()
        self.state.selected_polygon_idx = None
        self.invalidate_poly_bboxes()

    # ── I/O ───────────────────────────────────────────────────────────────

    def save(self, detect_path: str, segment_path: str) -> bool:
        """Save current annotations to YOLO-format label files."""
        s = self.state
        if s.img_width <= 0 or s.img_height <= 0:
            return False
        try:
            write_detect_labels(detect_path, s.boxes, s.img_width, s.img_height)
            write_segment_labels(segment_path, s.polygons, s.img_width, s.img_height)
            return True
        except OSError:
            return False

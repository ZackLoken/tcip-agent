"""AnnotationEngine — Annotation CRUD, spatial index, undo/redo.

GUI-free.  Operates on a :class:`tcip_annotation.state.AnnotationState` instance.
Can be instantiated headlessly for programmatic use (AI agents, training
pipelines, CLI, web backend).

Ported from yolo-annotator (yololabeler.annotation.engine) with adaptations:

  * Inputs use :mod:`tcip_annotation.state` dataclasses (``BBox``,
    ``Polygon``) instead of bare tuples.
  * File I/O uses explicit paths passed by the caller rather than reading
    ``labels_dir`` / ``detect_dir`` / ``segment_dir`` off the state.
  * No ``print`` warnings — failures raise or return a typed status.
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from typing import Optional

from tcip_annotation.json_io import write_detect as write_detect_labels
from tcip_annotation.json_io import write_segment as write_segment_labels
from tcip_annotation.state import AnnotationState, BBox, Polygon

logger = logging.getLogger(__name__)


# Snapshot tuple structure — kept compact for memory efficiency with long
# undo chains on dense annotation sets.
Snapshot = tuple[list[BBox], list[Polygon], Optional[int], list[str], list[str]]
UNDO_DEPTH = 30


class AnnotationEngine:
    """Annotation logic that operates on AnnotationState without any GUI dependency.

    Parameters
    ----------
    state : AnnotationState
        The data model this engine mutates. Mutations are in-place so the
        caller can subscribe to state changes by wrapping the engine or
        polling.
    box_authors : list[str] | None, optional
        Per-box username for multi-user audit. Parallel to ``state.boxes``.
        Will be kept in sync on every mutation. Created as a list of empty
        strings if not provided.
    polygon_authors : list[str] | None, optional
        Per-polygon username. Parallel to ``state.polygons``.
    current_user : str, optional
        Username assigned to newly-created annotations.
    """

    def __init__(
        self,
        state: AnnotationState,
        *,
        box_authors: Optional[list[str]] = None,
        polygon_authors: Optional[list[str]] = None,
        current_user: str = "",
    ) -> None:
        self.state = state
        self.box_authors: list[str] = (
            list(box_authors) if box_authors is not None else [""] * len(state.boxes)
        )
        self.polygon_authors: list[str] = (
            list(polygon_authors) if polygon_authors is not None else [""] * len(state.polygons)
        )
        self.current_user = current_user

    # ── Spatial index (polygon bounding-box cache) ────────────────────────

    def invalidate_poly_bboxes(self) -> None:
        self.state._poly_bboxes_dirty = True

    def ensure_poly_bboxes(self) -> None:
        s = self.state
        if not s._poly_bboxes_dirty:
            return
        s._poly_bboxes = []
        for poly in s.polygons:
            pts = poly.points
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                s._poly_bboxes.append((min(xs), min(ys), max(xs), max(ys)))
            else:
                s._poly_bboxes.append((0.0, 0.0, 0.0, 0.0))
        s._poly_bboxes_dirty = False

    # ── Undo / redo ───────────────────────────────────────────────────────

    def _snapshot(self) -> Snapshot:
        s = self.state
        return (
            [replace(b) for b in s.boxes],
            [replace(p, points=list(p.points)) for p in s.polygons],
            s.selected_polygon_idx,
            list(self.box_authors),
            list(self.polygon_authors),
        )

    def push_undo(self) -> None:
        """Snapshot current annotation state before a mutation."""
        s = self.state
        s._undo_stack.append(self._snapshot())
        s._redo_stack.clear()
        if len(s._undo_stack) > UNDO_DEPTH:
            s._undo_stack.pop(0)

    def undo_snapshot(self) -> bool:
        """Restore previous state from the undo stack.

        Returns ``True`` if a snapshot was restored, ``False`` if the stack
        was empty.
        """
        s = self.state
        if not s._undo_stack:
            return False
        s._redo_stack.append(self._snapshot())
        snap = s._undo_stack.pop()
        self._apply_snapshot(snap)
        return True

    def redo_snapshot(self) -> bool:
        """Restore the most recently undone snapshot.

        Returns ``True`` if a snapshot was restored, ``False`` if the redo
        stack was empty.
        """
        s = self.state
        if not s._redo_stack:
            return False
        s._undo_stack.append(self._snapshot())
        snap = s._redo_stack.pop()
        self._apply_snapshot(snap)
        return True

    def _apply_snapshot(self, snap: Snapshot) -> None:
        s = self.state
        boxes, polygons, sel, box_authors, poly_authors = snap
        s.boxes = boxes
        s.polygons = polygons
        s.selected_polygon_idx = sel
        self.box_authors = box_authors if box_authors else [""] * len(s.boxes)
        self.polygon_authors = poly_authors if poly_authors else [""] * len(s.polygons)
        self.invalidate_poly_bboxes()

    # ── Box CRUD ──────────────────────────────────────────────────────────

    def add_box(self, box: BBox, *, push_undo: bool = True) -> int:
        """Append a new box. Returns its index."""
        if push_undo:
            self.push_undo()
        self.state.boxes.append(box)
        self.box_authors.append(self.current_user)
        return len(self.state.boxes) - 1

    def update_box(self, idx: int, new_box: BBox, *, push_undo: bool = True) -> None:
        """Replace the box at ``idx``."""
        if not 0 <= idx < len(self.state.boxes):
            raise IndexError(f"box index out of range: {idx}")
        if push_undo:
            self.push_undo()
        self.state.boxes[idx] = new_box

    def delete_box(self, idx: int, *, push_undo: bool = True) -> None:
        """Remove the box at ``idx``."""
        if not 0 <= idx < len(self.state.boxes):
            raise IndexError(f"box index out of range: {idx}")
        if push_undo:
            self.push_undo()
        self.state.boxes.pop(idx)
        if idx < len(self.box_authors):
            self.box_authors.pop(idx)

    # ── Polygon CRUD ──────────────────────────────────────────────────────

    def close_current_polygon(self) -> bool:
        """Finalize the in-progress polygon.

        Clamps vertices to image bounds, pushes undo, appends polygon.
        Returns ``True`` if a polygon was added, ``False`` if the in-progress
        polygon had fewer than 3 vertices.
        """
        s = self.state
        if len(s.current_polygon) < 3:
            s.current_polygon = []
            return False
        w = max(s.img_width, 0)
        h = max(s.img_height, 0)
        clamped: list[tuple[float, float]] = []
        for x, y in s.current_polygon:
            cx = x if w == 0 else max(0.0, min(float(w), x))
            cy = y if h == 0 else max(0.0, min(float(h), y))
            clamped.append((cx, cy))
        self.push_undo()
        s.polygons.append(Polygon(points=clamped, class_id=s.active_class))
        self.polygon_authors.append(self.current_user)
        self.invalidate_poly_bboxes()
        s.current_polygon = []
        return True

    def add_polygon(self, polygon: Polygon, *, push_undo: bool = True) -> int:
        """Append a pre-built polygon. Returns its index."""
        if push_undo:
            self.push_undo()
        self.state.polygons.append(polygon)
        self.polygon_authors.append(self.current_user)
        self.invalidate_poly_bboxes()
        return len(self.state.polygons) - 1

    def update_polygon(self, idx: int, new_polygon: Polygon, *, push_undo: bool = True) -> None:
        if not 0 <= idx < len(self.state.polygons):
            raise IndexError(f"polygon index out of range: {idx}")
        if push_undo:
            self.push_undo()
        self.state.polygons[idx] = new_polygon
        self.invalidate_poly_bboxes()

    def delete_polygon(self, idx: int, *, push_undo: bool = True) -> None:
        if not 0 <= idx < len(self.state.polygons):
            raise IndexError(f"polygon index out of range: {idx}")
        if push_undo:
            self.push_undo()
        self.state.polygons.pop(idx)
        if idx < len(self.polygon_authors):
            self.polygon_authors.pop(idx)
        if self.state.selected_polygon_idx == idx:
            self.state.selected_polygon_idx = None
        elif (
            self.state.selected_polygon_idx is not None
            and self.state.selected_polygon_idx > idx
        ):
            self.state.selected_polygon_idx -= 1
        self.invalidate_poly_bboxes()

    # ── I/O ───────────────────────────────────────────────────────────────

    def save(self, *, detect_path: Optional[str] = None, segment_path: Optional[str] = None) -> bool:
        """Save the current boxes / polygons to YOLO label files.

        Either or both paths may be omitted; omitted tasks are skipped. The
        engine does not infer paths from the state — the caller (web backend
        or CLI) is responsible for resolving them.

        Returns ``True`` if every requested write succeeded. Invalid image
        dimensions skip the write and return ``False``.
        """
        s = self.state
        if s.img_width <= 0 or s.img_height <= 0:
            logger.warning(
                "Invalid image dimensions (%sx%s); skipping save",
                s.img_width,
                s.img_height,
            )
            return False

        ok = True
        if detect_path is not None:
            try:
                os.makedirs(os.path.dirname(detect_path) or ".", exist_ok=True)
                write_detect_labels(detect_path, s.boxes, s.img_width, s.img_height)
            except OSError:
                logger.exception("Could not write detect labels to %s", detect_path)
                ok = False
        if segment_path is not None:
            try:
                os.makedirs(os.path.dirname(segment_path) or ".", exist_ok=True)
                write_segment_labels(segment_path, s.polygons, s.img_width, s.img_height)
            except OSError:
                logger.exception("Could not write segment labels to %s", segment_path)
                ok = False
        return ok

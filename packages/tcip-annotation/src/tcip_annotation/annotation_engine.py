"""AnnotationEngine — Annotation CRUD, spatial index, undo/redo.

GUI-free. Operates on a :class:`tcip_annotation.state.AnnotationState` instance whose
``annotations`` is one flat list of :class:`~tcip_annotation.state.Annotation` records (a box or a
polygon geometry, plus the subject and attribute values it carries). Can be instantiated headlessly
for programmatic use (AI agents, training pipelines, CLI, web backend).

  * Mutations are in-place so the caller can subscribe by wrapping the engine or polling.
  * File I/O uses explicit paths passed by the caller — one merged per-image JSON, all subjects.
  * No ``print`` warnings — failures raise or return a typed status.
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from typing import Optional

from tcip_annotation.json_io import write_annotations
from tcip_annotation.state import Annotation, AnnotationState, BBox, Polygon

logger = logging.getLogger(__name__)


# Snapshot tuple structure — kept compact for memory efficiency with long
# undo chains on dense annotation sets.
Snapshot = tuple[list[Annotation], Optional[int], list[str]]
UNDO_DEPTH = 30


class AnnotationEngine:
    """Annotation logic that operates on AnnotationState without any GUI dependency.

    Parameters
    ----------
    state : AnnotationState
        The data model this engine mutates. Mutations are in-place so the
        caller can subscribe to state changes by wrapping the engine or
        polling.
    authors : list[str] | None, optional
        Per-annotation username for multi-user audit. Parallel to
        ``state.annotations``. Kept in sync on every mutation; created as a
        list of empty strings if not provided.
    current_user : str, optional
        Username assigned to newly-created annotations.
    """

    def __init__(
        self,
        state: AnnotationState,
        *,
        authors: Optional[list[str]] = None,
        current_user: str = "",
    ) -> None:
        self.state = state
        self.authors: list[str] = (
            list(authors) if authors is not None else [""] * len(state.annotations)
        )
        self.current_user = current_user

    # ── Spatial index (per-annotation bounding-box cache) ─────────────────

    def invalidate_poly_bboxes(self) -> None:
        self.state._poly_bboxes_dirty = True

    def ensure_poly_bboxes(self) -> None:
        s = self.state
        if not s._poly_bboxes_dirty:
            return
        s._poly_bboxes = []
        for ann in s.annotations:
            geom = ann.geometry
            if isinstance(geom, BBox):
                s._poly_bboxes.append((geom.x1, geom.y1, geom.x2, geom.y2))
            elif isinstance(geom, Polygon) and geom.points:
                xs = [p[0] for p in geom.points]
                ys = [p[1] for p in geom.points]
                s._poly_bboxes.append((min(xs), min(ys), max(xs), max(ys)))
            else:
                s._poly_bboxes.append((0.0, 0.0, 0.0, 0.0))
        s._poly_bboxes_dirty = False

    # ── Undo / redo ───────────────────────────────────────────────────────

    def _snapshot(self) -> Snapshot:
        s = self.state
        return (
            [replace(a, attributes=dict(a.attributes)) for a in s.annotations],
            s.selected_polygon_idx,
            list(self.authors),
        )

    def push_undo(self) -> None:
        """Snapshot current annotation state before a mutation."""
        s = self.state
        s._undo_stack.append(self._snapshot())
        s._redo_stack.clear()
        if len(s._undo_stack) > UNDO_DEPTH:
            s._undo_stack.pop(0)

    def undo_snapshot(self) -> bool:
        """Restore previous state from the undo stack. ``True`` if one was restored."""
        s = self.state
        if not s._undo_stack:
            return False
        s._redo_stack.append(self._snapshot())
        self._apply_snapshot(s._undo_stack.pop())
        return True

    def redo_snapshot(self) -> bool:
        """Restore the most recently undone snapshot. ``True`` if one was restored."""
        s = self.state
        if not s._redo_stack:
            return False
        s._undo_stack.append(self._snapshot())
        self._apply_snapshot(s._redo_stack.pop())
        return True

    def _apply_snapshot(self, snap: Snapshot) -> None:
        s = self.state
        annotations, sel, authors = snap
        s.annotations = annotations
        s.selected_polygon_idx = sel
        self.authors = authors if authors else [""] * len(s.annotations)
        self.invalidate_poly_bboxes()

    # ── Annotation CRUD ───────────────────────────────────────────────────

    def add_annotation(self, annotation: Annotation, *, push_undo: bool = True) -> int:
        """Append a pre-built annotation. Returns its index."""
        if push_undo:
            self.push_undo()
        self.state.annotations.append(annotation)
        self.authors.append(self.current_user)
        self.invalidate_poly_bboxes()
        return len(self.state.annotations) - 1

    def add_box(self, box: BBox, *, subject: str | None = None, push_undo: bool = True) -> int:
        """Append a box annotation under ``subject`` (defaults to the state's active subject)."""
        subj = subject if subject is not None else self.state.active_subject
        return self.add_annotation(Annotation(subject=subj, geometry=box), push_undo=push_undo)

    def add_polygon(self, polygon: Polygon, *, subject: str | None = None,
                    push_undo: bool = True) -> int:
        """Append a polygon annotation under ``subject`` (defaults to the active subject)."""
        subj = subject if subject is not None else self.state.active_subject
        return self.add_annotation(Annotation(subject=subj, geometry=polygon), push_undo=push_undo)

    def update_annotation(self, idx: int, annotation: Annotation, *, push_undo: bool = True) -> None:
        """Replace the annotation at ``idx``."""
        if not 0 <= idx < len(self.state.annotations):
            raise IndexError(f"annotation index out of range: {idx}")
        if push_undo:
            self.push_undo()
        self.state.annotations[idx] = annotation
        self.invalidate_poly_bboxes()

    def delete_annotation(self, idx: int, *, push_undo: bool = True) -> None:
        """Remove the annotation at ``idx``."""
        if not 0 <= idx < len(self.state.annotations):
            raise IndexError(f"annotation index out of range: {idx}")
        if push_undo:
            self.push_undo()
        self.state.annotations.pop(idx)
        if idx < len(self.authors):
            self.authors.pop(idx)
        sel = self.state.selected_polygon_idx
        if sel == idx:
            self.state.selected_polygon_idx = None
        elif sel is not None and sel > idx:
            self.state.selected_polygon_idx = sel - 1
        self.invalidate_poly_bboxes()

    def close_current_polygon(self) -> bool:
        """Finalize the in-progress polygon into an annotation under the active subject.

        Clamps vertices to image bounds, pushes undo, appends the annotation. Returns ``True`` if a
        polygon was added, ``False`` if the in-progress polygon had fewer than 3 vertices.
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
        self.add_polygon(Polygon(points=clamped))
        s.current_polygon = []
        return True

    # ── I/O ───────────────────────────────────────────────────────────────

    def save(self, *, path: Optional[str] = None) -> bool:
        """Save the current annotations to the single per-image JSON label file.

        The caller resolves the path (web backend or CLI); the engine does not infer it from the
        state. Returns ``True`` on success. Invalid image dimensions skip the write and return
        ``False``.
        """
        s = self.state
        if s.img_width <= 0 or s.img_height <= 0:
            logger.warning("Invalid image dimensions (%sx%s); skipping save", s.img_width, s.img_height)
            return False
        if path is None:
            return True
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            write_annotations(path, s.annotations, s.img_width, s.img_height)
        except OSError:
            logger.exception("Could not write labels to %s", path)
            return False
        return True

import type { StateCreator } from "zustand";

import { pathInDir } from "@/lib/paths";
import type { AppState } from "@/store/appState";
import type { Annotation, Box, ImageLabels, PointShape, PolygonShape } from "@/store/types";

/**
 * Local canvas state: per-image draft annotations shown on the canvas.
 * These are not synced to the backend until the user hits save.
 */
export interface CanvasState {
  imgWidth: number;
  imgHeight: number;
  boxes: Box[];
  polygons: PolygonShape[];
  points: PointShape[];
  // Geometry-less (image/plant-level) ratings; kept so they round-trip losslessly on save.
  imageAnnotations: Annotation[];
  currentPolygon: [number, number][];
  selectedPolygonIdx: number | null;
  selectedPointIdx: number | null;
  undoStack: CanvasSnapshot[];
  redoStack: CanvasSnapshot[];
  /** True when the canvas content differs from the last save (compared by content, so a
   *  net-zero edit like draw-then-delete is clean, not "changed"). */
  dirty: boolean;
  /** Serialized content of the last save/load: the baseline dirty is computed against. */
  savedSignature: string;
  /** Which image's labels the canvas holds: status writes must not read shapes that
   *  still belong to the previous image (or a failed load) mid-flip. */
  loadedImagePath: string | null;
}

/** The saved-content fields only: selection, undo stacks and draft state don't make a save. */
function contentSignature(c: {
  boxes: Box[];
  polygons: PolygonShape[];
  points: PointShape[];
  imageAnnotations: Annotation[];
}): string {
  return JSON.stringify([c.boxes, c.polygons, c.points, c.imageAnnotations]);
}

/** Recompute dirty from content vs the saved baseline (drags skip this per tick; see dragVertex). */
function withContentDirty(c: CanvasState): CanvasState {
  return { ...c, dirty: contentSignature(c) !== c.savedSignature };
}

interface CanvasSnapshot {
  boxes: Box[];
  polygons: PolygonShape[];
  points: PointShape[];
  imageAnnotations: Annotation[];
  selectedPolygonIdx: number | null;
  selectedPointIdx: number | null;
}

const EMPTY_CANVAS: CanvasState = {
  imgWidth: 0,
  imgHeight: 0,
  boxes: [],
  polygons: [],
  points: [],
  imageAnnotations: [],
  currentPolygon: [],
  selectedPolygonIdx: null,
  selectedPointIdx: null,
  undoStack: [],
  redoStack: [],
  dirty: false,
  savedSignature: contentSignature({ boxes: [], polygons: [], points: [], imageAnnotations: [] }),
  loadedImagePath: null,
};

/** Whether the loaded canvas belongs to the open dataset's own image directory: nothing clears
 *  the canvas when another project opens, so a project with no images would otherwise keep
 *  showing the previous project's image facts. The one place that fact is asked, so a status
 *  read and a future consumer can't drift into two different answers. */
export const selectCanvasMatchesDataset = (s: Pick<AppState, "gui" | "canvas">): boolean =>
  pathInDir(s.canvas.loadedImagePath, s.gui.dataset.images_dir);

function snapshot(c: CanvasState): CanvasSnapshot {
  return {
    boxes: c.boxes.slice(),
    polygons: c.polygons.slice(),
    points: c.points.slice(),
    imageAnnotations: c.imageAnnotations.slice(),
    selectedPolygonIdx: c.selectedPolygonIdx,
    selectedPointIdx: c.selectedPointIdx,
  };
}

export interface CanvasSlice {
  /** Canvas-local draft state (not persisted until save). */
  canvas: CanvasState;

  /** Canvas helpers. */
  loadLabelsIntoCanvas: (labels: ImageLabels) => void;
  clearCanvas: () => void;
  pushUndo: () => void;
  undo: () => void;
  redo: () => void;
  addBox: (box: Box) => void;
  updateBox: (idx: number, box: Box) => void;
  /** No-undo box mutation for a live resize/move drag; undo is captured once at drag
   *  start (see updateBox for the undo-pushing variant). */
  dragBox: (idx: number, box: Box) => void;
  deleteBox: (idx: number) => void;
  addPolygon: (polygon: PolygonShape) => void;
  updatePolygon: (idx: number, polygon: PolygonShape) => void;
  /** Move a single polygon vertex (of one ring) without pushing an undo snapshot. Used during a
   *  live vertex drag (undo is captured once at drag start) so a 50px drag doesn't
   *  push dozens of snapshots and evict the whole 30-entry undo history. */
  dragVertex: (
    polygonIdx: number,
    ringIdx: number,
    vertexIdx: number,
    point: [number, number],
  ) => void;
  deletePolygon: (idx: number) => void;
  /** Replaces the polygon at `idx` with the two pieces a cut produced: one undo snapshot, the
   *  first piece selected, the parent's provenance kept on each except the sign-off (dropped),
   *  and the hover index cleared since every later polygon's index has just shifted by one. */
  splitPolygon: (idx: number, rings: [[number, number][], [number, number][]]) => void;
  selectPolygon: (idx: number | null) => void;
  /** Point helpers. A point is one coordinate, so it has no vertex/ring variants: it is placed,
   *  dragged (no-undo, like dragBox/dragVertex: one snapshot per drag, taken at drag start),
   *  attribute-edited via updatePoint, and deleted whole. */
  addPoint: (point: PointShape) => void;
  updatePoint: (idx: number, point: PointShape) => void;
  dragPoint: (idx: number, x: number, y: number) => void;
  deletePoint: (idx: number) => void;
  selectPoint: (idx: number | null) => void;
  setCurrentPolygon: (pts: [number, number][]) => void;
  commitCurrentPolygon: () => boolean;
  /** Geometry-less (image/plant-level) rating helpers. */
  addImageAnnotation: (subject: string) => void;
  updateImageAnnotation: (idx: number, ann: Annotation) => void;
  deleteImageAnnotation: (idx: number) => void;
  markClean: () => void;
  /** Settle dirty from content after a drag (drags flag it per tick without comparing). */
  recomputeDirty: () => void;
}

export const createCanvasSlice: StateCreator<AppState, [], [], CanvasSlice> = (set, get) => ({
  canvas: EMPTY_CANVAS,

  loadLabelsIntoCanvas: (labels) =>
    set(() => {
      const content = {
        boxes: labels.boxes.slice(),
        polygons: labels.polygons.slice(),
        points: labels.points.slice(),
        imageAnnotations: labels.imageAnnotations.slice(),
      };
      return {
        canvas: {
          imgWidth: labels.img_width,
          imgHeight: labels.img_height,
          ...content,
          currentPolygon: [],
          selectedPolygonIdx: null,
          selectedPointIdx: null,
          undoStack: [],
          redoStack: [],
          dirty: false,
          savedSignature: contentSignature(content),
          loadedImagePath: labels.image_path || null,
        },
      };
    }),

  clearCanvas: () => set({ canvas: EMPTY_CANVAS }),

  pushUndo: () =>
    set((s) => ({
      canvas: {
        ...s.canvas,
        undoStack: [...s.canvas.undoStack, snapshot(s.canvas)].slice(-30),
        redoStack: [],
      },
    })),

  undo: () =>
    set((s) => {
      if (s.canvas.currentPolygon.length > 0) {
        return {
          canvas: {
            ...s.canvas,
            currentPolygon: s.canvas.currentPolygon.slice(0, -1),
          },
        };
      }
      const last = s.canvas.undoStack[s.canvas.undoStack.length - 1];
      if (!last) return s;
      return {
        canvas: withContentDirty({
          ...s.canvas,
          undoStack: s.canvas.undoStack.slice(0, -1),
          redoStack: [...s.canvas.redoStack, snapshot(s.canvas)],
          boxes: last.boxes,
          polygons: last.polygons,
          points: last.points,
          imageAnnotations: last.imageAnnotations,
          selectedPolygonIdx: last.selectedPolygonIdx,
          selectedPointIdx: last.selectedPointIdx,
        }),
      };
    }),

  redo: () =>
    set((s) => {
      const last = s.canvas.redoStack[s.canvas.redoStack.length - 1];
      if (!last) return s;
      return {
        canvas: withContentDirty({
          ...s.canvas,
          undoStack: [...s.canvas.undoStack, snapshot(s.canvas)],
          redoStack: s.canvas.redoStack.slice(0, -1),
          boxes: last.boxes,
          polygons: last.polygons,
          points: last.points,
          imageAnnotations: last.imageAnnotations,
          selectedPolygonIdx: last.selectedPolygonIdx,
          selectedPointIdx: last.selectedPointIdx,
        }),
      };
    }),

  addBox: (box) => {
    get().pushUndo();
    set((s) => ({
      canvas: withContentDirty({ ...s.canvas, boxes: [...s.canvas.boxes, box] }),
    }));
  },

  updateBox: (idx, box) => {
    get().pushUndo();
    set((s) => {
      const next = s.canvas.boxes.slice();
      next[idx] = box;
      return { canvas: withContentDirty({ ...s.canvas, boxes: next }) };
    });
  },

  deleteBox: (idx) => {
    get().pushUndo();
    set((s) => ({
      canvas: withContentDirty({
        ...s.canvas,
        boxes: s.canvas.boxes.filter((_, i) => i !== idx),
      }),
    }));
  },

  addPolygon: (polygon) => {
    get().pushUndo();
    set((s) => ({
      canvas: withContentDirty({ ...s.canvas, polygons: [...s.canvas.polygons, polygon] }),
    }));
  },

  updatePolygon: (idx, polygon) => {
    get().pushUndo();
    set((s) => {
      const next = s.canvas.polygons.slice();
      next[idx] = polygon;
      return { canvas: withContentDirty({ ...s.canvas, polygons: next }) };
    });
  },

  // The drag actions fire per mousemove: a per-tick content compare would re-serialize the
  // whole canvas at pointer rate, so they flag dirty and the release calls recomputeDirty.
  dragVertex: (polygonIdx, ringIdx, vertexIdx, point) =>
    set((s) => {
      const poly = s.canvas.polygons[polygonIdx];
      if (!poly?.rings[ringIdx]) return s;
      const pts = poly.rings[ringIdx].slice();
      pts[vertexIdx] = point;
      const rings = poly.rings.slice();
      rings[ringIdx] = pts;
      const next = s.canvas.polygons.slice();
      next[polygonIdx] = { ...poly, rings };
      return { canvas: { ...s.canvas, polygons: next, dirty: true } };
    }),

  dragBox: (idx, box) =>
    set((s) => {
      if (!s.canvas.boxes[idx]) return s;
      const next = s.canvas.boxes.slice();
      next[idx] = box;
      return { canvas: { ...s.canvas, boxes: next, dirty: true } };
    }),

  deletePolygon: (idx) => {
    get().pushUndo();
    set((s) => {
      const polys = s.canvas.polygons.filter((_, i) => i !== idx);
      let sel = s.canvas.selectedPolygonIdx;
      if (sel === idx) sel = null;
      else if (sel !== null && sel > idx) sel = sel - 1;
      return {
        canvas: withContentDirty({ ...s.canvas, polygons: polys, selectedPolygonIdx: sel }),
      };
    });
  },

  splitPolygon: (idx, rings) => {
    get().pushUndo();
    set((s) => {
      const parent = s.canvas.polygons[idx];
      if (!parent) return s;
      const pieces: PolygonShape[] = rings.map((ring) => ({
        rings: [ring],
        subject: parent.subject,
        attributes: { ...parent.attributes },
        created_by: parent.created_by,
        created_at: parent.created_at,
        accepted_by: null,
        accepted_at: null,
        authorship: parent.authorship,
      }));
      const polys = s.canvas.polygons.slice();
      polys.splice(idx, 1, ...pieces);
      return {
        canvas: withContentDirty({ ...s.canvas, polygons: polys, selectedPolygonIdx: idx }),
        annotateUi: { ...s.annotateUi, hoveredPolygonIdx: null },
      };
    });
  },

  selectPolygon: (selectedPolygonIdx) =>
    set((s) => ({ canvas: { ...s.canvas, selectedPolygonIdx } })),

  addPoint: (point) => {
    get().pushUndo();
    set((s) => ({
      canvas: withContentDirty({ ...s.canvas, points: [...s.canvas.points, point] }),
    }));
  },

  updatePoint: (idx, point) => {
    get().pushUndo();
    set((s) => {
      const next = s.canvas.points.slice();
      next[idx] = point;
      return { canvas: withContentDirty({ ...s.canvas, points: next }) };
    });
  },

  dragPoint: (idx, x, y) =>
    set((s) => {
      const p = s.canvas.points[idx];
      if (!p) return s;
      const next = s.canvas.points.slice();
      next[idx] = { ...p, x, y };
      return { canvas: { ...s.canvas, points: next, dirty: true } };
    }),

  deletePoint: (idx) => {
    get().pushUndo();
    set((s) => {
      const points = s.canvas.points.filter((_, i) => i !== idx);
      let sel = s.canvas.selectedPointIdx;
      if (sel === idx) sel = null;
      else if (sel !== null && sel > idx) sel = sel - 1;
      return { canvas: withContentDirty({ ...s.canvas, points, selectedPointIdx: sel }) };
    });
  },

  selectPoint: (selectedPointIdx) => set((s) => ({ canvas: { ...s.canvas, selectedPointIdx } })),

  setCurrentPolygon: (pts) => set((s) => ({ canvas: { ...s.canvas, currentPolygon: pts } })),

  commitCurrentPolygon: () => {
    const cur = get().canvas.currentPolygon;
    if (cur.length < 3) {
      set((s) => ({ canvas: { ...s.canvas, currentPolygon: [] } }));
      return false;
    }
    const subject = get().gui.active_subject;
    if (!subject) {
      // No subject selected: refuse to author a subjectless shape (the backend save rejects it).
      set((s) => ({ canvas: { ...s.canvas, currentPolygon: [] } }));
      return false;
    }
    const { imgWidth, imgHeight } = get().canvas;
    const clamped: [number, number][] = cur.map(([x, y]) => [
      imgWidth ? Math.max(0, Math.min(imgWidth, x)) : x,
      imgHeight ? Math.max(0, Math.min(imgHeight, y)) : y,
    ]);
    get().pushUndo();
    set((s) => ({
      canvas: withContentDirty({
        ...s.canvas,
        currentPolygon: [],
        // A hand-drawn shape is one contour: one ring (the canvas never draws a second by hand).
        polygons: [...s.canvas.polygons, { rings: [clamped], subject, attributes: {} }],
      }),
    }));
    return true;
  },

  addImageAnnotation: (subject) => {
    get().pushUndo();
    set((s) => ({
      canvas: withContentDirty({
        ...s.canvas,
        imageAnnotations: [...s.canvas.imageAnnotations, { subject, attributes: {} }],
      }),
    }));
  },

  updateImageAnnotation: (idx, ann) => {
    get().pushUndo();
    set((s) => {
      const next = s.canvas.imageAnnotations.slice();
      next[idx] = ann;
      return { canvas: withContentDirty({ ...s.canvas, imageAnnotations: next }) };
    });
  },

  deleteImageAnnotation: (idx) => {
    get().pushUndo();
    set((s) => ({
      canvas: withContentDirty({
        ...s.canvas,
        imageAnnotations: s.canvas.imageAnnotations.filter((_, i) => i !== idx),
      }),
    }));
  },

  // A save re-baselines: the just-saved content is what future edits compare against.
  markClean: () =>
    set((s) => ({
      canvas: { ...s.canvas, dirty: false, savedSignature: contentSignature(s.canvas) },
    })),

  recomputeDirty: () => set((s) => ({ canvas: withContentDirty(s.canvas) })),
});

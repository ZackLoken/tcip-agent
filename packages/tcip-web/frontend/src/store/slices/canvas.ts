import type { StateCreator } from "zustand";

import { currentImage, pathInDir } from "@/lib/paths";
import type { AppState } from "@/store/appState";
import type {
  Annotation,
  Box,
  Flag,
  ImageLabels,
  PointShape,
  PolygonShape,
  Mode,
  SubjectState,
} from "@/store/types";

/** The one focused item: an annotation of a tool's geometry by its canvas index, or a proposal by
 *  its index in the bucket's document. Null focuses nothing. */
export interface Focus {
  kind: Mode | "proposal";
  index: number;
}

/** The focus after the `kind` shape at `deleted` left its array: dropped when it was that shape,
 *  shifted down when it sat after it. */
function focusAfterDelete(focus: Focus | null, kind: Mode, deleted: number): Focus | null {
  if (focus?.kind !== kind) return focus;
  if (focus.index === deleted) return null;
  return focus.index > deleted ? { kind, index: focus.index - 1 } : focus;
}

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
  focus: Focus | null;
  undoStack: CanvasSnapshot[];
  redoStack: CanvasSnapshot[];
  /** True when the canvas content differs from the last save (compared by content, so a
   *  net-zero edit like draw-then-delete is clean, not "changed"). */
  dirty: boolean;
  /** Serialized content of the last save/load: the baseline dirty is computed against. */
  savedSignature: string;
  /** Which image's labels the canvas holds: a save must not read shapes that still belong to the
   *  previous image (or a failed load) mid-flip. */
  loadedImagePath: string | null;
  /** Each subject's state as the backend last served it for the loaded image. */
  completion: Record<string, SubjectState>;
  /** The image's flags as the backend last served them. */
  flags: Flag[];
  /** The save in flight, which holds the canvas against edits until its answer is adopted whole;
   *  null when none is. A load that replaces the canvas drops the hold. */
  saving: SaveHold | null;
}

/** A save's hold on the canvas: the image it writes, a request id minted once per page so a
 *  request from an unmounted editor never shares an id with a later one, and the controller of
 *  the request, which the hold owns. */
export interface SaveHold {
  image: string;
  id: number;
  controller: AbortController;
}

let lastSaveId = 0;

/** Whether `hold` is the hold the canvas has now, by the whole hold. */
export function holdsCanvas(canvas: Pick<CanvasState, "saving">, hold: SaveHold): boolean {
  return canvas.saving?.id === hold.id && canvas.saving.image === hold.image;
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

/** Recompute dirty from content vs the saved baseline. */
function withContentDirty(c: CanvasState): CanvasState {
  return { ...c, dirty: contentSignature(c) !== c.savedSignature };
}

interface CanvasSnapshot {
  boxes: Box[];
  polygons: PolygonShape[];
  points: PointShape[];
  imageAnnotations: Annotation[];
  focus: Focus | null;
  /** The focus context the snapshot was taken in (`focusContext`). */
  context: string;
}

/** What a focus names an index of: the image's labels and the bucket's document. Focus clears
 *  when this changes and survives a refresh of the same one; the one place that rule is stated. */
export const focusContext = (s: Pick<AppState, "gui">): string =>
  [s.gui.dataset.images_dir, currentImage(s.gui.dataset).name, s.gui.dataset.bucket].join("\0");

const EMPTY_CANVAS: CanvasState = {
  imgWidth: 0,
  imgHeight: 0,
  boxes: [],
  polygons: [],
  points: [],
  imageAnnotations: [],
  currentPolygon: [],
  focus: null,
  undoStack: [],
  redoStack: [],
  dirty: false,
  savedSignature: contentSignature({ boxes: [], polygons: [], points: [], imageAnnotations: [] }),
  loadedImagePath: null,
  completion: {},
  flags: [],
  saving: null,
};

/** Whether the loaded canvas belongs to the open dataset's own image directory: nothing clears
 *  the canvas when another project opens, so a project with no images would otherwise keep
 *  showing the previous project's image facts. The one place that fact is asked, so a status
 *  read and a future consumer can't drift into two different answers. */
export const selectCanvasMatchesDataset = (s: Pick<AppState, "gui" | "canvas">): boolean =>
  pathInDir(s.canvas.loadedImagePath, s.gui.dataset.images_dir);

function snapshot(s: AppState): CanvasSnapshot {
  const c = s.canvas;
  return {
    boxes: c.boxes.slice(),
    polygons: c.polygons.slice(),
    points: c.points.slice(),
    imageAnnotations: c.imageAnnotations.slice(),
    // A proposal's index names a bucket's document, which history does not own.
    focus: c.focus?.kind === "proposal" ? null : c.focus,
    context: focusContext(s),
  };
}

/** The focus a restored snapshot brings back: its own where the context is still the one it was
 *  taken in, else the current one, which the context rule owns. */
const restoredFocus = (s: AppState, last: CanvasSnapshot): Focus | null =>
  last.context === focusContext(s) ? last.focus : s.canvas.focus;

/** Restore the last history snapshot, whatever polygon is being drawn. */
function restoreLast(s: AppState): Partial<AppState> | AppState {
  const last = s.canvas.undoStack[s.canvas.undoStack.length - 1];
  if (!last) return s;
  return {
    canvas: withContentDirty({
      ...s.canvas,
      undoStack: s.canvas.undoStack.slice(0, -1),
      redoStack: [...s.canvas.redoStack, snapshot(s)],
      boxes: last.boxes,
      polygons: last.polygons,
      points: last.points,
      imageAnnotations: last.imageAnnotations,
      focus: restoredFocus(s, last),
    }),
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
  /** Restore the last history snapshot even while a polygon is being drawn, which `undo` would
   *  shorten instead; for a gesture that took its own snapshot and is reverted whole. */
  rollbackLast: () => void;
  redo: () => void;
  addBox: (box: Box) => void;
  updateBox: (idx: number, box: Box) => void;
  /** No-undo box mutation for a live resize/move drag; undo is captured once at drag
   *  start (see updateBox for the undo-pushing variant). */
  dragBox: (idx: number, box: Box) => void;
  deleteBox: (idx: number) => void;
  addPolygon: (polygon: PolygonShape) => void;
  updatePolygon: (idx: number, polygon: PolygonShape) => void;
  /** Move a single polygon vertex (of one ring) without pushing an undo snapshot. */
  dragVertex: (
    polygonIdx: number,
    ringIdx: number,
    vertexIdx: number,
    point: [number, number],
  ) => void;
  deletePolygon: (idx: number) => void;
  /** Replaces the polygon at `idx` with the two pieces a cut produced: one undo snapshot, the
   *  first piece focused, the parent's subject, attributes and crowd flag on each, and the hover
   *  index cleared since every later polygon's index has just shifted by one. */
  splitPolygon: (idx: number, rings: [[number, number][], [number, number][]]) => void;
  /** Focus one item, or nothing; the previous focus is replaced. */
  setFocus: (focus: Focus | null) => void;
  /** Point helpers. A point is one coordinate, so it has no vertex/ring variants: it is placed,
   *  dragged (no-undo, like dragBox/dragVertex: one snapshot per drag, taken at drag start),
   *  attribute-edited via updatePoint, and deleted whole. */
  addPoint: (point: PointShape) => void;
  updatePoint: (idx: number, point: PointShape) => void;
  dragPoint: (idx: number, x: number, y: number) => void;
  deletePoint: (idx: number) => void;
  setCurrentPolygon: (pts: [number, number][]) => void;
  commitCurrentPolygon: () => boolean;
  /** Geometry-less (image/plant-level) rating helpers. */
  addImageAnnotation: (subject: string) => void;
  updateImageAnnotation: (idx: number, ann: Annotation) => void;
  deleteImageAnnotation: (idx: number) => void;
  /** Hold the canvas against edits for a save of `image`, and return the hold; a load unlocks it. */
  holdForSave: (image: string) => SaveHold;
  /** Release the canvas if `hold` still holds it; a hold dropped by a load or taken by another
   *  save is left alone. */
  releaseSave: (hold: SaveHold) => void;
  /** Abort the request the hold owns and release the canvas, from whichever tab is mounted; the
   *  unsaved edits stay and the request's late answer finds no hold to adopt under. */
  cancelSave: () => void;
  /** Settle dirty from content after a drag (drags flag it per tick without comparing). */
  recomputeDirty: () => void;
}

/** Every action that changes the canvas' content or its history; inert while a save is in flight
 *  so the answer is adopted over the very content it was built from. Focus, loads, clearing and
 *  the dirty recount are not edits. */
const CONTENT_EDITS = [
  "pushUndo",
  "undo",
  "rollbackLast",
  "redo",
  "addBox",
  "updateBox",
  "dragBox",
  "deleteBox",
  "addPolygon",
  "updatePolygon",
  "dragVertex",
  "deletePolygon",
  "splitPolygon",
  "addPoint",
  "updatePoint",
  "dragPoint",
  "deletePoint",
  "setCurrentPolygon",
  "commitCurrentPolygon",
  "addImageAnnotation",
  "updateImageAnnotation",
  "deleteImageAnnotation",
] as const;

export const createCanvasSlice: StateCreator<AppState, [], [], CanvasSlice> = (set, get) => {
  const slice: CanvasSlice = {
    canvas: EMPTY_CANVAS,

    loadLabelsIntoCanvas: (labels) =>
      set((s) => {
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
            focus: s.canvas.focus,
            undoStack: [],
            redoStack: [],
            dirty: false,
            savedSignature: contentSignature(content),
            loadedImagePath: labels.image_path || null,
            completion: labels.completion,
            flags: labels.flags,
            saving: null,
          },
        };
      }),

    clearCanvas: () => set({ canvas: EMPTY_CANVAS }),

    pushUndo: () =>
      set((s) => ({
        canvas: {
          ...s.canvas,
          undoStack: [...s.canvas.undoStack, snapshot(s)].slice(-30),
          redoStack: [],
        },
      })),

    undo: () =>
      set((s) =>
        s.canvas.currentPolygon.length > 0
          ? {
              canvas: {
                ...s.canvas,
                currentPolygon: s.canvas.currentPolygon.slice(0, -1),
              },
            }
          : restoreLast(s),
      ),

    rollbackLast: () => set(restoreLast),

    redo: () =>
      set((s) => {
        const last = s.canvas.redoStack[s.canvas.redoStack.length - 1];
        if (!last) return s;
        return {
          canvas: withContentDirty({
            ...s.canvas,
            undoStack: [...s.canvas.undoStack, snapshot(s)],
            redoStack: s.canvas.redoStack.slice(0, -1),
            boxes: last.boxes,
            polygons: last.polygons,
            points: last.points,
            imageAnnotations: last.imageAnnotations,
            focus: restoredFocus(s, last),
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
          focus: focusAfterDelete(s.canvas.focus, "box", idx),
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
        return {
          canvas: withContentDirty({
            ...s.canvas,
            polygons: polys,
            focus: focusAfterDelete(s.canvas.focus, "polygon", idx),
          }),
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
          iscrowd: parent.iscrowd,
        }));
        const polys = s.canvas.polygons.slice();
        polys.splice(idx, 1, ...pieces);
        return {
          canvas: withContentDirty({
            ...s.canvas,
            polygons: polys,
            focus: { kind: "polygon", index: idx },
          }),
          annotateUi: { ...s.annotateUi, hoveredPolygonIdx: null },
        };
      });
    },

    setFocus: (focus) => set((s) => ({ canvas: { ...s.canvas, focus } })),

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
        return {
          canvas: withContentDirty({
            ...s.canvas,
            points,
            focus: focusAfterDelete(s.canvas.focus, "point", idx),
          }),
        };
      });
    },

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
          imageAnnotations: [
            ...s.canvas.imageAnnotations,
            { subject, attributes: {}, iscrowd: false },
          ],
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

    holdForSave: (image) => {
      const hold = { image, id: ++lastSaveId, controller: new AbortController() };
      set((s) => ({ canvas: { ...s.canvas, saving: hold } }));
      return hold;
    },

    releaseSave: (hold) =>
      set((s) => (holdsCanvas(s.canvas, hold) ? { canvas: { ...s.canvas, saving: null } } : s)),

    cancelSave: () => {
      const hold = get().canvas.saving;
      if (!hold) return;
      hold.controller.abort();
      get().releaseSave(hold);
    },

    recomputeDirty: () => set((s) => ({ canvas: withContentDirty(s.canvas) })),
  };
  for (const name of CONTENT_EDITS) {
    const edit = slice[name] as (...args: never[]) => unknown;
    slice[name] = ((...args: never[]) =>
      get().canvas.saving ? undefined : edit(...args)) as never;
  }
  return slice;
};

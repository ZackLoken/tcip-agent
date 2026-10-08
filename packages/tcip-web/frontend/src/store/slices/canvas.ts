import type { StateCreator } from "zustand";

import { currentImage, pathInDir } from "@/lib/paths";
import type { AppState } from "@/store/appState";
import {
  contentOf,
  NO_CONTENT,
  TOOL_ARRAY,
  type CanvasContent,
  type Flag,
  type ImageLabels,
  type PolygonShape,
  type Mode,
  type SubjectState,
} from "@/store/types";

/** The one focused item: an annotation of a tool's geometry by its canvas index, or a proposal by
 *  its index in the bucket's document. Null focuses nothing. */
export interface Focus {
  kind: Mode | "proposal";
  index: number;
}

/** A content array's name, and the item it holds. */
type ContentArray = keyof CanvasContent;
type ItemOf<A extends ContentArray> = CanvasContent[A][number];

/** The focus after the item at `deleted` left `array`: dropped when it was that item, shifted
 *  down when it sat after it. */
function focusAfterDelete(focus: Focus | null, array: ContentArray, deleted: number): Focus | null {
  if (!focus || focus.kind === "proposal" || TOOL_ARRAY[focus.kind] !== array) return focus;
  if (focus.index === deleted) return null;
  return focus.index > deleted ? { ...focus, index: focus.index - 1 } : focus;
}

/**
 * Local canvas state: per-image draft annotations shown on the canvas.
 * These are not synced to the backend until the user hits save.
 */
export interface CanvasState extends CanvasContent {
  imgWidth: number;
  imgHeight: number;
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
function contentSignature(c: CanvasContent): string {
  return JSON.stringify(contentOf(c));
}

/** Recompute dirty from content vs the saved baseline. */
function withContentDirty(c: CanvasState): CanvasState {
  return { ...c, dirty: contentSignature(c) !== c.savedSignature };
}

interface CanvasSnapshot extends CanvasContent {
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
  ...NO_CONTENT,
  currentPolygon: [],
  focus: null,
  undoStack: [],
  redoStack: [],
  dirty: false,
  savedSignature: contentSignature(NO_CONTENT),
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
    ...contentOf(c),
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
      ...contentOf(last),
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
  /** Append `item` to `array`, one undo snapshot first. */
  add: <A extends ContentArray>(array: A, item: ItemOf<A>) => void;
  /** Replace the item at `idx` of `array`, one undo snapshot first. */
  update: <A extends ContentArray>(array: A, idx: number, item: ItemOf<A>) => void;
  /** Replace the item at `idx` of `array` for a live drag: no undo snapshot (the drag takes one at
   *  its start), dirty flagged without a compare, and a missing index ignored. */
  drag: <A extends ContentArray>(array: A, idx: number, item: ItemOf<A>) => void;
  /** Remove the item at `idx` of `array`, one undo snapshot first; the focus follows it. */
  remove: (array: ContentArray, idx: number) => void;
  /** Move a single polygon vertex (of one ring) without pushing an undo snapshot. */
  dragVertex: (
    polygonIdx: number,
    ringIdx: number,
    vertexIdx: number,
    point: [number, number],
  ) => void;
  /** Replaces the polygon at `idx` with the two pieces a cut produced: one undo snapshot, the
   *  first piece focused, the parent's subject, attributes and crowd flag on each, and the hover
   *  index cleared since every later polygon's index has just shifted by one. */
  splitPolygon: (idx: number, rings: [[number, number][], [number, number][]]) => void;
  /** Focus one item, or nothing; the previous focus is replaced. */
  setFocus: (focus: Focus | null) => void;
  setCurrentPolygon: (pts: [number, number][]) => void;
  commitCurrentPolygon: () => boolean;
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
export const CONTENT_EDITS = [
  "pushUndo",
  "undo",
  "rollbackLast",
  "redo",
  "add",
  "update",
  "drag",
  "remove",
  "dragVertex",
  "splitPolygon",
  "setCurrentPolygon",
  "commitCurrentPolygon",
] as const;

type Item = ItemOf<ContentArray>;

const replaceAt = (items: readonly Item[], idx: number, item: Item): Item[] =>
  items.map((had, i) => (i === idx ? item : had));

export const createCanvasSlice: StateCreator<AppState, [], [], CanvasSlice> = (set, get) => {
  /** One undo snapshot, then `array` replaced by what `change` makes of it, and the focus by what
   *  `refocus` makes of it. */
  const edit = (
    array: ContentArray,
    change: (items: readonly Item[]) => Item[],
    refocus = (focus: Focus | null) => focus,
  ) => {
    get().pushUndo();
    set((s) => ({
      canvas: withContentDirty({
        ...s.canvas,
        [array]: change(s.canvas[array]),
        focus: refocus(s.canvas.focus),
      }),
    }));
  };

  const slice: CanvasSlice = {
    canvas: EMPTY_CANVAS,

    loadLabelsIntoCanvas: (labels) =>
      set((s) => ({
        canvas: {
          imgWidth: labels.img_width,
          imgHeight: labels.img_height,
          ...contentOf(labels),
          currentPolygon: [],
          focus: s.canvas.focus,
          undoStack: [],
          redoStack: [],
          dirty: false,
          savedSignature: contentSignature(labels),
          loadedImagePath: labels.image_path || null,
          completion: labels.completion,
          flags: labels.flags,
          saving: null,
        },
      })),

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
            ...contentOf(last),
            focus: restoredFocus(s, last),
          }),
        };
      }),

    add: (array, item) => edit(array, (items) => [...items, item]),

    update: (array, idx, item) => edit(array, (items) => replaceAt(items, idx, item)),

    remove: (array, idx) =>
      edit(
        array,
        (items) => items.filter((_, i) => i !== idx),
        (focus) => focusAfterDelete(focus, array, idx),
      ),

    // The drag actions fire per mousemove: a per-tick content compare would re-serialize the
    // whole canvas at pointer rate, so they flag dirty and the release calls recomputeDirty.
    drag: (array, idx, item) =>
      set((s) =>
        s.canvas[array][idx]
          ? { canvas: { ...s.canvas, [array]: replaceAt(s.canvas[array], idx, item), dirty: true } }
          : s,
      ),

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

    setCurrentPolygon: (pts) => set((s) => ({ canvas: { ...s.canvas, currentPolygon: pts } })),

    commitCurrentPolygon: () => {
      const { currentPolygon: cur, imgWidth, imgHeight } = get().canvas;
      const subject = get().gui.active_subject;
      set((s) => ({ canvas: { ...s.canvas, currentPolygon: [] } }));
      // No subject selected: refuse to author a subjectless shape (the backend save rejects it).
      if (cur.length < 3 || !subject) return false;
      const clamped: [number, number][] = cur.map(([x, y]) => [
        imgWidth ? Math.max(0, Math.min(imgWidth, x)) : x,
        imgHeight ? Math.max(0, Math.min(imgHeight, y)) : y,
      ]);
      // A hand-drawn shape is one contour: one ring (the canvas never draws a second by hand).
      get().add("polygons", { rings: [clamped], subject, attributes: {} });
      return true;
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
    const action = slice[name] as (...args: never[]) => unknown;
    slice[name] = ((...args: never[]) =>
      get().canvas.saving ? undefined : action(...args)) as never;
  }
  return slice;
};

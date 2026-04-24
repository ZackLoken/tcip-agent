import { create } from "zustand";

import type {
  Box,
  DatasetSelection,
  Detection,
  GuiState,
  ImageLabels,
  MatchesResponse,
  PolygonShape,
  PredictionReference,
  ReviewFilters,
  TabName,
  ViewState,
} from "@/store/types";

const DEFAULT_REVIEW: ReviewFilters = {
  iou_threshold: 0.5,
  conf_threshold: 0.25,
  filter_type: "all",
  filter_class: "all",
  status_filter: "all",
  detection_idx: 0,
};

const DEFAULT_DATASET: DatasetSelection = {
  project_root: null,
  dataset_root: null,
  annotation_type: null,
  date: null,
  image_list: [],
  current_image_index: 0,
  annotations_detect_dir: null,
  annotations_segment_dir: null,
  predictions_detect_dir: null,
  predictions_segment_dir: null,
};

const DEFAULT_STATE: GuiState = {
  active_tab: "annotate",
  dataset: DEFAULT_DATASET,
  view: { scale: 1, offset_x: 0, offset_y: 0 },
  class_names: { 0: "catkin" },
  class_colors: { 0: "#4CAF50" },
  mode: "box",
  active_class: 0,
  review: DEFAULT_REVIEW,
  pred_reference: null,
  training_runs: [],
  active_run_id: null,
  inference_jobs: [],
};

/**
 * Local canvas state — per-image draft annotations shown on the canvas.
 * These are NOT synced to the backend until the user hits save.
 */
export interface CanvasState {
  imgWidth: number;
  imgHeight: number;
  boxes: Box[];
  polygons: PolygonShape[];
  currentPolygon: [number, number][];
  selectedPolygonIdx: number | null;
  undoStack: { boxes: Box[]; polygons: PolygonShape[]; selectedPolygonIdx: number | null }[];
  redoStack: { boxes: Box[]; polygons: PolygonShape[]; selectedPolygonIdx: number | null }[];
  dirty: boolean;
}

const EMPTY_CANVAS: CanvasState = {
  imgWidth: 0,
  imgHeight: 0,
  boxes: [],
  polygons: [],
  currentPolygon: [],
  selectedPolygonIdx: null,
  undoStack: [],
  redoStack: [],
  dirty: false,
};

interface ReviewTabState {
  matches: MatchesResponse | null;
  loading: boolean;
}

export interface AppState {
  /** Server-synchronized state (mirrors backend GuiState). */
  gui: GuiState;
  wsStatus: "disconnected" | "connecting" | "connected" | "error";

  /** Canvas-local draft state (not persisted until save). */
  canvas: CanvasState;

  /** Review tab derived state. */
  review: ReviewTabState;

  /** Setters. */
  setGui: (next: GuiState) => void;
  patchGui: (partial: Partial<GuiState>) => void;
  setWsStatus: (s: AppState["wsStatus"]) => void;
  setActiveTab: (tab: TabName) => void;
  setView: (view: ViewState) => void;
  setMode: (mode: "box" | "polygon") => void;
  setActiveClass: (cid: number) => void;
  setPredReference: (p: PredictionReference | null) => void;

  /** Canvas helpers. */
  loadLabelsIntoCanvas: (labels: ImageLabels) => void;
  clearCanvas: () => void;
  pushUndo: () => void;
  undo: () => void;
  redo: () => void;
  addBox: (box: Box) => void;
  updateBox: (idx: number, box: Box) => void;
  deleteBox: (idx: number) => void;
  addPolygon: (polygon: PolygonShape) => void;
  updatePolygon: (idx: number, polygon: PolygonShape) => void;
  deletePolygon: (idx: number) => void;
  selectPolygon: (idx: number | null) => void;
  setCurrentPolygon: (pts: [number, number][]) => void;
  commitCurrentPolygon: () => boolean;
  markClean: () => void;

  /** Review helpers. */
  setMatches: (matches: MatchesResponse | null) => void;
  setReviewLoading: (loading: boolean) => void;
  setReviewDetectionIdx: (idx: number) => void;
  markDetectionReviewed: (idx: number, action: string) => void;
}

function snapshot(c: CanvasState) {
  return {
    boxes: c.boxes.slice(),
    polygons: c.polygons.slice(),
    selectedPolygonIdx: c.selectedPolygonIdx,
  };
}

export const useStore = create<AppState>()((set, get) => ({
  gui: DEFAULT_STATE,
  wsStatus: "disconnected",
  canvas: EMPTY_CANVAS,
  review: { matches: null, loading: false },

  setGui: (next) => set({ gui: next }),
  patchGui: (partial) => set((s) => ({ gui: { ...s.gui, ...partial } })),
  setWsStatus: (wsStatus) => set({ wsStatus }),
  setActiveTab: (active_tab) => set((s) => ({ gui: { ...s.gui, active_tab } })),
  setView: (view) => set((s) => ({ gui: { ...s.gui, view } })),
  setMode: (mode) => set((s) => ({ gui: { ...s.gui, mode } })),
  setActiveClass: (active_class) => set((s) => ({ gui: { ...s.gui, active_class } })),
  setPredReference: (pred_reference) => set((s) => ({ gui: { ...s.gui, pred_reference } })),

  loadLabelsIntoCanvas: (labels) =>
    set(() => ({
      canvas: {
        imgWidth: labels.img_width,
        imgHeight: labels.img_height,
        boxes: labels.boxes.slice(),
        polygons: labels.polygons.slice(),
        currentPolygon: [],
        selectedPolygonIdx: null,
        undoStack: [],
        redoStack: [],
        dirty: false,
      },
    })),

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
      const last = s.canvas.undoStack[s.canvas.undoStack.length - 1];
      if (!last) return s;
      return {
        canvas: {
          ...s.canvas,
          undoStack: s.canvas.undoStack.slice(0, -1),
          redoStack: [...s.canvas.redoStack, snapshot(s.canvas)],
          boxes: last.boxes,
          polygons: last.polygons,
          selectedPolygonIdx: last.selectedPolygonIdx,
          dirty: true,
        },
      };
    }),

  redo: () =>
    set((s) => {
      const last = s.canvas.redoStack[s.canvas.redoStack.length - 1];
      if (!last) return s;
      return {
        canvas: {
          ...s.canvas,
          undoStack: [...s.canvas.undoStack, snapshot(s.canvas)],
          redoStack: s.canvas.redoStack.slice(0, -1),
          boxes: last.boxes,
          polygons: last.polygons,
          selectedPolygonIdx: last.selectedPolygonIdx,
          dirty: true,
        },
      };
    }),

  addBox: (box) => {
    get().pushUndo();
    set((s) => ({ canvas: { ...s.canvas, boxes: [...s.canvas.boxes, box], dirty: true } }));
  },

  updateBox: (idx, box) => {
    get().pushUndo();
    set((s) => {
      const next = s.canvas.boxes.slice();
      next[idx] = box;
      return { canvas: { ...s.canvas, boxes: next, dirty: true } };
    });
  },

  deleteBox: (idx) => {
    get().pushUndo();
    set((s) => ({
      canvas: {
        ...s.canvas,
        boxes: s.canvas.boxes.filter((_, i) => i !== idx),
        dirty: true,
      },
    }));
  },

  addPolygon: (polygon) => {
    get().pushUndo();
    set((s) => ({
      canvas: { ...s.canvas, polygons: [...s.canvas.polygons, polygon], dirty: true },
    }));
  },

  updatePolygon: (idx, polygon) => {
    get().pushUndo();
    set((s) => {
      const next = s.canvas.polygons.slice();
      next[idx] = polygon;
      return { canvas: { ...s.canvas, polygons: next, dirty: true } };
    });
  },

  deletePolygon: (idx) => {
    get().pushUndo();
    set((s) => {
      const polys = s.canvas.polygons.filter((_, i) => i !== idx);
      let sel = s.canvas.selectedPolygonIdx;
      if (sel === idx) sel = null;
      else if (sel !== null && sel > idx) sel = sel - 1;
      return { canvas: { ...s.canvas, polygons: polys, selectedPolygonIdx: sel, dirty: true } };
    });
  },

  selectPolygon: (selectedPolygonIdx) =>
    set((s) => ({ canvas: { ...s.canvas, selectedPolygonIdx } })),

  setCurrentPolygon: (pts) =>
    set((s) => ({ canvas: { ...s.canvas, currentPolygon: pts } })),

  commitCurrentPolygon: () => {
    const cur = get().canvas.currentPolygon;
    if (cur.length < 3) {
      set((s) => ({ canvas: { ...s.canvas, currentPolygon: [] } }));
      return false;
    }
    const active_class = get().gui.active_class;
    const { imgWidth, imgHeight } = get().canvas;
    const clamped: [number, number][] = cur.map(([x, y]) => [
      imgWidth ? Math.max(0, Math.min(imgWidth, x)) : x,
      imgHeight ? Math.max(0, Math.min(imgHeight, y)) : y,
    ]);
    get().pushUndo();
    set((s) => ({
      canvas: {
        ...s.canvas,
        currentPolygon: [],
        polygons: [...s.canvas.polygons, { points: clamped, class_id: active_class }],
        dirty: true,
      },
    }));
    return true;
  },

  markClean: () => set((s) => ({ canvas: { ...s.canvas, dirty: false } })),

  setMatches: (matches) => set((s) => ({ review: { ...s.review, matches } })),
  setReviewLoading: (loading) => set((s) => ({ review: { ...s.review, loading } })),
  setReviewDetectionIdx: (idx) =>
    set((s) => ({
      gui: { ...s.gui, review: { ...s.gui.review, detection_idx: idx } },
    })),

  markDetectionReviewed: (idx, action) =>
    set((s) => {
      if (!s.review.matches) return s;
      const next: Detection[] = s.review.matches.detections.slice();
      if (next[idx]) {
        next[idx] = { ...next[idx], reviewed: true, reviewed_action: action };
      }
      return {
        review: {
          ...s.review,
          matches: { ...s.review.matches, detections: next },
        },
      };
    }),
}));

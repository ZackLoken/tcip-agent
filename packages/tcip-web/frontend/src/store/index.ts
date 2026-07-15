import { create } from "zustand";

import { autoColor, type ClassEntry, type ImageStatus } from "@/api/classes";
import { datasetKey, loadDatasetUi, saveDatasetUi } from "@/lib/datasetUiState";
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
  mode: "box",
  active_class: 0,
  review: DEFAULT_REVIEW,
  pred_reference: null,
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
  // Which image's labels the canvas holds — status writes must not read shapes that
  // still belong to the previous image (or a failed load) mid-flip.
  loadedImagePath: string | null;
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
  loadedImagePath: null,
};

const EMPTY_SESSION_TRACKING: SessionTrackingState = {
  currentImageName: null,
  imageEnterTimeMs: null,
  loadedAnnotationCount: 0,
  annotationsAddedDelta: 0,
  lastFlushedKey: null,
};

interface ReviewTabState {
  matches: MatchesResponse | null;
  loading: boolean;
  // One-shot: the detection index a `review_focus` command asked to center on. The reload
  // effect consumes it once (else it always jumps to first-unreviewed, dropping the agent's
  // "look at detection N" request).
  focusDetectionIdx: number | null;
}

interface ClassesState {
  /** Ordered list of classes (id, name, color). Source of truth for class colors. */
  list: ClassEntry[];
  /** Set after first successful load for the current project (prevents double-save on hydrate). */
  loaded: boolean;
}

interface AnnotateUiState {
  /** Show annotation overlays (Visible checkbox). */
  visible: boolean;
  /** Snap toggle (polygon mode). */
  snap: boolean;
  /** Stream toggle (polygon mode). */
  stream: boolean;
  /** Currently hovered polygon index (for vertex-handle rendering). */
  hoveredPolygonIdx: number | null;
  /** Active vertex drag: [polygonIdx, vertexIdx]. */
  draggingVertex: [number, number] | null;
}

interface SessionTrackingState {
  /** Active image for per-image annotation timing. */
  currentImageName: string | null;
  /** Epoch ms when the annotator entered this image. */
  imageEnterTimeMs: number | null;
  /** Count loaded from disk on image entry. */
  loadedAnnotationCount: number;
  /** Number of new annotations created during this image visit. */
  annotationsAddedDelta: number;
  /** Signature of the last flushed event to avoid duplicate emits. */
  lastFlushedKey: string | null;
}

interface PerImageStatusState {
  /** Loaded from backend on dataset select. */
  byImage: Record<string, ImageStatus>;
  /** Filter applied in top bar. */
  activeFilter: "all" | ImageStatus;
}

export interface AgentActivity {
  /** Increments per event so effects can react to the latest one. */
  seq: number;
  panel: string;
  eventType: string;
  data: Record<string, unknown>;
}

export interface Toast {
  id: number;
  message: string;
  level: "error" | "info" | "success";
}

export interface AppState {
  /** Server-synchronized state (mirrors backend GuiState). */
  gui: GuiState;
  wsStatus: "disconnected" | "connecting" | "connected" | "error";
  /** Highest backend state version applied — used to drop stale snapshot replays. */
  wsVersion: number;

  /** Canvas-local draft state (not persisted until save). */
  canvas: CanvasState;

  /** Review tab derived state. */
  review: ReviewTabState;

  /** Class registry + per-image status + annotate ui. */
  classes: ClassesState;
  imageStatus: PerImageStatusState;
  annotateUi: AnnotateUiState;
  sessionTracking: SessionTrackingState;

  /** Last panel event pushed by the MCP agent (via /ws/panel subscription). */
  agentActivity: AgentActivity | null;
  pushAgentActivity: (panel: string, eventType: string, data: Record<string, unknown>) => void;

  /** Transient user-facing notifications (API failures, etc.). */
  toasts: Toast[];
  pushToast: (message: string, level?: Toast["level"]) => void;
  dismissToast: (id: number) => void;

  /** Agent terminal rail visibility (persisted; the server session survives closing). */
  terminalOpen: boolean;
  setTerminalOpen: (open: boolean) => void;

  /** Setters. */
  setGui: (next: GuiState) => void;
  patchGui: (partial: Partial<GuiState>) => void;
  /** Clear the dataset selection, returning the GUI to the project front door. */
  clearDataset: () => void;
  /** Persist the current dataset's UI state (position/filters) before switching away. Call
   *  synchronously before the async /dataset/select so a broadcast can't move it mid-await. */
  saveCurrentDatasetUi: () => void;
  /** Adopt a new dataset selection, restoring its saved position/filters when the user has been
   *  here before (else the selection's own values). Establishes the new identity locally so a
   *  same-identity backend snapshot keeps the restored index instead of resetting it to 0. */
  applyRestoredDataset: (sel: DatasetSelection) => void;
  /**
   * Apply a backend state snapshot with ownership-aware merge (NOT a wholesale
   * replace, which used to clobber unsaved edits, the active tab, and the scroll
   * position). Backend owns the dataset selection; the browser owns
   * navigation/view/mode/class/review-filter state and keeps its own copy.
   */
  mergeSnapshot: (state: GuiState, version: number | null) => void;
  setWsStatus: (s: AppState["wsStatus"]) => void;
  setActiveTab: (tab: TabName) => void;
  setView: (view: ViewState) => void;
  setMode: (mode: "box" | "polygon") => void;
  setActiveClass: (cid: number) => void;
  setPredReference: (p: PredictionReference | null) => void;

  /** Class helpers. */
  setClasses: (list: ClassEntry[]) => void;
  upsertClass: (entry: ClassEntry) => void;
  removeClass: (id: number) => void;
  classColor: (cid: number) => string;
  className: (cid: number) => string;

  /** Per-image status helpers. */
  setImageStatuses: (byImage: Record<string, ImageStatus>) => void;
  setImageStatus: (image: string, status: ImageStatus) => void;
  setStatusFilter: (filter: "all" | ImageStatus) => void;

  /** Annotate UI flags. */
  setVisible: (v: boolean) => void;
  setSnap: (v: boolean) => void;
  setStream: (v: boolean) => void;
  setHoveredPolygon: (idx: number | null) => void;
  setDraggingVertex: (v: [number, number] | null) => void;

  /** Per-image session telemetry helpers. */
  startImageSessionTracking: (
    imageName: string,
    loadedAnnotationCount: number,
    imageEnterTimeMs?: number,
  ) => void;
  incrementAnnotationsAdded: (delta?: number) => void;
  markSessionFlushed: (key: string) => void;
  clearSessionTracking: () => void;

  /** Canvas helpers. */
  loadLabelsIntoCanvas: (labels: ImageLabels) => void;
  clearCanvas: () => void;
  pushUndo: () => void;
  undo: () => void;
  redo: () => void;
  addBox: (box: Box) => void;
  updateBox: (idx: number, box: Box) => void;
  /** No-undo box mutation for a live resize/move drag — undo is captured once at drag
   *  start (see updateBox for the undo-pushing variant). */
  dragBox: (idx: number, box: Box) => void;
  deleteBox: (idx: number) => void;
  addPolygon: (polygon: PolygonShape) => void;
  updatePolygon: (idx: number, polygon: PolygonShape) => void;
  /** Move a single polygon vertex WITHOUT pushing an undo snapshot. Used during a
   *  live vertex drag (undo is captured once at drag start) so a 50px drag doesn't
   *  push dozens of snapshots and evict the whole 30-entry undo history. */
  dragVertex: (polygonIdx: number, vertexIdx: number, point: [number, number]) => void;
  deletePolygon: (idx: number) => void;
  selectPolygon: (idx: number | null) => void;
  setCurrentPolygon: (pts: [number, number][]) => void;
  commitCurrentPolygon: () => boolean;
  markClean: () => void;

  /** Review helpers. */
  setMatches: (matches: MatchesResponse | null) => void;
  setReviewLoading: (loading: boolean) => void;
  setReviewDetectionIdx: (idx: number) => void;
  setReviewFocusIdx: (idx: number | null) => void;
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
  wsVersion: 0,
  canvas: EMPTY_CANVAS,
  review: { matches: null, loading: false, focusDetectionIdx: null },
  classes: { list: [], loaded: false },
  imageStatus: { byImage: {}, activeFilter: "all" },
  annotateUi: {
    visible: true,
    snap: false,
    stream: false,
    hoveredPolygonIdx: null,
    draggingVertex: null,
  },
  sessionTracking: EMPTY_SESSION_TRACKING,
  agentActivity: null,
  toasts: [],

  pushAgentActivity: (panel, eventType, data) =>
    set((s) => ({
      agentActivity: { seq: (s.agentActivity?.seq ?? 0) + 1, panel, eventType, data },
    })),

  pushToast: (message, level = "error") =>
    set((s) => {
      const id = (s.toasts[s.toasts.length - 1]?.id ?? 0) + 1;
      // Cap the stack so a failing poll can't flood the screen.
      return { toasts: [...s.toasts, { id, message, level }].slice(-4) };
    }),
  dismissToast: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),

  // Default open — the rail is the breeder's front door to the agent. Guarded:
  // this runs at module init, and blocked-storage browsers throw on access.
  terminalOpen: (() => {
    try {
      return localStorage.getItem("tcip.terminal_open") !== "0";
    } catch {
      return true;
    }
  })(),
  setTerminalOpen: (terminalOpen) => {
    try {
      localStorage.setItem("tcip.terminal_open", terminalOpen ? "1" : "0");
    } catch {
      /* preference just won't persist */
    }
    set({ terminalOpen });
  },

  setGui: (next) => set({ gui: next }),
  patchGui: (partial) => set((s) => ({ gui: { ...s.gui, ...partial } })),
  clearDataset: () => set((s) => ({ gui: { ...s.gui, dataset: DEFAULT_DATASET } })),

  saveCurrentDatasetUi: () => {
    const s = get();
    const key = datasetKey(s.gui.dataset);
    if (!key) return;
    saveDatasetUi(key, {
      index: s.gui.dataset.current_image_index,
      review: s.gui.review,
      statusFilter: s.imageStatus.activeFilter,
    });
  },

  applyRestoredDataset: (sel) =>
    set((s) => {
      const key = datasetKey(sel);
      const restored = key ? loadDatasetUi(key) : null;
      const index =
        restored && sel.image_list.length
          ? Math.max(0, Math.min(restored.index, sel.image_list.length - 1))
          : sel.current_image_index;
      return {
        gui: {
          ...s.gui,
          dataset: { ...sel, current_image_index: index },
          review: restored?.review ?? s.gui.review,
        },
        imageStatus: {
          ...s.imageStatus,
          activeFilter: restored?.statusFilter ?? s.imageStatus.activeFilter,
        },
      };
    }),

  mergeSnapshot: (incoming, version) =>
    set((s) => {
      // Drop a stale replay (older backend version than one already applied) —
      // e.g. a reconnecting socket resending an old snapshot.
      if (version != null && version < s.wsVersion) return s;
      const nextVersion = version != null ? Math.max(s.wsVersion, version) : s.wsVersion;
      const local = s.gui;
      const inDs = incoming.dataset;

      // A null/empty backend dataset must never clobber a populated client one:
      // this is what lets the browser survive a backend restart (the restarted
      // backend broadcasts an empty state before it knows the project).
      if (!inDs || !inDs.dataset_root) {
        return { wsVersion: nextVersion };
      }

      const identityChanged =
        inDs.dataset_root !== local.dataset.dataset_root ||
        inDs.date !== local.dataset.date ||
        inDs.annotation_type !== local.dataset.annotation_type;

      // Boot hydration: the browser has no dataset yet (fresh load / project open), so
      // there is no local state to protect — adopt the persisted tab/mode/filters too,
      // so a refresh resumes where the session left off instead of resetting to Annotate.
      if (!local.dataset.dataset_root) {
        return { gui: { ...incoming, pred_reference: null }, wsVersion: nextVersion };
      }

      if (identityChanged) {
        // New dataset selection: adopt it wholesale (including its index) and drop
        // any stale prediction-reference overlay. The active tab stays put.
        return { gui: { ...local, dataset: inDs, pred_reference: null }, wsVersion: nextVersion };
      }
      // Same dataset: accept backend-owned dataset fields (e.g. a changed model's
      // prediction dirs) but keep the user's navigation position AND the local
      // image_list reference (same identity => same list; reusing the ref avoids
      // spuriously re-firing effects keyed on it, like class/status hydration).
      // Everything else (active_tab / mode / active_class / view / review /
      // pred_reference) is client-owned — keep local.
      return {
        gui: {
          ...local,
          dataset: {
            ...inDs,
            image_list: local.dataset.image_list,
            current_image_index: local.dataset.current_image_index,
          },
        },
        wsVersion: nextVersion,
      };
    }),

  setWsStatus: (wsStatus) => set({ wsStatus }),
  setActiveTab: (active_tab) => set((s) => ({ gui: { ...s.gui, active_tab } })),
  setView: (view) => set((s) => ({ gui: { ...s.gui, view } })),
  setMode: (mode) => set((s) => ({ gui: { ...s.gui, mode } })),
  setActiveClass: (active_class) => set((s) => ({ gui: { ...s.gui, active_class } })),
  setPredReference: (pred_reference) => set((s) => ({ gui: { ...s.gui, pred_reference } })),

  setClasses: (list) =>
    set(() => ({
      classes: { list: list.slice().sort((a, b) => a.id - b.id), loaded: true },
    })),

  upsertClass: (entry) =>
    set((s) => {
      const map = new Map(s.classes.list.map((c) => [c.id, c]));
      map.set(entry.id, entry);
      const list = Array.from(map.values()).sort((a, b) => a.id - b.id);
      return { classes: { list, loaded: true } };
    }),

  removeClass: (id) =>
    set((s) => ({
      classes: {
        list: s.classes.list.filter((c) => c.id !== id),
        loaded: true,
      },
    })),

  classColor: (cid) => {
    const entry = get().classes.list.find((c) => c.id === cid);
    return entry?.color ?? autoColor(cid);
  },

  className: (cid) => {
    const entry = get().classes.list.find((c) => c.id === cid);
    return entry?.name ?? `class_${cid}`;
  },

  setImageStatuses: (byImage) => set(() => ({ imageStatus: { byImage, activeFilter: "all" } })),
  setImageStatus: (image, status) =>
    set((s) => ({
      imageStatus: {
        ...s.imageStatus,
        byImage: { ...s.imageStatus.byImage, [image]: status },
      },
    })),
  setStatusFilter: (activeFilter) =>
    set((s) => ({ imageStatus: { ...s.imageStatus, activeFilter } })),

  setVisible: (visible) => set((s) => ({ annotateUi: { ...s.annotateUi, visible } })),
  setSnap: (snap) => set((s) => ({ annotateUi: { ...s.annotateUi, snap } })),
  setStream: (stream) => set((s) => ({ annotateUi: { ...s.annotateUi, stream } })),
  setHoveredPolygon: (hoveredPolygonIdx) =>
    set((s) => ({ annotateUi: { ...s.annotateUi, hoveredPolygonIdx } })),
  setDraggingVertex: (draggingVertex) =>
    set((s) => ({ annotateUi: { ...s.annotateUi, draggingVertex } })),

  startImageSessionTracking: (imageName, loadedAnnotationCount, imageEnterTimeMs) =>
    set((s) => ({
      sessionTracking: {
        ...s.sessionTracking,
        currentImageName: imageName,
        imageEnterTimeMs: imageEnterTimeMs ?? Date.now(),
        loadedAnnotationCount: Math.max(0, loadedAnnotationCount),
        annotationsAddedDelta: 0,
        lastFlushedKey: null,
      },
    })),

  incrementAnnotationsAdded: (delta = 1) =>
    set((s) => {
      if (!s.sessionTracking.currentImageName) return s;
      return {
        sessionTracking: {
          ...s.sessionTracking,
          annotationsAddedDelta: s.sessionTracking.annotationsAddedDelta + Math.max(0, delta),
        },
      };
    }),

  markSessionFlushed: (key) =>
    set((s) => ({
      sessionTracking: {
        ...s.sessionTracking,
        lastFlushedKey: key,
      },
    })),

  clearSessionTracking: () =>
    set((s) => ({
      sessionTracking: {
        ...s.sessionTracking,
        currentImageName: null,
        imageEnterTimeMs: null,
        loadedAnnotationCount: 0,
        annotationsAddedDelta: 0,
      },
    })),

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
        loadedImagePath: labels.image_path || null,
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

  dragVertex: (polygonIdx, vertexIdx, point) =>
    set((s) => {
      const poly = s.canvas.polygons[polygonIdx];
      if (!poly) return s;
      const pts = poly.points.slice();
      pts[vertexIdx] = point;
      const next = s.canvas.polygons.slice();
      next[polygonIdx] = { ...poly, points: pts };
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
      return { canvas: { ...s.canvas, polygons: polys, selectedPolygonIdx: sel, dirty: true } };
    });
  },

  selectPolygon: (selectedPolygonIdx) =>
    set((s) => ({ canvas: { ...s.canvas, selectedPolygonIdx } })),

  setCurrentPolygon: (pts) => set((s) => ({ canvas: { ...s.canvas, currentPolygon: pts } })),

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
  setReviewFocusIdx: (idx) => set((s) => ({ review: { ...s.review, focusDetectionIdx: idx } })),

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

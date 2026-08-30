import { create } from "zustand";

import { subjectColor, type AttributeDef, type ImageStatus, type Registry } from "@/api/classes";
import type { ActionPayload } from "@/api/types.generated";
import type { BandSelection } from "@/lib/bandSelection";
import {
  datasetKey,
  loadDatasetUi,
  loadLastTab,
  recordLastTab,
  saveDatasetUi,
} from "@/lib/datasetUiState";
import type {
  Annotation,
  Box,
  DatasetSelection,
  Detection,
  GuiState,
  ImageLabels,
  MatchesResponse,
  Mode,
  PointShape,
  PolygonShape,
  ReviewFilters,
  ReviewImageStatus,
  ReviewStatusFilter,
  TabName,
  ViewState,
} from "@/store/types";

const DEFAULT_REVIEW: ReviewFilters = {
  iou_threshold: 0.5,
  conf_threshold: 0.25,
  filter_type: "all",
  filter_class: "all",
  detection_idx: 0,
};

const DEFAULT_DATASET: DatasetSelection = {
  project_root: null,
  dataset_root: null,
  subject: null,
  date: null,
  image_list: [],
  current_image_index: 0,
  images_dir: null,
  annotations_dir: null,
  predictions_dir: null,
};

const DEFAULT_STATE: GuiState = {
  active_tab: "annotate",
  dataset: DEFAULT_DATASET,
  view: { scale: 1, offset_x: 0, offset_y: 0 },
  mode: "box",
  active_subject: null,
  review: DEFAULT_REVIEW,
};

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
  // True when the canvas content differs from the last save (compared by content, so a
  // net-zero edit like draw-then-delete is clean, not "changed").
  dirty: boolean;
  // Serialized content of the last save/load: the baseline dirty is computed against.
  savedSignature: string;
  // Which image's labels the canvas holds: status writes must not read shapes that
  // still belong to the previous image (or a failed load) mid-flip.
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

const EMPTY_SESSION_TRACKING: SessionTrackingState = {
  currentImageName: null,
  imageEnterTimeMs: null,
  annotationsAddedDelta: 0,
  lastFlushedKey: null,
};

// Scoped to one dataset selection's own batch fetch; a switch to a different dataset must not
// carry a prior dataset's review-status facts forward.
const DEFAULT_REVIEW_STATUS: ReviewImageStatusState = {
  byImage: {},
  hasDetections: {},
  unreadable: [],
  activeFilter: "all",
};

/** True when two selections name a different (dataset_root, date, subject): the identity a
 *  review-status fetch is scoped to, so a fact fetched for one never gates navigation in the
 *  other. */
function datasetIdentityChanged(
  a: Pick<DatasetSelection, "dataset_root" | "date" | "subject">,
  b: Pick<DatasetSelection, "dataset_root" | "date" | "subject">,
): boolean {
  return a.dataset_root !== b.dataset_root || a.date !== b.date || a.subject !== b.subject;
}

interface BandSelectionState {
  /** The breeder's chosen band composite for each distinct band set this session has seen, keyed
   *  by bandSetSignature (the image's band names, in order). Absent means no change has been made
   *  for that set yet; useBandSelection is the one reader/writer, never the map directly. */
  byBandSet: Record<string, BandSelection>;
}

interface ReviewTabState {
  matches: MatchesResponse | null;
  loading: boolean;
  // One-shot: the detection index a `review_focus` command asked to center on. The reload
  // effect consumes it once (else it always jumps to first-unreviewed, dropping the agent's
  // "look at detection N" request).
  focusDetectionIdx: number | null;
  // Bumped by a review_focus command to force a matches refetch when image + paths are unchanged.
  refetchNonce: number;
}

interface RegistryState {
  /** The dataset's nested subject registry (subject -> {description?, attributes?}). No integer
   *  ids, no colours: colour is GUI-local (see subjectColor). Source of truth for the subject
   *  picker and per-instance attribute editing. */
  subjects: Registry;
  /** Set after the first successful load for the current dataset. */
  loaded: boolean;
  /** The stored registry's compare-and-set token as of the last load or save, or null when
   *  nothing has been saved yet. Carried back into the next save so a stale save is refused
   *  instead of silently overwriting a registry this browser never saw. */
  version: string | null;
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
  /** Active vertex drag: [polygonIdx, ringIdx, vertexIdx]; a vertex belongs to one ring of one
   *  polygon, so a multi-ring shape's second ring is addressable rather than uneditable. */
  draggingVertex: [number, number, number] | null;
}

interface SessionTrackingState {
  /** Active image for per-image annotation timing. */
  currentImageName: string | null;
  /** Epoch ms when the annotator entered this image. */
  imageEnterTimeMs: number | null;
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
  /** Confirmed names (complete/negative) whose derived token disagrees with the stored one: the
   *  label file changed since a human finished the image, so it needs a fresh look rather than a
   *  silent rewrite. Populated by the hydrate reconcile, cleared name-by-name as each is written. */
  staleMarks: string[];
}

interface ReviewImageStatusState {
  /** Per-image review completion status, batch-fetched from the ReviewEngine on dataset entry
   *  and kept live as verdicts land (untouched images default to "not_started"). */
  byImage: Record<string, ReviewImageStatus>;
  /** Whether each image stays reachable in Review navigation: true when it has anything to
   *  review, when its label document could not be read (stays navigable so the breeder can see
   *  why), or when the batch fetch hasn't resolved yet; false only for a readable image with
   *  zero detections, which Review navigation skips. */
  hasDetections: Record<string, boolean>;
  /** Label document paths (GT or prediction) that would not read, from the same batch fetch. */
  unreadable: string[];
  /** Image-level Reviewed/Unreviewed navigation filter for the Review tab. */
  activeFilter: ReviewStatusFilter;
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

/** A note the agent pushed for one panel, carrying the id of the event that delivered it. */
export interface Banner {
  id: string;
  text: string;
}

interface BannerState {
  /** The latest banner per panel; a panel with no entry has none. */
  byPanel: Record<string, Banner | null>;
  /** Banner event ids the human closed. The backend replays its whole per-panel event ring on
   *  every reconnect, so a dismissal has to be keyed on the event, not just cleared: without
   *  this the same note reappears on the next reconnect or reload. */
  dismissed: Set<string>;
}

export interface AppState {
  /** Server-synchronized state (mirrors backend GuiState). */
  gui: GuiState;
  wsStatus: "disconnected" | "connecting" | "connected" | "error";
  /** Highest backend state version applied; used to drop stale snapshot replays. */
  wsVersion: number;

  /** Canvas-local draft state (not persisted until save). */
  canvas: CanvasState;

  /** Review tab derived state. */
  review: ReviewTabState;

  /** Band-composite selection, held per band-set signature; see useBandSelection. */
  bandSelection: BandSelectionState;
  setBandSelectionFor: (signature: string, selection: BandSelection) => void;

  /** Dataset subject registry + per-image status + annotate ui. */
  registry: RegistryState;
  imageStatus: PerImageStatusState;
  reviewStatus: ReviewImageStatusState;
  annotateUi: AnnotateUiState;
  sessionTracking: SessionTrackingState;

  /** Last panel event pushed by the MCP agent (via /ws/panel subscription). */
  agentActivity: AgentActivity | null;
  pushAgentActivity: (panel: string, eventType: string, data: Record<string, unknown>) => void;

  /** Transient user-facing notifications (API failures, etc.). */
  toasts: Toast[];
  pushToast: (message: string, level?: Toast["level"]) => void;
  dismissToast: (id: number) => void;

  /** Agent-pushed per-tab notes (see TabBanner). Free text the agent chose, never a computed
   *  verdict about the data. */
  banners: BannerState;
  pushBanner: (panel: string, id: string, text: string) => void;
  dismissBanner: (id: string) => void;

  /** Agent terminal rail visibility (persisted; the server session survives closing). */
  terminalOpen: boolean;
  setTerminalOpen: (open: boolean) => void;

  /** A request staged for the agent terminal: any component that would otherwise make the
   *  breeder hand-author a CV/ML decision (a training config, a search space, a calibration
   *  trigger) calls `sendToAgentTerminal` instead of exposing the raw control. Opens the rail
   *  and stages the text here; TerminalRail sends it as terminal input once its socket is
   *  open, then clears it so it never resends. */
  pendingTerminalMessage: string | null;
  sendToAgentTerminal: (text: string) => void;
  clearPendingTerminalMessage: () => void;

  /** Current annotator/reviewer identity (persisted). Stamped as created_by/accepted_by on
   * everything this person authors; set on the workspace page, shown in the status bar. */
  user: string;
  setUser: (user: string) => void;

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
   * Apply a backend state snapshot with ownership-aware merge, not a wholesale
   * replace: a wholesale replace would clobber unsaved edits, the active tab, and
   * the scroll position. Backend owns the dataset selection; the browser owns
   * navigation/view/mode/subject/review-filter state and keeps its own copy.
   */
  mergeSnapshot: (state: GuiState, version: number | null) => void;
  setWsStatus: (s: AppState["wsStatus"]) => void;
  setActiveTab: (tab: TabName) => void;
  setView: (view: ViewState) => void;
  setMode: (mode: Mode) => void;
  setActiveSubject: (subject: string | null) => void;

  /** Registry helpers. ``version`` is the stored registry's compare-and-set token to carry into
   *  the next save; omitted (or null) for a caller with no version to assert, such as a test
   *  seeding the registry directly. */
  setRegistry: (subjects: Registry, version?: string | null) => void;
  subjectNames: () => string[];
  subjectAttributes: (subject: string | null) => Record<string, AttributeDef>;

  /** Per-image status helpers. */
  setImageStatuses: (byImage: Record<string, ImageStatus>, staleMarks?: string[]) => void;
  setImageStatus: (image: string, status: ImageStatus) => void;
  setStatusFilter: (filter: "all" | ImageStatus) => void;
  /** Drops every stale mark, e.g. on dataset selection so a prior dataset's marks are never read
   *  against a same-named image in the newly selected one. */
  clearStaleMarks: () => void;
  /** Re-adds a name to `staleMarks`, for a re-confirm write that failed to persist: the mark it
   *  was about to clear still describes reality, since nothing was actually confirmed. */
  markStale: (image: string) => void;

  /** Review-status helpers (image-level Reviewed/Unreviewed navigation). */
  setReviewImageStatuses: (
    byImage: Record<string, ReviewImageStatus>,
    hasDetections: Record<string, boolean>,
    unreadable?: string[],
  ) => void;
  setReviewImageStatus: (image: string, status: ReviewImageStatus) => void;
  setReviewStatusFilter: (filter: ReviewStatusFilter) => void;

  /** Annotate UI flags. */
  setVisible: (v: boolean) => void;
  setSnap: (v: boolean) => void;
  setStream: (v: boolean) => void;
  setHoveredPolygon: (idx: number | null) => void;
  setDraggingVertex: (v: [number, number, number] | null) => void;

  /** Per-image session telemetry helpers. */
  startImageSessionTracking: (imageName: string, imageEnterTimeMs?: number) => void;
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

  /** Review helpers. */
  setMatches: (matches: MatchesResponse | null) => void;
  setReviewLoading: (loading: boolean) => void;
  setReviewDetectionIdx: (idx: number) => void;
  setReviewFocusIdx: (idx: number | null) => void;
  /** Force a matches refetch even when image/paths are unchanged (re-focus on the open image). */
  bumpReviewRefetch: () => void;
  markDetectionReviewed: (idx: number, action: Exclude<ActionPayload["action"], "swept">) => void;
}

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

export const useStore = create<AppState>()((set, get) => ({
  gui: DEFAULT_STATE,
  wsStatus: "disconnected",
  wsVersion: 0,
  canvas: EMPTY_CANVAS,
  review: { matches: null, loading: false, focusDetectionIdx: null, refetchNonce: 0 },
  bandSelection: { byBandSet: {} },
  setBandSelectionFor: (signature, selection) =>
    set((s) => ({
      bandSelection: { byBandSet: { ...s.bandSelection.byBandSet, [signature]: selection } },
    })),
  registry: { subjects: {}, loaded: false, version: null },
  imageStatus: { byImage: {}, activeFilter: "all", staleMarks: [] },
  reviewStatus: DEFAULT_REVIEW_STATUS,
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

  banners: { byPanel: {}, dismissed: new Set<string>() },
  pushBanner: (panel, id, text) =>
    set((s) => ({
      banners: { ...s.banners, byPanel: { ...s.banners.byPanel, [panel]: { id, text } } },
    })),
  dismissBanner: (id) =>
    set((s) => ({
      banners: { ...s.banners, dismissed: new Set(s.banners.dismissed).add(id) },
    })),

  // Always closed on open, never restored across sessions: the canvas is the front door;
  // the rail opens on demand (the toggle, or an agent send via sendToAgentTerminal).
  terminalOpen: false,
  setTerminalOpen: (terminalOpen) => set({ terminalOpen }),

  pendingTerminalMessage: null,
  sendToAgentTerminal: (text) => {
    get().setTerminalOpen(true);
    set({ pendingTerminalMessage: text });
  },
  clearPendingTerminalMessage: () => set({ pendingTerminalMessage: null }),

  user: (() => {
    try {
      return localStorage.getItem("tcip.user") ?? "";
    } catch {
      return "";
    }
  })(),
  setUser: (user) => {
    try {
      localStorage.setItem("tcip.user", user);
    } catch {
      /* preference just won't persist */
    }
    set({ user });
  },

  setGui: (next) => set({ gui: next }),
  patchGui: (partial) => set((s) => ({ gui: { ...s.gui, ...partial } })),
  clearDataset: () =>
    set((s) => ({
      gui: { ...s.gui, dataset: DEFAULT_DATASET },
      reviewStatus: DEFAULT_REVIEW_STATUS,
    })),

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
          // Land on the project's last-used tab; a project never opened before gets Annotate.
          active_tab: loadLastTab(sel.project_root) ?? "annotate",
          dataset: { ...sel, current_image_index: index },
          review: restored?.review ?? s.gui.review,
        },
        imageStatus: {
          ...s.imageStatus,
          activeFilter: restored?.statusFilter ?? s.imageStatus.activeFilter,
          // A stale mark names an image in the dataset it was computed for; carrying it into a
          // newly selected dataset could flag a same-named image that was never checked.
          staleMarks: [],
        },
        reviewStatus: datasetIdentityChanged(s.gui.dataset, sel)
          ? DEFAULT_REVIEW_STATUS
          : s.reviewStatus,
      };
    }),

  mergeSnapshot: (incoming, version) =>
    set((s) => {
      // Drop a stale replay (older backend version than one already applied),
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

      const identityChanged = datasetIdentityChanged(inDs, local.dataset);

      // Boot hydration: no local dataset to protect, so adopt the persisted mode/filters/position;
      // the tab is the client's per-project record (backend active_tab only moves on agent focus).
      if (!local.dataset.dataset_root) {
        return {
          gui: {
            ...incoming,
            active_tab: loadLastTab(incoming.dataset.project_root) ?? "annotate",
            active_subject: incoming.active_subject ?? null,
          },
          reviewStatus: DEFAULT_REVIEW_STATUS,
          wsVersion: nextVersion,
        };
      }

      if (identityChanged) {
        // New dataset selection: adopt it wholesale (including its index) and drop
        // the stale reviewStatus. The active tab stays put.
        return {
          gui: { ...local, dataset: inDs },
          reviewStatus: DEFAULT_REVIEW_STATUS,
          wsVersion: nextVersion,
        };
      }
      // Same dataset: accept backend-owned dataset fields (e.g. a changed model's
      // prediction dir) but keep the user's navigation position and the local
      // image_list reference (same identity => same list; reusing the ref avoids
      // spuriously re-firing effects keyed on it, like registry/status hydration).
      // Everything else (active_tab / mode / active_subject / view / review) is
      // client-owned; keep local.
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
  setActiveTab: (active_tab) =>
    set((s) => {
      // Write-through so the next open of this project resumes on the tab last worked in.
      if (s.gui.dataset.project_root) recordLastTab(s.gui.dataset.project_root, active_tab);
      return { gui: { ...s.gui, active_tab } };
    }),
  setView: (view) => set((s) => ({ gui: { ...s.gui, view } })),
  setMode: (mode) => set((s) => ({ gui: { ...s.gui, mode } })),
  setActiveSubject: (active_subject) => set((s) => ({ gui: { ...s.gui, active_subject } })),

  setRegistry: (subjects, version = null) =>
    set(() => ({ registry: { subjects, loaded: true, version } })),

  subjectNames: () => Object.keys(get().registry.subjects),

  subjectAttributes: (subject) => {
    if (!subject) return {};
    return get().registry.subjects[subject]?.attributes ?? {};
  },

  setImageStatuses: (byImage, staleMarks = []) =>
    set(() => ({ imageStatus: { byImage, activeFilter: "all", staleMarks } })),
  setImageStatus: (image, status) =>
    set((s) => ({
      imageStatus: {
        ...s.imageStatus,
        byImage: { ...s.imageStatus.byImage, [image]: status },
        staleMarks: s.imageStatus.staleMarks.filter((name) => name !== image),
      },
    })),
  setStatusFilter: (activeFilter) =>
    set((s) => ({ imageStatus: { ...s.imageStatus, activeFilter } })),
  clearStaleMarks: () => set((s) => ({ imageStatus: { ...s.imageStatus, staleMarks: [] } })),
  markStale: (image) =>
    set((s) => ({
      imageStatus: {
        ...s.imageStatus,
        staleMarks: s.imageStatus.staleMarks.includes(image)
          ? s.imageStatus.staleMarks
          : [...s.imageStatus.staleMarks, image].sort(),
      },
    })),

  setReviewImageStatuses: (byImage, hasDetections, unreadable = []) =>
    set((s) => ({ reviewStatus: { ...s.reviewStatus, byImage, hasDetections, unreadable } })),
  setReviewImageStatus: (image, status) =>
    set((s) => ({
      reviewStatus: {
        ...s.reviewStatus,
        byImage: { ...s.reviewStatus.byImage, [image]: status },
      },
    })),
  setReviewStatusFilter: (activeFilter) =>
    set((s) => ({ reviewStatus: { ...s.reviewStatus, activeFilter } })),

  setVisible: (visible) => set((s) => ({ annotateUi: { ...s.annotateUi, visible } })),
  setSnap: (snap) => set((s) => ({ annotateUi: { ...s.annotateUi, snap } })),
  setStream: (stream) => set((s) => ({ annotateUi: { ...s.annotateUi, stream } })),
  setHoveredPolygon: (hoveredPolygonIdx) =>
    set((s) => ({ annotateUi: { ...s.annotateUi, hoveredPolygonIdx } })),
  setDraggingVertex: (draggingVertex) =>
    set((s) => ({ annotateUi: { ...s.annotateUi, draggingVertex } })),

  startImageSessionTracking: (imageName, imageEnterTimeMs) =>
    set((s) => ({
      sessionTracking: {
        ...s.sessionTracking,
        currentImageName: imageName,
        imageEnterTimeMs: imageEnterTimeMs ?? Date.now(),
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
        annotationsAddedDelta: 0,
      },
    })),

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

  setMatches: (matches) => set((s) => ({ review: { ...s.review, matches } })),
  setReviewLoading: (loading) => set((s) => ({ review: { ...s.review, loading } })),
  setReviewDetectionIdx: (idx) =>
    set((s) => ({
      gui: { ...s.gui, review: { ...s.gui.review, detection_idx: idx } },
    })),
  setReviewFocusIdx: (idx) => set((s) => ({ review: { ...s.review, focusDetectionIdx: idx } })),
  bumpReviewRefetch: () =>
    set((s) => ({ review: { ...s.review, refetchNonce: s.review.refetchNonce + 1 } })),

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

// Re-export so callers derive a subject's colour from one source (GUI-local, name-hashed).
export { subjectColor };

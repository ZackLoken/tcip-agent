import type { StateCreator } from "zustand";

import { setSubjectColorRegistry } from "@/api/classes";
import type { AttributeDef, ImageStatus, Registry } from "@/api/classes";
import type { AppState } from "@/store/appState";
import type { ReviewImageStatus, ReviewStatusFilter } from "@/store/types";

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
  /** Cut toggle (polygon mode): arms the two-click split gesture. */
  cut: boolean;
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
  /** Confirmed names (complete/negative) needing a fresh look: the label file's content changed
   *  since a human finished the image, or the subject's attribute schema did. Populated by the
   *  hydrate reconcile (content) unioned with the status route's stale_definition (schema),
   *  cleared name-by-name as each is written. */
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

const EMPTY_SESSION_TRACKING: SessionTrackingState = {
  currentImageName: null,
  imageEnterTimeMs: null,
  annotationsAddedDelta: 0,
  lastFlushedKey: null,
};

// Scoped to one dataset selection's own batch fetch; a switch to a different dataset must not
// carry a prior dataset's review-status facts forward.
export const DEFAULT_REVIEW_STATUS: ReviewImageStatusState = {
  byImage: {},
  hasDetections: {},
  unreadable: [],
  activeFilter: "all",
};

export interface RegistryStatusSlice {
  /** Dataset subject registry + per-image status + annotate ui. */
  registry: RegistryState;
  imageStatus: PerImageStatusState;
  reviewStatus: ReviewImageStatusState;
  annotateUi: AnnotateUiState;
  sessionTracking: SessionTrackingState;

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
  /** Re-adds a name to `staleMarks`: used both when a write's confirmation never reached the
   *  server and when it landed but its schema stamp did not, since either way the mark it was
   *  about to clear still describes reality. */
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
  setCut: (v: boolean) => void;
  setHoveredPolygon: (idx: number | null) => void;
  setDraggingVertex: (v: [number, number, number] | null) => void;

  /** Per-image session telemetry helpers. */
  startImageSessionTracking: (imageName: string, imageEnterTimeMs?: number) => void;
  incrementAnnotationsAdded: (delta?: number) => void;
  markSessionFlushed: (key: string) => void;
  clearSessionTracking: () => void;
}

export const createRegistryStatusSlice: StateCreator<AppState, [], [], RegistryStatusSlice> = (
  set,
  get,
) => ({
  registry: { subjects: {}, loaded: false, version: null },
  imageStatus: { byImage: {}, activeFilter: "all", staleMarks: [] },
  reviewStatus: DEFAULT_REVIEW_STATUS,
  annotateUi: {
    visible: true,
    snap: false,
    stream: false,
    cut: false,
    hoveredPolygonIdx: null,
    draggingVertex: null,
  },
  sessionTracking: EMPTY_SESSION_TRACKING,

  setRegistry: (subjects, version = null) => {
    setSubjectColorRegistry(Object.keys(subjects));
    set(() => ({ registry: { subjects, loaded: true, version } }));
  },

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
  setCut: (cut) => set((s) => ({ annotateUi: { ...s.annotateUi, cut } })),
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
});

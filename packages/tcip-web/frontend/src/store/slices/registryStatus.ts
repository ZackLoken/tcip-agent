import type { StateCreator } from "zustand";

import { setSubjectColorRegistry } from "@/api/subjects";
import type { AttributeDef, Registry } from "@/api/subjects";
import type { AppState } from "@/store/appState";

interface RegistryState {
  /** The dataset's nested subject registry (subject -> {description?, attributes?}). No integer
   *  ids, no colors: color is GUI-local (see subjectColor). Source of truth for the subject
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
  /** The proposal the agent pointed the person at, by its index in the bucket's document. */
  focusedProposal: number | null;
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

const EMPTY_SESSION_TRACKING: SessionTrackingState = {
  currentImageName: null,
  imageEnterTimeMs: null,
  annotationsAddedDelta: 0,
  lastFlushedKey: null,
};

export interface RegistryStatusSlice {
  /** Dataset subject registry + annotate ui + session telemetry. */
  registry: RegistryState;
  annotateUi: AnnotateUiState;
  sessionTracking: SessionTrackingState;

  /** Registry helpers. ``version`` is the stored registry's compare-and-set token to carry into
   *  the next save; omitted (or null) for a caller with no version to assert, such as a test
   *  seeding the registry directly. */
  setRegistry: (subjects: Registry, version?: string | null) => void;
  subjectNames: () => string[];
  subjectAttributes: (subject: string | null) => Record<string, AttributeDef>;

  /** Annotate UI flags. */
  setVisible: (v: boolean) => void;
  setSnap: (v: boolean) => void;
  setStream: (v: boolean) => void;
  setCut: (v: boolean) => void;
  setHoveredPolygon: (idx: number | null) => void;
  setDraggingVertex: (v: [number, number, number] | null) => void;
  setFocusedProposal: (index: number | null) => void;

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
  annotateUi: {
    visible: true,
    snap: false,
    stream: false,
    cut: false,
    hoveredPolygonIdx: null,
    draggingVertex: null,
    focusedProposal: null,
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

  setVisible: (visible) => set((s) => ({ annotateUi: { ...s.annotateUi, visible } })),
  setSnap: (snap) => set((s) => ({ annotateUi: { ...s.annotateUi, snap } })),
  setStream: (stream) => set((s) => ({ annotateUi: { ...s.annotateUi, stream } })),
  setCut: (cut) => set((s) => ({ annotateUi: { ...s.annotateUi, cut } })),
  setHoveredPolygon: (hoveredPolygonIdx) =>
    set((s) => ({ annotateUi: { ...s.annotateUi, hoveredPolygonIdx } })),
  setDraggingVertex: (draggingVertex) =>
    set((s) => ({ annotateUi: { ...s.annotateUi, draggingVertex } })),
  setFocusedProposal: (focusedProposal) =>
    set((s) => ({ annotateUi: { ...s.annotateUi, focusedProposal } })),

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

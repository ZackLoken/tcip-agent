import type { StateCreator } from "zustand";

import { setSubjectColorRegistry } from "@/api/subjects";
import type { AttributeDef, Registry } from "@/api/subjects";
import type { ImageEventPayload } from "@/api/types.generated";
import type { AppState } from "@/store/appState";

interface RegistryState {
  /** The dataset's nested subject registry (subject -> {description?, attributes?}), by name:
   *  no integer ids, no colors. */
  subjects: Registry;
  /** Set after the first successful load for the current dataset. */
  loaded: boolean;
  /** The stored registry's compare-and-set token as of the last load or save, or null when
   *  nothing has been saved yet. Carried back into the next save so a stale save is refused
   *  instead of silently overwriting a registry this browser never saw. */
  version: string | null;
  /** The subject names the dataset's labels hold, as the last load discovered them: a starting
   *  point for declaring subjects, never a registry. */
  discovered: string[];
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
  /** Whether a save during this image visit left the dataset subject a confirmed negative. */
  negativeMarked: boolean;
  /** The project the visit was opened under, which it keeps. */
  projectId: string | null;
}

const EMPTY_SESSION_TRACKING: SessionTrackingState = {
  currentImageName: null,
  imageEnterTimeMs: null,
  annotationsAddedDelta: 0,
  negativeMarked: false,
  projectId: null,
};

export interface RegistryStatusSlice {
  /** Dataset subject registry + annotate ui + session telemetry. */
  registry: RegistryState;
  annotateUi: AnnotateUiState;
  sessionTracking: SessionTrackingState;
  /** Finished image visits, each with its person, project and activity, not yet accepted by the
   *  backend. */
  heldContributions: ImageEventPayload[];

  /** Registry helpers. ``version`` is the stored registry's compare-and-set token to carry into
   *  the next save, null when none is asserted. ``discovered`` is what a load found in the
   *  labels. */
  setRegistry: (subjects: Registry, version?: string | null, discovered?: string[]) => void;
  subjectNames: () => string[];
  subjectAttributes: (subject: string | null) => Record<string, AttributeDef>;

  /** Annotate UI flags. */
  setVisible: (v: boolean) => void;
  setSnap: (v: boolean) => void;
  setStream: (v: boolean) => void;
  setCut: (v: boolean) => void;
  setHoveredPolygon: (idx: number | null) => void;
  setDraggingVertex: (v: [number, number, number] | null) => void;

  /** Per-image session telemetry helpers. A visit opens under the current person and project;
   *  closing it holds its contribution when a project was named, and drops it otherwise. */
  startImageSessionTracking: (imageName: string, imageEnterTimeMs?: number) => void;
  incrementAnnotationsAdded: (delta?: number) => void;
  markNegativeConfirmed: () => void;
  closeSessionInterval: () => void;
  retireContribution: (contribution: ImageEventPayload) => void;
}

export const createRegistryStatusSlice: StateCreator<AppState, [], [], RegistryStatusSlice> = (
  set,
  get,
) => ({
  registry: { subjects: {}, loaded: false, version: null, discovered: [] },
  annotateUi: {
    visible: true,
    snap: false,
    stream: false,
    cut: false,
    hoveredPolygonIdx: null,
    draggingVertex: null,
  },
  sessionTracking: EMPTY_SESSION_TRACKING,
  heldContributions: [],

  setRegistry: (subjects, version = null, discovered = []) => {
    setSubjectColorRegistry(Object.keys(subjects));
    set(() => ({ registry: { subjects, loaded: true, version, discovered } }));
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

  startImageSessionTracking: (imageName, imageEnterTimeMs) =>
    set((s) => ({
      sessionTracking: {
        currentImageName: imageName,
        imageEnterTimeMs: imageEnterTimeMs ?? Date.now(),
        annotationsAddedDelta: 0,
        negativeMarked: false,
        projectId: s.openProject?.id ?? null,
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

  markNegativeConfirmed: () =>
    set((s) =>
      s.sessionTracking.currentImageName
        ? { sessionTracking: { ...s.sessionTracking, negativeMarked: true } }
        : s,
    ),

  closeSessionInterval: () =>
    set((s) => {
      const t = s.sessionTracking;
      if (t.currentImageName === null || t.imageEnterTimeMs === null) return s;
      if (t.projectId === null) {
        return { sessionTracking: EMPTY_SESSION_TRACKING };
      }
      const contribution: ImageEventPayload = {
        image_name: t.currentImageName,
        seconds: Number((Math.max(0, Date.now() - t.imageEnterTimeMs) / 1000).toFixed(2)),
        annotations_added: t.annotationsAddedDelta,
        activity:
          t.annotationsAddedDelta > 0
            ? "new_annotation"
            : t.negativeMarked
              ? "negative_confirmation"
              : "review",
        user: s.user,
        project_id: t.projectId,
      };
      return {
        sessionTracking: EMPTY_SESSION_TRACKING,
        heldContributions: [...s.heldContributions, contribution],
      };
    }),

  retireContribution: (contribution) =>
    set((s) => ({ heldContributions: s.heldContributions.filter((c) => c !== contribution) })),
});

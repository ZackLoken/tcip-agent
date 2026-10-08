import type { StateCreator } from "zustand";

import { setSubjectColorRegistry } from "@/api/subjects";
import type { AttributeDef, Registry } from "@/api/subjects";
import type { ImageEventPayload, SessionRef } from "@/api/types.generated";
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
   *  polygon. */
  draggingVertex: [number, number, number] | null;
}

interface SessionTrackingState {
  /** Active image for per-image annotation timing. */
  currentImageName: string | null;
  /** Epoch ms the visit's clock last started: entering the image, or resuming after a pause;
   *  null while paused. */
  imageEnterTimeMs: number | null;
  /** Number of new annotations created since the visit's last contribution was held. */
  annotationsAddedDelta: number;
  /** Whether a save since the visit's last contribution was held left the dataset subject a
   *  confirmed negative. */
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
  /** Contributions of image visits, each a closed visit or a visit's part up to a pause, with
   *  its identity, person, project and activity, whose recording the backend has not
   *  acknowledged. */
  heldContributions: ImageEventPayload[];
  /** The held contributions sent at least once with no answer yet that settles whether they
   *  were recorded: one in flight, or one whose send failed or answered indeterminately. */
  unconfirmedContributions: ImageEventPayload[];
  /** The session named by the latest answer to a contribution of this page that named one, with
   *  its person: what the next visit of that person in that project names, and what the page
   *  names to end it; null until one is answered, and again once an end for it has been sent,
   *  whatever became of that send, or its project stops being the open one. */
  recordedSession: RecordedSession | null;

  /** Registry helpers. */
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
   *  closing it holds its contribution when a project was named, naming the recorded session
   *  when that is the same person's in the same project, and drops it otherwise. */
  startImageSessionTracking: (imageName: string, imageEnterTimeMs?: number) => void;
  incrementAnnotationsAdded: (delta?: number) => void;
  markNegativeConfirmed: () => void;
  closeSessionInterval: () => void;
  /** Hold the visit's contribution so far and keep its image open with its clock stopped, so
   *  what follows is counted from zero. */
  pauseSessionInterval: () => void;
  /** Restart the clock of a visit `pauseSessionInterval` stopped. */
  resumeSessionInterval: () => void;
  /** Close the visit open now and forget the recorded session, as `following` (null for none)
   *  becomes the open project; every held contribution of another project is retired and the
   *  person told in one notice per outcome: never sent, not recorded; sent with no settling
   *  answer, recording not confirmed. The live notice of unconfirmed contributions follows
   *  (`noticeUnconfirmed`). */
  leaveProjectSession: (following: string | null) => void;
  /** Retire `contribution` from the held and unconfirmed contributions; the live notice of
   *  unconfirmed contributions follows (`noticeUnconfirmed`). */
  retireContribution: (contribution: ImageEventPayload) => void;
  setRecordedSession: (session: RecordedSession | null) => void;
}

/** A session this page recorded, as the page names it to end it, and the person it is of. */
export interface RecordedSession extends SessionRef {
  user: string;
}

const UNCONFIRMED_CHANNEL = "unconfirmed-visits";

/** Refresh the notice of unconfirmed contributions from `s.unconfirmedContributions`, as a send
 *  fails and as contributions leave it: a standing notice is replaced to name every image now
 *  unconfirmed, or dismissed once none is; with `announce`, a notice is shown even when none
 *  stands. A send that only adds membership leaves the notice until its answer. */
export function noticeUnconfirmed(s: AppState, announce: boolean): void {
  const standing = s.toasts.find((t) => t.channel === UNCONFIRMED_CHANNEL);
  const images = s.unconfirmedContributions.map((c) => c.image_name).join(", ");
  if (!images) {
    if (standing) s.dismissToast(standing.id);
  } else if (standing || announce) {
    s.pushToast(
      `Could not confirm these visits were recorded; they are sent again with the next visit: ` +
        images,
      "error",
      UNCONFIRMED_CHANNEL,
    );
  }
}

/** Whether `list` holds the contribution `c`, by the identity it was minted with. */
export function holds(list: ImageEventPayload[], c: ImageEventPayload): boolean {
  return list.some((x) => x.contribution_id === c.contribution_id);
}

/** `s`'s held contributions with the open visit's contribution so far added after them, under
 *  an identity minted here that every send of it carries: none added when no visit is open, it
 *  names no project, or its clock is stopped (`pauseSessionInterval` already held its part) and
 *  nothing was counted since. Whether a contribution records anything is the backend's answer. */
function withVisitSoFar(s: AppState): ImageEventPayload[] {
  const t = s.sessionTracking;
  if (t.currentImageName === null || t.projectId === null) return s.heldContributions;
  if (t.imageEnterTimeMs === null && !t.annotationsAddedDelta && !t.negativeMarked) {
    return s.heldContributions;
  }
  const elapsed = t.imageEnterTimeMs === null ? 0 : Date.now() - t.imageEnterTimeMs;
  const seconds = Number((Math.max(0, elapsed) / 1000).toFixed(2));
  const recorded = s.recordedSession;
  return [
    ...s.heldContributions,
    {
      contribution_id: Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) =>
        b.toString(16).padStart(2, "0"),
      ).join(""),
      image_name: t.currentImageName,
      seconds,
      annotations_added: t.annotationsAddedDelta,
      activity:
        t.annotationsAddedDelta > 0
          ? "new_annotation"
          : t.negativeMarked
            ? "negative_confirmation"
            : "review",
      user: s.user,
      project_id: t.projectId,
      started:
        recorded?.project_id === t.projectId && recorded.user === s.user ? recorded.started : null,
    },
  ];
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
  unconfirmedContributions: [],
  recordedSession: null,

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
    set((s) => ({
      sessionTracking: EMPTY_SESSION_TRACKING,
      heldContributions: withVisitSoFar(s),
    })),

  pauseSessionInterval: () =>
    set((s) =>
      s.sessionTracking.currentImageName === null
        ? s
        : {
            sessionTracking: {
              ...s.sessionTracking,
              imageEnterTimeMs: null,
              annotationsAddedDelta: 0,
              negativeMarked: false,
            },
            heldContributions: withVisitSoFar(s),
          },
    ),

  resumeSessionInterval: () =>
    set((s) =>
      s.sessionTracking.currentImageName !== null && s.sessionTracking.imageEnterTimeMs === null
        ? { sessionTracking: { ...s.sessionTracking, imageEnterTimeMs: Date.now() } }
        : s,
    ),

  leaveProjectSession: (following) => {
    get().closeSessionInterval();
    const { heldContributions, unconfirmedContributions, pushToast } = get();
    const departed = heldContributions.filter((c) => c.project_id !== following);
    set({
      recordedSession: null,
      heldContributions: heldContributions.filter((c) => !holds(departed, c)),
      unconfirmedContributions: unconfirmedContributions.filter((c) => !holds(departed, c)),
    });
    noticeUnconfirmed(get(), false);
    const images = (sent: boolean) =>
      departed
        .filter((c) => holds(unconfirmedContributions, c) === sent)
        .map((c) => c.image_name)
        .join(", ");
    const [unsent, unconfirmed] = [images(false), images(true)];
    if (unsent) pushToast(`Not recorded, as their project was switched away from: ${unsent}`);
    if (unconfirmed) {
      pushToast(
        `Recording not confirmed before their project was switched away from: ${unconfirmed}`,
      );
    }
  },

  retireContribution: (contribution) => {
    set((s) => ({
      heldContributions: s.heldContributions.filter((c) => !holds([contribution], c)),
      unconfirmedContributions: s.unconfirmedContributions.filter((c) => !holds([contribution], c)),
    }));
    noticeUnconfirmed(get(), false);
  },

  setRecordedSession: (recordedSession) => set(() => ({ recordedSession })),
});

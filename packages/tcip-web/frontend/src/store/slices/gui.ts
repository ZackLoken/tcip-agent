import type { StateCreator } from "zustand";

import {
  datasetKey,
  loadDatasetUi,
  loadLastTab,
  recordLastTab,
  saveDatasetUi,
} from "@/lib/datasetUiState";
import type { AppState } from "@/store/appState";
import { DEFAULT_REVIEW_STATUS } from "@/store/slices/registryStatus";
import type {
  DatasetSelection,
  GuiState,
  Mode,
  ReviewFilters,
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

/** True when two selections name a different (dataset_root, date, subject): the identity a
 *  review-status fetch is scoped to, so a fact fetched for one never gates navigation in the
 *  other. */
function datasetIdentityChanged(
  a: Pick<DatasetSelection, "dataset_root" | "date" | "subject">,
  b: Pick<DatasetSelection, "dataset_root" | "date" | "subject">,
): boolean {
  return a.dataset_root !== b.dataset_root || a.date !== b.date || a.subject !== b.subject;
}

export interface GuiSlice {
  /** Server-synchronized state (mirrors backend GuiState). */
  gui: GuiState;
  wsStatus: "disconnected" | "connecting" | "connected" | "error";
  /** Highest backend state version applied; used to drop stale snapshot replays. */
  wsVersion: number;
  /** This backend process's launch identity, from the envelope; a change accepts a lower
   *  wsVersion instead of dropping it as a stale replay (a restarted backend's own snapshot). */
  wsEpoch: string | null;
  /** The canvas_open_binding generation the last select response or broadcast carried; the
   *  canvas pusher's write-authority token, adopted in the same store update as the dataset. */
  bindingGeneration: number | null;
  /** True while a dataset is selected but no binding_generation has been adopted yet (the
   *  canvas pusher's presence gate is blocking pushes): surfaces the condition rather than
   *  pushing silently against a binding that is not there to check the generation against. */
  canvasBindingMissing: boolean;

  setGui: (next: GuiState) => void;
  patchGui: (partial: Partial<GuiState>) => void;
  /** Clear the dataset selection, returning the GUI to the project front door. */
  clearDataset: () => void;
  /** Persist the current dataset's UI state (position/filters) before switching away. Call
   *  synchronously before the async /dataset/select so a broadcast can't move it mid-await. */
  saveCurrentDatasetUi: () => void;
  /** Adopt a new dataset selection and its binding generation in one store update, restoring
   *  the selection's saved position/filters when the user has been here before (else the
   *  selection's own values). Establishes the new identity locally so a same-identity backend
   *  snapshot keeps the restored index instead of resetting it to 0. */
  applyRestoredDataset: (sel: DatasetSelection, generation: number) => void;
  /**
   * Apply a backend state snapshot with ownership-aware merge, not a wholesale
   * replace: a wholesale replace would clobber unsaved edits, the active tab, and
   * the scroll position. Backend owns the dataset selection; the browser owns
   * navigation/view/mode/subject/review-filter state and keeps its own copy. ``generation``
   * and ``epoch`` come from the same envelope and are adopted in this one update, never set
   * separately from the dataset they describe.
   */
  mergeSnapshot: (
    state: GuiState,
    version: number | null,
    generation: number | null,
    epoch: string | null,
  ) => void;
  setWsStatus: (s: "disconnected" | "connecting" | "connected" | "error") => void;
  setActiveTab: (tab: TabName) => void;
  setView: (view: ViewState) => void;
  setMode: (mode: Mode) => void;
  setActiveSubject: (subject: string | null) => void;
  setCanvasBindingMissing: (missing: boolean) => void;
}

export const createGuiSlice: StateCreator<AppState, [], [], GuiSlice> = (set, get) => ({
  gui: DEFAULT_STATE,
  wsStatus: "disconnected",
  wsVersion: 0,
  wsEpoch: null,
  bindingGeneration: null,
  canvasBindingMissing: false,

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

  applyRestoredDataset: (sel, generation) =>
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
        // Adopted alongside the dataset in this one update: the pusher's next build reads a
        // generation that already names the same project as the dataset it fires against.
        bindingGeneration: generation,
        canvasBindingMissing: false,
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

  mergeSnapshot: (rawIncoming, version, generation, epoch) =>
    set((s) => {
      // pred_reference is gui_snapshot's frozen-format resident (state.py's GuiState); drop it here.
      const { pred_reference: _predReference, ...incoming } = rawIncoming as GuiState & {
        pred_reference?: unknown;
      };
      // A moved epoch is a restarted backend's own replay: accepted regardless of version, since
      // its lower-numbered first snapshot would otherwise drop as a stale one.
      const epochChanged = epoch != null && epoch !== s.wsEpoch;
      if (!epochChanged && version != null && version < s.wsVersion) return s;
      const nextVersion = epochChanged
        ? (version ?? 0)
        : version != null
          ? Math.max(s.wsVersion, version)
          : s.wsVersion;
      const nextEpoch = epoch ?? s.wsEpoch;
      const local = s.gui;
      const inDs = incoming.dataset;

      /** A null/empty backend dataset must never clobber a populated client one: this is what
       *  lets the browser survive a backend restart (the restarted backend broadcasts an empty
       *  state before it knows the project). An ordinary same-process broadcast (no epoch change)
       *  keeps the client's own still-good generation rather than adopting whatever this one
       *  empty envelope carries; a restart's replay (epoch changed) adopts the incoming value,
       *  which by then is the record-read generation (see StateStore.refresh_binding_generation_
       *  from_record), not a fresh process's stranding None. */
      if (!inDs || !inDs.dataset_root) {
        if (!epochChanged) {
          return {
            wsVersion: nextVersion,
            wsEpoch: nextEpoch,
            bindingGeneration: s.bindingGeneration,
          };
        }
        return {
          wsVersion: nextVersion,
          wsEpoch: nextEpoch,
          bindingGeneration: generation,
          canvasBindingMissing: generation == null,
        };
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
          wsEpoch: nextEpoch,
          bindingGeneration: generation,
          canvasBindingMissing: generation == null,
        };
      }

      if (identityChanged) {
        // New dataset selection: adopt it wholesale (including its index) and drop
        // the stale reviewStatus. The active tab stays put.
        return {
          gui: { ...local, dataset: inDs },
          reviewStatus: DEFAULT_REVIEW_STATUS,
          wsVersion: nextVersion,
          wsEpoch: nextEpoch,
          bindingGeneration: generation,
          canvasBindingMissing: generation == null,
        };
      }
      /** Same dataset: accept backend-owned dataset fields (e.g. a changed model's prediction
       *  dir) but keep the user's navigation position and the local image_list reference (same
       *  identity => same list; reusing the ref avoids spuriously re-firing effects keyed on it,
       *  like registry/status hydration). Everything else (active_tab / mode / active_subject /
       *  view / review) is client-owned; keep local. */
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
        wsEpoch: nextEpoch,
        bindingGeneration: generation,
        canvasBindingMissing: generation == null,
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
  setCanvasBindingMissing: (canvasBindingMissing) => set({ canvasBindingMissing }),
});

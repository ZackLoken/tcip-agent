import type { StateCreator } from "zustand";

import { GUI_STATE_DEFAULTS } from "@/api/types.generated";
import {
  datasetKey,
  loadDatasetUi,
  loadLastTab,
  recordLastTab,
  saveDatasetUi,
} from "@/lib/datasetUiState";
import type { AppState } from "@/store/appState";
import type {
  DatasetSelection,
  GuiState,
  Mode,
  OpenProject,
  TabName,
  ViewState,
} from "@/store/types";

const DEFAULT_DATASET: DatasetSelection = GUI_STATE_DEFAULTS.dataset;

/** True when two selections name a different (dataset_root, date, subject): a different dataset
 *  whose own position the browser adopts. */
function datasetIdentityChanged(
  a: Pick<DatasetSelection, "dataset_root" | "date" | "subject">,
  b: Pick<DatasetSelection, "dataset_root" | "date" | "subject">,
): boolean {
  return a.dataset_root !== b.dataset_root || a.date !== b.date || a.subject !== b.subject;
}

/** The tab a project opens on: the one last worked in, else the default tab. */
function landingTab(project: OpenProject | null): TabName {
  return (project && loadLastTab(project.id)) ?? GUI_STATE_DEFAULTS.active_tab;
}

/** The incoming snapshot's GUI state, on ``project``'s landing tab, with no subject unless named. */
function adoptedGui(incoming: GuiState, project: OpenProject | null): GuiState {
  return {
    ...incoming,
    active_tab: landingTab(project),
    active_subject: incoming.active_subject ?? null,
  };
}

/** The open project's directory, or null when the backend has none open. */
export const selectProjectRoot = (s: Pick<AppState, "openProject">): string | null =>
  s.openProject?.path ?? null;

export interface GuiSlice {
  /** Server-synchronized state (mirrors backend GuiState). */
  gui: GuiState;
  /** The project the backend has open, from its state envelope or the open request's answer. */
  openProject: OpenProject | null;
  wsStatus: "disconnected" | "connecting" | "connected" | "error";
  /** Highest backend state version applied; used to drop stale snapshot replays. */
  wsVersion: number;
  /** This backend process's launch identity, from the envelope; a change accepts a lower
   *  wsVersion instead of dropping it as a stale replay (a restarted backend's own snapshot). */
  wsEpoch: string | null;

  patchGui: (partial: Partial<GuiState>) => void;
  /** Clear the dataset selection, returning the GUI to the project front door. */
  clearDataset: () => void;
  /** Persist the current dataset's position before switching away. Call synchronously before the
   *  async /dataset/select so a broadcast can't move it mid-await. */
  saveCurrentDatasetUi: () => void;
  /** Adopt a new dataset selection of ``project`` in one store update, restoring the selection's
   *  saved position when the user has been here before (else the selection's own values).
   *  Establishes the new identity locally so a same-identity backend snapshot keeps the restored
   *  index instead of resetting it to 0. */
  applyRestoredDataset: (sel: DatasetSelection, project: OpenProject) => void;
  /**
   * Apply a backend state snapshot with ownership-aware merge, not a wholesale
   * replace: a wholesale replace would clobber unsaved edits, the active tab, and
   * the scroll position. Backend owns the dataset selection and the open project; the browser
   * owns navigation/view/mode/subject state and keeps its own copy. ``project``
   * and ``epoch`` come from the same envelope and are adopted in this one update.
   */
  mergeSnapshot: (
    state: GuiState,
    version: number | null,
    project: OpenProject | null,
    epoch: string | null,
  ) => void;
  setWsStatus: (s: "disconnected" | "connecting" | "connected" | "error") => void;
  setActiveTab: (tab: TabName) => void;
  setView: (view: ViewState) => void;
  setMode: (mode: Mode) => void;
  setActiveSubject: (subject: string | null) => void;
}

export const createGuiSlice: StateCreator<AppState, [], [], GuiSlice> = (set, get) => ({
  gui: GUI_STATE_DEFAULTS,
  openProject: null,
  wsStatus: "disconnected",
  wsVersion: 0,
  wsEpoch: null,

  patchGui: (partial) => set((s) => ({ gui: { ...s.gui, ...partial } })),
  clearDataset: () => set((s) => ({ gui: { ...s.gui, dataset: DEFAULT_DATASET } })),

  saveCurrentDatasetUi: () => {
    const s = get();
    const key = datasetKey(s.openProject, s.gui.dataset);
    if (!key) return;
    saveDatasetUi(key, { index: s.gui.dataset.current_image_index });
  },

  applyRestoredDataset: (sel, project) => {
    if (project.id !== get().openProject?.id) get().closeSessionInterval();
    set((s) => {
      const key = datasetKey(project, sel);
      const restored = key ? loadDatasetUi(key) : null;
      const index =
        restored && sel.image_list.length
          ? Math.max(0, Math.min(restored.index, sel.image_list.length - 1))
          : sel.current_image_index;
      return {
        gui: {
          ...s.gui,
          active_tab: landingTab(project),
          dataset: { ...sel, current_image_index: index },
        },
        openProject: project,
      };
    });
  },

  mergeSnapshot: (incoming, version, project, epoch) => {
    let accepted = false;
    let adoptedOther = false;
    set((s) => {
      // A moved epoch is a restarted backend's own replay: accepted regardless of version, since
      // its lower-numbered first snapshot would otherwise drop as a stale one.
      const epochChanged = epoch != null && epoch !== s.wsEpoch;
      if (!epochChanged && version != null && version < s.wsVersion) return s;
      accepted = true;
      const nextVersion = epochChanged
        ? (version ?? 0)
        : version != null
          ? Math.max(s.wsVersion, version)
          : s.wsVersion;
      const nextEpoch = epoch ?? s.wsEpoch;
      const local = s.gui;
      const inDs = incoming.dataset;

      // The backend names which project is open; a different one (or none) leaves nothing of
      // the previous project's dataset in the browser.
      if (project?.id !== s.openProject?.id) {
        adoptedOther = true;
        return {
          gui: project ? adoptedGui(incoming, project) : { ...local, dataset: DEFAULT_DATASET },
          openProject: project,
          wsVersion: nextVersion,
          wsEpoch: nextEpoch,
        };
      }

      if (!inDs.dataset_root) {
        return { wsVersion: nextVersion, wsEpoch: nextEpoch };
      }

      // Boot hydration: no local dataset to protect, so adopt the persisted mode/filters/position;
      // the tab is the client's per-project record (backend active_tab only moves on agent focus).
      if (!local.dataset.dataset_root) {
        return {
          gui: adoptedGui(incoming, project),
          wsVersion: nextVersion,
          wsEpoch: nextEpoch,
        };
      }

      if (datasetIdentityChanged(inDs, local.dataset)) {
        // New dataset selection: adopt it wholesale (including its index). The active tab stays
        // put.
        return {
          gui: { ...local, dataset: inDs },
          wsVersion: nextVersion,
          wsEpoch: nextEpoch,
        };
      }
      /** Same dataset: accept backend-owned dataset fields (e.g. a changed model's prediction
       *  dir) but keep the user's navigation position and the local image_list reference (same
       *  identity => same list; reusing the ref avoids spuriously re-firing effects keyed on it,
       *  like registry hydration). Everything else (active_tab / mode / active_subject / view)
       *  is client-owned; keep local. */
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
      };
    });
    if (accepted) get().recordAcceptedProject(project?.id ?? null);
    if (adoptedOther) {
      get().closeSessionInterval();
      get().supersedeOpen(project?.id ?? null);
    }
  },

  setWsStatus: (wsStatus) => set({ wsStatus }),
  setActiveTab: (active_tab) =>
    set((s) => {
      // Write-through so the next open of this project resumes on the tab last worked in.
      if (s.openProject) recordLastTab(s.openProject.id, active_tab);
      return { gui: { ...s.gui, active_tab } };
    }),
  setView: (view) => set((s) => ({ gui: { ...s.gui, view } })),
  // Focus belongs to the selected tool: switching tools drops it.
  setMode: (mode) =>
    set((s) =>
      mode === s.gui.mode ? s : { gui: { ...s.gui, mode }, canvas: { ...s.canvas, focus: null } },
    ),
  setActiveSubject: (active_subject) => set((s) => ({ gui: { ...s.gui, active_subject } })),
});

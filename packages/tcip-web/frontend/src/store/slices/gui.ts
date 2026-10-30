import type { StateCreator } from "zustand";

import {
  GUI_STATE_DEFAULTS,
  type SnapshotProject,
  type StateSnapshotMessage,
} from "@/api/types.generated";
import {
  datasetKey,
  loadDatasetUi,
  loadLastTab,
  recordLastTab,
  saveDatasetUi,
} from "@/lib/datasetUiState";
import type { AppState } from "@/store/appState";
import type { DatasetSelection, GuiState, Mode, TabName, ViewState } from "@/store/types";

/** True when two selections name a different (dataset_root, date, subject): a different dataset
 *  whose own position the browser adopts. */
function datasetIdentityChanged(a: DatasetSelection, b: DatasetSelection): boolean {
  return a.dataset_root !== b.dataset_root || a.date !== b.date || a.subject !== b.subject;
}

/** The tab a project opens on: the one last worked in, else the default tab. */
function landingTab(project: SnapshotProject | null): TabName {
  return (project && loadLastTab(project.id)) ?? GUI_STATE_DEFAULTS.active_tab;
}

/** The incoming snapshot's GUI state, on ``project``'s landing tab. */
function adoptedGui(incoming: GuiState, project: SnapshotProject | null): GuiState {
  return { ...incoming, active_tab: landingTab(project) };
}

/** The open project's directory, or null when the backend has none open. */
export const selectProjectRoot = (s: Pick<AppState, "openProject">): string | null =>
  s.openProject?.path ?? null;

/** The selected capture's dataset root, or null when no capture is selected. */
export const selectDatasetRoot = ({ gui: { dataset } }: Pick<AppState, "gui">): string | null =>
  dataset && dataset.dataset_root;

export interface GuiSlice {
  /** Server-synchronized state (mirrors backend GuiState). */
  gui: GuiState;
  /** The project the backend has open, from its state envelope or the open request's answer. */
  openProject: SnapshotProject | null;
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
  applyRestoredDataset: (sel: DatasetSelection | null, project: SnapshotProject) => void;
  /**
   * Apply a backend state snapshot by ownership: the backend owns the dataset selection and the
   * open project; the browser owns navigation/view/mode/subject state and keeps its own copy. The
   * envelope's project and epoch are adopted in this one update.
   */
  mergeSnapshot: (envelope: StateSnapshotMessage) => void;
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
  clearDataset: () => set((s) => ({ gui: { ...s.gui, dataset: null } })),

  saveCurrentDatasetUi: () => {
    const s = get();
    if (!s.openProject || !s.gui.dataset) return;
    saveDatasetUi(datasetKey(s.openProject, s.gui.dataset), {
      index: s.gui.dataset.current_image_index,
    });
  },

  applyRestoredDataset: (sel, project) => {
    if (project.id !== get().openProject?.id) get().leaveProjectSession(project.id);
    set((s) => {
      const restored = sel && loadDatasetUi(datasetKey(project, sel));
      return {
        gui: {
          ...s.gui,
          active_tab: landingTab(project),
          dataset:
            sel && restored && sel.images.length
              ? {
                  ...sel,
                  current_image_index: Math.max(0, Math.min(restored.index, sel.images.length - 1)),
                }
              : sel,
        },
        openProject: project,
      };
    });
  },

  mergeSnapshot: ({ state: incoming, version, project, epoch }) => {
    let adoptedOther = false;
    set((s) => {
      // A moved epoch is a restarted backend's own replay: accepted regardless of version, since
      // its lower-numbered first snapshot would otherwise drop as a stale one.
      const epochChanged = epoch !== s.wsEpoch;
      if (!epochChanged && version < s.wsVersion) return s;
      const nextVersion = epochChanged ? version : Math.max(s.wsVersion, version);
      const local = s.gui;
      const inDs = incoming.dataset;

      // The backend names which project is open; a different one (or none) leaves nothing of
      // the previous project's dataset in the browser.
      if (project?.id !== s.openProject?.id) {
        adoptedOther = true;
        return {
          gui: project ? adoptedGui(incoming, project) : { ...local, dataset: null },
          openProject: project,
          wsVersion: nextVersion,
          wsEpoch: epoch,
        };
      }

      if (!inDs) {
        return { gui: { ...local, dataset: null }, wsVersion: nextVersion, wsEpoch: epoch };
      }

      // Boot hydration: no local dataset to protect, so adopt the persisted mode/filters/position;
      // the tab is the client's per-project record (backend active_tab only moves on agent focus).
      if (!local.dataset) {
        return {
          gui: adoptedGui(incoming, project),
          wsVersion: nextVersion,
          wsEpoch: epoch,
        };
      }

      if (datasetIdentityChanged(inDs, local.dataset)) {
        // New dataset selection: adopt it wholesale (including its index). The active tab stays
        // put.
        return {
          gui: { ...local, dataset: inDs },
          wsVersion: nextVersion,
          wsEpoch: epoch,
        };
      }
      // Same capture: the backend's selection and images, at the user's own position in them.
      return {
        gui: {
          ...local,
          dataset: {
            ...inDs,
            current_image_index: Math.min(
              local.dataset.current_image_index,
              Math.max(0, inDs.images.length - 1),
            ),
          },
        },
        wsVersion: nextVersion,
        wsEpoch: epoch,
      };
    });
    if (adoptedOther) {
      get().leaveProjectSession(project?.id ?? null);
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

import type { StateCreator } from "zustand";

import type { AppState } from "@/store/appState";

/** The picker's chosen project and the capture, subject and bucket to open it on. */
export interface OpenChoice {
  projectId: string | null;
  date: string;
  subject: string;
  bucket: string;
}

export const NO_OPEN_CHOICE: OpenChoice = { projectId: null, date: "", subject: "", bucket: "" };

/** A failed open and the project it was started for. */
export interface OpenFailure {
  projectId: string;
  message: string;
}

/** The one open in flight: its request id (each open takes a new one) and its project. A result
 *  whose request is no longer the held one is dropped. */
export interface OpenHold {
  requestId: number;
  /** The project the open is for; null while an open is taken before its project is known. */
  projectId: string | null;
  /** The project the latest snapshot accepted since the hold was taken (null for none open), or
   *  null while no snapshot has been accepted. */
  accepted: { projectId: string | null } | null;
}

export interface ProjectOpenSlice {
  /** What the person chose to open and the open in flight, held here so they survive the picker
   *  being replaced when the person's name is committed. */
  openChoice: OpenChoice;
  opening: OpenHold | null;
  openError: OpenFailure | null;
  /** Whether the first-load opening of the backend's open project has been attempted. */
  initialOpenAttempted: boolean;
  /** Choosing another project than the one in flight supersedes that open. */
  patchOpenChoice: (patch: Partial<OpenChoice>) => void;
  patchOpenStatus: (patch: Partial<Pick<ProjectOpenSlice, "opening" | "openError">>) => void;
  /** Release the open in flight when it is bound to a project other than ``keepProjectId``. */
  supersedeOpen: (keepProjectId: string | null) => void;
  /** A snapshot was accepted naming ``projectId`` as open (null for none), whether or not it
   *  changed the project: record it on the open in flight, whose own listing decides. */
  recordAcceptedProject: (projectId: string | null) => void;
  /** The first-load attempt reached its listing and decided; no later picker repeats it. */
  consumeInitialOpen: () => void;
}

const boundElsewhere = (hold: OpenHold | null, projectId: string | null): boolean =>
  hold?.projectId != null && hold.projectId !== projectId;

export const createProjectOpenSlice: StateCreator<AppState, [], [], ProjectOpenSlice> = (
  set,
  get,
) => ({
  openChoice: NO_OPEN_CHOICE,
  opening: null,
  openError: null,
  initialOpenAttempted: false,
  patchOpenChoice: (patch) => {
    set((s) => ({ openChoice: { ...s.openChoice, ...patch } }));
    if (patch.projectId !== undefined) get().supersedeOpen(patch.projectId);
  },
  patchOpenStatus: (patch) => set(patch),
  supersedeOpen: (keepProjectId) =>
    set((s) => (boundElsewhere(s.opening, keepProjectId) ? { opening: null } : s)),
  recordAcceptedProject: (projectId) =>
    set((s) => (s.opening ? { opening: { ...s.opening, accepted: { projectId } } } : s)),
  consumeInitialOpen: () => set({ initialOpenAttempted: true }),
});

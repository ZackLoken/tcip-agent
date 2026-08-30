import type { StateCreator } from "zustand";

import type { AppState } from "@/store/appState";

export interface TerminalOpenSlice {
  /** Agent terminal rail visibility (persisted; the server session survives closing). */
  terminalOpen: boolean;
  setTerminalOpen: (open: boolean) => void;
}

export const createTerminalOpenSlice: StateCreator<AppState, [], [], TerminalOpenSlice> = (
  set,
) => ({
  // Always closed on open, never restored across sessions: the canvas is the front door;
  // the rail opens on demand (the toggle, or an agent send via sendToAgentTerminal).
  terminalOpen: false,
  setTerminalOpen: (terminalOpen) => set({ terminalOpen }),
});

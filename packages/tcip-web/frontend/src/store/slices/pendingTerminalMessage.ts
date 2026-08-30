import type { StateCreator } from "zustand";

import type { AppState } from "@/store/appState";

export interface PendingTerminalMessageSlice {
  /** A request staged for the agent terminal: any component that would otherwise make the
   *  breeder hand-author a CV/ML decision (a training config, a search space, a calibration
   *  trigger) calls `sendToAgentTerminal` instead of exposing the raw control. Opens the rail
   *  and stages the text here; TerminalRail sends it as terminal input once its socket is
   *  open, then clears it so it never resends. */
  pendingTerminalMessage: string | null;
  sendToAgentTerminal: (text: string) => void;
  clearPendingTerminalMessage: () => void;
}

export const createPendingTerminalMessageSlice: StateCreator<
  AppState,
  [],
  [],
  PendingTerminalMessageSlice
> = (set, get) => ({
  pendingTerminalMessage: null,
  sendToAgentTerminal: (text) => {
    get().setTerminalOpen(true);
    set({ pendingTerminalMessage: text });
  },
  clearPendingTerminalMessage: () => set({ pendingTerminalMessage: null }),
});

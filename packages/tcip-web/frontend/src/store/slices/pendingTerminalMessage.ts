import type { StateCreator } from "zustand";

import type { AppState } from "@/store/appState";

export interface PendingTerminalMessageSlice {
  /** Requests staged for the agent terminal, oldest first, until delivered. */
  pendingTerminalMessages: string[];
  /** Open the agent rail and stage `text` behind the requests already staged. */
  sendToAgentTerminal: (text: string) => void;
  /** Drop the `count` oldest staged requests. */
  dropDeliveredTerminalMessages: (count: number) => void;
  /** Stage `texts`, requests an ended agent was never given, ahead of every staged request. */
  restageTerminalMessages: (texts: string[]) => void;
}

export const createPendingTerminalMessageSlice: StateCreator<
  AppState,
  [],
  [],
  PendingTerminalMessageSlice
> = (set, get) => ({
  pendingTerminalMessages: [],
  sendToAgentTerminal: (text) => {
    get().setTerminalOpen(true);
    set({ pendingTerminalMessages: [...get().pendingTerminalMessages, text] });
  },
  dropDeliveredTerminalMessages: (count) =>
    set({ pendingTerminalMessages: get().pendingTerminalMessages.slice(count) }),
  restageTerminalMessages: (texts) =>
    set({ pendingTerminalMessages: [...texts, ...get().pendingTerminalMessages] }),
});

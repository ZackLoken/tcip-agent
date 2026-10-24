import type { StateCreator } from "zustand";

import type { AppState } from "@/store/appState";

const PROVIDER_KEY = "tcip.terminal_provider";

/** Where the rail's agent is in its life: live (the socket runs, and launches an agent when it
 *  holds no session), handing-off (a switch of harness waits for the agent's hand-off, the socket
 *  still running), ending (an End is under way) or ended (only Start launches another). */
export type TerminalPhase = "live" | "handing-off" | "ending" | "ended";

export interface TerminalOpenSlice {
  /** Agent terminal rail visibility (the server session survives closing). */
  terminalOpen: boolean;
  setTerminalOpen: (open: boolean) => void;
  /** Where the rail's agent is in its life, read by its render and its callbacks alike. */
  terminalPhase: TerminalPhase;
  setTerminalPhase: (phase: TerminalPhase) => void;
  /** The provider row the breeder chose for the rail, remembered across page loads. */
  terminalProvider: string | null;
  setTerminalProvider: (provider: string) => void;
}

function rememberedProvider(): string | null {
  try {
    return localStorage.getItem(PROVIDER_KEY);
  } catch {
    return null;
  }
}

export const createTerminalOpenSlice: StateCreator<AppState, [], [], TerminalOpenSlice> = (
  set,
) => ({
  // Always closed on open, never restored across sessions: the canvas is the front door;
  // the rail opens on demand (the toggle, or an agent send via sendToAgentTerminal).
  terminalOpen: false,
  setTerminalOpen: (terminalOpen) => set({ terminalOpen }),
  terminalPhase: "live",
  setTerminalPhase: (terminalPhase) => set({ terminalPhase }),
  terminalProvider: rememberedProvider(),
  setTerminalProvider: (terminalProvider) => {
    try {
      localStorage.setItem(PROVIDER_KEY, terminalProvider);
    } catch {
      /* the choice just won't persist */
    }
    set({ terminalProvider });
  },
});

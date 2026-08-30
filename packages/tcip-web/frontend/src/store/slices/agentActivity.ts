import type { StateCreator } from "zustand";

import type { AppState } from "@/store/appState";

export interface AgentActivity {
  /** Increments per event so effects can react to the latest one. */
  seq: number;
  panel: string;
  eventType: string;
  data: Record<string, unknown>;
}

export interface AgentActivitySlice {
  /** Last panel event pushed by the MCP agent (via /ws/panel subscription). */
  agentActivity: AgentActivity | null;
  pushAgentActivity: (panel: string, eventType: string, data: Record<string, unknown>) => void;
}

export const createAgentActivitySlice: StateCreator<AppState, [], [], AgentActivitySlice> = (
  set,
) => ({
  agentActivity: null,

  pushAgentActivity: (panel, eventType, data) =>
    set((s) => ({
      agentActivity: { seq: (s.agentActivity?.seq ?? 0) + 1, panel, eventType, data },
    })),
});

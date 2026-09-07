import type { StateCreator } from "zustand";

import type { AppState } from "@/store/appState";

export interface AgentActivity {
  /** Increments per event so effects can react to the latest one. */
  seq: number;
  panel: string;
  eventType: string;
  data: Record<string, unknown>;
  /** The harness that declared itself on this push (``agent_client_name``, its version appended
   * when declared), or null when the sender declared none: a plain process's own write. */
  actor: string | null;
}

export interface AgentActivitySlice {
  /** Last panel event pushed by the MCP agent (via /ws/panel subscription). */
  agentActivity: AgentActivity | null;
  pushAgentActivity: (
    panel: string,
    eventType: string,
    data: Record<string, unknown>,
    actor: string | null,
  ) => void;
}

export const createAgentActivitySlice: StateCreator<AppState, [], [], AgentActivitySlice> = (
  set,
) => ({
  agentActivity: null,

  pushAgentActivity: (panel, eventType, data, actor) =>
    set((s) => ({
      agentActivity: { seq: (s.agentActivity?.seq ?? 0) + 1, panel, eventType, data, actor },
    })),
});

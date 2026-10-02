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

/** The client a record or a push declared: its name with its version appended when declared, or
 * null when it declared none. */
export function declaredClient(fields: {
  agent_client_name?: unknown;
  agent_client_version?: unknown;
}): string | null {
  const name = typeof fields.agent_client_name === "string" ? fields.agent_client_name : null;
  if (!name) return null;
  const version = fields.agent_client_version;
  return typeof version === "string" && version ? `${name} ${version}` : name;
}

export interface AgentActivitySlice {
  /** The last panel event and the actor it declared, or null when none has arrived. */
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

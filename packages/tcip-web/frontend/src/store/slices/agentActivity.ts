import type { StateCreator } from "zustand";

import type { AGENT_IDENTITY_FIELDS } from "@/api/types.generated";
import type { AppState } from "@/store/appState";

/** The identity fields a record or a push declared (tcip_mcp.agent_identity), each null or
 * absent when not declared. */
export type DeclaredIdentity = Partial<
  Record<(typeof AGENT_IDENTITY_FIELDS)[number], string | null>
>;

export interface AgentActivity {
  /** Increments per event so effects can react to the latest one. */
  seq: number;
  panel: string;
  eventType: string;
  data: Record<string, unknown>;
  /** The harness that declared itself on this push (``agent_client_name``, its version appended
   * when declared), or null when the sender declared none: a plain process's own write. */
  client: string | null;
}

/** The client a record or a push declared: its name with its version appended when declared, or
 * null when it declared none. */
export function declaredClient(fields: DeclaredIdentity): string | null {
  const name = fields.agent_client_name;
  if (!name) return null;
  const version = fields.agent_client_version;
  return version ? `${name} ${version}` : name;
}

export interface AgentActivitySlice {
  /** The last panel event and the client it declared, or null when none has arrived. */
  agentActivity: AgentActivity | null;
  pushAgentActivity: (
    panel: string,
    eventType: string,
    data: Record<string, unknown>,
    client: string | null,
  ) => void;
}

export const createAgentActivitySlice: StateCreator<AppState, [], [], AgentActivitySlice> = (
  set,
) => ({
  agentActivity: null,

  pushAgentActivity: (panel, eventType, data, client) =>
    set((s) => ({
      agentActivity: { seq: (s.agentActivity?.seq ?? 0) + 1, panel, eventType, data, client },
    })),
});

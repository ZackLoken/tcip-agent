import type { StateCreator } from "zustand";

import type { Panel, PanelEvent, RecordedIdentity } from "@/api/types.generated";
import type { AppState } from "@/store/appState";

export interface AgentActivity {
  /** Increments per event so effects can react to the latest one. */
  seq: number;
  panel: Panel;
  event: PanelEvent;
  /** The harness that declared itself on this push (``agent_client_name``, its version appended
   * when declared), or null when the sender declared none: a plain process's own write. */
  client: string | null;
}

/** A declared client's name with its version appended when declared. */
export function clientName(name: string, version: string | null): string {
  return version ? `${name} ${version}` : name;
}

/** The client a push declared (`clientName`), or null when it declared none. */
export function declaredClient(identity: RecordedIdentity): string | null {
  const name = identity.agent_client_name;
  return name ? clientName(name, identity.agent_client_version) : null;
}

export interface AgentActivitySlice {
  /** The last panel event and the client it declared, or null when none has arrived. */
  agentActivity: AgentActivity | null;
  pushAgentActivity: (panel: Panel, event: PanelEvent, client: string | null) => void;
}

export const createAgentActivitySlice: StateCreator<AppState, [], [], AgentActivitySlice> = (
  set,
) => ({
  agentActivity: null,

  pushAgentActivity: (panel, event, client) =>
    set((s) => ({
      agentActivity: { seq: (s.agentActivity?.seq ?? 0) + 1, panel, event, client },
    })),
});

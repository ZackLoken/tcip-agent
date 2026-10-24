/** REST client for the embedded agent terminal's provider status, launches, ends and requests. */

import { getJson, postJson, wsUrl } from "@/api/http";
import { ROUTES } from "@/api/routes";
import type {
  CreateSessionRequest,
  EndSessionRequest,
  SubmitRequest,
  TerminalEnd,
  TerminalLaunch,
  TerminalStatus,
} from "@/api/types.generated";

export const terminalApi = {
  status: () => getJson<TerminalStatus>(ROUTES.getTerminalStatus),

  createSession: (launch: CreateSessionRequest) =>
    postJson<TerminalLaunch>(ROUTES.postTerminalSessions, launch),

  end: (id: string, request: EndSessionRequest) =>
    postJson<TerminalEnd>(ROUTES.postTerminalSessionsBySessionIdEnd(id), request),

  submit: (id: string, request: SubmitRequest) =>
    postJson<Record<string, never>>(ROUTES.postTerminalSessionsBySessionIdSubmit(id), request),
};

export function terminalWsUrl(sessionId: string): string {
  return wsUrl(ROUTES.socketTerminalWsBySessionId(sessionId));
}

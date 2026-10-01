/** REST client for the embedded agent terminal's provider status, launches and requests. */

import { getJson, postJson, wsUrl } from "@/api/http";
import { ROUTES } from "@/api/routes";
import type {
  CreateSessionRequest,
  SubmitRequest,
  TerminalLaunch,
  TerminalStatus,
} from "@/api/types.generated";

export const terminalApi = {
  status: () => getJson<TerminalStatus>(ROUTES.getTerminalStatus),

  createSession: (launch: CreateSessionRequest) =>
    postJson<TerminalLaunch>(ROUTES.postTerminalSessions, launch),

  restart: (id: string, launch: CreateSessionRequest) =>
    postJson<TerminalLaunch>(ROUTES.postTerminalSessionsBySessionIdRestart(id), launch),

  submit: (id: string, request: SubmitRequest) =>
    postJson<Record<string, never>>(ROUTES.postTerminalSessionsBySessionIdSubmit(id), request),
};

export function terminalWsUrl(sessionId: string): string {
  return wsUrl(ROUTES.socketTerminalWsBySessionId(sessionId));
}

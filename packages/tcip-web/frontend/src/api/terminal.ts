/** REST client for the embedded agent terminal's provider status, launches, hand-offs, ends and
 *  requests. */

import { getJson, postJson, wsUrl } from "@/api/http";
import { ROUTES } from "@/api/routes";
import type {
  CreateSessionRequest,
  EndSessionRequest,
  HandOffAnswer,
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

  /** Ask for the agent's hand-off before a switch: that it was asked, or the end of a session the
   *  backend ended at once. */
  handOff: (id: string, request: EndSessionRequest) =>
    postJson<HandOffAnswer>(ROUTES.postTerminalSessionsBySessionIdHandOff(id), request),

  submit: (id: string, request: SubmitRequest) =>
    postJson<null>(ROUTES.postTerminalSessionsBySessionIdSubmit(id), request),
};

export function terminalWsUrl(sessionId: string): string {
  return wsUrl(ROUTES.socketTerminalWsBySessionId(sessionId));
}

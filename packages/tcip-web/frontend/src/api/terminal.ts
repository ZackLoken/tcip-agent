/**
 * REST client for the embedded agent terminal (the real Claude Code CLI in a PTY).
 * The byte stream itself flows over the session socket; see TerminalRail.
 */

import { wsUrl } from "@/api/http";
import { ROUTES } from "@/api/routes";

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(`${resp.status} ${resp.statusText}: ${text}`);
  }
  return (await resp.json()) as T;
}

export const terminalApi = {
  status: () => json<{ available: boolean; reason?: string }>(ROUTES.getTerminalStatus),

  createSession: (rows: number, cols: number) =>
    json<{ session_id: string; existing: boolean }>(ROUTES.postTerminalSessions, {
      method: "POST",
      body: JSON.stringify({ rows, cols }),
    }),

  restart: (id: string, rows: number, cols: number) =>
    json<{ session_id: string; alive: boolean }>(
      ROUTES.postTerminalSessionsBySessionIdRestart(id),
      {
        method: "POST",
        body: JSON.stringify({ rows, cols }),
      },
    ),
};

export function terminalWsUrl(sessionId: string): string {
  return wsUrl(ROUTES.socketTerminalWsBySessionId(sessionId));
}

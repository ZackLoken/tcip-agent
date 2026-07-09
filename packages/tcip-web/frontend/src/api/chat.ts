/**
 * Chat REST client + WebSocket tail for the in-app agent chat.
 * Mirrors the panel-socket reconnect pattern in ws.ts.
 */

export interface ChatEvent {
  type: string;
  [k: string]: unknown;
}

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

export const chatApi = {
  status: () => json<{ available: boolean; reason?: string }>("/api/chat/status"),

  createSession: () => json<{ session_id: string }>("/api/chat/sessions", { method: "POST" }),

  messages: (id: string) => json<{ messages: ChatEvent[] }>(`/api/chat/sessions/${id}/messages`),

  send: (id: string, text: string, includeContext: boolean) =>
    json<{ status: string }>(`/api/chat/sessions/${id}/messages`, {
      method: "POST",
      body: JSON.stringify({ text, include_context: includeContext }),
    }),

  interrupt: (id: string) =>
    json<{ status: string }>(`/api/chat/sessions/${id}/interrupt`, { method: "POST" }),

  permission: (id: string, requestId: string, decision: "allow" | "deny", note = "") =>
    json<{ status: string }>(`/api/chat/sessions/${id}/permission`, {
      method: "POST",
      body: JSON.stringify({ request_id: requestId, decision, note }),
    }),
};

const MAX_BACKOFF_MS = 15_000;

function wsUrl(path: string): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}${path}`;
}

/** Live tail of a chat session with capped-backoff reconnect. */
export class ChatSocket {
  private ws: WebSocket | null = null;
  private closedByClient = false;
  private backoff = 500;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(
    private sessionId: string,
    private onEvent: (ev: ChatEvent) => void,
    // Fired when the socket (re)opens, BEFORE the backend replays the transcript — the
    // consumer resets its event list so a reconnect replay rebuilds it without duplicates.
    private onOpen?: () => void,
  ) {}

  connect() {
    if (this.closedByClient) return;
    const ws = new WebSocket(wsUrl(`/api/chat/ws/${this.sessionId}`));
    this.ws = ws;
    ws.onopen = () => {
      this.backoff = 500;
      this.onOpen?.();
    };
    ws.onmessage = (ev) => {
      try {
        this.onEvent(JSON.parse(ev.data));
      } catch {
        /* ignore */
      }
    };
    ws.onclose = () => {
      if (this.closedByClient) return;
      const delay = this.backoff;
      this.backoff = Math.min(this.backoff * 2, MAX_BACKOFF_MS);
      this.reconnectTimer = setTimeout(() => this.connect(), delay);
    };
  }

  close() {
    this.closedByClient = true;
    if (this.reconnectTimer !== null) clearTimeout(this.reconnectTimer);
    this.ws?.close();
  }
}

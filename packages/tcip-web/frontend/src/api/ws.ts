/**
 * WebSocket client that subscribes to GuiState snapshots + panel events.
 * Auto-reconnects with capped exponential backoff.
 */

import { wsUrl } from "@/api/http";
import { ROUTES } from "@/api/routes";
import type { GuiState } from "@/store/types";
import { useStore } from "@/store";

type IncomingMessage =
  | { type: "state_snapshot"; state: GuiState; version?: number }
  | { type: string; [k: string]: unknown };

const MAX_BACKOFF_MS = 15_000;

export class StateSocket {
  private ws: WebSocket | null = null;
  private closed = false;
  private backoffMs = 500;

  constructor(private path: string = ROUTES.socketWsState) {}

  connect() {
    this.closed = false;
    // Don't stack a second socket when one is already live/opening (React
    // StrictMode double-mounts, rapid reconnects); that produced duplicate live
    // WebSockets each driving their own reconnect loop.
    if (
      this.ws &&
      (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)
    ) {
      return;
    }
    useStore.getState().setWsStatus("connecting");
    const ws = new WebSocket(wsUrl(this.path));
    this.ws = ws;

    // Every handler no-ops if `ws` has been superseded (replaced by a newer
    // connect, or nulled by close), so a stale socket can't drive status or
    // trigger a reconnect.
    ws.onopen = () => {
      if (ws !== this.ws) return;
      useStore.getState().setWsStatus("connected");
      this.backoffMs = 500;
    };
    ws.onmessage = (ev) => {
      if (ws !== this.ws) return;
      let msg: IncomingMessage;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      this.handle(msg);
    };
    ws.onerror = () => {
      if (ws !== this.ws) return;
      useStore.getState().setWsStatus("error");
    };
    ws.onclose = () => {
      if (ws !== this.ws) return;
      this.ws = null;
      useStore.getState().setWsStatus("disconnected");
      if (!this.closed) {
        const delay = this.backoffMs;
        this.backoffMs = Math.min(this.backoffMs * 2, MAX_BACKOFF_MS);
        setTimeout(() => {
          if (!this.closed) this.connect();
        }, delay);
      }
    };
  }

  close() {
    this.closed = true;
    const ws = this.ws;
    this.ws = null; // supersede: its handlers become no-ops, so no reconnect fires
    ws?.close();
  }

  private handle(msg: IncomingMessage) {
    if (msg.type === "state_snapshot") {
      const m = msg as { state: GuiState; version?: number };
      useStore.getState().mergeSnapshot(m.state, m.version ?? null);
    }
  }

  /**
   * Subscribe to server-pushed events for one panel, reconnecting with capped
   * exponential backoff if the socket drops (backend restart, transient
   * network); without it, panel events silently stop for the rest of the
   * browser session. The returned unsubscribe closes the live socket and
   * cancels any pending reconnect.
   */
  subscribePanel(
    panel: string,
    handler: (event: {
      panel: string;
      event_type: string;
      data: Record<string, unknown>;
      // Stamped per event by the backend: the ring buffer replays on every reconnect, so a
      // handler that acts once per event (dismissing a banner) needs to tell them apart.
      event_id: string;
    }) => void,
  ) {
    let ws: WebSocket | null = null;
    let closedByClient = false;
    let backoff = 500;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      if (closedByClient) return;
      ws = new WebSocket(wsUrl(ROUTES.socketWsPanelByPanel(panel)));
      ws.onopen = () => {
        backoff = 500;
      };
      ws.onmessage = (ev) => {
        try {
          const m = JSON.parse(ev.data);
          handler(m);
        } catch {
          /* ignore */
        }
      };
      ws.onclose = () => {
        if (closedByClient) return;
        const delay = backoff;
        backoff = Math.min(backoff * 2, MAX_BACKOFF_MS);
        reconnectTimer = setTimeout(connect, delay);
      };
    };

    connect();
    return () => {
      closedByClient = true;
      if (reconnectTimer !== null) clearTimeout(reconnectTimer);
      ws?.close();
    };
  }
}

export const stateSocket = new StateSocket();

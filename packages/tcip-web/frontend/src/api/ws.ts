/**
 * WebSocket client that subscribes to GuiState snapshots + panel events.
 * Auto-reconnects with capped exponential backoff.
 */

import type { GuiState } from "@/store/types";
import { useStore } from "@/store";

type IncomingMessage =
  | { type: "state_snapshot"; state: GuiState; version?: number }
  | { type: string; [k: string]: unknown };

const MAX_BACKOFF_MS = 15_000;

function wsUrl(path: string): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}${path}`;
}

export class StateSocket {
  private ws: WebSocket | null = null;
  private closed = false;
  private backoffMs = 500;
  private panelHandlers = new Map<
    string,
    (event: { panel: string; event_type: string; data: Record<string, unknown> }) => void
  >();

  constructor(private path: string = "/ws/state") {}

  connect() {
    this.closed = false;
    // Don't stack a second socket when one is already live/opening (React
    // StrictMode double-mounts, rapid reconnects) — that produced duplicate live
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

  subscribePanel(
    panel: string,
    handler: (event: { panel: string; event_type: string; data: Record<string, unknown> }) => void,
  ) {
    this.panelHandlers.set(panel, handler);
    const panelWs = new WebSocket(wsUrl(`/ws/panel/${panel}`));
    panelWs.onmessage = (ev) => {
      try {
        const m = JSON.parse(ev.data);
        handler(m);
      } catch {
        /* ignore */
      }
    };
    return () => {
      panelWs.close();
      this.panelHandlers.delete(panel);
    };
  }
}

export const stateSocket = new StateSocket();

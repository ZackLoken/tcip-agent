/**
 * WebSocket client that subscribes to GuiState snapshots + panel events.
 * Auto-reconnects with capped exponential backoff.
 */

import type { GuiState } from "@/store/types";
import { useStore } from "@/store";

type IncomingMessage =
  | { type: "state_snapshot"; state: GuiState }
  | { type: "state_delta"; path: string; value: unknown }
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
    useStore.getState().setWsStatus("connecting");
    const ws = new WebSocket(wsUrl(this.path));
    this.ws = ws;

    ws.onopen = () => {
      useStore.getState().setWsStatus("connected");
      this.backoffMs = 500;
    };
    ws.onmessage = (ev) => {
      let msg: IncomingMessage;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      this.handle(msg);
    };
    ws.onerror = () => {
      useStore.getState().setWsStatus("error");
    };
    ws.onclose = () => {
      useStore.getState().setWsStatus("disconnected");
      if (!this.closed) {
        const delay = this.backoffMs;
        this.backoffMs = Math.min(this.backoffMs * 2, MAX_BACKOFF_MS);
        setTimeout(() => this.connect(), delay);
      }
    };
  }

  close() {
    this.closed = true;
    this.ws?.close();
  }

  private handle(msg: IncomingMessage) {
    if (msg.type === "state_snapshot") {
      const state = (msg as { state: GuiState }).state;
      useStore.getState().setGui(state);
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

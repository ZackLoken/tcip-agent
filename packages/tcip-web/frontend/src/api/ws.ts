/**
 * WebSocket client that subscribes to GuiState snapshots + panel events.
 * Auto-reconnects with capped exponential backoff.
 */

import { wsUrl } from "@/api/http";
import { ROUTES } from "@/api/routes";
import { createReconnectingSocket, type ReconnectingSocket } from "@/lib/reconnectingSocket";
import type { GuiState } from "@/store/types";
import { useStore } from "@/store";

type IncomingMessage =
  | { type: "state_snapshot"; state: GuiState; version?: number }
  | { type: string; [k: string]: unknown };

export class StateSocket {
  private socket: ReconnectingSocket;

  constructor(private path: string = ROUTES.socketWsState) {
    this.socket = createReconnectingSocket({
      url: wsUrl(this.path),
      onConnecting: () => useStore.getState().setWsStatus("connecting"),
      onOpen: () => useStore.getState().setWsStatus("connected"),
      onError: () => useStore.getState().setWsStatus("error"),
      onClose: () => useStore.getState().setWsStatus("disconnected"),
      onMessage: (data) => {
        let msg: IncomingMessage;
        try {
          msg = JSON.parse(data);
        } catch {
          return;
        }
        this.handle(msg);
      },
    });
  }

  connect() {
    this.socket.start();
  }

  close() {
    this.socket.stop();
  }

  private handle(msg: IncomingMessage) {
    if (msg.type === "state_snapshot") {
      const m = msg as { state: GuiState; version?: number };
      useStore.getState().mergeSnapshot(m.state, m.version ?? null);
    }
  }

  /** Subscribe to server-pushed events for one panel; without reconnect, panel events would
   *  silently stop for the rest of the browser session on a backend restart or a drop. The
   *  returned unsubscribe closes the live socket and cancels any pending reconnect. */
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
    const socket = createReconnectingSocket({
      url: wsUrl(ROUTES.socketWsPanelByPanel(panel)),
      onMessage: (data) => {
        try {
          handler(JSON.parse(data));
        } catch {
          /* ignore */
        }
      },
    });
    socket.start();
    return () => socket.stop();
  }
}

export const stateSocket = new StateSocket();

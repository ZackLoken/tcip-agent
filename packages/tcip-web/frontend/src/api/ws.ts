/**
 * WebSocket client that subscribes to GuiState snapshots + panel events.
 * Auto-reconnects with capped exponential backoff.
 */

import { wsUrl } from "@/api/http";
import { ROUTES } from "@/api/routes";
import type { Panel, StateSnapshotMessage } from "@/api/types.generated";
import { createReconnectingSocket, type ReconnectingSocket } from "@/lib/reconnectingSocket";
import { useStore } from "@/store";
import type { DeclaredIdentity } from "@/store/slices/agentActivity";

export class StateSocket {
  private socket: ReconnectingSocket;

  constructor(private path: string = ROUTES.socketWsState) {
    this.socket = createReconnectingSocket({
      url: () => wsUrl(this.path),
      onConnecting: () => useStore.getState().setWsStatus("connecting"),
      onOpen: () => useStore.getState().setWsStatus("connected"),
      onError: () => useStore.getState().setWsStatus("error"),
      onClose: () => useStore.getState().setWsStatus("disconnected"),
      onMessage: (data) => {
        let envelope: StateSnapshotMessage;
        try {
          envelope = JSON.parse(data) as StateSnapshotMessage;
        } catch {
          return;
        }
        useStore.getState().mergeSnapshot(envelope);
      },
    });
  }

  connect() {
    this.socket.start();
  }

  close() {
    this.socket.stop();
  }

  /** Force a fresh connect-time replay, re-delivering the authoritative dataset and open project
   *  in one update. */
  resync() {
    this.socket.stop();
    this.socket.start();
  }

  /** Subscribe to server-pushed events for one panel; without reconnect, panel events would
   *  silently stop for the rest of the browser session on a backend restart or a drop. The
   *  returned unsubscribe closes the live socket and cancels any pending reconnect. */
  subscribePanel(
    panel: Panel,
    // The declared identity is spread at the top level by the backend's own broadcast payload.
    handler: (
      event: {
        panel: Panel;
        event_type: string;
        data: Record<string, unknown>;
        // Stamped per event by the backend: the ring buffer replays on every reconnect, so a
        // handler that acts once per event (dismissing a banner) needs to tell them apart.
        event_id: string;
      } & DeclaredIdentity,
    ) => void,
  ) {
    const socket = createReconnectingSocket({
      url: wsUrl(ROUTES.socketWsPanelByPanel(panel)),
      onMessage: (data) => {
        try {
          handler(JSON.parse(data) as Parameters<typeof handler>[0]);
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

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CONNECTION_LOST, StateSocket } from "@/api/ws";
import { useStore } from "@/store";
import { holdingSentence } from "@/test/notifications";
import { PANEL_EVENTS } from "@/test/panelEvents";
import { PRODUCED_ENVELOPE } from "@/test/store";
import { FakeWebSocket, lastSocket } from "@/test/webSocket";

const initialStoreState = useStore.getState();

describe("StateSocket.connect", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    FakeWebSocket.instances = [];
    vi.stubGlobal("WebSocket", FakeWebSocket);
  });

  afterEach(() => vi.useRealTimers());

  it("opens the state socket once and does not stack a second one while it is live", () => {
    const socket = new StateSocket();
    socket.connect();
    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(lastSocket().url).toContain("/ws/state");

    lastSocket().open();
    socket.connect();
    expect(FakeWebSocket.instances).toHaveLength(1);
    socket.close();
  });

  it("reconnects after a drop and holds the delay at 15 seconds once the backoff saturates", () => {
    const socket = new StateSocket();
    socket.connect();
    // Never opens, so each delay doubles: 500, 1000, 2000, 4000, 8000, then the cap.
    for (const delay of [500, 1000, 2000, 4000, 8000]) {
      lastSocket().drop();
      vi.advanceTimersByTime(delay);
    }
    expect(FakeWebSocket.instances).toHaveLength(6);

    lastSocket().drop();
    vi.advanceTimersByTime(14_999);
    expect(FakeWebSocket.instances).toHaveLength(6);
    vi.advanceTimersByTime(1);
    expect(FakeWebSocket.instances).toHaveLength(7);

    // An open alone puts the next delay back to 500ms.
    lastSocket().open();
    lastSocket().drop();
    vi.advanceTimersByTime(500);
    expect(FakeWebSocket.instances).toHaveLength(8);
    socket.close();
  });

  it("close suppresses any further reconnect", () => {
    const socket = new StateSocket();
    socket.connect();
    lastSocket().open();

    socket.close();
    vi.advanceTimersByTime(60_000);
    expect(FakeWebSocket.instances).toHaveLength(1);
  });
});

describe("StateSocket state_snapshot dispatch", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    FakeWebSocket.instances = [];
    vi.stubGlobal("WebSocket", FakeWebSocket);
  });

  afterEach(() => vi.useRealTimers());

  it("hands the backend's envelope whole to one mergeSnapshot call", () => {
    const mergeSnapshot = vi
      .spyOn(useStore.getState(), "mergeSnapshot")
      .mockImplementation(() => {});
    const socket = new StateSocket();
    socket.connect();
    lastSocket().open();

    lastSocket().message(JSON.stringify(PRODUCED_ENVELOPE));

    expect(mergeSnapshot).toHaveBeenCalledWith(PRODUCED_ENVELOPE);
    socket.close();
  });

  it("supplies no field an envelope lacks: one missing its epoch reaches mergeSnapshot missing it", () => {
    const mergeSnapshot = vi
      .spyOn(useStore.getState(), "mergeSnapshot")
      .mockImplementation(() => {});
    const socket = new StateSocket();
    socket.connect();
    lastSocket().open();
    // The produced envelope with its epoch removed, a frame the backend never sends.
    const { epoch: _removed, ...withoutEpoch } = PRODUCED_ENVELOPE;

    lastSocket().message(JSON.stringify(withoutEpoch));

    expect(mergeSnapshot.mock.calls[0][0]).toStrictEqual(withoutEpoch);
    socket.close();
  });

  it("holds the stale bundle with the sentence an envelope states, one entry while it repeats, and ends it once a frame states none", () => {
    vi.spyOn(useStore.getState(), "mergeSnapshot").mockImplementation(() => {});
    const socket = new StateSocket();
    socket.connect();
    lastSocket().open();
    const listed = useStore.getState().notifications.length;

    lastSocket().message(JSON.stringify({ ...PRODUCED_ENVELOPE, bundle: "rebuild the bundle" }));
    lastSocket().message(JSON.stringify({ ...PRODUCED_ENVELOPE, bundle: "rebuild the bundle" }));
    expect(holdingSentence("bundle")).toBe("rebuild the bundle");
    expect(useStore.getState().notifications).toHaveLength(listed + 1);

    lastSocket().message(JSON.stringify(PRODUCED_ENVELOPE));
    expect(useStore.getState().holding.bundle).toBeUndefined();
    socket.close();
  });

  it("holds an envelope's stale-bundle sentence even when mergeSnapshot throws on the envelope", () => {
    vi.spyOn(useStore.getState(), "mergeSnapshot").mockImplementation(() => {
      throw new TypeError("an envelope shape this bundle was not built for");
    });
    const socket = new StateSocket();
    socket.connect();
    lastSocket().open();

    expect(() =>
      lastSocket().message(JSON.stringify({ ...PRODUCED_ENVELOPE, bundle: "rebuild the bundle" })),
    ).toThrow(TypeError);
    expect(holdingSentence("bundle")).toBe("rebuild the bundle");

    useStore.getState().holdCondition("bundle", null);
    socket.close();
  });

  it("resync stops and re-starts the socket, forcing a fresh connect-time replay", () => {
    const socket = new StateSocket();
    socket.connect();
    const first = lastSocket();
    first.open();

    socket.resync();
    expect(first.readyState).toBe(FakeWebSocket.CLOSED); // the old socket was closed
    expect(FakeWebSocket.instances).toHaveLength(2); // a fresh connect happened synchronously
    expect(lastSocket()).not.toBe(first);

    socket.close();
  });
});

describe("StateSocket lost connection", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    FakeWebSocket.instances = [];
    vi.stubGlobal("WebSocket", FakeWebSocket);
    useStore.setState(initialStoreState, true);
  });

  afterEach(() => vi.useRealTimers());

  it("holds one lost connection from an error through the drop and every retry, and ends it at the next open", () => {
    const socket = new StateSocket();
    socket.connect();
    lastSocket().open();
    expect(useStore.getState().holding.connection).toBeUndefined();

    lastSocket().error();
    expect(holdingSentence("connection")).toBe(CONNECTION_LOST);
    lastSocket().drop();
    vi.advanceTimersByTime(500);
    lastSocket().drop();
    expect(useStore.getState().notifications).toHaveLength(1);

    vi.advanceTimersByTime(1000);
    lastSocket().open();
    expect(useStore.getState().holding.connection).toBeUndefined();
    socket.close();
  });

  it("raises nothing for a connect that is slow to open", () => {
    const socket = new StateSocket();
    socket.connect();
    vi.advanceTimersByTime(5000);
    lastSocket().open();

    expect(useStore.getState().notifications).toEqual([]);
    expect(useStore.getState().announced).toEqual([]);
    socket.close();
  });

  it("raises the lost connection at once when a first connect closes without opening", () => {
    const socket = new StateSocket();
    socket.connect();
    lastSocket().drop();

    expect(holdingSentence("connection")).toBe(CONNECTION_LOST);
    socket.close();
  });

  it("ends the lost connection on a deliberate close, and raises none from it", () => {
    const socket = new StateSocket();
    socket.connect();
    lastSocket().drop();
    expect(useStore.getState().holding.connection).toBeDefined();

    socket.close();
    vi.advanceTimersByTime(60_000);
    expect(useStore.getState().holding.connection).toBeUndefined();
    expect(useStore.getState().notifications).toHaveLength(1);
  });

  it.each([
    ["close", (socket: StateSocket) => socket.close()],
    ["resync", (socket: StateSocket) => socket.resync()],
  ])("raises no lost connection from a %s of an open socket", (_name, end) => {
    const socket = new StateSocket();
    socket.connect();
    lastSocket().open();

    end(socket);
    vi.advanceTimersByTime(60_000);

    expect(useStore.getState().notifications).toEqual([]);
    expect(useStore.getState().announced).toEqual([]);
    socket.close();
  });
});

describe("StateSocket.subscribePanel", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    FakeWebSocket.instances = [];
    vi.stubGlobal("WebSocket", FakeWebSocket);
  });

  afterEach(() => vi.useRealTimers());

  it("opens a socket for the panel and dispatches parsed events to the handler", () => {
    const handler = vi.fn();
    new StateSocket().subscribePanel("annotate", handler);

    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(lastSocket().url).toContain("/ws/panel/annotate");

    lastSocket().open();
    const envelope = PANEL_EVENTS.labels_written;
    lastSocket().message(JSON.stringify(envelope));
    expect(handler).toHaveBeenCalledWith(envelope);
  });

  it("ignores malformed frames without breaking the subscription", () => {
    const handler = vi.fn();
    new StateSocket().subscribePanel("annotate", handler);
    lastSocket().open();

    lastSocket().message("not json");
    expect(handler).not.toHaveBeenCalled();

    lastSocket().message(JSON.stringify(PANEL_EVENTS.labels_written));
    expect(handler).toHaveBeenCalledTimes(1);
  });

  it("reconnects after an unexpected drop with exponential backoff", () => {
    const handler = vi.fn();
    new StateSocket().subscribePanel("annotate", handler);
    lastSocket().open();

    lastSocket().drop();
    expect(FakeWebSocket.instances).toHaveLength(1);
    vi.advanceTimersByTime(500);
    expect(FakeWebSocket.instances).toHaveLength(2);

    // The retry never opened, so the next delay doubles to 1000ms.
    lastSocket().drop();
    vi.advanceTimersByTime(999);
    expect(FakeWebSocket.instances).toHaveLength(2);
    vi.advanceTimersByTime(1);
    expect(FakeWebSocket.instances).toHaveLength(3);

    // Events on the reconnected socket still reach the original handler.
    lastSocket().open();
    lastSocket().message(JSON.stringify(PANEL_EVENTS.labels_written));
    expect(handler).toHaveBeenCalledTimes(1);
  });

  it("caps the backoff at 15s and resets it after a successful reopen", () => {
    new StateSocket().subscribePanel("annotate", vi.fn());
    // Drop repeatedly without ever opening: 500 → 1000 → ... → capped at 15000.
    for (let i = 0; i < 8; i++) {
      lastSocket().drop();
      vi.advanceTimersByTime(15_000);
    }
    const saturated = FakeWebSocket.instances.length;
    lastSocket().drop();
    vi.advanceTimersByTime(14_999);
    expect(FakeWebSocket.instances).toHaveLength(saturated);
    vi.advanceTimersByTime(1);
    expect(FakeWebSocket.instances).toHaveLength(saturated + 1);

    // A successful reopen alone resets the backoff to 500ms.
    lastSocket().open();
    lastSocket().drop();
    vi.advanceTimersByTime(500);
    expect(FakeWebSocket.instances).toHaveLength(saturated + 2);
  });

  it("unsubscribe closes the socket and suppresses reconnection", () => {
    const unsubscribe = new StateSocket().subscribePanel("annotate", vi.fn());
    lastSocket().open();

    unsubscribe();
    expect(lastSocket().readyState).toBe(FakeWebSocket.CLOSED);
    vi.advanceTimersByTime(60_000);
    expect(FakeWebSocket.instances).toHaveLength(1);
  });

  it("unsubscribe cancels a pending reconnect", () => {
    const unsubscribe = new StateSocket().subscribePanel("annotate", vi.fn());
    lastSocket().drop(); // schedules a reconnect in 500ms

    unsubscribe();
    vi.advanceTimersByTime(60_000);
    expect(FakeWebSocket.instances).toHaveLength(1);
  });
});

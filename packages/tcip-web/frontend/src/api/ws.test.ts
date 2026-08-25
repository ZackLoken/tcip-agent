import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { StateSocket } from "@/api/ws";
import { useStore } from "@/store";

/** Minimal WebSocket stand-in: records instances, lets tests drive events. */
class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;

  url: string;
  readyState = FakeWebSocket.CONNECTING;
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }

  open() {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.();
  }

  message(data: string) {
    this.onmessage?.({ data });
  }

  /** Server-side drop (unexpected close). */
  drop() {
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.();
  }

  error() {
    this.onerror?.();
  }

  /** Client-initiated close: the browser still fires the close event. */
  close() {
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.();
  }
}

function lastSocket(): FakeWebSocket {
  return FakeWebSocket.instances[FakeWebSocket.instances.length - 1];
}

describe("StateSocket.connect", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    FakeWebSocket.instances = [];
    vi.stubGlobal("WebSocket", FakeWebSocket);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

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

    // A successful open puts the next delay back to 500ms.
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

describe("StateSocket wsStatus transitions", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    FakeWebSocket.instances = [];
    vi.stubGlobal("WebSocket", FakeWebSocket);
    useStore.setState({ wsStatus: "disconnected" });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it("drives connecting, connected, error and an unexpected drop's disconnected", () => {
    const socket = new StateSocket();
    socket.connect();
    expect(useStore.getState().wsStatus).toBe("connecting");

    lastSocket().open();
    expect(useStore.getState().wsStatus).toBe("connected");

    lastSocket().error();
    expect(useStore.getState().wsStatus).toBe("error");

    lastSocket().drop();
    expect(useStore.getState().wsStatus).toBe("disconnected");
    socket.close();
  });

  it("a deliberate close does not flip status to disconnected: supersession suppresses it", () => {
    const socket = new StateSocket();
    socket.connect();
    lastSocket().open();
    expect(useStore.getState().wsStatus).toBe("connected");

    socket.close();
    expect(useStore.getState().wsStatus).toBe("connected");
  });
});

describe("StateSocket.subscribePanel", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    FakeWebSocket.instances = [];
    vi.stubGlobal("WebSocket", FakeWebSocket);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it("opens a socket for the panel and dispatches parsed events to the handler", () => {
    const handler = vi.fn();
    new StateSocket().subscribePanel("annotate", handler);

    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(lastSocket().url).toContain("/ws/panel/annotate");

    lastSocket().open();
    const event = { panel: "annotate", event_type: "labels_written", data: {} };
    lastSocket().message(JSON.stringify(event));
    expect(handler).toHaveBeenCalledWith(event);
  });

  it("ignores malformed frames without breaking the subscription", () => {
    const handler = vi.fn();
    new StateSocket().subscribePanel("annotate", handler);
    lastSocket().open();

    lastSocket().message("not json");
    expect(handler).not.toHaveBeenCalled();

    lastSocket().message(JSON.stringify({ panel: "annotate", event_type: "e", data: {} }));
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
    lastSocket().message(JSON.stringify({ panel: "annotate", event_type: "e", data: {} }));
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

    // A successful open resets the backoff to 500ms.
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

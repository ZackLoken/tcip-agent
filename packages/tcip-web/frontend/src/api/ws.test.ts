import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { StateSocket } from "@/api/ws";

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

  /** Client-initiated close — the browser still fires the close event. */
  close() {
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.();
  }
}

function lastSocket(): FakeWebSocket {
  return FakeWebSocket.instances[FakeWebSocket.instances.length - 1];
}

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

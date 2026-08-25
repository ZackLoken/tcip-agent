import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createReconnectingSocket } from "@/lib/reconnectingSocket";

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
  onclose: ((ev: { code: number }) => void) | null = null;
  sent: string[] = [];

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

  drop(code = 1006) {
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.({ code });
  }

  close() {
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.({ code: 1000 });
  }

  send(data: string) {
    this.sent.push(data);
  }
}

function lastSocket(): FakeWebSocket {
  return FakeWebSocket.instances[FakeWebSocket.instances.length - 1];
}

beforeEach(() => {
  vi.useFakeTimers();
  FakeWebSocket.instances = [];
  vi.stubGlobal("WebSocket", FakeWebSocket);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("createReconnectingSocket", () => {
  it("connects, reconnects on drop with capped backoff, and stop suppresses further attempts", () => {
    const onConnecting = vi.fn();
    const onOpen = vi.fn();
    const onClose = vi.fn();
    const socket = createReconnectingSocket({
      url: "ws://x/one",
      onMessage: vi.fn(),
      onConnecting,
      onOpen,
      onClose,
    });

    socket.start();
    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(onConnecting).toHaveBeenCalledTimes(1);

    lastSocket().open();
    expect(onOpen).toHaveBeenCalledTimes(1);

    lastSocket().drop();
    expect(onClose).toHaveBeenCalledWith({ code: 1006 }, true);
    vi.advanceTimersByTime(499);
    expect(FakeWebSocket.instances).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(FakeWebSocket.instances).toHaveLength(2);

    socket.stop();
    vi.advanceTimersByTime(60_000);
    expect(FakeWebSocket.instances).toHaveLength(2);
  });

  it("guards re-entry while a socket is already open or connecting", () => {
    const socket = createReconnectingSocket({ url: "ws://x/two", onMessage: vi.fn() });
    socket.start();
    socket.start(); // still CONNECTING: must not stack a second attempt
    expect(FakeWebSocket.instances).toHaveLength(1);

    lastSocket().open();
    socket.start(); // OPEN: still must not stack
    expect(FakeWebSocket.instances).toHaveLength(1);
  });

  it("supersedes a replaced socket: a stale instance's late events are no-ops", () => {
    const onMessage = vi.fn();
    const socket = createReconnectingSocket({ url: "ws://x/three", onMessage });
    socket.start();
    const stale = lastSocket();
    stale.open();

    socket.stop();
    stale.message("late frame"); // fired on the superseded instance directly
    expect(onMessage).not.toHaveBeenCalled();
  });

  it("restarts after stop, re-arming a fresh attempt", () => {
    const socket = createReconnectingSocket({ url: "ws://x/four", onMessage: vi.fn() });
    socket.start();
    lastSocket().open();
    socket.stop();
    expect(FakeWebSocket.instances).toHaveLength(1);

    socket.start();
    expect(FakeWebSocket.instances).toHaveLength(2);
  });

  it("stops reconnecting once a frame marks the stream terminal", () => {
    const socket = createReconnectingSocket({
      url: "ws://x/five",
      onMessage: vi.fn(),
      isTerminal: (data) => data === "done",
    });
    socket.start();
    lastSocket().open();
    lastSocket().message("done");

    lastSocket().drop();
    vi.advanceTimersByTime(60_000);
    expect(FakeWebSocket.instances).toHaveLength(1);
  });

  it("an async URL provider that throws reports through onError with no reconnect scheduled", async () => {
    const onError = vi.fn();
    const socket = createReconnectingSocket({
      url: () => Promise.reject(new Error("no session")),
      onMessage: vi.fn(),
      onError,
    });
    socket.start();
    // Fake timers only intercept macrotasks; flush the microtask queue the rejected
    // provider promise resolves on, with no timer to advance.
    await Promise.resolve();
    await Promise.resolve();
    expect(onError).toHaveBeenCalledTimes(1);
    vi.advanceTimersByTime(60_000);
    expect(FakeWebSocket.instances).toHaveLength(0);
  });

  it("sends only while the socket is open", () => {
    const socket = createReconnectingSocket({ url: "ws://x/six", onMessage: vi.fn() });
    socket.start();
    socket.send("too early");
    expect(lastSocket().sent).toEqual([]);

    lastSocket().open();
    socket.send("hello");
    expect(lastSocket().sent).toEqual(["hello"]);
  });
});

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { openInferenceStream } from "@/api/inference";
import { openTrainingStream } from "@/api/training";

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

  close() {
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.();
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

describe("openInferenceStream", () => {
  it("opens the job's stream over ws, percent-escaping the job id", () => {
    openInferenceStream("job 1/a", vi.fn());
    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(lastSocket().url.startsWith("ws://")).toBe(true);
    expect(lastSocket().url).toContain("/api/inference/jobs/job%201%2Fa/stream");
  });

  it("dispatches parsed frames and ignores malformed ones", () => {
    const onMessage = vi.fn();
    openInferenceStream("j1", onMessage);
    lastSocket().open();

    lastSocket().message("not json");
    expect(onMessage).not.toHaveBeenCalled();

    lastSocket().message(JSON.stringify({ type: "progress", done: 3, total: 10 }));
    expect(onMessage).toHaveBeenCalledTimes(1);
    expect(onMessage).toHaveBeenCalledWith({ type: "progress", done: 3, total: 10 });
  });

  it("reconnects after a mid-run drop so progress does not silently stop", () => {
    openInferenceStream("j1", vi.fn());
    lastSocket().open();
    lastSocket().message(JSON.stringify({ type: "progress", done: 3, total: 10 }));

    lastSocket().drop();
    vi.advanceTimersByTime(499);
    expect(FakeWebSocket.instances).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(FakeWebSocket.instances).toHaveLength(2);
  });

  it("holds the reconnect delay at 15 seconds once the backoff has saturated", () => {
    openInferenceStream("j1", vi.fn());
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
  });

  it("stops reconnecting once the terminal frame has arrived", () => {
    const onMessage = vi.fn();
    openInferenceStream("j1", onMessage);
    lastSocket().open();
    lastSocket().message(JSON.stringify({ type: "final", status: "completed" }));
    expect(onMessage).toHaveBeenCalledWith({ type: "final", status: "completed" });

    lastSocket().drop();
    vi.advanceTimersByTime(60_000);
    expect(FakeWebSocket.instances).toHaveLength(1);
  });

  it("stops reconnecting once the not-found frame arrives, since it is typed final", () => {
    const onMessage = vi.fn();
    openInferenceStream("j1", onMessage);
    lastSocket().open();
    lastSocket().message(JSON.stringify({ type: "final", error: "job not found" }));
    expect(onMessage).toHaveBeenCalledWith({ type: "final", error: "job not found" });

    lastSocket().drop();
    vi.advanceTimersByTime(60_000);
    expect(FakeWebSocket.instances).toHaveLength(1);
  });

  it("closing the stream suppresses any further reconnect", () => {
    const stop = openInferenceStream("j1", vi.fn());
    lastSocket().open();
    stop();
    vi.advanceTimersByTime(60_000);
    expect(FakeWebSocket.instances).toHaveLength(1);
  });

  it("parses each frame once, not once for the terminal check and again for the handler", () => {
    openInferenceStream("j1", vi.fn());
    lastSocket().open();
    const parseSpy = vi.spyOn(JSON, "parse");
    lastSocket().message(JSON.stringify({ type: "progress", done: 1, total: 4 }));
    expect(parseSpy).toHaveBeenCalledTimes(1);
    parseSpy.mockRestore();
  });
});

describe("openTrainingStream", () => {
  it("opens the run's stream over ws, percent-escaping the run id and project root", () => {
    openTrainingStream("/data/proj", "run 1/a", vi.fn());
    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(lastSocket().url.startsWith("ws://")).toBe(true);
    expect(lastSocket().url).toContain(
      "/api/training/runs/run%201%2Fa/stream?project_root=%2Fdata%2Fproj",
    );
  });

  it("reconnects after a mid-run drop so metric rows resume", () => {
    openTrainingStream("/data/proj", "r1", vi.fn());
    lastSocket().open();
    lastSocket().message(JSON.stringify({ type: "row", run_id: "r1", row: { epoch: 2 } }));

    lastSocket().drop();
    vi.advanceTimersByTime(499);
    expect(FakeWebSocket.instances).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(FakeWebSocket.instances).toHaveLength(2);
  });

  it("holds the reconnect delay at 15 seconds once the backoff has saturated", () => {
    openTrainingStream("/data/proj", "r1", vi.fn());
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
  });

  it("stops reconnecting once a terminal status frame has arrived", () => {
    openTrainingStream("/data/proj", "r1", vi.fn());
    lastSocket().open();
    lastSocket().message(
      JSON.stringify({ type: "status", run_id: "r1", status: { status: "completed" } }),
    );

    lastSocket().drop();
    vi.advanceTimersByTime(60_000);
    expect(FakeWebSocket.instances).toHaveLength(1);
  });

  it("stops reconnecting once an error frame has arrived", () => {
    const onMessage = vi.fn();
    openTrainingStream("/data/proj", "r1", onMessage);
    lastSocket().open();
    lastSocket().message(JSON.stringify({ type: "error", run_id: "r1", error: "unknown run" }));
    expect(onMessage).toHaveBeenCalledWith({ type: "error", run_id: "r1", error: "unknown run" });

    lastSocket().drop();
    vi.advanceTimersByTime(60_000);
    expect(FakeWebSocket.instances).toHaveLength(1);
  });

  it("closing the stream suppresses any further reconnect", () => {
    const stop = openTrainingStream("/data/proj", "r1", vi.fn());
    lastSocket().open();
    stop();
    vi.advanceTimersByTime(60_000);
    expect(FakeWebSocket.instances).toHaveLength(1);
  });

  it("parses each frame once, not once for the terminal check and again for the handler", () => {
    openTrainingStream("/data/proj", "r1", vi.fn());
    lastSocket().open();
    const parseSpy = vi.spyOn(JSON, "parse");
    lastSocket().message(JSON.stringify({ type: "row", run_id: "r1", row: { epoch: 2 } }));
    expect(parseSpy).toHaveBeenCalledTimes(1);
    parseSpy.mockRestore();
  });
});

import { afterEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, renderHook } from "@testing-library/react";

import { useEmbeddedToolRetry } from "@/hooks/useEmbeddedToolRetry";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("useEmbeddedToolRetry", () => {
  it("retries a failing step on the given cadence until it reports done", async () => {
    vi.useFakeTimers();
    try {
      const step = vi
        .fn()
        .mockResolvedValueOnce({ url: null, error: "not found", done: false })
        .mockResolvedValueOnce({ url: "http://localhost:6006", error: null, done: true });

      const { result } = renderHook(() => useEmbeddedToolRetry("a", true, 0, step, 1000));
      await act(async () => {
        await Promise.resolve();
        await Promise.resolve();
      });
      expect(step).toHaveBeenCalledTimes(1);
      expect(result.current).toEqual({ url: null, error: "not found" });

      await act(async () => {
        await vi.advanceTimersByTimeAsync(1000);
      });

      expect(step).toHaveBeenCalledTimes(2);
      expect(result.current).toEqual({ url: "http://localhost:6006", error: null });
    } finally {
      vi.useRealTimers();
    }
  });

  it("never calls the step while inactive", async () => {
    const step = vi.fn().mockResolvedValue({ url: null, error: null, done: true });
    renderHook(() => useEmbeddedToolRetry("a", false, 0, step));
    await Promise.resolve();
    expect(step).not.toHaveBeenCalled();
  });

  it("resets the outcome and re-runs the step when attempt changes", async () => {
    const step = vi
      .fn()
      .mockResolvedValueOnce({ url: null, error: "first failure", done: true })
      .mockResolvedValueOnce({ url: "http://localhost:6007", error: null, done: true });

    const { result, rerender } = renderHook(
      ({ attempt }) => useEmbeddedToolRetry("a", true, attempt, step),
      { initialProps: { attempt: 0 } },
    );
    await vi.waitFor(() => expect(result.current.error).toBe("first failure"));

    rerender({ attempt: 1 });
    await vi.waitFor(() => expect(result.current.url).toBe("http://localhost:6007"));
    expect(step).toHaveBeenCalledTimes(2);
  });

  it("resets the outcome and restarts the step when the key changes with active held true throughout", async () => {
    // Switching directly between two selected runs/sweeps: active stays true across the switch.
    const step = vi
      .fn()
      .mockResolvedValueOnce({ url: "http://localhost:6006", error: null, done: true })
      .mockResolvedValueOnce({ url: "http://localhost:6007", error: null, done: true });

    const { result, rerender } = renderHook(({ key }) => useEmbeddedToolRetry(key, true, 0, step), {
      initialProps: { key: "run-a" },
    });
    await vi.waitFor(() => expect(result.current.url).toBe("http://localhost:6006"));

    rerender({ key: "run-b" });
    expect(result.current).toEqual({ url: null, error: null });
    await vi.waitFor(() => expect(result.current.url).toBe("http://localhost:6007"));
    expect(step).toHaveBeenCalledTimes(2);
  });
});

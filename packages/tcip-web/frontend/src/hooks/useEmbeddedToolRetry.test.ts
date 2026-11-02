import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { PENDING, useEmbeddedToolRetry } from "@/hooks/useEmbeddedToolRetry";
import { TENSORBOARD_FAILURES, TENSORBOARD_SERVINGS } from "@/test/tensorboard";

afterEach(cleanup);

const [FIRST, SECOND] = TENSORBOARD_SERVINGS;
const [PORTLESS, DYING] = TENSORBOARD_FAILURES;

describe("useEmbeddedToolRetry", () => {
  it("retries a failing step on the given cadence until it reports done", async () => {
    vi.useFakeTimers();
    try {
      const step = vi
        .fn()
        .mockResolvedValueOnce({ outcome: PORTLESS, done: false })
        .mockResolvedValueOnce({ outcome: FIRST, done: true });

      const { result } = renderHook(() => useEmbeddedToolRetry("a", true, 0, step, 1000));
      await act(async () => {
        await Promise.resolve();
        await Promise.resolve();
      });
      expect(step).toHaveBeenCalledTimes(1);
      const held = result.current;
      if (held.kind !== "failed") throw new Error(`the hook holds a ${held.kind} outcome`);
      expect(held.error).toBe(PORTLESS.error);

      await act(async () => {
        await vi.advanceTimersByTimeAsync(1000);
      });

      expect(step).toHaveBeenCalledTimes(2);
      expect(result.current).toEqual(FIRST);
    } finally {
      vi.useRealTimers();
    }
  });

  it("never calls the step while inactive", async () => {
    const step = vi.fn().mockResolvedValue({ outcome: PENDING, done: true });
    renderHook(() => useEmbeddedToolRetry("a", false, 0, step));
    await Promise.resolve();
    expect(step).not.toHaveBeenCalled();
  });

  it("resets the outcome and re-runs the step when attempt changes", async () => {
    const step = vi
      .fn()
      .mockResolvedValueOnce({ outcome: DYING, done: true })
      .mockResolvedValueOnce({ outcome: SECOND, done: true });

    const { result, rerender } = renderHook(
      ({ attempt }) => useEmbeddedToolRetry("a", true, attempt, step),
      { initialProps: { attempt: 0 } },
    );
    await vi.waitFor(() => expect(result.current).toEqual(DYING));

    rerender({ attempt: 1 });
    await vi.waitFor(() => expect(result.current).toEqual(SECOND));
    expect(step).toHaveBeenCalledTimes(2);
  });

  it("resets the outcome and restarts the step when the key changes with active held true throughout", async () => {
    // Switching directly between two selected runs/sweeps: active stays true across the switch.
    const step = vi
      .fn()
      .mockResolvedValueOnce({ outcome: FIRST, done: true })
      .mockResolvedValueOnce({ outcome: SECOND, done: true });

    const { result, rerender } = renderHook(({ key }) => useEmbeddedToolRetry(key, true, 0, step), {
      initialProps: { key: "run-a" },
    });
    await vi.waitFor(() => expect(result.current).toEqual(FIRST));

    rerender({ key: "run-b" });
    expect(result.current).toEqual(PENDING);
    await vi.waitFor(() => expect(result.current).toEqual(SECOND));
    expect(step).toHaveBeenCalledTimes(2);
  });
});

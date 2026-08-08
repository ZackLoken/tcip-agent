import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderHook, act } from "@testing-library/react";

import { api } from "@/api/client";
import { useActiveTabSync } from "@/hooks/useActiveTabSync";
import { useStore } from "@/store";

const initialStoreState = useStore.getState();

beforeEach(() => {
  useStore.setState(initialStoreState, true);
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("useActiveTabSync", () => {
  it("a store tab change fires one debounced POST carrying the settled tab", () => {
    const spy = vi.spyOn(api.state, "tab").mockResolvedValue({ status: "ok" });
    renderHook(() => useActiveTabSync());

    act(() => useStore.getState().setActiveTab("review"));
    act(() => useStore.getState().setActiveTab("training"));
    expect(spy).not.toHaveBeenCalled();

    vi.advanceTimersByTime(400);
    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy).toHaveBeenCalledWith("training");
  });
});

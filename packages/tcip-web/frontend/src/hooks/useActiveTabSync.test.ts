import { renderHook, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "@/api/client";
import { useActiveTabSync } from "@/hooks/useActiveTabSync";
import { useStore } from "@/store";

const initialStoreState = useStore.getState();

beforeEach(() => {
  useStore.setState(initialStoreState, true);
  vi.useFakeTimers();
});

afterEach(() => vi.useRealTimers());

describe("useActiveTabSync", () => {
  it("a store tab change fires one debounced POST carrying the settled tab", () => {
    const spy = vi.spyOn(api.state, "tab").mockResolvedValue(null);
    renderHook(() => useActiveTabSync());

    act(() => useStore.getState().setActiveTab("results"));
    act(() => useStore.getState().setActiveTab("training"));
    expect(spy).not.toHaveBeenCalled();

    vi.advanceTimersByTime(400);
    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy).toHaveBeenCalledWith({ active_tab: "training" });
  });
});

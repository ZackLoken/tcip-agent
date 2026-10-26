import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { useDisplayPixels } from "@/hooks/useDisplayPixels";

function setScreen(width: number, height: number) {
  vi.spyOn(window.screen, "width", "get").mockReturnValue(width);
  vi.spyOn(window.screen, "height", "get").mockReturnValue(height);
}

describe("useDisplayPixels", () => {
  it("reads the screen's device pixels and reads them again when the window resizes", () => {
    const { result } = renderHook(() => useDisplayPixels());
    expect(result.current).toBe(1920 * 1080 * window.devicePixelRatio ** 2);
    act(() => {
      setScreen(2560, 1440);
      window.dispatchEvent(new Event("resize"));
    });
    expect(result.current).toBe(2560 * 1440 * window.devicePixelRatio ** 2);
  });
});

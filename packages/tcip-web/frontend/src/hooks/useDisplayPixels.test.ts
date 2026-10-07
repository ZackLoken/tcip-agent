import { afterEach, describe, expect, it } from "vitest";
import { act, renderHook } from "@testing-library/react";

import { useDisplayPixels } from "@/hooks/useDisplayPixels";

function setScreen(width: number, height: number) {
  Object.defineProperty(window.screen, "width", { value: width, configurable: true });
  Object.defineProperty(window.screen, "height", { value: height, configurable: true });
}

afterEach(() => setScreen(1920, 1080));

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

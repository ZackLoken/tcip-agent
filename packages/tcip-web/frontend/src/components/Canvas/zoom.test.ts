import { describe, expect, it } from "vitest";

import { MAX_SCALE, MIN_SCALE, ZOOM_LEVELS } from "@/components/Canvas/zoom";

describe("zoom ladder", () => {
  it("bounds the view between the 5% and 1000% stops", () => {
    expect(ZOOM_LEVELS).toHaveLength(20);
    expect(MIN_SCALE).toBe(0.05);
    expect(MAX_SCALE).toBe(10);
  });

  it("steps upward with no repeated stop", () => {
    expect(ZOOM_LEVELS.length).toBeGreaterThan(1);
    for (let i = 1; i < ZOOM_LEVELS.length; i++) {
      expect(ZOOM_LEVELS[i]).toBeGreaterThan(ZOOM_LEVELS[i - 1]);
    }
  });
});

import { describe, expect, it } from "vitest";

import { jumpTarget, stepTarget } from "@/hooks/useImageNav";

describe("stepTarget (shared by arrows + Prev/Next)", () => {
  it("steps by delta and clamps at the ends", () => {
    expect(stepTarget(4, 0, 1)).toBe(1);
    expect(stepTarget(4, 2, -1)).toBe(1);
    expect(stepTarget(4, 3, 1)).toBeNull();
    expect(stepTarget(4, 0, -1)).toBeNull();
    expect(stepTarget(0, 0, 1)).toBeNull();
  });
});

describe("jumpTarget (counter box)", () => {
  it("maps a 1-based position to an index, clamped", () => {
    expect(jumpTarget(4, 2)).toBe(1);
    expect(jumpTarget(4, 99)).toBe(3);
    expect(jumpTarget(4, 0)).toBe(0);
    expect(jumpTarget(0, 1)).toBeNull();
  });
});

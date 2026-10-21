import { describe, expect, it } from "vitest";

import { jumpTarget, stepTarget } from "@/hooks/useImageNav";
import { filteredIndices } from "@/lib/imageFilter";

describe("stepTarget (shared by arrows + Prev/Next)", () => {
  it("steps by delta and clamps at the ends", () => {
    const all = [0, 1, 2, 3];
    expect(stepTarget(all, 0, 1)).toBe(1);
    expect(stepTarget(all, 2, -1)).toBe(1);
    expect(stepTarget(all, 3, 1)).toBeNull();
    expect(stepTarget(all, 0, -1)).toBeNull();
    expect(stepTarget([], 0, 1)).toBeNull();
  });

  it("steps along the kept images only, and from a dropped one to the nearest kept", () => {
    const kept = [1, 4];
    expect(stepTarget(kept, 1, 1)).toBe(4);
    expect(stepTarget(kept, 4, -1)).toBe(1);
    expect(stepTarget(kept, 2, 1)).toBe(4);
    expect(stepTarget(kept, 2, -1)).toBe(1);
    expect(stepTarget(kept, 5, 1)).toBeNull();
    expect(stepTarget(kept, 0, -1)).toBeNull();
  });
});

describe("jumpTarget (counter box)", () => {
  it("maps a 1-based position among the kept images to an index, clamped", () => {
    expect(jumpTarget([0, 1, 2, 3], 2)).toBe(1);
    expect(jumpTarget([0, 1, 2, 3], 99)).toBe(3);
    expect(jumpTarget([0, 1, 2, 3], 0)).toBe(0);
    expect(jumpTarget([2, 5], 2)).toBe(5);
    expect(jumpTarget([], 1)).toBeNull();
  });
});

describe("filteredIndices", () => {
  it("keeps the images whose state the filter names, an unknown state only under all", () => {
    const list = ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"];
    const states = {
      "a.jpg": "unannotated" as const,
      "b.jpg": "partial" as const,
      "c.jpg": "complete" as const,
      "d.jpg": "negative" as const,
    };
    expect(filteredIndices(list, states, "all")).toEqual([0, 1, 2, 3, 4]);
    expect(filteredIndices(list, states, "unannotated")).toEqual([0]);
    expect(filteredIndices(list, states, "in_progress")).toEqual([1]);
    expect(filteredIndices(list, states, "complete")).toEqual([2, 3]);
  });
});

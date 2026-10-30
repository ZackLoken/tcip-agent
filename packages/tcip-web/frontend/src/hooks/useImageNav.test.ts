import { describe, expect, it } from "vitest";

import { jumpTarget, stepTarget } from "@/hooks/useImageNav";
import { filteredIndices } from "@/lib/imageFilter";
import { ANNOTATE_ANSWERS } from "@/test/annotate";
import { RAW_SELECTION } from "@/test/store";

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
    const list = RAW_SELECTION.images;
    // The produced states: the first negative, the second partial, the fifth complete.
    const { states } = ANNOTATE_ANSWERS.marked_states;
    // The fourth photograph's state left unstated, as one the answer does not know.
    const stated = Object.fromEntries(
      Object.entries(states).filter(([name]) => name !== list[3].name),
    );
    expect(filteredIndices(list, stated, "all")).toEqual([0, 1, 2, 3, 4]);
    expect(filteredIndices(list, stated, "unannotated")).toEqual([2]);
    expect(filteredIndices(list, stated, "in_progress")).toEqual([1]);
    expect(filteredIndices(list, stated, "complete")).toEqual([0, 4]);
  });
});

import { describe, expect, it } from "vitest";

import { computeFilteredIndices, jumpTarget, stepTarget } from "@/hooks/useImageNav";

const LIST = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"];
const STATUS = {
  "a.jpg": "complete",
  "b.jpg": "partial",
  "c.jpg": "complete",
  "d.jpg": "unannotated",
} as const;

describe("computeFilteredIndices", () => {
  it("returns every index for the 'all' filter", () => {
    expect(computeFilteredIndices(LIST, STATUS, "all")).toEqual([0, 1, 2, 3]);
  });
  it("keeps only indices matching the active status", () => {
    expect(computeFilteredIndices(LIST, STATUS, "complete")).toEqual([0, 2]);
  });
  it("skips images the isNavigable predicate rejects (Review's zero-detection skip)", () => {
    // "all" filter but c.jpg has nothing to review -> excluded from the walk.
    const navigable = (name: string) => name !== "c.jpg";
    expect(computeFilteredIndices(LIST, STATUS, "all", navigable)).toEqual([0, 1, 3]);
  });
  it("applies isNavigable and the status filter together", () => {
    const navigable = (name: string) => name !== "a.jpg"; // drop one complete image
    expect(computeFilteredIndices(LIST, STATUS, "complete", navigable)).toEqual([2]);
  });

  // K23: an explicit `order` (e.g. an active-learning priority ranking) replaces positional
  // traversal, with the same filter/isNavigable predicates still applied on top of it.
  it("omitting order behaves exactly as before (regression pin)", () => {
    expect(computeFilteredIndices(LIST, STATUS, "all", undefined, undefined)).toEqual([0, 1, 2, 3]);
  });
  it("traverses in the supplied order instead of positional order", () => {
    const priority = [3, 0, 2, 1]; // d, a, c, b — most-informative first
    expect(computeFilteredIndices(LIST, STATUS, "all", undefined, priority)).toEqual([3, 0, 2, 1]);
  });
  it("still applies the status filter and isNavigable on top of a supplied order", () => {
    const priority = [3, 0, 2, 1];
    const navigable = (name: string) => name !== "d.jpg";
    // "complete" keeps a.jpg/c.jpg; isNavigable additionally drops d.jpg (moot here, already
    // excluded by the status filter) — order among survivors is still priority's, not positional.
    expect(computeFilteredIndices(LIST, STATUS, "complete", navigable, priority)).toEqual([0, 2]);
  });
});

describe("stepTarget (shared by arrows + Prev/Next)", () => {
  it("walks the filtered set, not raw indices", () => {
    const filtered = [0, 2]; // complete
    expect(stepTarget(filtered, 0, 1)).toBe(2); // Next skips b.jpg (filtered out)
    expect(stepTarget(filtered, 2, -1)).toBe(0);
  });
  it("clamps at the ends (no wrap)", () => {
    expect(stepTarget([0, 2], 2, 1)).toBeNull(); // already last -> no move
    expect(stepTarget([0, 2], 0, -1)).toBeNull(); // already first -> no move
  });
  it("enters the filtered set at the nearest member in the travel direction", () => {
    // current index 1 isn't in {0,2}: Next -> first member after it, Prev -> last before.
    expect(stepTarget([0, 2], 1, 1)).toBe(2);
    expect(stepTarget([0, 2], 1, -1)).toBe(0);
    // A wider gap: members {2,5,8}, current 6 (a non-member between 5 and 8).
    expect(stepTarget([2, 5, 8], 6, 1)).toBe(8);
    expect(stepTarget([2, 5, 8], 6, -1)).toBe(5);
  });
  it("returns null for an empty set", () => {
    expect(stepTarget([], 0, 1)).toBeNull();
  });
  it("walks a non-monotonic (priority-ranked) order exactly as given, not by numeric value", () => {
    const priority = [3, 0, 2, 1];
    expect(stepTarget(priority, 3, 1)).toBe(0);
    expect(stepTarget(priority, 0, 1)).toBe(2);
    expect(stepTarget(priority, 0, -1)).toBe(3);
  });
});

describe("jumpTarget (counter box)", () => {
  it("maps a 1-based filtered position to the real image index", () => {
    expect(jumpTarget([0, 2], 2)).toBe(2); // 2nd complete image is index 2
  });
  it("clamps out-of-range input", () => {
    expect(jumpTarget([0, 2], 99)).toBe(2);
    expect(jumpTarget([0, 2], 0)).toBe(0);
  });
});

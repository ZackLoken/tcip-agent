import { describe, expect, it } from "vitest";

import type { ImageStatus } from "@/api/classes";
import { canvasHoldsSubject, reconcileImageStatuses } from "@/lib/imageStatus";

function statuses(pairs: [string, ImageStatus][]): Record<string, ImageStatus> {
  return Object.fromEntries(pairs);
}

const EMPTY_SHAPES = { boxes: [], polygons: [], points: [], imageAnnotations: [] };

describe("reconcileImageStatuses", () => {
  it("writes an unconfirmed name whose derived token differs from the stored one", () => {
    const stored = statuses([["a.jpg", "unannotated"]]);
    const derived = statuses([["a.jpg", "partial"]]);
    const result = reconcileImageStatuses(stored, derived, []);
    expect(result.writes).toEqual({ "a.jpg": "partial" });
    expect(result.staleMarks).toEqual([]);
  });

  it("flags a confirmed name whose derived token now disagrees, rather than writing it", () => {
    const stored = statuses([["a.jpg", "complete"]]);
    const derived = statuses([["a.jpg", "negative"]]);
    const result = reconcileImageStatuses(stored, derived, ["a.jpg"]);
    expect(result.writes).toEqual({});
    expect(result.staleMarks).toEqual(["a.jpg"]);
  });

  it("flags the reverse disagreement too: a confirmed negative that now derives complete", () => {
    const stored = statuses([["a.jpg", "negative"]]);
    const derived = statuses([["a.jpg", "complete"]]);
    const result = reconcileImageStatuses(stored, derived, ["a.jpg"]);
    expect(result.writes).toEqual({});
    expect(result.staleMarks).toEqual(["a.jpg"]);
  });

  it("writes nothing and marks nothing when every derived token matches what is stored", () => {
    const stored = statuses([
      ["a.jpg", "complete"],
      ["b.jpg", "partial"],
    ]);
    const derived = statuses([
      ["a.jpg", "complete"],
      ["b.jpg", "partial"],
    ]);
    const result = reconcileImageStatuses(stored, derived, ["a.jpg"]);
    expect(result.writes).toEqual({});
    expect(result.staleMarks).toEqual([]);
  });

  it("heals an unconfirmed name from partial back to unannotated", () => {
    const stored = statuses([["a.jpg", "partial"]]);
    const derived = statuses([["a.jpg", "unannotated"]]);
    const result = reconcileImageStatuses(stored, derived, []);
    expect(result.writes).toEqual({ "a.jpg": "unannotated" });
    expect(result.staleMarks).toEqual([]);
  });

  it("never treats a name outside confirmed as stale, even when it changes", () => {
    const stored = statuses([["a.jpg", "unannotated"]]);
    const derived = statuses([["a.jpg", "partial"]]);
    const result = reconcileImageStatuses(stored, derived, ["b.jpg"]);
    expect(result.staleMarks).toEqual([]);
    expect(result.writes).toEqual({ "a.jpg": "partial" });
  });
});

describe("canvasHoldsSubject", () => {
  it("is false with no subject selected, regardless of canvas content", () => {
    expect(canvasHoldsSubject({ ...EMPTY_SHAPES, boxes: [{ subject: "a" }] }, null)).toBe(false);
  });

  it("is false when the canvas holds only another subject's shapes", () => {
    expect(canvasHoldsSubject({ ...EMPTY_SHAPES, boxes: [{ subject: "b" }] }, "a")).toBe(false);
  });

  it("is true for a box of the given subject", () => {
    expect(canvasHoldsSubject({ ...EMPTY_SHAPES, boxes: [{ subject: "a" }] }, "a")).toBe(true);
  });

  it("is true for an image-level rating of the given subject, with no geometry", () => {
    expect(canvasHoldsSubject({ ...EMPTY_SHAPES, imageAnnotations: [{ subject: "a" }] }, "a")).toBe(
      true,
    );
  });
});

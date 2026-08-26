import { describe, expect, it } from "vitest";

import type { ImageStatus } from "@/api/classes";
import { reconcileImageStatuses } from "@/lib/imageStatus";

function statuses(pairs: [string, ImageStatus][]): Record<string, ImageStatus> {
  return Object.fromEntries(pairs);
}

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

  it("heals an unconfirmed name from unannotated to partial and back", () => {
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

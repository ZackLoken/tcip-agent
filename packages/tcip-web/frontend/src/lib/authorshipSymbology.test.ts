import { describe, expect, it } from "vitest";

import { authorshipLabel, dashPattern } from "@/lib/authorshipSymbology";

describe("dashPattern", () => {
  it("draws a tool's own unaccepted shape dotted", () => {
    expect(dashPattern("tool", 2)).toEqual([2, 6]);
  });

  it("draws a polygon's derived box dashed, a distinct pattern from the tool's dots", () => {
    expect(dashPattern("derived", 2)).toEqual([12, 8]);
  });
});

describe("authorshipLabel", () => {
  it("names a person's shape by its subject alone", () => {
    expect(authorshipLabel("fruit", "person")).toBe("fruit");
  });

  it("appends the tool suffix for a shape no person has accepted", () => {
    expect(authorshipLabel("fruit", "tool")).toBe("fruit, tool");
  });

  it("appends the accepted-tool suffix once a person has accepted it", () => {
    expect(authorshipLabel("fruit", "tool_accepted")).toBe("fruit, accepted tool");
  });

  it("names an unattributed shape by its subject alone", () => {
    expect(authorshipLabel("fruit", "unattributed")).toBe("fruit");
  });
});

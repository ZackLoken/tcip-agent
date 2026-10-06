import { describe, expect, it } from "vitest";

import { subjectColor } from "@/api/subjects";
import {
  authorshipLabel,
  dashFor,
  legendRows,
  lineStyleOf,
  outlineColor,
  proposalLabel,
  REVIEW_STATUSES,
  STATUS_COLORS,
  STATUS_WORDS,
  strokeWidths,
} from "@/lib/symbology";

describe("line style says whose shape it is", () => {
  it("a tool's unaccepted shape is dotted; a person's, an accepted tool's and an unattributed one are solid", () => {
    expect(lineStyleOf("tool")).toBe("dotted");
    for (const authorship of ["person", "tool_accepted", "unattributed", null, undefined]) {
      expect(lineStyleOf(authorship)).toBe("solid");
    }
  });

  it("a dotted stroke's dash scales with the width and a solid stroke has none", () => {
    expect(dashFor("dotted", 2)).toEqual([2, 6]);
    expect(dashFor("solid", 2)).toBeUndefined();
  });
});

describe("outline color says status while reviewing and subject otherwise", () => {
  it("answers the status color for a status and the subject color for none", () => {
    for (const status of REVIEW_STATUSES) {
      expect(outlineColor("fruit", status)).toBe(STATUS_COLORS[status]);
    }
    expect(outlineColor("fruit", null)).toBe(subjectColor("fruit"));
  });

  it("the three status colors are distinct from each other", () => {
    expect(new Set(Object.values(STATUS_COLORS)).size).toBe(REVIEW_STATUSES.length);
  });
});

describe("labels", () => {
  it("names a person's shape by its subject alone and a tool's by its authorship", () => {
    expect(authorshipLabel("fruit", "person")).toBe("fruit");
    expect(authorshipLabel("fruit", "tool")).toBe("fruit, tool");
    expect(authorshipLabel("fruit", "tool_accepted")).toBe("fruit, accepted tool");
    expect(authorshipLabel("fruit", "unattributed")).toBe("fruit");
  });

  it("labels an unpaired proposal always and a paired one only while focused", () => {
    const unpaired = { subject: "fruit", score: 0.874, paired: null };
    const paired = { subject: "fruit", score: null, paired: 3 };
    expect(proposalLabel(unpaired, false)).toBe("fruit proposal 0.87");
    expect(proposalLabel(paired, false)).toBeNull();
    expect(proposalLabel(paired, true)).toBe("fruit proposal (pairs with an annotation)");
  });
});

describe("line widths scale with zoom", () => {
  it("grows on screen with the zoom and never with the image", () => {
    const far = strokeWidths(0.25);
    const near = strokeWidths(4);
    // On-screen width (image width times scale) grows mildly between the zooms.
    expect(near.boxStroke * 4).toBeGreaterThan(far.boxStroke * 0.25);
    // In image units it shrinks as the zoom grows, so a line never thickens with magnification.
    expect(near.boxStroke).toBeLessThan(far.boxStroke);
    expect(near.pointTickOuter * 4).toBeCloseTo(far.pointTickOuter * 0.25);
  });
});

describe("the legend is drawn from the constants", () => {
  it("lists the subjects when not reviewing, then the line styles and the halo", () => {
    const rows = legendRows(["fruit", "leaf"], false);
    expect(rows.slice(0, 2).map((r) => [r.text, r.color, r.subject])).toEqual([
      ["fruit", subjectColor("fruit"), "fruit"],
      ["leaf", subjectColor("leaf"), "leaf"],
    ]);
    expect(rows.slice(2).map((r) => [r.style, !!r.halo])).toEqual([
      ["solid", false],
      ["dotted", false],
      ["solid", true],
    ]);
  });

  it("lists every review status in its color while reviewing, naming no subject", () => {
    const rows = legendRows(["fruit"], true);
    expect(rows.slice(0, REVIEW_STATUSES.length).map((r) => [r.text, r.color])).toEqual(
      REVIEW_STATUSES.map((s) => [STATUS_WORDS[s], STATUS_COLORS[s]]),
    );
    expect(rows.every((r) => r.subject === undefined)).toBe(true);
  });
});

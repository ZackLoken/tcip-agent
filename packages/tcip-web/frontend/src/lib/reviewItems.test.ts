import { describe, expect, it } from "vitest";

import {
  annotationStatus,
  keptItems,
  nearestNeighborOrder,
  reviewItems,
  reviewStatuses,
  scopedOrder,
  stepTarget,
  type ReviewItem,
} from "@/lib/reviewItems";
import type { Proposal } from "@/store/types";

const proposal = (over: Partial<Proposal>): Proposal => ({
  subject: "fruit",
  attributes: {},
  iscrowd: false,
  score: 0.9,
  index: 0,
  paired: null,
  decision: null,
  bbox: [0, 0, 10, 10],
  ...over,
});

const box = (x: number, y: number, index?: number, subject = "fruit") => ({
  x1: x,
  y1: y,
  x2: x + 10,
  y2: y + 10,
  subject,
  attributes: {},
  index,
});

describe("annotationStatus", () => {
  const proposals = [
    proposal({ index: 0, paired: 0, decision: "accepted" }),
    proposal({ index: 1, paired: 1, decision: null }),
    proposal({ index: 2, paired: 2, decision: "rejected" }),
  ];

  it("confirmed when a decision accepted a proposal pairing with it", () => {
    expect(annotationStatus(0, proposals)).toBe("confirmed");
  });

  it("undecided while a pairing proposal awaits a decision", () => {
    expect(annotationStatus(1, proposals)).toBe("undecided");
  });

  it("nothing proposed when no live proposal pairs with it, or it was drawn since the load", () => {
    expect(annotationStatus(2, proposals)).toBe("unproposed");
    expect(annotationStatus(7, proposals)).toBe("unproposed");
    expect(annotationStatus(undefined, proposals)).toBe("unproposed");
  });

  it("reviewStatuses aligns with the canvas arrays and is all null while not reviewing", () => {
    const canvas = { boxes: [box(0, 0, 0), box(20, 0)], polygons: [], points: [] };
    expect(reviewStatuses(canvas, proposals, true)).toEqual({
      boxes: ["confirmed", "unproposed"],
      polygons: [],
      points: [],
    });
    expect(reviewStatuses(canvas, proposals, false).boxes).toEqual([null, null]);
  });
});

describe("reviewItems", () => {
  const canvas = {
    boxes: [box(0, 0, 0)],
    polygons: [
      {
        rings: [
          [
            [50, 50],
            [60, 50],
            [60, 60],
          ],
        ] as [number, number][][],
        subject: "leaf",
        attributes: {},
        index: 1,
      },
    ],
    points: [{ x: 90, y: 90, subject: "fruit", attributes: {}, index: 2 }],
  };
  const proposals = [
    proposal({ index: 0, bbox: [70, 70, 80, 80] }),
    proposal({ index: 1, bbox: [1, 1, 9, 9], paired: 0, decision: "accepted" }),
  ];

  it("lists every annotation and, while reviewing, the undecided proposals only", () => {
    const items = reviewItems(canvas, proposals, true);
    // `ref` is the index into each canvas array, never the document index.
    expect(items.map((i) => [i.kind, i.shape, i.ref, i.status])).toEqual([
      ["annotation", "box", 0, "confirmed"],
      ["annotation", "polygon", 0, "unproposed"],
      ["annotation", "point", 0, "unproposed"],
      ["proposal", "box", 0, "undecided"],
    ]);
    expect(items[1].bbox).toEqual([50, 50, 60, 60]);
    expect(items[2].bbox).toEqual([90, 90, 90, 90]);
  });

  it("lists the annotations alone with no status while not reviewing", () => {
    const items = reviewItems(canvas, proposals, false);
    expect(items).toHaveLength(3);
    expect(items.every((i) => i.status === null)).toBe(true);
  });

  it("keeps annotations through the confidence floor and drops proposals under it", () => {
    const items = reviewItems(
      canvas,
      [proposal({ index: 0, score: 0.3 }), proposal({ index: 1, score: 0.8 })],
      true,
    );
    const kept = keptItems(items, { confidence: 0.5, shape: "all" });
    expect(kept.filter((i) => i.kind === "proposal").map((i) => i.ref)).toEqual([1]);
    expect(kept.filter((i) => i.kind === "annotation")).toHaveLength(3);
  });

  it("the type filter keeps one shape", () => {
    const kept = keptItems(reviewItems(canvas, proposals, true), {
      confidence: null,
      shape: "polygon",
    });
    expect(kept.map((i) => i.shape)).toEqual(["polygon"]);
  });
});

describe("the review path", () => {
  const item = (ref: number, x: number, y: number, status: ReviewItem["status"]): ReviewItem => ({
    kind: "annotation",
    shape: "box",
    ref,
    subject: "fruit",
    bbox: [x, y, x + 2, y + 2],
    status,
    score: null,
  });

  it("tours the items by nearest neighbor from the top-left corner", () => {
    const order = nearestNeighborOrder([
      item(0, 90, 90, null),
      item(1, 0, 0, null),
      item(2, 10, 0, null),
      item(3, 80, 90, null),
    ]);
    expect(order.map((i) => i.ref)).toEqual([1, 2, 3, 0]);
  });

  it("the status scope takes a subsequence of the order without re-touring it", () => {
    const order = nearestNeighborOrder([
      item(0, 0, 0, "confirmed"),
      item(1, 10, 0, "undecided"),
      item(2, 20, 0, "confirmed"),
      item(3, 30, 0, "undecided"),
    ]);
    expect(scopedOrder(order, "all").map((i) => i.ref)).toEqual([0, 1, 2, 3]);
    expect(scopedOrder(order, "undecided").map((i) => i.ref)).toEqual([1, 3]);
  });

  it("steps along the order, wrapping, and starts at either end when nothing is focused", () => {
    const order = [item(0, 0, 0, null), item(1, 10, 0, null), item(2, 20, 0, null)];
    expect(stepTarget(order, null, 1)?.ref).toBe(0);
    expect(stepTarget(order, null, -1)?.ref).toBe(2);
    expect(stepTarget(order, order[2], 1)?.ref).toBe(0);
    expect(stepTarget(order, order[0], -1)?.ref).toBe(2);
    expect(stepTarget([], null, 1)).toBeNull();
  });
});

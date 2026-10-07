import { describe, expect, it } from "vitest";

import { boxDraft, draftStrokes } from "@/lib/draftStrokes";
import { NO_SUBJECT_DRAFT_COLOR } from "@/lib/symbology";

describe("boxDraft", () => {
  it("orders the bounds whichever corner the drag started at and takes its subject's color", () => {
    const scene = boxDraft({ x1: 50, y1: 9, x2: 10, y2: 40, subject: "bud" }, (s) => `#${s}`);
    expect(scene).toEqual({ bounds: [10, 9, 50, 40], color: "#bud" });
  });
});

const none = {
  mode: "polygon",
  currentPolygon: [] as [number, number][],
  polygonColor: "#112233",
  cutStart: null,
  cursor: null,
};

describe("draftStrokes", () => {
  it("lays the polygon in progress as dotted vertices and a tail to the cursor with none", () => {
    const laid: [number, number][] = [
      [1, 1],
      [2, 2],
    ];
    expect(draftStrokes({ ...none, currentPolygon: laid })).toEqual([
      { points: laid, color: "#112233", vertices: true, label: "drawing" },
    ]);
    expect(draftStrokes({ ...none, currentPolygon: laid, cursor: [9, 9] })).toEqual([
      { points: laid, color: "#112233", vertices: true, label: "drawing" },
      {
        points: [
          [2, 2],
          [9, 9],
        ],
        color: "#112233",
        vertices: false,
      },
    ]);
  });

  it("draws in the neutral draft color while no subject is active", () => {
    const [stroke] = draftStrokes({ ...none, currentPolygon: [[1, 1]], polygonColor: null });
    expect(stroke.color).toBe(NO_SUBJECT_DRAFT_COLOR);
  });

  it("lays a pending cut as its start with a tail to the cursor, in the cut's own color", () => {
    const cutStart = { point: [3, 4] as [number, number], color: "#445566" };
    expect(draftStrokes({ ...none, cutStart, cursor: [5, 6] })).toEqual([
      { points: [[3, 4]], color: "#445566", vertices: true, label: "cut" },
      {
        points: [
          [3, 4],
          [5, 6],
        ],
        color: "#445566",
        vertices: false,
      },
    ]);
  });

  it("strokes nothing outside polygon mode or with nothing started", () => {
    expect(draftStrokes({ ...none, mode: "box", currentPolygon: [[1, 1]] })).toEqual([]);
    expect(draftStrokes({ ...none, cursor: [5, 6] })).toEqual([]);
  });
});

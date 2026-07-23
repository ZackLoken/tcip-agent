import { describe, expect, it } from "vitest";

import {
  computePolygonBboxes,
  findHoveredPolygon,
  pointInPolygon,
  polygonBbox,
} from "@/lib/polygonGeometry";

const SQUARE: [number, number][] = [
  [0, 0],
  [10, 0],
  [10, 10],
  [0, 10],
];

describe("pointInPolygon", () => {
  it("detects inside vs outside", () => {
    expect(pointInPolygon([5, 5], SQUARE)).toBe(true);
    expect(pointInPolygon([15, 5], SQUARE)).toBe(false);
  });
});

describe("polygonBbox / computePolygonBboxes", () => {
  it("computes the axis-aligned extent", () => {
    expect(polygonBbox(SQUARE)).toEqual([0, 0, 10, 10]);
  });
  it("maps each polygon to its bbox in order", () => {
    const polys = [
      { points: SQUARE },
      {
        points: [
          [20, 20],
          [30, 25],
        ] as [number, number][],
      },
    ];
    expect(computePolygonBboxes(polys)).toEqual([
      [0, 0, 10, 10],
      [20, 20, 30, 25],
    ]);
  });

  it("agrees with the backend's COCO-xywh box derivation (cross-codebase pin)", () => {
    // polygonBbox (TS) and bbox_of + the _annotation_record xywh conversion (Python) are separate
    // copies of the same min/max math; they must not drift. This polygon and its expected xywh
    // mirror tests/test_json_io.py's SQUARE, where the backend writes bbox = [10, 20, 100, 200].
    const square: [number, number][] = [
      [10, 20],
      [110, 20],
      [110, 220],
      [10, 220],
    ];
    const [minX, minY, maxX, maxY] = polygonBbox(square);
    const cocoXywh = [minX, minY, maxX - minX, maxY - minY];
    expect(cocoXywh).toEqual([10, 20, 100, 200]);
  });
});

describe("findHoveredPolygon (bbox-prefiltered hover scan)", () => {
  const polys = [
    { points: SQUARE }, // 0: covers (0,0)-(10,10)
    {
      points: [
        [100, 100],
        [110, 100],
        [110, 110],
        [100, 110],
      ] as [number, number][],
    }, // 1: far away
  ];
  const bboxes = computePolygonBboxes(polys);

  it("returns the containing polygon's index", () => {
    expect(findHoveredPolygon([5, 5], polys, bboxes)).toBe(0);
    expect(findHoveredPolygon([105, 105], polys, bboxes)).toBe(1);
  });
  it("returns null when the point is in no polygon", () => {
    expect(findHoveredPolygon([50, 50], polys, bboxes)).toBeNull();
  });
  it("matches a brute-force ray-cast over every polygon (bbox pre-filter is sound)", () => {
    const pts: [number, number][] = [
      [5, 5],
      [105, 105],
      [50, 50],
      [0, 0],
      [10, 10],
    ];
    for (const pt of pts) {
      let brute: number | null = null;
      for (let i = 0; i < polys.length; i++) {
        if (pointInPolygon(pt, polys[i].points)) {
          brute = i;
          break;
        }
      }
      expect(findHoveredPolygon(pt, polys, bboxes)).toBe(brute);
    }
  });
});

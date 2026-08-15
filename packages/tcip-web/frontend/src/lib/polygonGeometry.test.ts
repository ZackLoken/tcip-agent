import { describe, expect, it } from "vitest";

import {
  computePolygonBboxes,
  findHitPoint,
  findHoveredPolygon,
  pointInPolygon,
  pointInRings,
  polygonBbox,
  ringsBbox,
} from "@/lib/polygonGeometry";

const SQUARE: [number, number][] = [
  [0, 0],
  [10, 0],
  [10, 10],
  [0, 10],
];

const FAR_SQUARE: [number, number][] = [
  [100, 100],
  [110, 100],
  [110, 110],
  [100, 110],
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
      { rings: [SQUARE] },
      {
        rings: [
          [
            [20, 20],
            [30, 25],
          ],
        ] as [number, number][][],
      },
    ];
    expect(computePolygonBboxes(polys)).toEqual([
      [0, 0, 10, 10],
      [20, 20, 30, 25],
    ]);
  });

  it("a multi-ring polygon's bbox spans every ring, not just the first", () => {
    // The derived box a breeder sees in box mode is the whole object's footprint; ring-0-only
    // bounds would draw a box around one half of an occlusion-split subject_a.
    expect(ringsBbox([SQUARE, FAR_SQUARE])).toEqual([0, 0, 110, 110]);
    expect(computePolygonBboxes([{ rings: [SQUARE, FAR_SQUARE] }])).toEqual([[0, 0, 110, 110]]);
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

describe("pointInRings", () => {
  it("hits any part of a multi-part shape (rings are disjoint parts, never holes)", () => {
    const rings = [SQUARE, FAR_SQUARE];
    expect(pointInRings([5, 5], rings)).toBe(true); // part 1
    expect(pointInRings([105, 105], rings)).toBe(true); // part 2, hit-testing ring 0 alone misses
    expect(pointInRings([50, 50], rings)).toBe(false);
  });
});

describe("findHitPoint (point selection, zoom-aware)", () => {
  const points = [
    { x: 100, y: 100 },
    { x: 108, y: 100 },
    { x: 400, y: 400 },
  ];
  // The canvas passes screen-px / view.scale, so the grab target is the same size on screen at
  // every zoom: 11 screen px is 11 image px at 1x, ~1.1 image px at 10x.
  const HIT_CANVAS = 11;
  const radiusAt = (scale: number) => HIT_CANVAS / scale;

  it("selects a point clicked within the radius and nothing outside it", () => {
    expect(findHitPoint([102, 102], points, radiusAt(1))).toBe(0); // ~2.8px away at 1x
    expect(findHitPoint([100, 130], points, radiusAt(1))).toBeNull();
  });

  it("zoomed in, the same image-space near-miss stops hitting and only a closer click does", () => {
    // At 10x, 11 screen px is ~1.1 image px: the 2.8-image-px-away click that hit at 1x must miss,
    // which is what makes precise placement possible instead of grabbing whatever is nearby.
    expect(findHitPoint([102, 102], points, radiusAt(10))).toBeNull();
    expect(findHitPoint([100.5, 100.5], points, radiusAt(10))).toBe(0);
  });

  it("returns the nearest point in range, not the first (two points a few px apart)", () => {
    // Both are inside an 11px radius from (107, 100); the one aimed at is index 1.
    expect(findHitPoint([107, 100], points, radiusAt(1))).toBe(1);
    expect(findHitPoint([101, 100], points, radiusAt(1))).toBe(0);
  });

  it("returns null for an empty point list", () => {
    expect(findHitPoint([100, 100], [], radiusAt(1))).toBeNull();
  });
});

describe("findHoveredPolygon (bbox-prefiltered hover scan)", () => {
  const polys = [
    { rings: [SQUARE] }, // 0: covers (0,0)-(10,10)
    { rings: [FAR_SQUARE] }, // 1: far away
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
        if (polys[i].rings.some((ring) => pointInPolygon(pt, ring))) {
          brute = i;
          break;
        }
      }
      expect(findHoveredPolygon(pt, polys, bboxes)).toBe(brute);
    }
  });

  it("hovers a multi-ring shape from any of its parts", () => {
    // One annotation, two disjoint regions: pointing at the second must select the same shape,
    // not read as empty canvas (which is how a click on a part-2 region used to deselect).
    const multi = [{ rings: [SQUARE, FAR_SQUARE] }];
    const bb = computePolygonBboxes(multi);
    expect(findHoveredPolygon([5, 5], multi, bb)).toBe(0);
    expect(findHoveredPolygon([105, 105], multi, bb)).toBe(0);
    expect(findHoveredPolygon([50, 50], multi, bb)).toBeNull(); // between the parts, inside the bbox
  });
});

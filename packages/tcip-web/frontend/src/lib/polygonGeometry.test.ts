import { describe, expect, it } from "vitest";

import {
  computePolygonBboxes,
  CUT_ALONG_EDGE_REFUSAL,
  CUT_ENDPOINT_INSIDE_REFUSAL,
  CUT_MISSES_REFUSAL,
  CUT_PARTITION_FAILED_REFUSAL,
  CUT_PIECE_TOO_SMALL_REFUSAL,
  CUT_TOO_MANY_CROSSINGS_REFUSAL,
  CUT_ZERO_LENGTH_REFUSAL,
  cutRing,
  derivedBoxFromPolygon,
  findHitPoint,
  findHoveredPolygon,
  MIN_BOX_SIDE,
  pointInPolygon,
  pointInRings,
  pointToSegmentDist,
  polygonBbox,
  ringsBbox,
  withRing,
} from "@/lib/polygonGeometry";
import type { PolygonShape } from "@/store/types";

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

describe("derivedBoxFromPolygon", () => {
  it("carries the polygon's subject onto its footprint box", () => {
    const p: PolygonShape = { rings: [SQUARE], subject: "trunk", attributes: {} };
    expect(derivedBoxFromPolygon(p)).toEqual({
      x1: 0,
      y1: 0,
      x2: 10,
      y2: 10,
      subject: "trunk",
      attributes: {},
    });
  });

  it("spans every ring, not just the first (an occlusion-split shape's whole footprint)", () => {
    const p: PolygonShape = { rings: [SQUARE, FAR_SQUARE], subject: "trunk", attributes: {} };
    expect(derivedBoxFromPolygon(p)).toEqual({
      x1: 0,
      y1: 0,
      x2: 110,
      y2: 110,
      subject: "trunk",
      attributes: {},
    });
  });
});

describe("withRing", () => {
  it("replaces one ring, leaving the rest of the annotation untouched", () => {
    const p: PolygonShape = {
      rings: [SQUARE, FAR_SQUARE],
      subject: "trunk",
      attributes: { health: "good" },
    };
    const replaced: [number, number][] = [
      [0, 0],
      [5, 0],
      [5, 5],
    ];
    const next = withRing(p, 0, replaced);
    expect(next.rings).toEqual([replaced, FAR_SQUARE]);
    expect(next.subject).toBe("trunk");
    expect(next.attributes).toEqual({ health: "good" });
    expect(p.rings[0]).toEqual(SQUARE); // the input is not mutated
  });
});

function area(ring: [number, number][]): number {
  let sum = 0;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    sum += ring[j][0] * ring[i][1] - ring[i][0] * ring[j][1];
  }
  return Math.abs(sum) / 2;
}

describe("cutRing", () => {
  it("a square cut vertically yields two four-vertex rectangles whose areas sum to the square's", () => {
    const r = cutRing(SQUARE, [5, -5], [5, 15]);
    if ("reason" in r) throw new Error(`expected a cut, got refused: ${r.reason}`);
    const [p1, p2] = r.rings;
    expect(p1).toHaveLength(4);
    expect(p2).toHaveLength(4);
    expect(area(p1) + area(p2)).toBeCloseTo(area(SQUARE));
  });

  it("a cut through two vertices transversally is admitted, each vertex counted once", () => {
    // The vertical line x=0 passes exactly through B and E, each transversal (opposite-side neighbors).
    const hex: [number, number][] = [
      [-5, 0], // A
      [0, -5], // B
      [5, 0], // C
      [5, 10], // D
      [0, 15], // E
      [-5, 10], // F
    ];
    const r = cutRing(hex, [0, -15], [0, 25]);
    if ("reason" in r) throw new Error(`expected a cut, got refused: ${r.reason}`);
    const [p1, p2] = r.rings;
    expect(area(p1) + area(p2)).toBeCloseTo(area(hex));
    expect(p1.length + p2.length).toBe(hex.length + 2); // B and E each shared by both pieces
  });

  it("the two-spike tangency refuses as a miss (two grazing contacts are zero crossings)", () => {
    // A naive count-the-touches implementation reads the two tangent spike tips as crossings.
    const comb: [number, number][] = [
      [0, 0],
      [5, 10],
      [10, 0],
      [15, 10],
      [20, 0],
      [20, -10],
      [0, -10],
    ];
    const r = cutRing(comb, [-5, 10], [25, 10]);
    expect(r).toEqual({ reason: CUT_MISSES_REFUSAL });
  });

  it("a segment that misses the ring refuses with its own reason", () => {
    const r = cutRing(SQUARE, [50, 50], [60, 60]);
    expect(r).toEqual({ reason: CUT_MISSES_REFUSAL });
  });

  it("a segment starting inside the ring refuses with its own reason", () => {
    const r = cutRing(SQUARE, [5, 5], [20, 5]);
    expect(r).toEqual({ reason: CUT_ENDPOINT_INSIDE_REFUSAL });
  });

  it("a segment laid along an edge refuses with its own reason, not the plain miss", () => {
    const r = cutRing(SQUARE, [-5, 0], [15, 0]);
    expect(r).toEqual({ reason: CUT_ALONG_EDGE_REFUSAL });
  });

  it("a zero-length segment (both clicks at the same point) refuses with its own reason", () => {
    const r = cutRing(SQUARE, [50, 50], [50, 50]);
    expect(r).toEqual({ reason: CUT_ZERO_LENGTH_REFUSAL });
  });

  const U_SHAPE: [number, number][] = [
    [0, 0],
    [30, 0],
    [30, 30],
    [20, 30],
    [20, 10],
    [10, 10],
    [10, 30],
    [0, 30],
  ];

  it("a U cut across both arms refuses with its own more-than-twice reason", () => {
    const r = cutRing(U_SHAPE, [-5, 20], [35, 20]);
    expect(r).toEqual({ reason: CUT_TOO_MANY_CROSSINGS_REFUSAL });
  });

  it("a U cut through one arm is admitted", () => {
    const r = cutRing(U_SHAPE, [25, -5], [25, 35]);
    expect("reason" in r).toBe(false);
  });

  it("a crafted walk whose piece areas do not sum to the parent's refuses on the post-condition", () => {
    // A bowtie's signed area is 0 (its lobes cancel); cut through its own self-crossing, the
    // pieces are valid triangles whose areas sum to 50, so the post-condition refuses it.
    const bowtie: [number, number][] = [
      [0, 0],
      [10, 10],
      [10, 0],
      [0, 10],
    ];
    const r = cutRing(bowtie, [5, -5], [5, 15]);
    expect(r).toEqual({ reason: CUT_PARTITION_FAILED_REFUSAL });
  });

  it("a piece thinner than MIN_BOX_SIDE refuses rather than writing a sliver", () => {
    // 1px from the square's left edge: one piece's bbox is 1px wide, below the three-pixel floor.
    const r = cutRing(SQUARE, [1, -5], [1, 15]);
    expect(r).toEqual({ reason: CUT_PIECE_TOO_SMALL_REFUSAL });
  });

  it("a piece at exactly MIN_BOX_SIDE is admitted, not refused (a floor, not a margin)", () => {
    const r = cutRing(SQUARE, [MIN_BOX_SIDE, -5], [MIN_BOX_SIDE, 15]);
    expect("reason" in r).toBe(false);
  });

  it("a large-area cut is not spuriously refused by shoelace rounding at that scale", () => {
    // A fixed epsilon this tiny is far below the rounding drift double-precision summation
    // produces here; the tolerance scales with the parent's own area instead.
    const big = 1_000_000;
    const bigSquare: [number, number][] = [
      [0, 0],
      [big, 0],
      [big, big],
      [0, big],
    ];
    const r = cutRing(bigSquare, [big / 2, -5], [big / 2, big + 5]);
    expect("reason" in r).toBe(false);
  });
});

describe("pointToSegmentDist", () => {
  it("projects onto the segment interior and reports its fraction", () => {
    const r = pointToSegmentDist(5, 5, 0, 0, 10, 0);
    expect(r.dist).toBe(5);
    expect(r.t).toBe(0.5);
    expect(r.proj).toEqual([5, 0]);
  });

  it("clamps the projection to an endpoint past the segment", () => {
    const r = pointToSegmentDist(-5, 5, 0, 0, 10, 0);
    expect(r.t).toBe(0);
    expect(r.proj).toEqual([0, 0]);
    expect(r.dist).toBeCloseTo(Math.hypot(5, 5));
  });

  it("degenerates to point distance when the segment has zero length", () => {
    const r = pointToSegmentDist(3, 4, 0, 0, 0, 0);
    expect(r.dist).toBe(5);
    expect(r.proj).toEqual([0, 0]);
  });
});

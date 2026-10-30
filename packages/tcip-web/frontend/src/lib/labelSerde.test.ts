import { describe, expect, it } from "vitest";

import { annotationsToCanvas, serializeCanvas } from "@/lib/labelSerde";
import type { CanvasContent } from "@/store/types";
import { LABEL_ANSWERS, servedAnnotation } from "@/test/labels";

const canvasToAnnotations = (labels: CanvasContent) => serializeCanvas(labels).annotations;

/** An annotation as a test states it, the rest of its served record its default. */
type Loaded = Parameters<typeof servedAnnotation>[0];

/** The canvas the load routes' answer for `annotations` splits into. */
const toCanvas = (annotations: Loaded[]) =>
  annotationsToCanvas(annotations.map((a) => servedAnnotation(a)));

describe("serializeCanvas positions", () => {
  it("name where each canvas shape lands in the saved list", () => {
    const labels = {
      boxes: [
        { x1: 0, y1: 0, x2: 5, y2: 5, subject: "box0", attributes: {}, iscrowd: false },
        { x1: 0, y1: 0, x2: 5, y2: 5, subject: "box1", attributes: {}, iscrowd: false },
      ],
      polygons: [
        {
          rings: [
            [
              [0, 0],
              [9, 0],
              [9, 9],
            ] as [number, number][],
          ],
          subject: "poly0",
          attributes: {},
          iscrowd: false,
        },
      ],
      points: [{ x: 1, y: 1, subject: "point0", attributes: {}, iscrowd: false }],
      imageAnnotations: [],
    };
    const { annotations, positions } = serializeCanvas(labels);
    expect(annotations[positions.box[1]].subject).toBe("box1");
    expect(annotations[positions.polygon[0]].subject).toBe("poly0");
    expect(annotations[positions.point[0]].subject).toBe("point0");
  });
});

/** The routes' load of the document the generator saved: a box, a polygon, a point and a
 *  geometry-less rating. */
const { loaded: PRODUCED_LOAD } = LABEL_ANSWERS;

describe("labelSerde round-trip", () => {
  it("splits a unified list by each annotation's own geometry, then reassembles it symmetrically", () => {
    // One of each kind in one file, including a geometry-less rating that must not be dropped.
    const annotations: Loaded[] = [
      {
        subject: "subject_a",
        iscrowd: false,
        bbox: [10, 20, 30, 40],
        attributes: { growth_stage: "extended" },
      },
      {
        subject: "subject_a",
        iscrowd: false,
        rings: [
          [
            [0, 0],
            [10, 0],
            [10, 10],
          ],
        ],
        attributes: {},
      },
      { subject: "tip", iscrowd: false, point: [7, 9], attributes: {} },
      { subject: "efb", iscrowd: false, attributes: { severity: "moderate" } },
    ];

    const canvas = toCanvas(annotations);
    expect(canvas.boxes).toHaveLength(1);
    expect(canvas.polygons).toHaveLength(1);
    expect(canvas.points).toHaveLength(1);
    expect(canvas.imageAnnotations).toHaveLength(1); // the geometry-less rating survives the split
    expect(canvas.boxes[0].subject).toBe("subject_a");
    expect(canvas.imageAnnotations[0].subject).toBe("efb");

    // Save reassembles every bucket: a box stays a box, a polygon a polygon, a point a point.
    const back = canvasToAnnotations(canvas);
    expect(back).toHaveLength(4);
    const box = back.find((a) => a.bbox);
    const poly = back.find((a) => a.points);
    const pt = back.find((a) => a.point);
    const rating = back.find((a) => !a.bbox && !a.points && !a.rings && !a.point);
    expect(pt).toMatchObject({ subject: "tip", point: [7, 9] });
    expect(box).toMatchObject({
      subject: "subject_a",
      bbox: [10, 20, 30, 40],
      attributes: { growth_stage: "extended" },
    });
    expect(poly).toMatchObject({
      subject: "subject_a",
      points: [
        [0, 0],
        [10, 0],
        [10, 10],
      ],
    });
    expect(rating).toMatchObject({ subject: "efb", attributes: { severity: "moderate" } });
  });

  it("keeps the routes' geometry-less rating through the split and the save it reassembles", () => {
    const canvas = annotationsToCanvas(PRODUCED_LOAD.annotations);
    expect(canvas.imageAnnotations).toHaveLength(1);
    const ratings = canvasToAnnotations(canvas).filter(
      (a) => !a.bbox && !a.points && !a.rings && !a.point,
    );
    expect(ratings).toHaveLength(1);
    expect(ratings[0].subject).toBe(canvas.imageAnnotations[0].subject);
  });

  it("buckets a both-geometry annotation (rings and bbox) to one polygon, zero boxes", () => {
    // Measurement-critical: a polygon record carries its derived bbox on disk. The split is
    // rings-first (if/else-if), so it must produce exactly one polygon and no box: a two-ifs
    // regression would emit both and double-count the subject_a.
    const annotations: Loaded[] = [
      {
        subject: "subject_a",
        iscrowd: false,
        rings: [
          [
            [0, 0],
            [10, 0],
            [10, 10],
          ],
        ],
        bbox: [0, 0, 10, 10],
        attributes: {},
      },
    ];
    const canvas = toCanvas(annotations);
    expect(canvas.polygons).toHaveLength(1);
    expect(canvas.boxes).toHaveLength(0);
  });

  it("a polygon-with-bbox saves as one polygon record with no bbox, which the routes load as one polygon", () => {
    const original: Loaded[] = [
      {
        subject: "subject_a",
        iscrowd: false,
        rings: [
          [
            [0, 0],
            [10, 0],
            [10, 10],
          ],
        ],
        bbox: [0, 0, 10, 10],
        attributes: {},
      },
    ];
    const saved = canvasToAnnotations(toCanvas(original));
    // Save emits the polygon only, no bbox, no extra record, so the box is never authored.
    expect(saved).toHaveLength(1);
    expect(saved[0].points).toBeDefined();
    expect(saved[0].bbox).toBeNull();

    const polygon = PRODUCED_LOAD.annotations.filter((a) => a.rings !== null);
    const reloaded = annotationsToCanvas(polygon);
    expect(reloaded.polygons).toHaveLength(1);
    expect(reloaded.boxes).toHaveLength(0);
  });
});

describe("labelSerde multi-ring polygons", () => {
  // An occlusion-split instance_seg shape (a subject_a behind a branch): two disjoint regions, one
  // annotation. Both routes' load side always sends every ring.
  const twoRings: [number, number][][] = [
    [
      [0, 0],
      [10, 0],
      [10, 10],
    ],
    [
      [40, 40],
      [60, 40],
      [60, 60],
    ],
  ];

  it("loads every ring into one canvas polygon (no ring dropped, no shape split in two)", () => {
    const canvas = toCanvas([
      { subject: "subject_a", iscrowd: false, rings: twoRings, attributes: {} },
    ]);
    expect(canvas.polygons).toHaveLength(1);
    expect(canvas.polygons[0].rings).toEqual(twoRings);
  });

  it("deep-copies the rings so canvas edits never mutate the loaded response", () => {
    const loaded: Loaded[] = [
      { subject: "subject_a", iscrowd: false, rings: twoRings, attributes: {} },
    ];
    const canvas = toCanvas(loaded);
    canvas.polygons[0].rings[1][0] = [999, 999];
    expect(twoRings[1][0]).toEqual([40, 40]);
  });

  it("saves a multi-ring shape as `rings` (all of them) and never as `points`", () => {
    const saved = canvasToAnnotations({
      boxes: [],
      polygons: [{ rings: twoRings, subject: "subject_a", attributes: {}, iscrowd: false }],
      points: [],
      imageAnnotations: [],
    });
    expect(saved).toHaveLength(1);
    expect(saved[0].rings).toEqual(twoRings);
    expect(saved[0].points).toBeNull(); // `points` would carry one contour = a lost region
  });

  it("saves a one-ring shape as `points`, the field a hand-drawn/edited contour belongs in", () => {
    const saved = canvasToAnnotations({
      boxes: [],
      polygons: [{ rings: [twoRings[0]], subject: "subject_a", attributes: {}, iscrowd: false }],
      points: [],
      imageAnnotations: [],
    });
    expect(saved[0].points).toEqual(twoRings[0]);
    expect(saved[0].rings).toBeNull(); // never both: the backend prefers `rings`
  });

  it("loads a multi-ring shape as one polygon and saves it back with every ring", () => {
    const original: Loaded[] = [
      { subject: "subject_a", iscrowd: false, rings: twoRings, attributes: {} },
    ];
    const canvas = toCanvas(original);
    expect(canvas.polygons).toHaveLength(1);
    expect(canvas.polygons[0].rings).toEqual(twoRings);
    expect(canvas.points).toHaveLength(0);
    expect(canvasToAnnotations(canvas).map((a) => a.rings)).toEqual([twoRings]);
  });
});

describe("labelSerde crowd flag", () => {
  it("keeps a crowd region a crowd region through the load split and the save, on every shape kind", () => {
    const ring: [number, number][] = [
      [0, 0],
      [10, 0],
      [10, 10],
    ];
    // As the load routes send them: every annotation states its flag.
    const original: Loaded[] = [
      { subject: "bur", bbox: [0, 0, 10, 10], attributes: {}, iscrowd: true },
      { subject: "bur", rings: [ring], attributes: {}, iscrowd: true },
      { subject: "bur", bbox: [20, 20, 30, 30], attributes: {}, iscrowd: false },
    ];
    const canvas = toCanvas(original);
    expect(canvas.boxes.map((b) => b.iscrowd)).toEqual([true, false]);
    expect(canvas.polygons[0].iscrowd).toBe(true);

    const saved = canvasToAnnotations(canvas);
    expect(saved.map((a) => a.iscrowd)).toEqual([true, false, true]);
  });
});

describe("labelSerde authorship", () => {
  it("carries authorship onto every canvas shape kind, from the load response", () => {
    const canvas = toCanvas([
      {
        subject: "subject_a",
        iscrowd: false,
        bbox: [0, 0, 10, 10],
        attributes: {},
        authorship: "tool",
      },
      {
        subject: "subject_a",
        iscrowd: false,
        rings: [
          [
            [0, 0],
            [10, 0],
            [10, 10],
          ],
        ],
        attributes: {},
        authorship: "person",
      },
      { subject: "tip", iscrowd: false, point: [1, 2], attributes: {}, authorship: "unattributed" },
    ]);
    expect(canvas.boxes[0].authorship).toBe("tool");
    expect(canvas.polygons[0].authorship).toBe("person");
    expect(canvas.points[0].authorship).toBe("unattributed");
  });

  it("never carries authorship back into a save payload, on any shape kind", () => {
    const canvas = toCanvas([
      {
        subject: "subject_a",
        iscrowd: false,
        bbox: [0, 0, 10, 10],
        attributes: {},
        authorship: "tool",
      },
      {
        subject: "subject_a",
        iscrowd: false,
        rings: [
          [
            [0, 0],
            [10, 0],
            [10, 10],
          ],
        ],
        attributes: {},
        authorship: "tool",
      },
      { subject: "tip", iscrowd: false, point: [1, 2], attributes: {}, authorship: "tool" },
    ]);
    const saved = canvasToAnnotations(canvas);
    expect(saved.every((a) => !("authorship" in a))).toBe(true);
  });
});

describe("labelSerde cut pieces", () => {
  it("two single-ring pieces from a cut serialize as two points records with the parent's values", () => {
    // What splitPolygon hands to save: two PolygonShapes sharing the parent's subject and values.
    const pieceA: [number, number][] = [
      [0, 0],
      [5, 0],
      [5, 5],
    ];
    const pieceB: [number, number][] = [
      [5, 0],
      [10, 0],
      [10, 10],
    ];
    const saved = canvasToAnnotations({
      boxes: [],
      polygons: [
        { rings: [pieceA], subject: "subject_a", attributes: { health: "good" }, iscrowd: false },
        { rings: [pieceB], subject: "subject_a", attributes: { health: "good" }, iscrowd: false },
      ],
      points: [],
      imageAnnotations: [],
    });
    expect(saved).toHaveLength(2);
    for (const [i, ring] of [pieceA, pieceB].entries()) {
      expect(saved[i]).toMatchObject({
        subject: "subject_a",
        points: ring,
        attributes: { health: "good" },
      });
      expect(saved[i].rings).toBeNull();
    }
  });
});

describe("labelSerde points", () => {
  it("loads a point annotation into the points bucket, never a box or a polygon", () => {
    // A point has no extent: bucketing it as a box would hand a fabricated zero-area target to
    // every downstream consumer that reads canvas.boxes.
    const canvas = toCanvas([
      {
        subject: "tip",
        point: [12.5, 40],
        attributes: { stage: "open" },
        iscrowd: false,
        authorship: "unattributed",
        index: 0,
      },
    ]);
    expect(canvas.points).toEqual([
      {
        x: 12.5,
        y: 40,
        subject: "tip",
        attributes: { stage: "open" },
        iscrowd: false,
        authorship: "unattributed",
        index: 0,
      },
    ]);
    expect(canvas.boxes).toHaveLength(0);
    expect(canvas.polygons).toHaveLength(0);
    expect(canvas.imageAnnotations).toHaveLength(0); // and not mistaken for a geometry-less rating
  });

  it("saves a point as `point` only, no bbox, no points/rings contour", () => {
    const saved = canvasToAnnotations({
      boxes: [],
      polygons: [],
      points: [{ x: 3, y: 4, subject: "tip", attributes: {}, iscrowd: false }],
      imageAnnotations: [],
    });
    expect(saved).toHaveLength(1);
    expect(saved[0].point).toEqual([3, 4]);
    expect(saved[0].bbox).toBeNull();
    expect(saved[0].points).toBeNull();
    expect(saved[0].rings).toBeNull();
  });

  it("loads a point (position, subject, attributes) as one point and saves it back as one", () => {
    const original: Loaded[] = [
      { subject: "tip", iscrowd: false, point: [101.5, 202.25], attributes: { stage: "open" } },
    ];
    const canvas = toCanvas(original);
    expect(canvas.points).toHaveLength(1);
    expect(canvas.points[0]).toMatchObject({
      x: 101.5,
      y: 202.25,
      subject: "tip",
      attributes: { stage: "open" },
    });
    // Neither side multiplies the annotation into a second geometry kind.
    expect(canvas.boxes).toHaveLength(0);
    expect(canvas.polygons).toHaveLength(0);
    expect(canvas.imageAnnotations).toHaveLength(0);
    expect(canvasToAnnotations(canvas)).toEqual([
      expect.objectContaining({
        subject: "tip",
        point: [101.5, 202.25],
        attributes: { stage: "open" },
        bbox: null,
        points: null,
        rings: null,
      }),
    ]);
  });

  it("keeps a point and a geometry-less rating distinct (a point is not an image-level label)", () => {
    const canvas = toCanvas([
      { subject: "tip", iscrowd: false, point: [1, 2], attributes: {} },
      { subject: "efb", iscrowd: false, attributes: { severity: "severe" } },
    ]);
    expect(canvas.points).toHaveLength(1);
    expect(canvas.imageAnnotations).toHaveLength(1);
    const saved = canvasToAnnotations(canvas);
    expect(saved.filter((a) => a.point)).toHaveLength(1);
    expect(saved.filter((a) => !a.point && !a.bbox && !a.points && !a.rings)).toHaveLength(1);
  });
});

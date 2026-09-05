import { describe, expect, it } from "vitest";

import { annotationsToCanvas, canvasToAnnotations } from "@/lib/labelSerde";
import type { Annotation, AnnotationPayload } from "@/store/types";

/** What the load routes hand back for a payload just saved: the save/load asymmetry made explicit,
 *  `_to_annotation` builds one Polygon from `rings` or `points`, and `_ann_dict` always emits
 *  `rings`. A save payload is therefore never a load payload; going through this is what makes a
 *  round-trip test a round-trip. `point` is symmetric (same field both ways), so it rides through
 *  the rest spread untouched. */
function asLoaded(saved: AnnotationPayload[]): Annotation[] {
  return saved.map(({ points, rings, ...rest }) => ({
    ...rest,
    rings: rings ?? (points ? [points] : null),
  })) as unknown as Annotation[];
}

describe("labelSerde round-trip", () => {
  it("splits a unified list by each annotation's own geometry, then reassembles it symmetrically", () => {
    // One of each kind in one file, including a geometry-less rating that must not be dropped.
    const annotations: Annotation[] = [
      { subject: "subject_a", bbox: [10, 20, 30, 40], attributes: { growth_stage: "extended" } },
      {
        subject: "subject_a",
        rings: [
          [
            [0, 0],
            [10, 0],
            [10, 10],
          ],
        ],
        attributes: {},
      },
      { subject: "tip", point: [7, 9], attributes: {} },
      { subject: "efb", attributes: { severity: "moderate" }, created_by: "user:breeder" },
    ];

    const canvas = annotationsToCanvas(annotations);
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
    // Provenance travels through the round-trip.
    expect(rating?.created_by).toBe("user:breeder");
  });

  it("preserves a geometry-less rating across a load->save->load cycle (never silently dropped)", () => {
    const original: Annotation[] = [{ subject: "efb", attributes: { severity: "severe" } }];
    const saved = canvasToAnnotations(annotationsToCanvas(original));
    // The saved payload, read back, still yields exactly the geometry-less rating.
    const reloaded = annotationsToCanvas(asLoaded(saved));
    expect(reloaded.boxes).toHaveLength(0);
    expect(reloaded.polygons).toHaveLength(0);
    expect(reloaded.imageAnnotations).toHaveLength(1);
    expect(reloaded.imageAnnotations[0]).toMatchObject({
      subject: "efb",
      attributes: { severity: "severe" },
    });
  });

  it("buckets a both-geometry annotation (rings and bbox) to one polygon, zero boxes", () => {
    // Measurement-critical: a polygon record carries its derived bbox on disk. The split is
    // rings-first (if/else-if), so it must produce exactly one polygon and no box: a two-ifs
    // regression would emit both and double-count the subject_a.
    const annotations: Annotation[] = [
      {
        subject: "subject_a",
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
    const canvas = annotationsToCanvas(annotations);
    expect(canvas.polygons).toHaveLength(1);
    expect(canvas.boxes).toHaveLength(0);
  });

  it("a polygon-with-bbox stays one polygon and emits no bbox/second record across load->save->load", () => {
    const original: Annotation[] = [
      {
        subject: "subject_a",
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
    const saved = canvasToAnnotations(annotationsToCanvas(original));
    // Save emits the polygon only, no bbox, no extra record, so the box is never authored.
    expect(saved).toHaveLength(1);
    expect(saved[0].points).toBeDefined();
    expect(saved[0].bbox).toBeUndefined();

    const reloaded = annotationsToCanvas(asLoaded(saved));
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
    const canvas = annotationsToCanvas([{ subject: "subject_a", rings: twoRings, attributes: {} }]);
    expect(canvas.polygons).toHaveLength(1);
    expect(canvas.polygons[0].rings).toEqual(twoRings);
  });

  it("deep-copies the rings so canvas edits never mutate the loaded response", () => {
    const loaded: Annotation[] = [{ subject: "subject_a", rings: twoRings, attributes: {} }];
    const canvas = annotationsToCanvas(loaded);
    canvas.polygons[0].rings[1][0] = [999, 999];
    expect(twoRings[1][0]).toEqual([40, 40]);
  });

  it("saves a multi-ring shape as `rings` (all of them) and never as `points`", () => {
    const saved = canvasToAnnotations({
      boxes: [],
      polygons: [{ rings: twoRings, subject: "subject_a", attributes: {} }],
      points: [],
      imageAnnotations: [],
    });
    expect(saved).toHaveLength(1);
    expect(saved[0].rings).toEqual(twoRings);
    expect(saved[0].points).toBeUndefined(); // `points` would carry one contour = a lost region
  });

  it("saves a one-ring shape as `points`, the field a hand-drawn/edited contour belongs in", () => {
    const saved = canvasToAnnotations({
      boxes: [],
      polygons: [{ rings: [twoRings[0]], subject: "subject_a", attributes: {} }],
      points: [],
      imageAnnotations: [],
    });
    expect(saved[0].points).toEqual(twoRings[0]);
    expect(saved[0].rings).toBeUndefined(); // never both: the backend prefers `rings`
  });

  it("round-trips a multi-ring shape unchanged through save -> load (points bucket empty)", () => {
    const original: Annotation[] = [
      { subject: "subject_a", rings: twoRings, attributes: {}, created_by: "user:breeder" },
    ];
    const saved = canvasToAnnotations(annotationsToCanvas(original));
    const reloaded = annotationsToCanvas(asLoaded(saved));
    expect(reloaded.polygons).toHaveLength(1);
    expect(reloaded.polygons[0].rings).toEqual(twoRings);
    expect(reloaded.polygons[0].created_by).toBe("user:breeder");
    expect(reloaded.points).toHaveLength(0);
  });
});

describe("labelSerde authorship", () => {
  it("carries authorship onto every canvas shape kind, from the load response", () => {
    const canvas = annotationsToCanvas([
      { subject: "subject_a", bbox: [0, 0, 10, 10], attributes: {}, authorship: "tool" },
      {
        subject: "subject_a",
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
      { subject: "tip", point: [1, 2], attributes: {}, authorship: "unattributed" },
    ]);
    expect(canvas.boxes[0].authorship).toBe("tool");
    expect(canvas.polygons[0].authorship).toBe("person");
    expect(canvas.points[0].authorship).toBe("unattributed");
  });

  it("never carries authorship back into a save payload, on any shape kind", () => {
    const canvas = annotationsToCanvas([
      { subject: "subject_a", bbox: [0, 0, 10, 10], attributes: {}, authorship: "tool" },
      {
        subject: "subject_a",
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
      { subject: "tip", point: [1, 2], attributes: {}, authorship: "tool" },
    ]);
    const saved = canvasToAnnotations(canvas);
    expect(saved.every((a) => !("authorship" in a))).toBe(true);
  });
});

describe("labelSerde points", () => {
  it("loads a point annotation into the points bucket, never a box or a polygon", () => {
    // A point has no extent: bucketing it as a box would hand a fabricated zero-area target to
    // every downstream consumer that reads canvas.boxes.
    const canvas = annotationsToCanvas([
      {
        subject: "tip",
        point: [12.5, 40],
        attributes: { stage: "open" },
        created_by: "user:breeder",
      },
    ]);
    expect(canvas.points).toEqual([
      {
        x: 12.5,
        y: 40,
        subject: "tip",
        attributes: { stage: "open" },
        created_by: "user:breeder",
        created_at: null,
        accepted_by: null,
        accepted_at: null,
        authorship: null,
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
      points: [{ x: 3, y: 4, subject: "tip", attributes: {} }],
      imageAnnotations: [],
    });
    expect(saved).toHaveLength(1);
    expect(saved[0].point).toEqual([3, 4]);
    expect(saved[0].bbox).toBeUndefined();
    expect(saved[0].points).toBeUndefined();
    expect(saved[0].rings).toBeUndefined();
  });

  it("round-trips a point (position, subject, attributes, provenance) through save -> load", () => {
    const original: Annotation[] = [
      {
        subject: "tip",
        point: [101.5, 202.25],
        attributes: { stage: "open" },
        created_by: "user:breeder",
      },
    ];
    const reloaded = annotationsToCanvas(
      asLoaded(canvasToAnnotations(annotationsToCanvas(original))),
    );
    expect(reloaded.points).toHaveLength(1);
    expect(reloaded.points[0]).toMatchObject({
      x: 101.5,
      y: 202.25,
      subject: "tip",
      attributes: { stage: "open" },
      created_by: "user:breeder",
    });
    // A round-trip must not multiply the annotation into a second geometry kind.
    expect(reloaded.boxes).toHaveLength(0);
    expect(reloaded.polygons).toHaveLength(0);
    expect(reloaded.imageAnnotations).toHaveLength(0);
  });

  it("keeps a point and a geometry-less rating distinct (a point is not an image-level label)", () => {
    const canvas = annotationsToCanvas([
      { subject: "tip", point: [1, 2], attributes: {} },
      { subject: "efb", attributes: { severity: "severe" } },
    ]);
    expect(canvas.points).toHaveLength(1);
    expect(canvas.imageAnnotations).toHaveLength(1);
    const saved = canvasToAnnotations(canvas);
    expect(saved.filter((a) => a.point)).toHaveLength(1);
    expect(saved.filter((a) => !a.point && !a.bbox && !a.points && !a.rings)).toHaveLength(1);
  });
});

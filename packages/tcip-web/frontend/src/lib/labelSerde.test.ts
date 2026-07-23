import { describe, expect, it } from "vitest";

import { annotationsToCanvas, canvasToAnnotations } from "@/lib/labelSerde";
import type { Annotation } from "@/store/types";

describe("labelSerde round-trip", () => {
  it("splits a unified list by each annotation's own geometry, then reassembles it symmetrically", () => {
    // One of each kind in one file, including a geometry-less rating that must NOT be dropped.
    const annotations: Annotation[] = [
      { subject: "catkin", bbox: [10, 20, 30, 40], attributes: { elongation: "elongated" } },
      {
        subject: "catkin",
        points: [
          [0, 0],
          [10, 0],
          [10, 10],
        ],
        attributes: {},
      },
      { subject: "efb", attributes: { severity: "moderate" }, created_by: "user:zack" },
    ];

    const canvas = annotationsToCanvas(annotations);
    expect(canvas.boxes).toHaveLength(1);
    expect(canvas.polygons).toHaveLength(1);
    expect(canvas.imageAnnotations).toHaveLength(1); // the geometry-less rating survives the split
    expect(canvas.boxes[0].subject).toBe("catkin");
    expect(canvas.imageAnnotations[0].subject).toBe("efb");

    // Save reassembles all three buckets — a box stays a box, a polygon a polygon, the rating kept.
    const back = canvasToAnnotations(canvas);
    expect(back).toHaveLength(3);
    const box = back.find((a) => a.bbox);
    const poly = back.find((a) => a.points);
    const rating = back.find((a) => !a.bbox && !a.points);
    expect(box).toMatchObject({
      subject: "catkin",
      bbox: [10, 20, 30, 40],
      attributes: { elongation: "elongated" },
    });
    expect(poly).toMatchObject({
      subject: "catkin",
      points: [
        [0, 0],
        [10, 0],
        [10, 10],
      ],
    });
    expect(rating).toMatchObject({ subject: "efb", attributes: { severity: "moderate" } });
    // Provenance travels through the round-trip.
    expect(rating?.created_by).toBe("user:zack");
  });

  it("preserves a geometry-less rating across a load->save->load cycle (never silently dropped)", () => {
    const original: Annotation[] = [{ subject: "efb", attributes: { severity: "severe" } }];
    const saved = canvasToAnnotations(annotationsToCanvas(original));
    // The saved payload, read back, still yields exactly the geometry-less rating.
    const reloaded = annotationsToCanvas(saved as unknown as Annotation[]);
    expect(reloaded.boxes).toHaveLength(0);
    expect(reloaded.polygons).toHaveLength(0);
    expect(reloaded.imageAnnotations).toHaveLength(1);
    expect(reloaded.imageAnnotations[0]).toMatchObject({
      subject: "efb",
      attributes: { severity: "severe" },
    });
  });

  it("buckets a both-geometry annotation (points AND bbox) to ONE polygon, ZERO boxes", () => {
    // Measurement-critical: a polygon record carries its derived bbox on disk. The split is
    // points-first (if/else-if), so it must produce exactly one polygon and no box — a two-ifs
    // regression would emit BOTH and double-count the catkin.
    const annotations: Annotation[] = [
      {
        subject: "catkin",
        points: [
          [0, 0],
          [10, 0],
          [10, 10],
        ],
        bbox: [0, 0, 10, 10],
        attributes: {},
      },
    ];
    const canvas = annotationsToCanvas(annotations);
    expect(canvas.polygons).toHaveLength(1);
    expect(canvas.boxes).toHaveLength(0);
  });

  it("a polygon-with-bbox stays one polygon and emits NO bbox/second record across load->save->load", () => {
    const original: Annotation[] = [
      {
        subject: "catkin",
        points: [
          [0, 0],
          [10, 0],
          [10, 10],
        ],
        bbox: [0, 0, 10, 10],
        attributes: {},
      },
    ];
    const saved = canvasToAnnotations(annotationsToCanvas(original));
    // Save emits the polygon only — no bbox, no extra record — so the box is never authored.
    expect(saved).toHaveLength(1);
    expect(saved[0].points).toBeDefined();
    expect(saved[0].bbox).toBeUndefined();

    const reloaded = annotationsToCanvas(saved as unknown as Annotation[]);
    expect(reloaded.polygons).toHaveLength(1);
    expect(reloaded.boxes).toHaveLength(0);
  });
});

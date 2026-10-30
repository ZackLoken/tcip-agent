/**
 * The single mapping between the unified name-based label document (one annotation list per image)
 * and the Annotate canvas' drawing model (boxes + polygons + points + geometry-less ratings). Load
 * and save share this so the round-trip is symmetric: a box stays a box, a polygon stays a polygon,
 * a point stays a point, and a geometry-less rating is never silently dropped on the next save.
 */

import type { AnnotationPayload, ServedAnnotation } from "@/api/types.generated";
import { NO_CONTENT, type CanvasAnnotation, type CanvasContent, type Mode } from "@/store/types";

/** The served annotation's own facts its canvas shape carries: subject, attribute values, crowd
 *  flag, and the authorship and index that travel onto the shape and never back. */
function shapeOf(a: ServedAnnotation): CanvasAnnotation {
  return {
    subject: a.subject,
    attributes: { ...a.attributes },
    iscrowd: a.iscrowd,
    authorship: a.authorship,
    index: a.index,
  };
}

/** Split a unified annotation list into the canvas' four buckets, keyed on each annotation's own
 *  geometry: rings -> polygon, else bbox -> box, else point -> point, else geometry-less rating.
 *  Every ring of a polygon is kept: dropping the rest of an occlusion-split shape would show the
 *  reviewer a part of the object and save it back as the whole. */
export function annotationsToCanvas(annotations: ServedAnnotation[]): CanvasContent {
  const content = structuredClone(NO_CONTENT);
  for (const a of annotations) {
    if (a.rings !== null && a.rings.length) {
      content.polygons.push({
        rings: a.rings.map((ring) => ring.map(([x, y]): [number, number] => [x, y])),
        ...shapeOf(a),
      });
    } else if (a.bbox !== null) {
      const [x1, y1, x2, y2] = a.bbox;
      content.boxes.push({ x1, y1, x2, y2, ...shapeOf(a) });
    } else if (a.point !== null) {
      const [x, y] = a.point;
      content.points.push({ x, y, ...shapeOf(a) });
    } else {
      content.imageAnnotations.push(shapeOf(a));
    }
  }
  return content;
}

/** The canvas' four buckets reassembled into one unified annotation list for save, and where each
 *  shape of a tool landed in it: `positions[tool][i]` is the list position of the tool's ith
 *  shape. */
export function serializeCanvas(labels: CanvasContent): {
  annotations: AnnotationPayload[];
  positions: Record<Mode, number[]>;
} {
  const out: AnnotationPayload[] = [];
  const positions: Record<Mode, number[]> = { box: [], polygon: [], point: [] };
  // A shape's payload, stating its one geometry and none of the others.
  const payload = (
    a: CanvasAnnotation,
    geometry: Partial<Pick<AnnotationPayload, "bbox" | "points" | "rings" | "point">>,
  ): AnnotationPayload => ({
    subject: a.subject,
    bbox: null,
    points: null,
    rings: null,
    point: null,
    ...geometry,
    attributes: a.attributes,
    iscrowd: a.iscrowd,
  });
  const push = (tool: Mode, shape: AnnotationPayload) => {
    positions[tool].push(out.length);
    out.push(shape);
  };
  for (const b of labels.boxes) push("box", payload(b, { bbox: [b.x1, b.y1, b.x2, b.y2] }));
  for (const p of labels.polygons) {
    // One contour goes back as `points`, more than one as `rings`; never both.
    const geometry =
      p.rings.length === 1
        ? { points: p.rings[0].map(([x, y]) => [x, y]) }
        : { rings: p.rings.map((ring) => ring.map(([x, y]) => [x, y])) };
    push("polygon", payload(p, geometry));
  }
  // A point is one coordinate pair: it has no contour, so neither `points` nor `rings` applies.
  for (const p of labels.points) push("point", payload(p, { point: [p.x, p.y] }));
  for (const a of labels.imageAnnotations) out.push(payload(a, {}));
  return { annotations: out, positions };
}

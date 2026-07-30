/**
 * The single mapping between the unified name-based label file (one Annotation list per image)
 * and the Annotate canvas' drawing model (boxes + polygons + points + geometry-less ratings). Load
 * and save share this so the round-trip is symmetric — a box stays a box, a polygon stays a polygon,
 * a point stays a point, and a geometry-less rating is never silently dropped on the next save.
 */

import type { Annotation, AnnotationPayload, Box, PointShape, PolygonShape } from "@/store/types";

function provenance(a: {
  created_by?: string | null;
  created_at?: string | null;
  accepted_by?: string | null;
  accepted_at?: string | null;
}) {
  return {
    created_by: a.created_by ?? null,
    created_at: a.created_at ?? null,
    accepted_by: a.accepted_by ?? null,
    accepted_at: a.accepted_at ?? null,
  };
}

export interface CanvasLabels {
  boxes: Box[];
  polygons: PolygonShape[];
  points: PointShape[];
  imageAnnotations: Annotation[];
}

/** Split a unified annotation list into the canvas' four buckets, keyed on each annotation's own
 *  geometry: rings -> polygon, else bbox -> box, else point -> point, else geometry-less rating.
 *  Every ring of a polygon is kept — dropping the rest of an occlusion-split shape would show the
 *  reviewer a part of the object and save it back as the whole. */
export function annotationsToCanvas(annotations: Annotation[]): CanvasLabels {
  const boxes: Box[] = [];
  const polygons: PolygonShape[] = [];
  const points: PointShape[] = [];
  const imageAnnotations: Annotation[] = [];
  for (const a of annotations) {
    const attributes = { ...(a.attributes ?? {}) };
    if (a.rings && a.rings.length) {
      polygons.push({
        rings: a.rings.map((ring) => ring.map(([x, y]): [number, number] => [x, y])),
        subject: a.subject,
        attributes,
        ...provenance(a),
      });
    } else if (a.bbox) {
      const [x1, y1, x2, y2] = a.bbox;
      boxes.push({ x1, y1, x2, y2, subject: a.subject, attributes, ...provenance(a) });
    } else if (a.point) {
      const [x, y] = a.point;
      points.push({ x, y, subject: a.subject, attributes, ...provenance(a) });
    } else {
      imageAnnotations.push({ ...a, attributes });
    }
  }
  return { boxes, polygons, points, imageAnnotations };
}

/** Reassemble the canvas' four buckets into one unified annotation list for save. */
export function canvasToAnnotations(labels: CanvasLabels): AnnotationPayload[] {
  const out: AnnotationPayload[] = [];
  for (const b of labels.boxes) {
    out.push({
      subject: b.subject,
      bbox: [b.x1, b.y1, b.x2, b.y2],
      attributes: b.attributes ?? {},
      ...provenance(b),
    });
  }
  for (const p of labels.polygons) {
    // One contour goes back as `points` — the field a hand-drawn/hand-edited shape belongs in, and
    // the only one the Review edit route (`edited_points`) has. More than one goes back as `rings`,
    // which is the only field that can carry them. Never both: the backend prefers `rings`.
    const geometry =
      p.rings.length === 1
        ? { points: p.rings[0].map(([x, y]) => [x, y]) }
        : { rings: p.rings.map((ring) => ring.map(([x, y]) => [x, y])) };
    out.push({
      subject: p.subject,
      ...geometry,
      attributes: p.attributes ?? {},
      ...provenance(p),
    });
  }
  for (const p of labels.points) {
    // One coordinate pair, always — a point has no contour, so neither `points` nor `rings` applies.
    out.push({
      subject: p.subject,
      point: [p.x, p.y],
      attributes: p.attributes ?? {},
      ...provenance(p),
    });
  }
  for (const a of labels.imageAnnotations) {
    out.push({ subject: a.subject, attributes: a.attributes ?? {}, ...provenance(a) });
  }
  return out;
}

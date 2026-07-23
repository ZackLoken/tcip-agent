/**
 * The single mapping between the unified name-based label file (one Annotation list per image)
 * and the Annotate canvas' drawing model (boxes + polygons + geometry-less ratings). Load and
 * save share this so the round-trip is symmetric — a box stays a box, a polygon stays a polygon,
 * and a geometry-less rating is never silently dropped on the next save.
 */

import type { Annotation, AnnotationPayload, Box, PolygonShape } from "@/store/types";

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
  imageAnnotations: Annotation[];
}

/** Split a unified annotation list into the canvas' three buckets, keyed on each annotation's own
 *  geometry: points -> polygon, else bbox -> box, else geometry-less rating. */
export function annotationsToCanvas(annotations: Annotation[]): CanvasLabels {
  const boxes: Box[] = [];
  const polygons: PolygonShape[] = [];
  const imageAnnotations: Annotation[] = [];
  for (const a of annotations) {
    const attributes = { ...(a.attributes ?? {}) };
    if (a.points && a.points.length) {
      polygons.push({
        points: a.points.map(([x, y]): [number, number] => [x, y]),
        subject: a.subject,
        attributes,
        ...provenance(a),
      });
    } else if (a.bbox) {
      const [x1, y1, x2, y2] = a.bbox;
      boxes.push({ x1, y1, x2, y2, subject: a.subject, attributes, ...provenance(a) });
    } else {
      imageAnnotations.push({ ...a, attributes });
    }
  }
  return { boxes, polygons, imageAnnotations };
}

/** Reassemble the canvas' three buckets into one unified annotation list for save. */
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
    out.push({
      subject: p.subject,
      points: p.points.map(([x, y]) => [x, y]),
      attributes: p.attributes ?? {},
      ...provenance(p),
    });
  }
  for (const a of labels.imageAnnotations) {
    out.push({ subject: a.subject, attributes: a.attributes ?? {}, ...provenance(a) });
  }
  return out;
}

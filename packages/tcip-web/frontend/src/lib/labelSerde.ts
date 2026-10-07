/**
 * The single mapping between the unified name-based label document (one Annotation list per image)
 * and the Annotate canvas' drawing model (boxes + polygons + points + geometry-less ratings). Load
 * and save share this so the round-trip is symmetric: a box stays a box, a polygon stays a polygon,
 * a point stays a point, and a geometry-less rating is never silently dropped on the next save.
 */

import type {
  Annotation,
  AnnotationPayload,
  Box,
  CarriedFields,
  Mode,
  PointShape,
  PolygonShape,
} from "@/store/types";

/** The facts a shape carries through the canvas unchanged ({@link CarriedFields}). */
function carried(a: CarriedFields): CarriedFields {
  return { iscrowd: a.iscrowd };
}

/** The load route's facts about an annotation that travel onto its canvas shape and never back. */
function loaded(a: Annotation): { authorship: string | null; index?: number } {
  return { authorship: a.authorship ?? null, index: a.index };
}

export interface CanvasLabels {
  boxes: Box[];
  polygons: PolygonShape[];
  points: PointShape[];
  imageAnnotations: Annotation[];
}

/** Split a unified annotation list into the canvas' four buckets, keyed on each annotation's own
 *  geometry: rings -> polygon, else bbox -> box, else point -> point, else geometry-less rating.
 *  Every ring of a polygon is kept: dropping the rest of an occlusion-split shape would show the
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
        ...carried(a),
        ...loaded(a),
      });
    } else if (a.bbox) {
      const [x1, y1, x2, y2] = a.bbox;
      boxes.push({
        x1,
        y1,
        x2,
        y2,
        subject: a.subject,
        attributes,
        ...carried(a),
        ...loaded(a),
      });
    } else if (a.point) {
      const [x, y] = a.point;
      points.push({
        x,
        y,
        subject: a.subject,
        attributes,
        ...carried(a),
        ...loaded(a),
      });
    } else {
      imageAnnotations.push({ ...a, attributes });
    }
  }
  return { boxes, polygons, points, imageAnnotations };
}

/** The canvas' four buckets reassembled into one unified annotation list for save, and where each
 *  shape of a tool landed in it: `positions[tool][i]` is the list position of the tool's ith
 *  shape. */
export function serializeCanvas(labels: CanvasLabels): {
  annotations: AnnotationPayload[];
  positions: Record<Mode, number[]>;
} {
  const out: AnnotationPayload[] = [];
  const positions: Record<Mode, number[]> = { box: [], polygon: [], point: [] };
  const push = (tool: Mode, payload: AnnotationPayload) => {
    positions[tool].push(out.length);
    out.push(payload);
  };
  for (const b of labels.boxes) {
    push("box", {
      subject: b.subject,
      bbox: [b.x1, b.y1, b.x2, b.y2],
      attributes: b.attributes ?? {},
      ...carried(b),
    });
  }
  for (const p of labels.polygons) {
    // One contour goes back as `points`, more than one as `rings`; never both.
    const geometry =
      p.rings.length === 1
        ? { points: p.rings[0].map(([x, y]) => [x, y]) }
        : { rings: p.rings.map((ring) => ring.map(([x, y]) => [x, y])) };
    push("polygon", {
      subject: p.subject,
      ...geometry,
      attributes: p.attributes ?? {},
      ...carried(p),
    });
  }
  for (const p of labels.points) {
    // One coordinate pair, always: a point has no contour, so neither `points` nor `rings` applies.
    push("point", {
      subject: p.subject,
      point: [p.x, p.y],
      attributes: p.attributes ?? {},
      ...carried(p),
    });
  }
  for (const a of labels.imageAnnotations) {
    out.push({ subject: a.subject, attributes: a.attributes ?? {}, ...carried(a) });
  }
  return { annotations: out, positions };
}

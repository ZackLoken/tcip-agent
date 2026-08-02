/**
 * The single source of a review detection's geometry. A unified label file mixes bbox and polygon
 * annotations (possibly of different subjects) in one image, so the review overlay must render each
 * detection by its own annotation's geometry, never by a per-image "review kind" that would hide a
 * whole geometry kind (an unreviewed false-negative, measurement-critical).
 *
 * Both the on-canvas overlay (ReviewTab) and the agent's shape mirror (canvasSync.buildReviewShapes)
 * key on these helpers, so the two renders agree by construction.
 */

import type { Annotation, Detection, MatchesResponse } from "@/store/types";

export type ReviewGeom =
  | { kind: "box"; box: [number, number, number, number] }
  | { kind: "polygon"; rings: [number, number][][] }
  | { kind: "point"; point: [number, number] };

/** The geometry an annotation draws as: its own polygon if it has one, else its box, else its
 *  point. A geometry-less annotation (image/plant-level rating) has nothing to draw. Every drawable
 *  ring travels: a prediction awaiting review can be occlusion-split, and a reviewer accepting it
 *  must have seen all of it. A point stays a point: no enclosing box is derived for it, so nothing
 *  downstream can read a fabricated extent off a location. (compute_matches ignores points, so no
 *  detection references one yet; this keeps the render honest for when one does.) */
export function annotationGeometry(ann: Annotation | null | undefined): ReviewGeom | null {
  if (!ann) return null;
  const rings = (ann.rings ?? []).filter((ring) => ring.length >= 2);
  if (rings.length) {
    return {
      kind: "polygon",
      rings: rings.map((ring) => ring.map(([x, y]): [number, number] => [x, y])),
    };
  }
  if (ann.bbox) {
    const [x1, y1, x2, y2] = ann.bbox;
    return { kind: "box", box: [x1, y1, x2, y2] };
  }
  if (ann.point) {
    const [x, y] = ann.point;
    return { kind: "point", point: [x, y] };
  }
  return null;
}

/** The ground-truth annotation a detection references (set for TP/FN), or null. */
export function detGtAnnotation(d: Detection, m: MatchesResponse): Annotation | null {
  return d.gt_idx != null ? (m.gt[d.gt_idx] ?? null) : null;
}

/** The prediction annotation a detection references (set for TP/FP), or null. */
export function detPredAnnotation(d: Detection, m: MatchesResponse): Annotation | null {
  return d.pred_idx != null ? (m.preds[d.pred_idx] ?? null) : null;
}

/** The geometry to draw as a detection's outcome shape: the prediction for an FP, the ground
 *  truth for a TP/FN. */
export function detOutcomeGeometry(d: Detection, m: MatchesResponse): ReviewGeom | null {
  return annotationGeometry(d.det_type === "fp" ? detPredAnnotation(d, m) : detGtAnnotation(d, m));
}

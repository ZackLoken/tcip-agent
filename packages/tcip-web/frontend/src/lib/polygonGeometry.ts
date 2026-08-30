/**
 * Pure hit-testing and structural-edit helpers for the annotate canvas' polygon geometry.
 * Extracted so the per-mouse-move hover scan (the hot path under dense annotation workloads) can
 * be unit tested and micro-benchmarked, and so an axis-aligned bounding-box pre-filter can skip
 * the expensive ray-cast for the vast majority of polygons.
 */

import type { Box, PolygonShape } from "@/store/types";

/** Axis-aligned bounding box: [minX, minY, maxX, maxY]. */
export type Bbox = [number, number, number, number];

/** Ray-casting point-in-polygon test (even-odd rule). */
export function pointInPolygon(pt: [number, number], poly: [number, number][]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > pt[1] !== yj > pt[1] && pt[0] < ((xj - xi) * (pt[1] - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

export function polygonBbox(points: [number, number][]): Bbox {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const [x, y] of points) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  return [minX, minY, maxX, maxY];
}

/** The extent of every ring of one polygon combined: its whole footprint, not just ring 0. */
export function ringsBbox(rings: [number, number][][]): Bbox {
  let out: Bbox = [Infinity, Infinity, -Infinity, -Infinity];
  for (const ring of rings) {
    const [minX, minY, maxX, maxY] = polygonBbox(ring);
    out = [
      Math.min(out[0], minX),
      Math.min(out[1], minY),
      Math.max(out[2], maxX),
      Math.max(out[3], maxY),
    ];
  }
  return out;
}

/** True when `pt` falls in any ring. Rings of one annotation are disjoint parts of the same object
 *  (an occlusion-split instance), never holes, so containment in one part is containment. */
export function pointInRings(pt: [number, number], rings: [number, number][][]): boolean {
  return rings.some((ring) => pointInPolygon(pt, ring));
}

/** Precompute one bbox per polygon (memoize on the polygon list; O(vertices) once). */
export function computePolygonBboxes(polygons: { rings: [number, number][][] }[]): Bbox[] {
  return polygons.map((p) => ringsBbox(p.rings));
}

/** A polygon's read-only derived box (the axis-aligned bounds of every ring), for box-mode display
 *  only. Reuses ringsBbox (the same min/max the loader and COCO export re-derive), so it can't
 *  drift. */
export function derivedBoxFromPolygon(p: PolygonShape): Box {
  const [x1, y1, x2, y2] = ringsBbox(p.rings);
  return { x1, y1, x2, y2, subject: p.subject, attributes: {} };
}

/** One ring replaced, the rest of the annotation untouched (an edit belongs to one contour). */
export function withRing(p: PolygonShape, ringIdx: number, ring: [number, number][]): PolygonShape {
  const rings = p.rings.slice();
  rings[ringIdx] = ring;
  return { ...p, rings };
}

/** Distance from a point to a line segment, plus the interpolation fraction and the projected
 *  point: used to find the nearest polygon edge for the closest-edge vertex-insert gesture. */
export function pointToSegmentDist(
  px: number,
  py: number,
  ax: number,
  ay: number,
  bx: number,
  by: number,
): { dist: number; t: number; proj: [number, number] } {
  const dx = bx - ax;
  const dy = by - ay;
  const len_sq = dx * dx + dy * dy;
  if (len_sq === 0) {
    const d = Math.hypot(px - ax, py - ay);
    return { dist: d, t: 0, proj: [ax, ay] };
  }
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len_sq));
  const proj: [number, number] = [ax + t * dx, ay + t * dy];
  const d = Math.hypot(px - proj[0], py - proj[1]);
  return { dist: d, t, proj };
}

/**
 * Index of the point annotation nearest `pt` within `radius`, or null. A point has no interior to
 * fall inside, so its whole hit test is this proximity check: the caller passes the radius in image
 * units (screen px / view scale), which keeps the grab target the same size on screen at every zoom.
 * Nearest, not first within radius, so two points a few px apart hand you the one you aimed at
 * (the same rule as hitTestEdit's handles).
 */
export function findHitPoint(
  pt: [number, number],
  points: { x: number; y: number }[],
  radius: number,
): number | null {
  const [px, py] = pt;
  let best: number | null = null;
  let bestD = radius;
  for (let i = 0; i < points.length; i++) {
    const d = Math.hypot(points[i].x - px, points[i].y - py);
    if (d <= bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

/**
 * Index of the first polygon (in list order) containing `pt`, or null. The bbox
 * pre-filter rejects most polygons with four comparisons before the O(vertices) ray-cast
 * runs: the win that keeps hover responsive when hundreds–thousands of polygons are on
 * screen. `bboxes[i]` must correspond to `polygons[i]` (see computePolygonBboxes).
 */
export function findHoveredPolygon(
  pt: [number, number],
  polygons: { rings: [number, number][][] }[],
  bboxes: Bbox[],
): number | null {
  const [px, py] = pt;
  for (let i = 0; i < polygons.length; i++) {
    const bb = bboxes[i];
    if (!bb || px < bb[0] || px > bb[2] || py < bb[1] || py > bb[3]) continue;
    if (pointInRings(pt, polygons[i].rings)) return i;
  }
  return null;
}

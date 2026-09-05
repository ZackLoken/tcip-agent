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

/** One ring's boundary: a closed loop of vertices, the first not repeated at the end. */
export type Ring = [number, number][];

/** `cutRing`'s outcome: the two pieces the cut produced, or the reason it could not answer for it. */
export type CutRingResult = { rings: [Ring, Ring] } | { reason: string };

/** The refusal `cutRing` returns for every crossing count but two, and for a piece its own
 *  post-conditions reject: a segment that misses, one that starts inside, one laid along an edge,
 *  a cut across both arms of a concave shape, and a walk whose pieces fail to partition the
 *  parent all read the same to a breeder, since none of them is a cut the geometry can answer for. */
export const CUT_RING_REFUSAL =
  "The cut must cross the selected outline exactly twice, starting and ending outside it. " +
  "Cut a concave shape in more than one pass.";

const AREA_TOLERANCE = 1e-6;

function cross2(ax: number, ay: number, bx: number, by: number): number {
  return ax * by - ay * bx;
}

/** The ring's signed area (shoelace formula); a caller that wants a plain area takes its absolute value. */
function shoelaceArea(ring: Ring): number {
  let sum = 0;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    sum += ring[j][0] * ring[i][1] - ring[i][0] * ring[j][1];
  }
  return sum / 2;
}

/** Drops a closing duplicate of the first vertex and any run of consecutive duplicates, so a
 *  degenerate walk can't pass the vertex-count post-condition on repeated points. */
function distinctRing(ring: Ring): Ring {
  const out: Ring = [];
  for (const p of ring) {
    const last = out[out.length - 1];
    if (!last || last[0] !== p[0] || last[1] !== p[1]) out.push(p);
  }
  while (out.length > 1) {
    const first = out[0];
    const last = out[out.length - 1];
    if (first[0] === last[0] && first[1] === last[1]) out.pop();
    else break;
  }
  return out;
}

function isValidPiece(piece: Ring): boolean {
  const distinct = distinctRing(piece);
  return distinct.length >= 3 && Math.abs(shoelaceArea(distinct)) > AREA_TOLERANCE;
}

/**
 * Splits `ring` into the two pieces the segment `a`-`b` cuts it into, or refuses. The cut is valid
 * only when both endpoints fall outside the ring and the segment crosses its boundary exactly
 * twice (an interior crossing of an edge, or a vertex whose two neighboring edges lie on opposite
 * sides of the line through `a`/`b`; a vertex touched tangentially, with both neighbors on the
 * same side, is not a crossing, and an edge collinear with the segment contributes none either).
 * The two crossings, in ring order, become the shared chord between the two returned pieces, each
 * the walk along the original boundary from one crossing to the other.
 */
export function cutRing(ring: Ring, a: [number, number], b: [number, number]): CutRingResult {
  const n = ring.length;
  if (n < 3) return { reason: CUT_RING_REFUSAL };
  if (pointInPolygon(a, ring) || pointInPolygon(b, ring)) return { reason: CUT_RING_REFUSAL };

  const [ax, ay] = a;
  const [bx, by] = b;
  const dx = bx - ax;
  const dy = by - ay;
  const abLenSq = dx * dx + dy * dy;
  if (abLenSq === 0) return { reason: CUT_RING_REFUSAL };

  // Which side of the line through a/b each vertex falls on; 0 means exactly on that line.
  const sides = ring.map(([x, y]) => Math.sign(cross2(dx, dy, x - ax, y - ay)));
  // Each vertex's own projection fraction onto segment a-b (meaningful only where sides[k] === 0).
  const params = ring.map(([x, y]) => ((x - ax) * dx + (y - ay) * dy) / abLenSq);

  interface Crossing {
    order: number; // vertex index for a vertex touch; edgeIndex + edge-fraction for an edge interior
    point: [number, number];
  }
  const crossings: Crossing[] = [];

  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    const si = sides[i];
    const sj = sides[j];
    if (si === 0 || sj === 0) continue; // a vertex on the line: resolved once, below, never here
    if (si === sj) continue; // both strictly on the same side: this edge doesn't cross the line
    const [x1, y1] = ring[i];
    const [x2, y2] = ring[j];
    const ex = x2 - x1;
    const ey = y2 - y1;
    const denom = dx * ey - dy * ex;
    if (denom === 0) continue; // parallel; shouldn't happen with opposite sides, but stay defensive
    const t = ((x1 - ax) * ey - (y1 - ay) * ex) / denom;
    const u = ((x1 - ax) * dy - (y1 - ay) * dx) / denom;
    if (t <= 0 || t >= 1 || u <= 0 || u >= 1) continue; // meets the infinite line, not segment a-b
    crossings.push({ order: i + u, point: [ax + t * dx, ay + t * dy] });
  }

  for (let k = 0; k < n; k++) {
    if (sides[k] !== 0) continue;
    const t = params[k];
    if (t <= 0 || t >= 1) continue; // collinear with the line through a/b, but off the segment itself
    const prev = sides[(k - 1 + n) % n];
    const next = sides[(k + 1) % n];
    if (prev === 0 || next === 0) continue; // part of an edge collinear with a/b: no intersection
    if (prev !== next) crossings.push({ order: k, point: ring[k] }); // transversal: counts once
    // prev === next: a tangency (both neighbors on the same side), not a crossing.
  }

  if (crossings.length !== 2) return { reason: CUT_RING_REFUSAL };
  crossings.sort((c1, c2) => c1.order - c2.order);
  const [c1, c2] = crossings;

  // The ring in walk order, with each edge-interior crossing inserted at its own edge; a vertex
  // crossing needs no insertion; it is already ring[k], found by its integer order below.
  const seq: { order: number; point: [number, number] }[] = ring.map((p, i) => ({
    order: i,
    point: p,
  }));
  for (const c of crossings) {
    if (!Number.isInteger(c.order)) seq.push({ order: c.order, point: c.point });
  }
  seq.sort((p, q) => p.order - q.order);
  const idx1 = seq.findIndex((p) => p.order === c1.order);
  const idx2 = seq.findIndex((p) => p.order === c2.order);

  const pieceA: Ring = seq.slice(idx1, idx2 + 1).map((p) => p.point);
  const pieceB: Ring = seq
    .slice(idx2)
    .concat(seq.slice(0, idx1 + 1))
    .map((p) => p.point);

  if (!isValidPiece(pieceA) || !isValidPiece(pieceB)) return { reason: CUT_RING_REFUSAL };
  const parentArea = Math.abs(shoelaceArea(ring));
  const sumArea = Math.abs(shoelaceArea(pieceA)) + Math.abs(shoelaceArea(pieceB));
  if (Math.abs(sumArea - parentArea) > AREA_TOLERANCE) return { reason: CUT_RING_REFUSAL };

  return { rings: [pieceA, pieceB] };
}

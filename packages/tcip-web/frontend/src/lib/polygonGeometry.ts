/**
 * Pure polygon hit-testing helpers, shared by the annotate canvas. Extracted so the
 * per-mouse-move hover scan (the hot path under dense catkin workloads) can be unit
 * tested and micro-benchmarked, and so an axis-aligned bounding-box pre-filter can skip
 * the expensive ray-cast for the vast majority of polygons.
 */

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

/** Precompute one bbox per polygon (memoize on the polygon list; O(vertices) once). */
export function computePolygonBboxes(polygons: { points: [number, number][] }[]): Bbox[] {
  return polygons.map((p) => polygonBbox(p.points));
}

/**
 * Index of the first polygon (in list order) containing `pt`, or null. The bbox
 * pre-filter rejects most polygons with four comparisons before the O(vertices) ray-cast
 * runs — the win that keeps hover responsive when hundreds–thousands of polygons are on
 * screen. `bboxes[i]` must correspond to `polygons[i]` (see computePolygonBboxes).
 */
export function findHoveredPolygon(
  pt: [number, number],
  polygons: { points: [number, number][] }[],
  bboxes: Bbox[],
): number | null {
  const [px, py] = pt;
  for (let i = 0; i < polygons.length; i++) {
    const bb = bboxes[i];
    if (!bb || px < bb[0] || px > bb[2] || py < bb[1] || py > bb[3]) continue;
    if (pointInPolygon(pt, polygons[i].points)) return i;
  }
  return null;
}

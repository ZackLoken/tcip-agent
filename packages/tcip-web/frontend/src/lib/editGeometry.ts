/**
 * Pure geometry for in-place box/polygon editing, shared by the Annotate and Review tabs'
 * editors. Extracted so the drag math (corner-anchored resize, bounds-clamped moves, handle hit
 * tests) is unit-testable without Konva.
 */

import { MIN_BOX_SIDE, pointInPolygon, polygonBbox } from "@/lib/polygonGeometry";
import type { ReviewGeom } from "@/lib/reviewGeometry";

// Re-exported under this module's established name; the constant itself lives in
// polygonGeometry.ts so the cut tool's piece floor can share it without a circular import.
export { MIN_BOX_SIDE };

export type EditShape =
  | { kind: "box"; box: [number, number, number, number] }
  | { kind: "polygon"; points: [number, number][] };

export type EditDrag =
  | { mode: "corner"; ax: number; ay: number }
  | { mode: "vertex"; idx: number }
  | { mode: "move"; lastX: number; lastY: number };

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

/** Clamp a seeded shape into the image so drag math starts from a valid state
 *  (tiled predictions can carry slightly out-of-bounds coordinates). */
export function clampShapeToImage(shape: EditShape, w: number, h: number): EditShape {
  if (shape.kind === "box") {
    const [x1, y1, x2, y2] = shape.box;
    return {
      kind: "box",
      box: [clamp(x1, 0, w), clamp(y1, 0, h), clamp(x2, 0, w), clamp(y2, 0, h)],
    };
  }
  return {
    kind: "polygon",
    points: shape.points.map(([px, py]) => [clamp(px, 0, w), clamp(py, 0, h)]),
  };
}

/** What a mouse-down at (x, y) grabs: the nearest handle within `tol`, else the shape
 *  body (move), else nothing. Nearest, not first within tolerance, so small boxes and
 *  dense polygons hand you the corner you aimed at. */
export function hitTestEdit(shape: EditShape, x: number, y: number, tol: number): EditDrag | null {
  if (shape.kind === "box") {
    const [x1, y1, x2, y2] = shape.box;
    const corners: [number, number][] = [
      [x1, y1],
      [x2, y1],
      [x2, y2],
      [x1, y2],
    ];
    let best = -1;
    let bestDist = tol;
    for (let i = 0; i < 4; i++) {
      const d = Math.hypot(x - corners[i][0], y - corners[i][1]);
      if (d <= bestDist) {
        bestDist = d;
        best = i;
      }
    }
    if (best >= 0) {
      const [ax, ay] = corners[(best + 2) % 4]; // the opposite corner stays anchored
      return { mode: "corner", ax, ay };
    }
    if (x >= x1 && x <= x2 && y >= y1 && y <= y2) return { mode: "move", lastX: x, lastY: y };
    return null;
  }
  let best = -1;
  let bestDist = tol;
  for (let i = 0; i < shape.points.length; i++) {
    const d = Math.hypot(x - shape.points[i][0], y - shape.points[i][1]);
    if (d <= bestDist) {
      bestDist = d;
      best = i;
    }
  }
  if (best >= 0) return { mode: "vertex", idx: best };
  if (pointInPolygon([x, y], shape.points)) return { mode: "move", lastX: x, lastY: y };
  return null;
}

/** Apply one pointer move to the dragged shape, clamped to the image. Returns the same
 *  shape reference when nothing changed so callers can skip a re-render. */
export function applyEditDrag(
  shape: EditShape,
  drag: EditDrag,
  x: number,
  y: number,
  w: number,
  h: number,
): { shape: EditShape; drag: EditDrag } {
  const cx = clamp(x, 0, w);
  const cy = clamp(y, 0, h);
  if (shape.kind === "box") {
    if (drag.mode === "corner") {
      return {
        shape: {
          kind: "box",
          box: [
            Math.min(drag.ax, cx),
            Math.min(drag.ay, cy),
            Math.max(drag.ax, cx),
            Math.max(drag.ay, cy),
          ],
        },
        drag,
      };
    }
    if (drag.mode === "move") {
      const [x1, y1, x2, y2] = shape.box;
      const dx = clamp(x - drag.lastX, -x1, w - x2);
      const dy = clamp(y - drag.lastY, -y1, h - y2);
      const next: EditDrag = { mode: "move", lastX: x, lastY: y };
      if (!dx && !dy) return { shape, drag: next };
      return { shape: { kind: "box", box: [x1 + dx, y1 + dy, x2 + dx, y2 + dy] }, drag: next };
    }
    return { shape, drag };
  }
  if (drag.mode === "vertex") {
    const pts = shape.points.map((p, i): [number, number] => (i === drag.idx ? [cx, cy] : p));
    return { shape: { kind: "polygon", points: pts }, drag };
  }
  if (drag.mode === "move") {
    const [minX, minY, maxX, maxY] = polygonBbox(shape.points);
    const dx = clamp(x - drag.lastX, -minX, w - maxX);
    const dy = clamp(y - drag.lastY, -minY, h - maxY);
    const next: EditDrag = { mode: "move", lastX: x, lastY: y };
    if (!dx && !dy) return { shape, drag: next };
    return {
      shape: { kind: "polygon", points: shape.points.map(([px, py]) => [px + dx, py + dy]) },
      drag: next,
    };
  }
  return { shape, drag };
}

/** The shape Edit picks up, from the geometry the detection draws as (the matched GT for a TP/FN,
 *  what a save replaces, or the prediction for an FP, which a save adds). Deep-copied so dragging
 *  never mutates matches. Single-ring by construction: hand-editing adjusts one contour, and
 *  ``/review/action``'s ``edited_points`` carries exactly one (startEdit turns a multi-part shape
 *  away rather than seeding one part and saving it as the whole object). A point never gets here:
 *  the editor authors an outline, and there is no outline a location could stand in for. */
export function seedEditShape(geom: Exclude<ReviewGeom, { kind: "point" }>): EditShape {
  if (geom.kind === "box") return { kind: "box", box: geom.box };
  return { kind: "polygon", points: geom.rings[0].map((p): [number, number] => [p[0], p[1]]) };
}

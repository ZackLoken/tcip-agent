/**
 * The review path over one image: the items on the canvas (its annotations and, while a
 * bucket's proposals are shown, the undecided proposals), each item's review status, the
 * filters that keep a subset, the nearest-neighbor order laid over the kept items, and the
 * status scope that steps through a subsequence of that order.
 */

import { ringsBbox, type Bbox } from "@/lib/polygonGeometry";
import type { ReviewStatus } from "@/lib/symbology";
import type { Box, Mode, PointShape, PolygonShape, Proposal } from "@/store/types";

/** An item's geometry, named by the tool mode that draws and edits it. */
export type ItemShape = Mode;

export interface ReviewItem {
  kind: "annotation" | "proposal";
  shape: ItemShape;
  /** The index into the canvas array of its shape, or the proposal's index in its bucket. */
  ref: number;
  subject: string;
  bbox: Bbox;
  /** Null while no bucket's proposals are shown. */
  status: ReviewStatus | null;
  score: number | null;
  /** An annotation's index in the document it was loaded from; none for one drawn since. */
  index?: number;
}

export interface CanvasArrays {
  boxes: Box[];
  polygons: PolygonShape[];
  points: PointShape[];
}

/** An annotation's status under the shown bucket, by the proposals pairing with its document
 *  index: confirmed when a decision accepted one, undecided while one awaits a decision, else
 *  nothing proposed. A shape drawn since the load has no index and nothing proposed. */
export function annotationStatus(index: number | undefined, proposals: Proposal[]): ReviewStatus {
  if (index === undefined) return "unproposed";
  const pairing = proposals.filter((p) => p.paired === index);
  if (pairing.some((p) => p.decision === "accepted")) return "confirmed";
  if (pairing.some((p) => p.decision === null)) return "undecided";
  return "unproposed";
}

/** Each canvas array's statuses, aligned with it; every entry null while not reviewing. */
export function reviewStatuses(
  canvas: CanvasArrays,
  proposals: Proposal[],
  reviewing: boolean,
): {
  boxes: (ReviewStatus | null)[];
  polygons: (ReviewStatus | null)[];
  points: (ReviewStatus | null)[];
} {
  const of = (shape: { index?: number }) =>
    reviewing ? annotationStatus(shape.index, proposals) : null;
  return {
    boxes: canvas.boxes.map(of),
    polygons: canvas.polygons.map(of),
    points: canvas.points.map(of),
  };
}

export function boxBbox(b: Box): Bbox {
  return [b.x1, b.y1, b.x2, b.y2];
}

function proposalItems(proposals: Proposal[]): ReviewItem[] {
  const items: ReviewItem[] = [];
  for (const p of proposals) {
    if (p.decision !== null) continue;
    const bbox: Bbox | null = p.rings?.length
      ? ringsBbox(p.rings)
      : p.bbox
        ? p.bbox
        : p.point
          ? [p.point[0], p.point[1], p.point[0], p.point[1]]
          : null;
    if (!bbox) continue;
    items.push({
      kind: "proposal",
      shape: p.rings?.length ? "polygon" : p.bbox ? "box" : "point",
      ref: p.index,
      subject: p.subject,
      bbox,
      status: "undecided",
      score: p.score ?? null,
    });
  }
  return items;
}

/** Every item of the image: its boxes, polygons and points in canvas order, then, while
 *  reviewing, the shown bucket's undecided proposals. */
export function reviewItems(
  canvas: CanvasArrays,
  proposals: Proposal[],
  reviewing: boolean,
): ReviewItem[] {
  const statuses = reviewStatuses(canvas, proposals, reviewing);
  const annotation = (
    shape: ItemShape,
    ref: number,
    of: { subject: string; index?: number },
    bbox: Bbox,
    status: ReviewStatus | null,
  ): ReviewItem => ({
    kind: "annotation",
    shape,
    ref,
    subject: of.subject,
    bbox,
    status,
    score: null,
    index: of.index,
  });
  return [
    ...canvas.boxes.map((b, i) => annotation("box", i, b, boxBbox(b), statuses.boxes[i])),
    ...canvas.polygons.map((p, i) =>
      annotation("polygon", i, p, ringsBbox(p.rings), statuses.polygons[i]),
    ),
    ...canvas.points.map((p, i) =>
      annotation("point", i, p, [p.x, p.y, p.x, p.y], statuses.points[i]),
    ),
    ...(reviewing ? proposalItems(proposals) : []),
  ];
}

export interface ItemFilters {
  /** Proposals scoring below it are dropped; null keeps every proposal. Annotations always pass. */
  confidence: number | null;
  shape: ItemShape | "all";
}

export function keptItems(items: ReviewItem[], filters: ItemFilters): ReviewItem[] {
  return items.filter((item) => {
    if (filters.shape !== "all" && item.shape !== filters.shape) return false;
    if (item.kind === "proposal" && filters.confidence !== null) {
      return item.score !== null && item.score >= filters.confidence;
    }
    return true;
  });
}

function centroid(bbox: Bbox): [number, number] {
  return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2];
}

/** The items in a greedy nearest-neighbor tour of their centroids, starting from the one nearest
 *  the image's top-left corner, so stepping sweeps the image without crossing it. */
export function nearestNeighborOrder(items: ReviewItem[]): ReviewItem[] {
  const left = items.map((item, i) => ({ item, i, c: centroid(item.bbox) }));
  const order: ReviewItem[] = [];
  let at: [number, number] = [0, 0];
  while (left.length) {
    let best = 0;
    let bestD = Infinity;
    for (let k = 0; k < left.length; k++) {
      const d = Math.hypot(left[k].c[0] - at[0], left[k].c[1] - at[1]);
      if (d < bestD || (d === bestD && left[k].i < left[best].i)) {
        bestD = d;
        best = k;
      }
    }
    const [next] = left.splice(best, 1);
    order.push(next.item);
    at = next.c;
  }
  return order;
}

/** What the stepper steps through: every kept item, or the ones awaiting a decision. */
export const STATUS_SCOPES = ["all", "undecided"] as const;
export type StatusScope = (typeof STATUS_SCOPES)[number];

/** The order's items the scope keeps, in the order's own sequence. */
export function scopedOrder(order: ReviewItem[], scope: StatusScope): ReviewItem[] {
  return scope === "all" ? order : order.filter((item) => item.status === "undecided");
}

/** The item as the stepper names it: its subject and what it is, a proposal with its score. */
export function itemName(item: ReviewItem | null): string | null {
  if (!item) return null;
  const score = item.score !== null ? ` ${item.score.toFixed(2)}` : "";
  const what = item.kind === "proposal" ? `proposal${score}` : item.shape;
  return `${item.subject} ${what}`;
}

export function sameItem(a: ReviewItem | null, b: ReviewItem | null): boolean {
  return !!a && !!b && a.kind === b.kind && a.shape === b.shape && a.ref === b.ref;
}

/** The item `delta` steps from `focused` along `order`, wrapping; the first item when nothing is
 *  focused or the focused item left the order; null for an empty order. */
export function stepTarget(
  order: ReviewItem[],
  focused: ReviewItem | null,
  delta: number,
): ReviewItem | null {
  if (!order.length) return null;
  const at = order.findIndex((item) => sameItem(item, focused));
  if (at < 0) return order[delta < 0 ? order.length - 1 : 0];
  return order[(((at + delta) % order.length) + order.length) % order.length];
}

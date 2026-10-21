/**
 * The review path over one image: the items of every geometry (the annotations and, while a
 * bucket's proposals are shown, the undecided proposals pairing with none of them),
 * each item's match type, whether it has been reviewed and whether it is flagged, the filters
 * that keep a subset, the nearest-neighbor order laid over the kept items, and the scope that
 * steps through a subsequence of that order.
 */

import { pointInRings, ringsBbox, type Bbox } from "@/lib/polygonGeometry";
import { personBacked, type MatchType } from "@/lib/symbology";
import type { Focus } from "@/store/slices/canvas";
import {
  NO_CONTENT,
  TOOL_ARRAY,
  type Flag,
  type FlagRequest,
  type Mode,
  type Proposal,
  type ToolContent,
} from "@/store/types";

export interface ReviewItem {
  kind: "annotation" | "proposal";
  /** The item's geometry, named by the tool mode that draws and edits it. */
  shape: Mode;
  /** The index into the canvas array of its shape, or the proposal's index in its bucket. */
  ref: number;
  /** A proposal's bucket, when several buckets' proposals are shown together. */
  bucket?: string;
  subject: string;
  bbox: Bbox;
  /** Null while no bucket's proposals are shown. */
  match: MatchType | null;
  /** Whether a person stands behind it: for an annotation, the backend's authorship says a
   *  person made it or signed it off (`personBacked`), with or without a bucket shown; a proposal
   *  shown is always awaiting a decision. */
  reviewed: boolean;
  score: number | null;
  /** An annotation's index in the document it was loaded from; none for one drawn since. */
  index?: number;
  /** On a matched annotation: the undecided proposal pairing with it, if any, by its index in
   *  its bucket. */
  pairing?: { index: number; bucket?: string };
  /** The place a flag on this item names, inside the item. */
  at: [number, number];
  /** The open flags on it. */
  flags: Flag[];
}

/** An annotation's match under the shown bucket, by the proposals pairing with its document
 *  index. A shape drawn since the load has no index, so nothing pairs with it. */
export function annotationMatch(index: number | undefined, proposals: Proposal[]): MatchType {
  return pairOf(index, proposals) ? "matched" : "annotation_only";
}

/** The proposal pairing with the annotation of document `index`: the matcher pairs one to one,
 *  so there is at most one. */
function pairOf(index: number | undefined, proposals: Proposal[]): Proposal | undefined {
  return index === undefined ? undefined : proposals.find((p) => p.paired === index);
}

export function proposalMatch(p: Proposal): MatchType {
  return p.paired !== null ? "matched" : "proposal_only";
}

/** Each canvas array's match types, aligned with it; every entry null while not reviewing. */
export type MatchTypes = Record<keyof ToolContent, (MatchType | null)[]>;

/** A producer of the canvas's match types that recomputes an array's only when that array, the
 *  proposals or the review state changed since its last call, and otherwise hands back the array
 *  it computed then. */
export function matchTypes(): (
  canvas: ToolContent,
  proposals: Proposal[],
  reviewing: boolean,
) => MatchTypes {
  let last:
    | { canvas: ToolContent; proposals: Proposal[]; reviewing: boolean; matches: MatchTypes }
    | undefined;
  return (canvas, proposals, reviewing) => {
    const kept = last?.proposals === proposals && last.reviewing === reviewing ? last : undefined;
    const of = (array: keyof ToolContent) =>
      kept?.canvas[array] === canvas[array]
        ? kept.matches[array]
        : canvas[array].map((shape) =>
            reviewing ? annotationMatch(shape.index, proposals) : null,
          );
    const matches = Object.fromEntries(
      Object.values(TOOL_ARRAY).map((array) => [array, of(array)]),
    ) as MatchTypes;
    last = { canvas, proposals, reviewing, matches };
    return matches;
  };
}

/** No array's match types: every array empty, so a lookup misses and the consumer reads none. */
export const NO_MATCHES = matchTypes()(NO_CONTENT, [], false);

function center(bbox: Bbox): [number, number] {
  return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2];
}

/** The shape a proposal is drawn and reviewed as, with its bounding box; null for one carrying
 *  no geometry. */
export function proposalGeometry(p: Proposal): { shape: Mode; bbox: Bbox } | null {
  if (p.rings?.length) return { shape: "polygon", bbox: ringsBbox(p.rings) };
  if (p.bbox) return { shape: "box", bbox: p.bbox };
  if (p.point) return { shape: "point", bbox: [p.point[0], p.point[1], p.point[0], p.point[1]] };
  return null;
}

/** Every item of every geometry: the boxes, polygons and points in canvas order, each matched
 *  one carrying the undecided proposal pairing with it, then, while reviewing, the undecided
 *  proposals that pair with nothing. `flags` are the image's; each item holds its open ones. */
export function reviewItems(args: {
  canvas: ToolContent;
  proposals: Proposal[];
  /** The canvas's match types (`matchTypes`), resolved once for every consumer. */
  matches: MatchTypes;
  reviewing: boolean;
  flags: Flag[];
  bucket: string | null;
}): ReviewItem[] {
  const { canvas, proposals, reviewing } = args;
  const open = args.flags.filter((f) => f.resolved_by === null);
  const annotation = (
    shape: Mode,
    ref: number,
    of: { subject: string; index?: number; authorship?: string | null },
    bbox: Bbox,
    at: [number, number],
  ): ReviewItem => {
    const match = args.matches[TOOL_ARRAY[shape]][ref] ?? null;
    const pair = match === "matched" ? pairOf(of.index, proposals) : undefined;
    return {
      kind: "annotation",
      shape,
      ref,
      subject: of.subject,
      bbox,
      match,
      reviewed: personBacked(of.authorship),
      score: null,
      index: of.index,
      pairing:
        pair?.decision === null
          ? { index: pair.index, ...(pair.bucket !== undefined ? { bucket: pair.bucket } : {}) }
          : undefined,
      at,
      flags: open.filter(
        (f) =>
          f.point !== null &&
          f.subject === of.subject &&
          f.point[0] >= bbox[0] &&
          f.point[0] <= bbox[2] &&
          f.point[1] >= bbox[1] &&
          f.point[1] <= bbox[3],
      ),
    };
  };
  const annotations: ReviewItem[] = [
    ...canvas.boxes.map((b, i) => {
      const bbox: Bbox = [b.x1, b.y1, b.x2, b.y2];
      return annotation("box", i, b, bbox, center(bbox));
    }),
    ...canvas.polygons.map((p, i) => {
      const bbox = ringsBbox(p.rings);
      const mid = center(bbox);
      // The bbox center of a concave shape can fall outside it; a vertex never does.
      return annotation("polygon", i, p, bbox, pointInRings(mid, p.rings) ? mid : p.rings[0][0]);
    }),
    ...canvas.points.map((p, i) => annotation("point", i, p, [p.x, p.y, p.x, p.y], [p.x, p.y])),
  ];
  if (!reviewing) return annotations;
  const unpaired: ReviewItem[] = [];
  for (const p of proposals) {
    const geometry = proposalGeometry(p);
    if (p.decision !== null || p.paired !== null || !geometry) continue;
    unpaired.push({
      kind: "proposal",
      shape: geometry.shape,
      ref: p.index,
      ...(p.bucket !== undefined ? { bucket: p.bucket } : {}),
      subject: p.subject,
      bbox: geometry.bbox,
      match: "proposal_only",
      reviewed: false,
      score: p.score ?? null,
      at: center(geometry.bbox),
      flags: open.filter(
        (f) =>
          f.proposal !== null &&
          f.proposal[0] === (p.bucket ?? args.bucket) &&
          f.proposal[1] === p.index,
      ),
    });
  }
  return [...annotations, ...unpaired];
}

/** Where each of `flags` draws its mark: at its own point, or at the center of the proposal it
 *  names when that proposal is one of `proposals` under `bucket`; a flag on the image as a whole
 *  has no place. */
export function flagPlaces(
  flags: Flag[],
  proposals: Proposal[],
  bucket: string | null,
): { at: [number, number]; flag: Flag }[] {
  const places: { at: [number, number]; flag: Flag }[] = [];
  for (const flag of flags) {
    if (flag.point) places.push({ at: flag.point, flag });
    else if (flag.proposal) {
      const [flagBucket, index] = flag.proposal;
      const named = proposals.find((p) => p.index === index && (p.bucket ?? bucket) === flagBucket);
      const geometry = named ? proposalGeometry(named) : null;
      if (geometry) places.push({ at: center(geometry.bbox), flag });
    }
  }
  return places;
}

/** The open flags on the image as a whole. */
export function imageFlags(flags: Flag[]): Flag[] {
  return flags.filter((f) => f.resolved_by === null && f.point === null && f.proposal === null);
}

/** The flag a comment on `item` raises, or one on the image as a whole for no item. */
export function flagRequest(
  text: string,
  item: ReviewItem | null,
  bucket: string | null,
): FlagRequest {
  if (!item) return { text };
  const own = item.bucket ?? bucket;
  if (item.kind === "proposal" && own) return { text, proposal: [own, item.ref] };
  return { text, point: item.at, subject: item.subject };
}

export interface ItemFilters {
  /** Proposals scoring below it are dropped; null keeps every proposal. Annotations always pass. */
  confidence: number | null;
  match: MatchType | "all";
}

function clearsFloor(score: number | null | undefined, filters: ItemFilters): boolean {
  return filters.confidence === null || (score != null && score >= filters.confidence);
}

export function keptItems(items: ReviewItem[], filters: ItemFilters): ReviewItem[] {
  return items.filter(
    (item) =>
      (filters.match === "all" || item.match === filters.match) &&
      (item.kind !== "proposal" || clearsFloor(item.score, filters)),
  );
}

/** The proposals the canvas draws: the undecided ones carrying a geometry that clear the
 *  confidence floor and the match filter, the ones pairing with an annotation among them. */
export function shownProposals(proposals: Proposal[], filters: ItemFilters) {
  return proposals.filter(
    (p) =>
      p.decision === null &&
      proposalGeometry(p) !== null &&
      clearsFloor(p.score, filters) &&
      (filters.match === "all" || proposalMatch(p) === filters.match),
  );
}

/** The items in a greedy nearest-neighbor tour of their centers, starting from the one nearest
 *  the image's top-left corner, so stepping sweeps the image without crossing it. */
export function nearestNeighborOrder(items: ReviewItem[]): ReviewItem[] {
  const left = items.map((item, i) => ({ item, i, c: center(item.bbox) }));
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

/** Whether the item awaits a decision: an undecided proposal, paired or not, or an annotation no
 *  person stands behind. A person's own annotation carries no verdict on a bucket's proposal. */
export function awaitsDecision(item: ReviewItem): boolean {
  return item.kind === "proposal" || item.pairing !== undefined || !item.reviewed;
}

/** What the stepper steps through: every kept item, the unreviewed ones, or the flagged ones. */
export const STEP_SCOPES = ["all", "unreviewed", "flagged"] as const;
export type StepScope = (typeof STEP_SCOPES)[number];

export const STEP_SCOPE_WORDS: Record<StepScope, string> = {
  all: "all",
  unreviewed: "unreviewed",
  flagged: "flagged",
};

/** The order's items the scope keeps, in the order's own sequence. */
export function scopedOrder(order: ReviewItem[], scope: StepScope): ReviewItem[] {
  if (scope === "unreviewed") return order.filter(awaitsDecision);
  if (scope === "flagged") return order.filter((item) => item.flags.length > 0);
  return order;
}

/** The item as the stepper names it: its subject and what it is, a proposal with its score. */
export function itemName(item: ReviewItem | null): string | null {
  if (!item) return null;
  const score = item.score !== null ? ` ${item.score.toFixed(2)}` : "";
  const what = item.kind === "proposal" ? `proposal${score}` : item.shape;
  return `${item.subject} ${what}`;
}

export function sameItem(a: ReviewItem | null, b: ReviewItem | null): boolean {
  return (
    !!a &&
    !!b &&
    a.kind === b.kind &&
    a.shape === b.shape &&
    a.ref === b.ref &&
    a.bucket === b.bucket
  );
}

/** What an accept or a reject acts on: the focused item, or the first unreviewed one when nothing
 *  is focused; never another item than the focused one. An item's proposal is itself or the
 *  undecided one pairing with it. An accept on an annotation with no proposal to decide confirms
 *  it when no person stands behind it; a reject acts on proposals only, since removing an
 *  annotation is the delete key's act. */
export function decisionTarget(
  action: "accept" | "reject",
  focused: ReviewItem | null,
  unreviewed: ReviewItem[],
):
  | { kind: "proposal"; index: number; bucket?: string }
  | { kind: "confirm"; item: ReviewItem }
  | null {
  const item = focused ?? unreviewed[0];
  if (!item) return null;
  const proposal =
    item.kind === "proposal" ? { index: item.ref, bucket: item.bucket } : item.pairing;
  if (proposal !== undefined) {
    return {
      kind: "proposal",
      index: proposal.index,
      ...(proposal.bucket !== undefined ? { bucket: proposal.bucket } : {}),
    };
  }
  if (action === "accept" && !item.reviewed) {
    return { kind: "confirm", item };
  }
  return null;
}

/** Whether the focused item is the annotation of the tool `shape` at canvas index `ref`. */
export function focusesAnnotation(focused: ReviewItem | null, shape: Mode, ref: number): boolean {
  return focused?.kind === "annotation" && focused.shape === shape && focused.ref === ref;
}

/** Whether the focused item is the proposal at `index` in its bucket's document; `bucket`
 *  tells two buckets' proposals apart when both are shown. */
export function focusesProposal(
  focused: ReviewItem | null,
  index: number,
  bucket?: string,
): boolean {
  return (
    focused?.kind === "proposal" &&
    focused.ref === index &&
    (bucket === undefined || focused.bucket === undefined || focused.bucket === bucket)
  );
}

/** The item `focus` names, or null when none of `items` is it. A proposal pairing with an
 *  annotation is that annotation's item, never one of its own. */
export function focusedItem(
  items: ReviewItem[],
  proposals: Proposal[],
  focus: Focus | null,
): ReviewItem | null {
  if (!focus) return null;
  const { kind, index } = focus;
  if (kind !== "proposal") return items.find((i) => focusesAnnotation(i, kind, index)) ?? null;
  const named = proposals.find(
    (p) => p.index === index && (focus.bucket === undefined || p.bucket === focus.bucket),
  );
  const paired = named?.paired ?? null;
  return (
    items.find((i) => focusesProposal(i, index, focus.bucket)) ??
    items.find((i) => i.kind === "annotation" && paired !== null && i.index === paired) ??
    null
  );
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

/**
 * The review path over one image: the items of every geometry (the annotations and, while a
 * bucket's proposals are shown, the undecided proposals pairing with none of them),
 * each item's match type, whether it has been reviewed and whether it is flagged, the filters
 * that keep a subset, the nearest-neighbor order laid over the kept items, and the scope that
 * steps through a subsequence of that order.
 */

import type { FlagRequest, ServedFlag, ServedProposal } from "@/api/types.generated";
import { pointInRings, ringsBbox, type Bbox } from "@/lib/polygonGeometry";
import { personBacked, proposalLabel, type MatchType } from "@/lib/symbology";
import type { Focus } from "@/store/slices/canvas";
import {
  NO_CONTENT,
  TOOL_ARRAY,
  type CanvasAnnotation,
  type Mode,
  type Proposal,
  type ToolContent,
} from "@/store/types";

/** What an item of either kind carries. */
interface ItemBase {
  /** The item's geometry, named by the tool mode that draws and edits it. */
  shape: Mode;
  bbox: Bbox;
  /** Null while no bucket's proposals are shown. */
  match: MatchType | null;
  /** The place a flag on this item names, inside the item. */
  at: [number, number];
  /** The open flags on it. */
  flags: ServedFlag[];
}

/** An undecided proposal pairing with no annotation. */
export interface ProposalItem extends ItemBase {
  kind: "proposal";
  proposal: Proposal;
}

export interface AnnotationItem extends ItemBase {
  kind: "annotation";
  /** The index into the canvas array of its shape. */
  ref: number;
  /** The canvas annotation it is: its subject, its authorship and its index in the document it
   *  was loaded from (none for one drawn since). */
  annotation: CanvasAnnotation;
  /** On a matched annotation: the undecided proposal pairing with it, if any. */
  pairing?: Proposal;
}

export type ReviewItem = AnnotationItem | ProposalItem;

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
export function proposalGeometry(p: ServedProposal): { shape: Mode; bbox: Bbox } | null {
  if (p.rings?.length) return { shape: "polygon", bbox: ringsBbox(p.rings) };
  if (p.bbox) return { shape: "box", bbox: p.bbox };
  if (p.point) return { shape: "point", bbox: [p.point[0], p.point[1], p.point[0], p.point[1]] };
  return null;
}

/** Every item of every geometry: the boxes, polygons and points in canvas order, each matched
 *  one carrying the undecided proposal pairing with it, then, while reviewing, the undecided
 *  proposals that pair with nothing. `flags` are the image's; each item holds its open ones, an
 *  annotation those the document served as on its index. */
export function reviewItems(args: {
  canvas: ToolContent;
  proposals: Proposal[];
  /** The canvas's match types (`matchTypes`), resolved once for every consumer. */
  matches: MatchTypes;
  reviewing: boolean;
  flags: ServedFlag[];
}): ReviewItem[] {
  const { canvas, proposals, reviewing } = args;
  const open = args.flags.filter((f) => f.resolution === null);
  const annotation = (
    shape: Mode,
    ref: number,
    of: CanvasAnnotation,
    bbox: Bbox,
    at: [number, number],
  ): AnnotationItem => {
    const match = args.matches[TOOL_ARRAY[shape]][ref] ?? null;
    const pair = match === "matched" ? pairOf(of.index, proposals) : undefined;
    return {
      kind: "annotation",
      shape,
      ref,
      annotation: of,
      bbox,
      match,
      pairing: pair?.decision === null ? pair : undefined,
      at,
      flags: open.filter((f) => of.index !== undefined && f.covered_by.includes(of.index)),
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
  const unpaired: ProposalItem[] = [];
  for (const p of proposals) {
    const geometry = proposalGeometry(p);
    if (p.decision !== null || p.paired !== null || !geometry) continue;
    unpaired.push({
      kind: "proposal",
      proposal: p,
      shape: geometry.shape,
      bbox: geometry.bbox,
      match: "proposal_only",
      at: center(geometry.bbox),
      flags: open.filter(
        ({ request: { target } }) =>
          target !== null &&
          "bucket" in target &&
          target.bucket === p.bucket &&
          target.index === p.index,
      ),
    });
  }
  return [...annotations, ...unpaired];
}

/** Where each of `flags` draws its mark: at its own point, or at the center of the proposal it
 *  names when that proposal is one of `proposals`; a flag on the image as a whole has no place. */
export function flagPlaces(
  flags: ServedFlag[],
  proposals: Proposal[],
): { at: [number, number]; flag: ServedFlag }[] {
  const places: { at: [number, number]; flag: ServedFlag }[] = [];
  for (const flag of flags) {
    const { target } = flag.request;
    if (target === null) continue;
    if ("point" in target) places.push({ at: target.point, flag });
    else {
      const named = proposals.find((p) => p.index === target.index && p.bucket === target.bucket);
      const geometry = named ? proposalGeometry(named) : null;
      if (geometry) places.push({ at: center(geometry.bbox), flag });
    }
  }
  return places;
}

/** The open flags on the image as a whole. */
export function imageFlags(flags: ServedFlag[]): ServedFlag[] {
  return flags.filter((f) => f.resolution === null && f.request.target === null);
}

/** The flag a comment on `item` raises, or one on the image as a whole for no item. */
export function flagRequest(text: string, item: ReviewItem | null): FlagRequest {
  if (!item) return { text, target: null };
  if (item.kind === "proposal") {
    const { bucket, index } = item.proposal;
    return { text, target: { bucket, index } };
  }
  return { text, target: { point: item.at, subject: item.annotation.subject } };
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
      (item.kind !== "proposal" || clearsFloor(item.proposal.score, filters)),
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
  return (
    item.kind === "proposal" ||
    item.pairing !== undefined ||
    !personBacked(item.annotation.authorship)
  );
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

/** The item as the stepper names it: an annotation by its subject and shape, a proposal by the
 *  label it carries when focused. */
export function itemName(item: ReviewItem | null): string | null {
  if (!item) return null;
  if (item.kind === "annotation") return `${item.annotation.subject} ${item.shape}`;
  return proposalLabel(item.proposal, true);
}

export function sameItem(a: ReviewItem | null, b: ReviewItem | null): boolean {
  if (a?.kind === "proposal") return focusesProposal(b, a.proposal.index, a.proposal.bucket);
  return a?.kind === "annotation" && focusesAnnotation(b, a.shape, a.ref);
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
): { kind: "proposal"; proposal: Proposal } | { kind: "confirm"; item: AnnotationItem } | null {
  const item = focused ?? unreviewed[0];
  if (!item) return null;
  if (item.kind === "proposal") return { kind: "proposal", proposal: item.proposal };
  if (item.pairing !== undefined) return { kind: "proposal", proposal: item.pairing };
  if (action === "accept" && !personBacked(item.annotation.authorship)) {
    return { kind: "confirm", item };
  }
  return null;
}

/** Whether the focused item is the annotation of the tool `shape` at canvas index `ref`. */
export function focusesAnnotation(focused: ReviewItem | null, shape: Mode, ref: number): boolean {
  return focused?.kind === "annotation" && focused.shape === shape && focused.ref === ref;
}

/** Whether the focused item is the proposal at `index` in the document of `bucket`. */
export function focusesProposal(
  focused: ReviewItem | null,
  index: number,
  bucket: string,
): boolean {
  return (
    focused?.kind === "proposal" &&
    focused.proposal.index === index &&
    focused.proposal.bucket === bucket
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
  if (focus.kind !== "proposal") {
    const { kind, index } = focus;
    return items.find((i) => focusesAnnotation(i, kind, index)) ?? null;
  }
  const { index } = focus;
  const named = proposals.find((p) => p.index === index && p.bucket === focus.bucket);
  const paired = named?.paired ?? null;
  return (
    items.find((i) => focusesProposal(i, index, focus.bucket)) ??
    items.find(
      (i) => i.kind === "annotation" && paired !== null && i.annotation.index === paired,
    ) ??
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

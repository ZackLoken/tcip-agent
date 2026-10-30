import { describe, expect, it } from "vitest";

import type { Authorship, FlagRequest, ServedFlag } from "@/api/types.generated";
import {
  annotationMatch,
  awaitsDecision,
  decisionTarget,
  flagPlaces,
  flagRequest,
  focusedItem,
  imageFlags,
  keptItems,
  matchTypes,
  NO_MATCHES,
  nearestNeighborOrder,
  reviewItems,
  scopedOrder,
  shownProposals,
  stepTarget,
  type AnnotationItem,
  type ReviewItem,
} from "@/lib/reviewItems";
import type { Proposal } from "@/store/types";
import { ANNOTATE_ANSWERS } from "@/test/annotate";
import { servedFlag, servedProposal, servedResolution } from "@/test/labels";

/** The bucket the produced review's proposals were staged under, and another staging's. */
const { review: BUCKET, shaped: OTHER_BUCKET } = ANNOTATE_ANSWERS.buckets;

const proposal = (over: Partial<Proposal>): Proposal => ({
  ...servedProposal({ subject: "fruit", score: 0.9, bbox: [0, 0, 10, 10], ...over }),
  bucket: BUCKET,
});

const box = (x: number, y: number, index?: number, subject = "fruit") => ({
  x1: x,
  y1: y,
  x2: x + 10,
  y2: y + 10,
  subject,
  attributes: {},
  iscrowd: false,
  index,
});

const flag = (request: Partial<FlagRequest>, over: Partial<ServedFlag> = {}): ServedFlag =>
  servedFlag(request, { id: "f1", ...over });

const canvas = {
  boxes: [box(0, 0, 0), box(40, 0, 1), box(80, 0)],
  polygons: [
    {
      rings: [
        [
          [50, 50],
          [60, 50],
          [60, 60],
        ],
      ] as [number, number][][],
      subject: "leaf",
      attributes: {},
      iscrowd: false,
      index: 2,
    },
  ],
  points: [{ x: 90, y: 90, subject: "fruit", attributes: {}, iscrowd: false, index: 3 }],
};

const proposals = [
  proposal({ index: 0, bbox: [1, 1, 9, 9], paired: 0, decision: "accepted" }),
  proposal({ index: 1, bbox: [41, 1, 49, 9], paired: 1 }),
  proposal({ index: 2, bbox: [70, 70, 80, 80], score: 0.3 }),
  proposal({ index: 3, bbox: [20, 70, 30, 80], decision: "rejected" }),
  proposal({
    index: 4,
    bbox: null,
    rings: [
      [
        [0, 50],
        [10, 50],
        [10, 60],
      ],
    ],
  }),
];

/** The item's canvas index, or its proposal's index in its bucket. */
const refOf = (i: ReviewItem) => (i.kind === "proposal" ? i.proposal.index : i.ref);

const items = (reviewing = true, flags: ServedFlag[] = []) =>
  reviewItems({
    canvas,
    proposals,
    matches: matchTypes()(canvas, proposals, reviewing),
    reviewing,
    flags,
  });

describe("match types", () => {
  it("an annotation is matched when a proposal pairs with it and annotation-only otherwise", () => {
    expect(annotationMatch(0, proposals)).toBe("matched");
    expect(annotationMatch(7, proposals)).toBe("annotation_only");
    expect(annotationMatch(undefined, proposals)).toBe("annotation_only");
  });

  it("matchTypes aligns with each canvas array and is all null while not reviewing", () => {
    expect(matchTypes()(canvas, proposals, true).boxes).toEqual([
      "matched",
      "matched",
      "annotation_only",
    ]);
    expect(matchTypes()(canvas, proposals, false).boxes).toEqual([null, null, null]);
  });

  it("matchTypes keeps an unchanged array's match types when another array changes", () => {
    const matchesOf = matchTypes();
    const before = matchesOf(canvas, proposals, true);
    const after = matchesOf({ ...canvas, boxes: [...canvas.boxes, box(0, 40)] }, proposals, true);
    expect(after.boxes).not.toBe(before.boxes);
    expect(after.boxes.at(-1)).toBe("annotation_only");
    expect(after.polygons).toBe(before.polygons);
    expect(after.points).toBe(before.points);
    expect(matchesOf(canvas, [...proposals], true).polygons).not.toBe(before.polygons);
  });
});

describe("reviewItems", () => {
  it("lists every geometry's annotations in canvas order, then the unpaired proposals", () => {
    expect(items().map((i) => [i.kind, i.shape, refOf(i), i.match])).toEqual([
      ["annotation", "box", 0, "matched"],
      ["annotation", "box", 1, "matched"],
      ["annotation", "box", 2, "annotation_only"],
      ["annotation", "polygon", 0, "annotation_only"],
      ["annotation", "point", 0, "annotation_only"],
      ["proposal", "box", 2, "proposal_only"],
      ["proposal", "polygon", 4, "proposal_only"],
    ]);
    expect(items()[2]).toMatchObject({ annotation: canvas.boxes[2] });
  });

  it("a matched annotation carries the undecided proposal pairing with it, as one item", () => {
    const [decided, pending] = items();
    expect(decided).toMatchObject({ kind: "annotation", pairing: undefined });
    expect(pending).toMatchObject({ kind: "annotation", pairing: proposals[1] });
  });

  it("an annotation awaits review until a person stands behind it, with or without a bucket", () => {
    const reviewedWith = (authorship: Authorship | undefined, reviewing: boolean) =>
      !awaitsDecision(
        reviewItems({
          canvas: { ...canvas, boxes: [{ ...box(80, 0), authorship }] },
          proposals: [],
          matches: matchTypes()({ ...canvas, boxes: [box(80, 0)] }, [], reviewing),
          reviewing,
          flags: [],
        })[0],
      );
    for (const reviewing of [true, false]) {
      expect(reviewedWith("tool", reviewing)).toBe(false);
      expect(reviewedWith("unattributed", reviewing)).toBe(false);
      expect(reviewedWith("tool_accepted", reviewing)).toBe(true);
      expect(reviewedWith("person", reviewing)).toBe(true);
      expect(reviewedWith(undefined, reviewing)).toBe(true);
    }
  });

  it("a matched annotation reads the same fact as any other, whatever its proposals' decisions", () => {
    const tools = { ...canvas, boxes: [{ ...box(0, 0, 0), authorship: "tool" as const }] };
    const decided = [proposal({ index: 0, paired: 0, decision: "accepted" })];
    const listed = reviewItems({
      canvas: tools,
      proposals: decided,
      matches: matchTypes()(tools, decided, true),
      reviewing: true,
      flags: [],
    });
    expect(awaitsDecision(listed[0])).toBe(true);
  });

  it("a person's annotation with an undecided proposal pairing it still awaits that decision", () => {
    const listed = items();
    const personsWithUndecided = listed[1];
    expect(personsWithUndecided).toMatchObject({
      annotation: canvas.boxes[1],
      pairing: { index: 1 },
    });
    expect(scopedOrder(listed, "unreviewed").map((i) => [i.kind, refOf(i)])).toEqual([
      ["annotation", 1],
      ["proposal", 2],
      ["proposal", 4],
    ]);
    expect(awaitsDecision(listed[0])).toBe(false);
    expect(awaitsDecision(listed[2])).toBe(false);
  });

  it("lists the annotations alone, with no match, while not reviewing", () => {
    const listed = items(false);
    expect(listed).toHaveLength(5);
    expect(listed.every((i) => i.match === null)).toBe(true);
  });

  it("the match filter keeps one match type and the floor drops proposals under it", () => {
    const all = items();
    expect(keptItems(all, { confidence: null, match: "proposal_only" }).map(refOf)).toEqual([2, 4]);
    expect(
      keptItems(all, { confidence: 0.5, match: "all" }).filter((i) => i.kind === "proposal"),
    ).toEqual([all[6]]);
    expect(keptItems(all, { confidence: 0.95, match: "all" })).toHaveLength(5);
  });

  it("draws the undecided proposals of every geometry that clear the floor and the filter", () => {
    const drawn = (match: "all" | "matched" | "proposal_only", confidence: number | null) =>
      shownProposals(proposals, { confidence, match }).map((p) => p.index);
    expect(drawn("all", null)).toEqual([1, 2, 4]);
    expect(drawn("all", 0.5)).toEqual([1, 4]);
    expect(drawn("matched", null)).toEqual([1]);
    expect(drawn("proposal_only", null)).toEqual([2, 4]);
  });
});

describe("focusedItem", () => {
  const focusOn = (focus: Parameters<typeof focusedItem>[2]) =>
    focusedItem(items(), proposals, focus);

  it("names an annotation by its tool and canvas index, and a proposal by its own index", () => {
    expect(focusOn({ kind: "box", index: 2 })).toMatchObject({ kind: "annotation", ref: 2 });
    expect(focusOn({ kind: "polygon", index: 0 })).toMatchObject({ shape: "polygon", ref: 0 });
    expect(focusOn({ kind: "proposal", index: 2, bucket: BUCKET })).toMatchObject({
      kind: "proposal",
      proposal: proposals[2],
    });
  });

  it("a proposal pairing with an annotation is that annotation's item", () => {
    expect(focusOn({ kind: "proposal", index: 1, bucket: BUCKET })).toMatchObject({
      kind: "annotation",
      ref: 1,
    });
  });

  it("names nothing for no focus, an index no shape holds, a decided unpaired proposal or another bucket's", () => {
    expect(focusOn(null)).toBeNull();
    expect(focusOn({ kind: "polygon", index: 5 })).toBeNull();
    expect(focusOn({ kind: "proposal", index: 3, bucket: BUCKET })).toBeNull();
    expect(focusOn({ kind: "proposal", index: 2, bucket: OTHER_BUCKET })).toBeNull();
  });
});

describe("decisionTarget", () => {
  // The third box a tool drew and no person signed off.
  const tooled = {
    ...canvas,
    boxes: canvas.boxes.map((b, i) => (i === 2 ? { ...b, authorship: "tool" as const } : b)),
  };
  const unconfirmed = reviewItems({
    canvas: tooled,
    proposals,
    matches: matchTypes()(tooled, proposals, true),
    reviewing: true,
    flags: [],
  })[2];
  const unpaired = items()[5];

  it("acts on the focused item's proposal, whichever way it pairs", () => {
    expect(decisionTarget("accept", items()[1], [])).toEqual({
      kind: "proposal",
      proposal: proposals[1],
    });
    expect(decisionTarget("reject", unpaired, [])).toEqual({
      kind: "proposal",
      proposal: proposals[2],
    });
  });

  it("with nothing focused acts on the first unreviewed item", () => {
    expect(decisionTarget("accept", null, [unpaired])).toEqual({
      kind: "proposal",
      proposal: proposals[2],
    });
    expect(decisionTarget("accept", null, [])).toBeNull();
  });

  it("an accept confirms an annotation no person stands behind; a reject never acts on it", () => {
    expect(decisionTarget("accept", unconfirmed, [unpaired])).toEqual({
      kind: "confirm",
      item: unconfirmed,
    });
    expect(decisionTarget("reject", unconfirmed, [unpaired])).toBeNull();
  });

  it("an annotation a person stands behind has nothing to accept", () => {
    expect(decisionTarget("accept", items()[2], [unpaired])).toBeNull();
  });
});

describe("flags on items", () => {
  const flags = [
    flag({ target: { point: [5, 5], subject: "fruit" } }, { id: "on-box", covered_by: [0] }),
    flag({ target: { bucket: BUCKET, index: 2 } }, { id: "on-proposal" }),
    flag({}, { id: "on-image" }),
    flag(
      { target: { point: [45, 5], subject: "fruit" } },
      { id: "done", covered_by: [1], resolution: servedResolution() },
    ),
    flag({ target: { bucket: OTHER_BUCKET, index: 2 } }, { id: "other-bucket" }),
  ];

  it("an annotation holds no flag served on no annotation, though it lies in its bounding box", () => {
    // Inside the triangle's bounding box and outside its rings, which the backend serves on none.
    const listed = items(true, [
      flag({ target: { point: [51, 59], subject: "leaf" } }, { id: "beside" }),
    ]);
    expect(listed.flatMap((i) => i.flags)).toEqual([]);
  });

  it("each item holds its open flags: by the annotation served, or by bucket and index", () => {
    const listed = items(true, flags);
    expect(listed.map((i) => i.flags.map((f) => f.id))).toEqual([
      ["on-box"],
      [],
      [],
      [],
      [],
      ["on-proposal"],
      [],
    ]);
    expect(imageFlags(flags).map((f) => f.id)).toEqual(["on-image"]);
  });

  it("a comment lands on the focused item's own place, its proposal, or the image", () => {
    const [annotation, , , , , unpaired] = items();
    expect(flagRequest("x", annotation)).toEqual({
      text: "x",
      target: { point: [5, 5], subject: "fruit" },
    });
    expect(flagRequest("x", unpaired)).toEqual({
      text: "x",
      target: { bucket: BUCKET, index: 2 },
    });
    expect(flagRequest("x", null)).toEqual({ text: "x", target: null });
  });

  it("flags a concave polygon at a place inside it, never its empty bounding-box center", () => {
    const concave = {
      boxes: [],
      points: [],
      polygons: [
        {
          rings: [
            [
              [0, 0],
              [100, 0],
              [100, 100],
              [90, 100],
              [90, 10],
              [0, 10],
            ],
          ] as [number, number][][],
          subject: "leaf",
          attributes: {},
          iscrowd: false,
        },
      ],
    };
    const [item] = reviewItems({
      canvas: concave,
      proposals: [],
      matches: NO_MATCHES,
      reviewing: false,
      flags: [],
    });
    expect(item.at).toEqual([0, 0]);
  });

  it("places each open flag's mark at its point or its proposal's center", () => {
    const open = flags.filter((f) => f.resolution === null);
    expect(flagPlaces(open, proposals).map((p) => [p.flag.id, p.at])).toEqual([
      ["on-box", [5, 5]],
      ["on-proposal", [75, 75]],
    ]);
  });
});

describe("the review path", () => {
  const item = (
    ref: number,
    x: number,
    y: number,
    over: Partial<AnnotationItem> = {},
  ): AnnotationItem => ({
    kind: "annotation",
    shape: "box",
    ref,
    annotation: box(x, y),
    bbox: [x, y, x + 2, y + 2],
    match: null,
    at: [x + 1, y + 1],
    flags: [],
    ...over,
  });

  it("tours the items by nearest neighbor from the top-left corner", () => {
    const order = nearestNeighborOrder([
      item(0, 90, 90),
      item(1, 0, 0),
      item(2, 10, 0),
      item(3, 80, 90),
    ]);
    expect(order.map(refOf)).toEqual([1, 2, 3, 0]);
  });

  it("each step goes to the item nearest the one just left, not nearest the corner", () => {
    const order = nearestNeighborOrder([
      item(0, 0, 0),
      item(1, 100, 0),
      item(2, 100, 10),
      item(3, 0, 60),
    ]);
    expect(order.map(refOf)).toEqual([0, 3, 2, 1]);
  });

  it("a scope takes a subsequence of the order without re-touring it", () => {
    const tool = (x: number) => ({ ...box(x, 0), authorship: "tool" as const });
    const order = nearestNeighborOrder([
      item(0, 0, 0),
      item(1, 10, 0, { annotation: tool(10) }),
      item(2, 20, 0, { flags: [flag({})] }),
      item(3, 30, 0, { annotation: tool(30), flags: [flag({})] }),
    ]);
    expect(scopedOrder(order, "all").map(refOf)).toEqual([0, 1, 2, 3]);
    expect(scopedOrder(order, "unreviewed").map(refOf)).toEqual([1, 3]);
    expect(scopedOrder(order, "flagged").map(refOf)).toEqual([2, 3]);
  });

  it("steps along the order, wrapping, and starts at either end when nothing is focused", () => {
    const order = [item(0, 0, 0), item(1, 10, 0), item(2, 20, 0)];
    expect(stepTarget(order, null, 1)).toBe(order[0]);
    expect(stepTarget(order, null, -1)).toBe(order[2]);
    expect(stepTarget(order, order[2], 1)).toBe(order[0]);
    expect(stepTarget(order, order[0], -1)).toBe(order[2]);
    expect(stepTarget([], null, 1)).toBeNull();
  });
});

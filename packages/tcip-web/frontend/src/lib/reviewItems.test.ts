import { describe, expect, it } from "vitest";

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
  nearestNeighborOrder,
  reviewItems,
  scopedOrder,
  shownProposals,
  stepTarget,
  type ReviewItem,
} from "@/lib/reviewItems";
import type { Flag, Mode, Proposal } from "@/store/types";

const BUCKET = "m1/2026-01-01";

const proposal = (over: Partial<Proposal>): Proposal => ({
  subject: "fruit",
  attributes: {},
  iscrowd: false,
  score: 0.9,
  index: 0,
  paired: null,
  decision: null,
  bbox: [0, 0, 10, 10],
  ...over,
});

const box = (x: number, y: number, index?: number, subject = "fruit") => ({
  x1: x,
  y1: y,
  x2: x + 10,
  y2: y + 10,
  subject,
  attributes: {},
  index,
});

const flag = (over: Partial<Flag>): Flag => ({
  id: "f1",
  text: "second look",
  by: "user:a",
  at: "2026-01-01T00:00:00+00:00",
  point: null,
  subject: null,
  proposal: null,
  resolved_by: null,
  resolved_at: null,
  reply: "",
  removed: false,
  ...over,
});

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
      index: 2,
    },
  ],
  points: [{ x: 90, y: 90, subject: "fruit", attributes: {}, index: 3 }],
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

const items = (mode: Mode, reviewing = true, flags: Flag[] = []) =>
  reviewItems({
    canvas,
    proposals,
    matches: matchTypes(canvas, proposals, reviewing),
    reviewing,
    mode,
    flags,
    bucket: BUCKET,
  });

describe("match types", () => {
  it("an annotation is matched when a proposal pairs with it and annotation-only otherwise", () => {
    expect(annotationMatch(0, proposals)).toBe("matched");
    expect(annotationMatch(7, proposals)).toBe("annotation_only");
    expect(annotationMatch(undefined, proposals)).toBe("annotation_only");
  });

  it("matchTypes aligns with the canvas arrays and is all null while not reviewing", () => {
    expect(matchTypes(canvas, proposals, true).boxes).toEqual([
      "matched",
      "matched",
      "annotation_only",
    ]);
    expect(matchTypes(canvas, proposals, false).boxes).toEqual([null, null, null]);
  });
});

describe("reviewItems", () => {
  it("lists the selected tool's geometry only: its annotations, then its unpaired proposals", () => {
    expect(items("box").map((i) => [i.kind, i.ref, i.match, i.reviewed])).toEqual([
      ["annotation", 0, "matched", true],
      ["annotation", 1, "matched", true],
      ["annotation", 2, "annotation_only", true],
      ["proposal", 2, "proposal_only", false],
    ]);
    expect(items("polygon").map((i) => [i.kind, i.ref, i.match])).toEqual([
      ["annotation", 0, "annotation_only"],
      ["proposal", 4, "proposal_only"],
    ]);
    expect(items("point").map((i) => [i.kind, i.shape])).toEqual([["annotation", "point"]]);
  });

  it("a matched annotation carries the undecided proposal pairing with it, as one item", () => {
    const [decided, pending] = items("box");
    expect(decided.pairing).toBeUndefined();
    expect(pending.pairing).toBe(1);
  });

  it("an annotation awaits review until a person stands behind it, with or without a bucket", () => {
    const reviewedWith = (authorship: string | null | undefined, reviewing: boolean) =>
      reviewItems({
        canvas: { ...canvas, boxes: [{ ...box(80, 0), authorship }] },
        proposals: [],
        matches: matchTypes({ ...canvas, boxes: [{ ...box(80, 0), authorship }] }, [], reviewing),
        reviewing,
        mode: "box",
        flags: [],
        bucket: reviewing ? BUCKET : null,
      })[0].reviewed;
    for (const reviewing of [true, false]) {
      expect(reviewedWith("tool", reviewing)).toBe(false);
      expect(reviewedWith("unattributed", reviewing)).toBe(false);
      expect(reviewedWith("tool_accepted", reviewing)).toBe(true);
      expect(reviewedWith("person", reviewing)).toBe(true);
      expect(reviewedWith(undefined, reviewing)).toBe(true);
    }
  });

  it("a matched annotation reads the same fact as any other, whatever its proposals' decisions", () => {
    const tools = { ...canvas, boxes: [{ ...box(0, 0, 0), authorship: "tool" }] };
    const decided = [proposal({ index: 0, paired: 0, decision: "accepted" })];
    const listed = reviewItems({
      canvas: tools,
      proposals: decided,
      matches: matchTypes(tools, decided, true),
      reviewing: true,
      mode: "box",
      flags: [],
      bucket: BUCKET,
    });
    expect(listed[0].reviewed).toBe(false);
  });

  it("a person's annotation with an undecided proposal pairing it still awaits that decision", () => {
    const listed = items("box");
    const personsWithUndecided = listed[1];
    expect(personsWithUndecided).toMatchObject({ reviewed: true, pairing: 1 });
    expect(scopedOrder(listed, "unreviewed").map((i) => [i.kind, i.ref])).toEqual([
      ["annotation", 1],
      ["proposal", 2],
    ]);
    expect(awaitsDecision(listed[0])).toBe(false);
    expect(awaitsDecision(listed[2])).toBe(false);
  });

  it("lists the annotations alone, with no match, while not reviewing", () => {
    const listed = items("box", false);
    expect(listed).toHaveLength(3);
    expect(listed.every((i) => i.match === null)).toBe(true);
  });

  it("the match filter keeps one match type and the floor drops proposals under it", () => {
    const all = items("box");
    expect(keptItems(all, { confidence: null, match: "proposal_only" }).map((i) => i.ref)).toEqual([
      2,
    ]);
    expect(
      keptItems(all, { confidence: 0.5, match: "all" }).filter((i) => i.kind === "proposal"),
    ).toEqual([]);
    expect(keptItems(all, { confidence: 0.5, match: "all" })).toHaveLength(3);
  });

  it("draws the undecided proposals of the tool's geometry that clear the floor and the filter", () => {
    const drawn = (match: "all" | "matched" | "proposal_only", confidence: number | null) =>
      shownProposals(proposals, { confidence, match }, "box").map((p) => p.index);
    expect(drawn("all", null)).toEqual([1, 2]);
    expect(drawn("all", 0.5)).toEqual([1]);
    expect(drawn("matched", null)).toEqual([1]);
    expect(drawn("proposal_only", null)).toEqual([2]);
    expect(shownProposals(proposals, { confidence: null, match: "all" }, "polygon")).toHaveLength(
      1,
    );
  });
});

describe("focusedItem", () => {
  const focusOn = (focus: Parameters<typeof focusedItem>[2]) =>
    focusedItem(items("box"), proposals, focus);

  it("names an annotation by its tool and canvas index, and a proposal by its own index", () => {
    expect(focusOn({ kind: "box", index: 2 })).toMatchObject({ kind: "annotation", ref: 2 });
    expect(focusOn({ kind: "proposal", index: 2 })).toMatchObject({ kind: "proposal", ref: 2 });
  });

  it("a proposal pairing with an annotation is that annotation's item", () => {
    expect(focusOn({ kind: "proposal", index: 1 })).toMatchObject({ kind: "annotation", ref: 1 });
  });

  it("names nothing for no focus, another tool's shape or a decided unpaired proposal", () => {
    expect(focusOn(null)).toBeNull();
    expect(focusOn({ kind: "polygon", index: 0 })).toBeNull();
    expect(focusOn({ kind: "proposal", index: 3 })).toBeNull();
  });
});

describe("decisionTarget", () => {
  const unconfirmed: ReviewItem = {
    ...items("box")[2],
    reviewed: false,
  };
  const pairedProposalOf = (reviewed: boolean): ReviewItem => ({
    ...items("box")[1],
    reviewed,
  });
  const unpaired = items("box")[3];

  it("acts on the focused item's proposal, whichever way it pairs", () => {
    expect(decisionTarget("accept", pairedProposalOf(true), [])).toEqual({
      kind: "proposal",
      index: 1,
    });
    expect(decisionTarget("reject", unpaired, [])).toEqual({ kind: "proposal", index: 2 });
  });

  it("with nothing focused acts on the first unreviewed item", () => {
    expect(decisionTarget("accept", null, [unpaired])).toEqual({ kind: "proposal", index: 2 });
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
    expect(decisionTarget("accept", items("box")[2], [unpaired])).toBeNull();
  });
});

describe("flags on items", () => {
  const flags = [
    flag({ id: "on-box", point: [5, 5], subject: "fruit" }),
    flag({ id: "on-proposal", proposal: [BUCKET, 2] }),
    flag({ id: "on-image" }),
    flag({ id: "done", point: [45, 5], subject: "fruit", resolved_by: "user:b" }),
    flag({ id: "other-bucket", proposal: ["m2/2026-01-01", 2] }),
  ];

  it("a flag at an item's place on another subject's mark is not that item's", () => {
    const listed = items("box", true, [flag({ id: "other", point: [5, 5], subject: "leaf" })]);
    expect(listed.flatMap((i) => i.flags)).toEqual([]);
  });

  it("each item holds its open flags: by place and subject, or by bucket and index", () => {
    const listed = items("box", true, flags);
    expect(listed.map((i) => i.flags.map((f) => f.id))).toEqual([
      ["on-box"],
      [],
      [],
      ["on-proposal"],
    ]);
    expect(imageFlags(flags).map((f) => f.id)).toEqual(["on-image"]);
  });

  it("a comment lands on the focused item's own place, its proposal, or the image", () => {
    const [annotation, , , unpaired] = items("box");
    expect(flagRequest("x", annotation, BUCKET)).toEqual({
      text: "x",
      point: [5, 5],
      subject: "fruit",
    });
    expect(flagRequest("x", unpaired, BUCKET)).toEqual({ text: "x", proposal: [BUCKET, 2] });
    expect(flagRequest("x", null, BUCKET)).toEqual({ text: "x" });
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
        },
      ],
    };
    const [item] = reviewItems({
      canvas: concave,
      proposals: [],
      matches: matchTypes(concave, [], false),
      reviewing: false,
      mode: "polygon",
      flags: [],
      bucket: null,
    });
    expect(item.at).toEqual([0, 0]);
  });

  it("places each open flag's mark at its point or its proposal's center", () => {
    const open = flags.filter((f) => f.resolved_by === null);
    expect(flagPlaces(open, proposals, BUCKET).map((p) => [p.flag.id, p.at])).toEqual([
      ["on-box", [5, 5]],
      ["on-proposal", [75, 75]],
    ]);
  });
});

describe("the review path", () => {
  const item = (ref: number, x: number, y: number, over: Partial<ReviewItem> = {}): ReviewItem => ({
    kind: "annotation",
    shape: "box",
    ref,
    subject: "fruit",
    bbox: [x, y, x + 2, y + 2],
    match: null,
    reviewed: true,
    score: null,
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
    expect(order.map((i) => i.ref)).toEqual([1, 2, 3, 0]);
  });

  it("each step goes to the item nearest the one just left, not nearest the corner", () => {
    const order = nearestNeighborOrder([
      item(0, 0, 0),
      item(1, 100, 0),
      item(2, 100, 10),
      item(3, 0, 60),
    ]);
    expect(order.map((i) => i.ref)).toEqual([0, 3, 2, 1]);
  });

  it("a scope takes a subsequence of the order without re-touring it", () => {
    const order = nearestNeighborOrder([
      item(0, 0, 0, { reviewed: true }),
      item(1, 10, 0, { reviewed: false }),
      item(2, 20, 0, { flags: [flag({})] }),
      item(3, 30, 0, { reviewed: false, flags: [flag({})] }),
    ]);
    expect(scopedOrder(order, "all").map((i) => i.ref)).toEqual([0, 1, 2, 3]);
    expect(scopedOrder(order, "unreviewed").map((i) => i.ref)).toEqual([1, 3]);
    expect(scopedOrder(order, "flagged").map((i) => i.ref)).toEqual([2, 3]);
  });

  it("steps along the order, wrapping, and starts at either end when nothing is focused", () => {
    const order = [item(0, 0, 0), item(1, 10, 0), item(2, 20, 0)];
    expect(stepTarget(order, null, 1)?.ref).toBe(0);
    expect(stepTarget(order, null, -1)?.ref).toBe(2);
    expect(stepTarget(order, order[2], 1)?.ref).toBe(0);
    expect(stepTarget(order, order[0], -1)?.ref).toBe(2);
    expect(stepTarget([], null, 1)).toBeNull();
  });
});

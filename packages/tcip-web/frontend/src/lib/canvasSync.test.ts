import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { subjectColor } from "@/api/subjects";
import {
  buildAnnotateShapes,
  computeViewport,
  createCanvasPusher,
  measureCanvasHost,
  notifyCanvasStateRequest,
  onCanvasStateRequest,
  shapeVisible,
  type CanvasStateBody,
} from "@/lib/canvasSync";
import { ringsBbox } from "@/lib/polygonGeometry";
import type { ReviewItem } from "@/lib/reviewItems";
import { FLAG_MARK, FOCUS_HALO, MATCH_COLORS } from "@/lib/symbology";
import type { Proposal } from "@/store/types";

const RED = subjectColor("subject_a");
const OTHER = subjectColor("other");

const focusOn = (shape: ReviewItem["shape"], ref: number): ReviewItem => ({
  kind: "annotation",
  shape,
  ref,
  subject: "",
  bbox: [0, 0, 0, 0],
  match: null,
  reviewed: null,
  score: null,
  at: [0, 0],
  flags: [],
});

const HALO = {
  color: FOCUS_HALO.color,
  opacity: FOCUS_HALO.opacity,
  width_factor: FOCUS_HALO.widthFactor,
};

describe("computeViewport", () => {
  it("maps pan/zoom to the visible image region", () => {
    const v = computeViewport(
      { scale: 2, offset_x: -100, offset_y: -40 },
      { w: 400, h: 200 },
      1000,
      800,
    );
    expect(v).toEqual({ x: 50, y: 20, w: 200, h: 100, scale: 2 });
  });

  it("clamps to the image bounds", () => {
    const v = computeViewport(
      { scale: 1, offset_x: 50, offset_y: 50 },
      { w: 400, h: 200 },
      100,
      80,
    );
    expect(v).toEqual({ x: 0, y: 0, w: 100, h: 80, scale: 1 });
  });

  it("returns null when nothing of the image is visible", () => {
    const v = computeViewport(
      { scale: 1, offset_x: -5000, offset_y: 0 },
      { w: 400, h: 200 },
      100,
      80,
    );
    expect(v).toBeNull();
  });
});

describe("measureCanvasHost", () => {
  afterEach(() => {
    document.body.innerHTML = "";
  });

  const mountHost = (width: number, height: number) => {
    const el = document.createElement("div");
    el.setAttribute("data-canvas-host", "");
    // jsdom lays nothing out, so the measured rect has to be supplied by the test.
    el.getBoundingClientRect = () => ({ width, height, x: 0, y: 0, top: 0, left: 0 }) as DOMRect;
    document.body.appendChild(el);
    return el;
  };

  it("reports the mounted host's width and height, keeping them in that order", () => {
    mountHost(640, 360);
    expect(measureCanvasHost()).toEqual({ w: 640, h: 360 });
  });

  it("returns null when no canvas host is mounted", () => {
    expect(measureCanvasHost()).toBeNull();
  });

  it("returns null when the host has collapsed to a sliver", () => {
    mountHost(1, 360);
    expect(measureCanvasHost()).toBeNull();
  });
});

describe("buildAnnotateShapes", () => {
  const base = {
    boxes: [] as {
      x1: number;
      y1: number;
      x2: number;
      y2: number;
      subject: string;
      attributes: Record<string, string>;
    }[],
    polygons: [
      {
        rings: [
          [
            [0, 0],
            [10, 0],
            [10, 10],
          ],
        ] as [number, number][][],
        subject: "subject_a",
        attributes: {},
      },
      {
        rings: [
          [
            [20, 20],
            [30, 20],
            [30, 30],
          ],
        ] as [number, number][][],
        subject: "other",
        attributes: {},
      },
    ],
    currentPolygon: [] as [number, number][],
    mode: "polygon",
    activeSubject: "subject_a",
    visible: true,
    colorFor: subjectColor,
  };

  it("filters polygon mode to the active subject, colors from the GUI, unlabeled at rest", () => {
    const shapes = buildAnnotateShapes(base);
    expect(shapes).toHaveLength(1); // "other" filtered out (not focused)
    expect(shapes[0]).toMatchObject({ kind: "polygon", color: RED, tag: "gt" });
    expect(shapes[0].label).toBeUndefined();
    expect(shapes[0].halo).toBeUndefined();
  });

  it("a focused polygon of another subject is included, haloed and labeled, its color kept", () => {
    const shapes = buildAnnotateShapes({ ...base, focused: focusOn("polygon", 1) });
    expect(shapes).toHaveLength(2);
    expect(shapes[1]).toMatchObject({ color: OTHER, halo: HALO, label: "other" });
  });

  it("an in-progress drawing rides along as a dashed polyline in the active subject's color", () => {
    // The mirror must match the real canvas's InProgressPolygon stroke, the active subject's color:
    // a divergence here is what capture_live_canvas would show that the breeder's own screen does not.
    const shapes = buildAnnotateShapes({
      ...base,
      currentPolygon: [
        [1, 1],
        [2, 2],
      ],
    });
    expect(shapes.at(-1)).toMatchObject({
      kind: "polyline",
      tag: "in_progress",
      dashed: true,
      color: RED,
    });
  });

  it("falls back to amber only when nothing is selected (drawing is otherwise blocked)", () => {
    const shapes = buildAnnotateShapes({
      ...base,
      activeSubject: "",
      currentPolygon: [
        [1, 1],
        [2, 2],
      ],
    });
    expect(shapes.at(-1)).toMatchObject({ kind: "polyline", color: "#FFE7B1" });
  });

  it("a pending cut start rides as an in_progress polyline labeled cut, start and cursor", () => {
    const shapes = buildAnnotateShapes({
      ...base,
      cutStart: { point: [3, 4], color: "#123456" },
      cursor: [5, 6],
    });
    expect(shapes.at(-1)).toMatchObject({
      kind: "polyline",
      tag: "in_progress",
      label: "cut",
      dashed: true,
      color: "#123456",
      points: [
        [3, 4],
        [5, 6],
      ],
    });
  });

  it("a pending cut start with no cursor yet rides as the start point alone", () => {
    const shapes = buildAnnotateShapes({
      ...base,
      cutStart: { point: [3, 4], color: "#123456" },
      cursor: null,
    });
    expect(shapes.at(-1)).toMatchObject({ kind: "polyline", label: "cut", points: [[3, 4]] });
  });

  it("no cut polyline rides when no start is pending", () => {
    const shapes = buildAnnotateShapes({ ...base, cutStart: null, cursor: [5, 6] });
    expect(shapes.some((s) => s.label === "cut")).toBe(false);
  });

  it("the labels toggle hides everything, exactly like the canvas", () => {
    expect(buildAnnotateShapes({ ...base, visible: false })).toEqual([]);
  });

  it("box mode renders the active-subject editable boxes solid (the 'other' subject filtered out)", () => {
    const shapes = buildAnnotateShapes({
      ...base,
      mode: "box",
      polygons: [], // isolate the editable-box behavior from the derived-box overlay
      boxes: [
        { x1: 12, y1: 7, x2: 41, y2: 23, subject: base.activeSubject, attributes: {} },
        { x1: 60, y1: 3, x2: 71, y2: 19, subject: "other", attributes: {} },
      ],
    });
    // Only the active subject's real box renders, solid (editable).
    expect(shapes).toHaveLength(1);
    expect(shapes[0]).toMatchObject({ kind: "box", xyxy: [12, 7, 41, 23] });
    expect(shapes[0].dashed).toBeFalsy();
  });

  it("an editable box keeps x before y in the wire tuple its renderer reads", () => {
    // Pairwise-distinct coordinates so a slip in the server-consumed [x1, y1, x2, y2] order cannot hide.
    const shapes = buildAnnotateShapes({
      ...base,
      mode: "box",
      polygons: [],
      boxes: [
        { x1: 33.04, y1: 6.06, x2: 90.11, y2: 58.02, subject: base.activeSubject, attributes: {} },
      ],
    });
    expect(shapes).toHaveLength(1);
    expect(shapes[0].xyxy).toEqual([33, 6.1, 90.1, 58]);
  });

  it("box mode adds one read-only derived box per active-subject polygon, === ringsBbox, unlabeled", () => {
    // Mirrors the canvas overlay: a polygon's detection footprint shows while boxing, and its coords
    // are exactly ringsBbox, never a stored box; it draws in the polygon's own line style.
    const shapes = buildAnnotateShapes({ ...base, mode: "box", boxes: [] });
    const derived = shapes.filter((s) => s.kind === "box");
    expect(derived).toHaveLength(1); // only the active "subject_a" polygon; "other" is filtered out
    expect(derived[0].xyxy).toEqual(ringsBbox(base.polygons[0].rings));
    expect(derived[0].label).toBeUndefined();
    expect(derived[0].dashed).toBeFalsy();
  });

  it("pushes every ring of a multi-ring polygon, sharing its color, labeled once when focused", () => {
    // The agent's view of the canvas must not drop a region either: an occlusion-split subject_a is one
    // annotation drawn as two paths (render_canvas_state draws one path per shape entry).
    const multi = {
      rings: [
        [
          [0, 0],
          [10, 0],
          [10, 10],
        ],
        [
          [40, 40],
          [60, 40],
          [60, 60],
        ],
      ] as [number, number][][],
      subject: "subject_a",
      attributes: {},
    };
    const shapes = buildAnnotateShapes({
      ...base,
      polygons: [multi],
      focused: focusOn("polygon", 0),
    });
    expect(shapes).toHaveLength(2);
    expect(shapes.map((s) => s.points)).toEqual(multi.rings);
    expect(shapes.every((s) => s.color === RED && s.tag === "gt" && s.halo)).toBe(true);
    // Labeled once: a two-part subject_a is one subject_a, not two.
    expect(shapes.filter((s) => s.label === "subject_a")).toHaveLength(1);
  });

  it("box mode derives one box spanning every ring of a multi-ring polygon", () => {
    const multi = {
      rings: [
        [
          [0, 0],
          [10, 0],
          [10, 10],
        ],
        [
          [40, 40],
          [60, 40],
          [60, 60],
        ],
      ] as [number, number][][],
      subject: "subject_a",
      attributes: {},
    };
    const shapes = buildAnnotateShapes({ ...base, mode: "box", boxes: [], polygons: [multi] });
    const derived = shapes.filter((s) => s.kind === "box");
    expect(derived).toHaveLength(1);
    expect(derived[0].xyxy).toEqual([0, 0, 60, 60]);
  });

  it("point mode pushes each active-subject point as its own point shape (no box, no path)", () => {
    // The agent's view of the canvas has to include placed points, and it must not see a box the
    // annotation never claimed: a fabricated extent here is the exact hazard Point warns about.
    const shapes = buildAnnotateShapes({
      ...base,
      mode: "point",
      polygons: [],
      points: [
        { x: 5.06, y: 7.04, subject: "subject_a", attributes: {} },
        { x: 50, y: 60, subject: "other", attributes: {} },
      ],
    });
    expect(shapes).toHaveLength(1); // "other" filtered out, exactly like the box/polygon rules
    expect(shapes[0]).toMatchObject({
      kind: "point",
      points: [[5.1, 7]], // rounded like every other pushed coordinate
      color: RED,
      tag: "gt",
    });
    expect(shapes[0].xyxy).toBeUndefined();
  });

  it("the focused point is pushed haloed in its own color, and no point draws outside point mode", () => {
    const points = [{ x: 5, y: 7, subject: "other", attributes: {} }];
    const focused = buildAnnotateShapes({
      ...base,
      mode: "point",
      polygons: [],
      points,
      focused: focusOn("point", 0),
    });
    expect(focused).toHaveLength(1); // included despite the subject filter, like a focused polygon
    expect(focused[0]).toMatchObject({ color: OTHER, halo: HALO, label: "other" });

    // Focus belongs to the tool: in box mode no point draws, focused or not.
    const inBoxMode = buildAnnotateShapes({
      ...base,
      mode: "box",
      boxes: [],
      polygons: [],
      points: [...points, { x: 9, y: 9, subject: "subject_a", attributes: {} }],
      focused: focusOn("point", 0),
    });
    expect(inBoxMode.filter((s) => s.kind === "point")).toHaveLength(0);
  });

  it("point mode draws no boxes and no derived boxes (nothing but its own points)", () => {
    const shapes = buildAnnotateShapes({
      ...base,
      mode: "point",
      boxes: [{ x1: 0, y1: 0, x2: 5, y2: 5, subject: "subject_a", attributes: {} }],
      points: [{ x: 1, y: 1, subject: "subject_a", attributes: {} }],
    });
    expect(shapes.filter((s) => s.kind === "box")).toHaveLength(0);
    expect(shapes.filter((s) => s.kind === "point")).toHaveLength(1);
  });

  it("the labels toggle hides points too", () => {
    expect(
      buildAnnotateShapes({
        ...base,
        mode: "point",
        visible: false,
        points: [{ x: 1, y: 1, subject: "subject_a", attributes: {} }],
      }),
    ).toEqual([]);
  });

  it("a tool's box pushes dashed and, when focused, the authorship suffix in its label", () => {
    const boxes = [
      { x1: 0, y1: 0, x2: 5, y2: 5, subject: "subject_a", attributes: {}, authorship: "tool" },
    ];
    const atRest = buildAnnotateShapes({ ...base, mode: "box", polygons: [], boxes });
    expect(atRest).toHaveLength(1);
    expect(atRest[0]).toMatchObject({ dashed: true });
    expect(atRest[0].label).toBeUndefined();
    const focused = buildAnnotateShapes({
      ...base,
      mode: "box",
      polygons: [],
      boxes,
      focused: focusOn("box", 0),
    });
    expect(focused[0]).toMatchObject({ dashed: true, halo: HALO, label: "subject_a, tool" });
  });

  it("a person's box pushes solid", () => {
    const shapes = buildAnnotateShapes({
      ...base,
      mode: "box",
      polygons: [],
      boxes: [
        { x1: 0, y1: 0, x2: 5, y2: 5, subject: "subject_a", attributes: {}, authorship: "person" },
      ],
    });
    expect(shapes[0].dashed).toBeFalsy();
  });

  it("a tool's polygon and a tool's point push dashed", () => {
    const polygon = buildAnnotateShapes({
      ...base,
      polygons: [{ ...base.polygons[0], authorship: "tool" }],
    });
    expect(polygon[0].dashed).toBe(true);
    const point = buildAnnotateShapes({
      ...base,
      mode: "point",
      polygons: [],
      points: [{ x: 1, y: 1, subject: "subject_a", attributes: {}, authorship: "tool" }],
    });
    expect(point[0].dashed).toBe(true);
  });

  it("a tool's polygon's derived box draws dashed like the polygon it belongs to", () => {
    const shapes = buildAnnotateShapes({
      ...base,
      mode: "box",
      boxes: [],
      polygons: [{ ...base.polygons[0], authorship: "tool" }],
    });
    const derived = shapes.find((s) => s.kind === "box")!;
    expect(derived.dashed).toBe(true);
    expect(derived.label).toBeUndefined();
  });

  it("box mode draws no polygon outline, and the rubber-band box rides along", () => {
    const shapes = buildAnnotateShapes({
      ...base,
      mode: "box",
      boxes: [],
      drawingBox: { x1: 50, y1: 50, x2: 40, y2: 60 },
    });
    expect(shapes.some((s) => s.kind === "polygon")).toBe(false);
    const rubber = shapes.find((s) => s.tag === "in_progress")!;
    expect(rubber).toMatchObject({ kind: "box", xyxy: [40, 50, 50, 60], dashed: true });
  });

  it("pushes each placed flag as a mark carrying its comment", () => {
    const shapes = buildAnnotateShapes({
      ...base,
      flagMarks: [
        {
          at: [12.04, 30],
          flag: {
            id: "f1",
            text: "open?",
            by: "user:a",
            at: "2026-01-01T00:00:00+00:00",
            point: [12.04, 30],
            subject: "subject_a",
            proposal: null,
            resolved_by: null,
            resolved_at: null,
            reply: "",
            removed: false,
          },
        },
      ],
    });
    expect(shapes.at(-1)).toMatchObject({
      kind: "point",
      points: [[12, 30]],
      color: FLAG_MARK.color,
      tag: "flag",
    });
    expect(shapes.at(-1)?.label).toContain("open?");
  });

  it("colors every shape by its match type while reviewing, the subject color otherwise", () => {
    const boxes = [
      { x1: 0, y1: 0, x2: 5, y2: 5, subject: "subject_a", attributes: {}, index: 0 },
      { x1: 10, y1: 10, x2: 15, y2: 15, subject: "subject_a", attributes: {}, index: 1 },
    ];
    const reviewing = buildAnnotateShapes({
      ...base,
      mode: "box",
      polygons: [],
      boxes,
      matches: { boxes: ["matched", "annotation_only"], polygons: [], points: [] },
    });
    expect(reviewing.map((s) => s.color)).toEqual([
      MATCH_COLORS.matched,
      MATCH_COLORS.annotation_only,
    ]);
    const plain = buildAnnotateShapes({ ...base, mode: "box", polygons: [], boxes });
    expect(plain.every((s) => s.color === RED)).toBe(true);
  });
});

describe("buildAnnotateShapes proposals", () => {
  const base = {
    boxes: [],
    polygons: [],
    currentPolygon: [] as [number, number][],
    mode: "polygon",
    activeSubject: "subject_a",
    visible: true,
    colorFor: subjectColor,
  };
  const proposal = (over: Partial<Proposal>): Proposal => ({
    subject: "subject_a",
    attributes: {},
    iscrowd: false,
    score: 0.9,
    index: 0,
    paired: null,
    decision: null,
    ...over,
  });

  it("mirrors the shown proposals dotted in their match color, every ring, the focused one haloed", () => {
    const shapes = buildAnnotateShapes({
      ...base,
      proposals: [
        proposal({ index: 0, bbox: [1, 2, 11, 12] }),
        proposal({
          index: 1,
          paired: 4,
          rings: [
            [
              [0, 0],
              [10, 0],
              [10, 10],
            ],
            [
              [40, 40],
              [60, 40],
              [60, 60],
            ],
          ],
        }),
      ],
      focused: {
        kind: "proposal",
        shape: "polygon",
        ref: 1,
        subject: "subject_a",
        bbox: [0, 0, 60, 60],
        match: "proposal_only",
        reviewed: false,
        score: 0.9,
        at: [30, 30],
        flags: [],
      },
    });
    const shown = shapes.filter((s) => s.tag === "proposal");
    expect(shown).toHaveLength(3);
    expect(shown[0]).toMatchObject({ kind: "box", xyxy: [1, 2, 11, 12] });
    expect(shown.every((s) => s.dashed)).toBe(true);
    expect(shown.map((s) => s.color)).toEqual([
      MATCH_COLORS.proposal_only,
      MATCH_COLORS.matched,
      MATCH_COLORS.matched,
    ]);
    expect(shown.slice(1).every((s) => s.kind === "polygon" && s.halo)).toBe(true);
    // One label per proposal: the unpaired one at rest, the paired one because it is focused.
    expect(shown.filter((s) => s.label)).toHaveLength(2);
    expect(shown[1].label).toContain("pairs with an annotation");
  });

  it("a paired proposal carries no label unless focused", () => {
    const shapes = buildAnnotateShapes({
      ...base,
      proposals: [proposal({ bbox: [1, 2, 11, 12], paired: 0 })],
    });
    expect(shapes[0].label).toBeUndefined();
  });

  it("the labels toggle hides proposals too", () => {
    expect(
      buildAnnotateShapes({
        ...base,
        visible: false,
        proposals: [proposal({ bbox: [1, 2, 11, 12] })],
      }),
    ).toEqual([]);
  });
});

describe("shapeVisible", () => {
  // The Annotate canvas imports this predicate instead of restating it, so the GUI and the agent's
  // mirror cannot disagree about which shapes are on screen.
  const at = (kind: "box" | "derived" | "polygon" | "point", mode: string, subject: string) =>
    shapeVisible({ kind, mode, subject, activeSubject: "subject_a", focused: false });

  it("shows each kind for the active subject in its own tool mode only", () => {
    for (const kind of ["box", "polygon", "point"] as const) {
      for (const mode of ["box", "polygon", "point"]) {
        expect(at(kind, mode, "subject_a")).toBe(kind === mode);
      }
      expect(at(kind, kind, "other")).toBe(false);
    }
  });

  it("shows a polygon's derived box with the boxes", () => {
    expect(at("derived", "box", "subject_a")).toBe(true);
    expect(at("derived", "polygon", "subject_a")).toBe(false);
    expect(at("derived", "box", "other")).toBe(false);
  });

  it("shows the focused shape whatever its subject, and only in its own tool mode", () => {
    for (const kind of ["box", "polygon", "point"] as const) {
      for (const mode of ["box", "polygon", "point"]) {
        expect(
          shapeVisible({ kind, mode, subject: "other", activeSubject: "subject_a", focused: true }),
        ).toBe(kind === mode);
      }
    }
  });
});

describe("createCanvasPusher", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  const body = (): CanvasStateBody => ({
    project_id: "a1b2c3d4e5f6",
    tab: "annotate",
    image_path: "/p/img.jpg",
    image: "img.jpg",
    img_width: 100,
    img_height: 80,
    viewport: null,
    classes: [],
    shapes: [{ kind: "box", xyxy: [0, 0, 1, 1], color: "#fff" }],
  });

  it("carries cut_armed through to the post untouched (the pusher never special-cases it)", () => {
    const posts: CanvasStateBody[] = [];
    const p = createCanvasPusher(
      (b) => {
        posts.push(b);
      },
      { debounceMs: 100, maxWaitMs: 1000 },
    );
    p.schedule(() => ({ ...body(), cut_armed: true }), true);
    vi.advanceTimersByTime(150);
    expect(posts[0].cut_armed).toBe(true);
  });

  it("coalesces bursts; a full flag anywhere in the burst keeps the geometry", () => {
    const posts: CanvasStateBody[] = [];
    const p = createCanvasPusher(
      (b) => {
        posts.push(b);
      },
      { debounceMs: 100, maxWaitMs: 1000 },
    );
    p.schedule(body, true);
    p.schedule(body, false);
    vi.advanceTimersByTime(150);
    expect(posts).toHaveLength(1);
    expect(posts[0].shapes).not.toBeNull(); // full won the burst
  });

  it("a heartbeat-only burst sends shapes: null", () => {
    const posts: CanvasStateBody[] = [];
    const p = createCanvasPusher(
      (b) => {
        posts.push(b);
      },
      { debounceMs: 100, maxWaitMs: 1000 },
    );
    p.schedule(body, false);
    vi.advanceTimersByTime(150);
    expect(posts[0].shapes).toBeNull();
  });

  it("continuous activity still surfaces at the maxWait cadence", () => {
    const posts: CanvasStateBody[] = [];
    const p = createCanvasPusher(
      (b) => {
        posts.push(b);
      },
      { debounceMs: 100, maxWaitMs: 500 },
    );
    for (let i = 0; i < 12; i++) {
      p.schedule(body, false);
      vi.advanceTimersByTime(80); // re-schedules faster than the debounce can fire
    }
    expect(posts.length).toBeGreaterThanOrEqual(1); // maxWait forced a send mid-burst
  });

  it("flush sends immediately", () => {
    const posts: CanvasStateBody[] = [];
    const p = createCanvasPusher(
      (b) => {
        posts.push(b);
      },
      { debounceMs: 5000, maxWaitMs: 10000 },
    );
    p.schedule(body, true);
    p.flush();
    expect(posts).toHaveLength(1);
  });

  it("a failed full post re-arms the geometry so heartbeats can't mask the loss", async () => {
    const posts: CanvasStateBody[] = [];
    let fail = true;
    const p = createCanvasPusher(
      (b) => {
        if (fail) return Promise.reject(new Error("boom"));
        posts.push(b);
        return Promise.resolve();
      },
      { debounceMs: 100, maxWaitMs: 1000 },
    );
    p.schedule(body, true);
    vi.advanceTimersByTime(150); // fires; the post rejects
    await Promise.resolve(); // let the rejection handler run
    fail = false;
    p.schedule(body, false); // a mere heartbeat follows...
    vi.advanceTimersByTime(150);
    expect(posts).toHaveLength(1);
    expect(posts[0].shapes).not.toBeNull(); // ...but the owed geometry ships with it
  });

  it("a conflict-resolved full post re-arms the geometry, the masquerade case", async () => {
    // pushState resolves {status:"conflict"} on a 409 rather than rejecting; the re-arm must
    // still fire, or a later heartbeat pairs fresh meta with the pre-conflict geometry.
    const posts: CanvasStateBody[] = [];
    let conflict = true;
    const p = createCanvasPusher(
      (b) => {
        if (conflict) return Promise.resolve({ status: "conflict" as const });
        posts.push(b);
        return Promise.resolve({ status: "ok" as const, shapes_written: true });
      },
      { debounceMs: 100, maxWaitMs: 1000 },
    );
    p.schedule(body, true);
    vi.advanceTimersByTime(150); // fires; the post resolves as a conflict
    await Promise.resolve();
    await Promise.resolve();
    conflict = false;
    p.schedule(body, false); // a mere heartbeat follows...
    vi.advanceTimersByTime(150);
    expect(posts).toHaveLength(1);
    expect(posts[0].shapes).not.toBeNull(); // ...but the owed geometry ships with it
  });

  it("the refresh ping reaches the registered handler and unsubscribes cleanly", () => {
    const seen: number[] = [];
    const off = onCanvasStateRequest(() => seen.push(1));
    notifyCanvasStateRequest();
    off();
    notifyCanvasStateRequest();
    expect(seen).toEqual([1]);
  });
});

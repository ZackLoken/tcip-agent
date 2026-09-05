import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  buildAnnotateShapes,
  buildReviewShapes,
  computeViewport,
  createCanvasPusher,
  measureCanvasHost,
  notifyCanvasStateRequest,
  onCanvasStateRequest,
  pointShapeVisible,
  type CanvasStateBody,
} from "@/lib/canvasSync";
import { ringsBbox } from "@/lib/polygonGeometry";
import type { ReviewColors } from "@/lib/reviewColors";
import type { MatchesResponse } from "@/store/types";

const COLORS: ReviewColors = {
  tp: "#22C55E",
  fp: "#EF4444",
  fn: "#F59E0B",
  active: "#00BFFF",
} as ReviewColors;

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
        created_by: "user:breeder",
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
    selectedPolygonIdx: null,
    mode: "polygon",
    activeSubject: "subject_a",
    visible: true,
    colorFor: (subject: string) => (subject === "subject_a" ? "#FF0000" : "#00FF00"),
  };

  it("filters polygon mode to the active subject, keeps provenance, colors from the GUI", () => {
    const shapes = buildAnnotateShapes(base);
    expect(shapes).toHaveLength(1); // "other" filtered out (not selected)
    expect(shapes[0]).toMatchObject({
      kind: "polygon",
      color: "#FF0000",
      label: "subject_a",
      tag: "gt",
      created_by: "user:breeder",
    });
  });

  it("a selected polygon of another subject is included and highlighted", () => {
    const shapes = buildAnnotateShapes({ ...base, selectedPolygonIdx: 1 });
    expect(shapes).toHaveLength(2);
    expect(shapes[1].color).toBe("#00BFFF");
    expect(shapes[1].label).toBe("other");
  });

  it("an in-progress drawing rides along as a dashed polyline in the active subject's colour", () => {
    // Regression: the mirror used to hardcode amber here while the real canvas's
    // InProgressPolygon stroke already used the active subject's colour, a divergence the
    // agent's capture_live_canvas view would show that the breeder's own screen never did.
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
      color: "#FF0000", // base.colorFor("subject_a"), base.activeSubject
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

  it("a pending cut start rides as an in_progress polyline labelled cut, start and cursor", () => {
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
    expect(shapes[0]).toMatchObject({
      kind: "box",
      xyxy: [12, 7, 41, 23],
      label: base.activeSubject,
    });
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

  it("box mode adds one read-only derived box per active-subject polygon, dashed, === ringsBbox", () => {
    // Mirrors the canvas overlay: a polygon's detection footprint shows while boxing, and its coords
    // are exactly ringsBbox, never a stored box. Dashed distinguishes it from a real editable box
    // (solid), the same convention in-progress/under-review shapes already use.
    const shapes = buildAnnotateShapes({ ...base, mode: "box", boxes: [] });
    const derived = shapes.filter((s) => s.kind === "box");
    expect(derived).toHaveLength(1); // only the active "subject_a" polygon; "other" is filtered out
    expect(derived[0].xyxy).toEqual(ringsBbox(base.polygons[0].rings));
    expect(derived[0].label).toBe("subject_a");
    expect(derived[0].dashed).toBe(true); // dashed = derived/read-only, not a real editable box
  });

  it("pushes every ring of a multi-ring polygon, sharing its colour, labelled once", () => {
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
      created_by: "user:breeder",
    };
    const shapes = buildAnnotateShapes({ ...base, polygons: [multi] });
    expect(shapes).toHaveLength(2);
    expect(shapes.map((s) => s.points)).toEqual(multi.rings);
    expect(shapes.every((s) => s.color === "#FF0000" && s.tag === "gt")).toBe(true);
    // Labelled once: a two-part subject_a is one subject_a, not two.
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
        { x: 5.06, y: 7.04, subject: "subject_a", attributes: {}, created_by: "user:breeder" },
        { x: 50, y: 60, subject: "other", attributes: {} },
      ],
    });
    expect(shapes).toHaveLength(1); // "other" filtered out, exactly like the box/polygon rules
    expect(shapes[0]).toMatchObject({
      kind: "point",
      points: [[5.1, 7]], // rounded like every other pushed coordinate
      color: "#FF0000",
      label: "subject_a",
      tag: "gt",
      created_by: "user:breeder",
    });
    expect(shapes[0].xyxy).toBeUndefined();
  });

  it("the selected point is pushed highlighted, and follows the selection out of point mode", () => {
    const points = [{ x: 5, y: 7, subject: "other", attributes: {} }];
    const selected = buildAnnotateShapes({
      ...base,
      mode: "point",
      polygons: [],
      points,
      selectedPointIdx: 0,
    });
    expect(selected).toHaveLength(1); // included despite the subject filter, like a selected polygon
    expect(selected[0].color).toBe("#00BFFF");

    // Box mode: only the selection survives, the shape being inspected stays on screen.
    const inBoxMode = buildAnnotateShapes({
      ...base,
      mode: "box",
      boxes: [],
      polygons: [],
      points: [...points, { x: 9, y: 9, subject: "subject_a", attributes: {} }],
      selectedPointIdx: 0,
    });
    expect(inBoxMode.filter((s) => s.kind === "point")).toHaveLength(1);
    expect(inBoxMode.filter((s) => s.kind === "point")[0].color).toBe("#00BFFF");
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

  it("a tool's box pushes dashed with the tool pattern, the hover-label suffix, and its authorship", () => {
    const shapes = buildAnnotateShapes({
      ...base,
      mode: "box",
      polygons: [],
      boxes: [
        {
          x1: 0,
          y1: 0,
          x2: 5,
          y2: 5,
          subject: "subject_a",
          attributes: {},
          authorship: "tool",
        },
      ],
    });
    expect(shapes).toHaveLength(1);
    expect(shapes[0]).toMatchObject({
      dashed: true,
      dash_kind: "tool",
      label: "subject_a, tool",
      authorship: "tool",
    });
  });

  it("a person's box pushes solid with no dash_kind", () => {
    const shapes = buildAnnotateShapes({
      ...base,
      mode: "box",
      polygons: [],
      boxes: [
        { x1: 0, y1: 0, x2: 5, y2: 5, subject: "subject_a", attributes: {}, authorship: "person" },
      ],
    });
    expect(shapes[0].dashed).toBeFalsy();
    expect(shapes[0].dash_kind).toBeUndefined();
    expect(shapes[0].label).toBe("subject_a");
  });

  it("a tool's polygon pushes dashed with the tool pattern and its authorship", () => {
    const shapes = buildAnnotateShapes({
      ...base,
      polygons: [{ ...base.polygons[0], authorship: "tool" }],
    });
    expect(shapes[0]).toMatchObject({
      dashed: true,
      dash_kind: "tool",
      label: "subject_a, tool",
      authorship: "tool",
    });
  });

  it("a tool's point pushes dashed with the tool pattern and its authorship", () => {
    const shapes = buildAnnotateShapes({
      ...base,
      mode: "point",
      polygons: [],
      points: [{ x: 1, y: 1, subject: "subject_a", attributes: {}, authorship: "tool" }],
    });
    expect(shapes[0]).toMatchObject({
      dashed: true,
      dash_kind: "tool",
      label: "subject_a, tool",
      authorship: "tool",
    });
  });

  it("a polygon's derived box carries the derived dash_kind and the polygon's own authorship label", () => {
    const shapes = buildAnnotateShapes({
      ...base,
      mode: "box",
      boxes: [],
      polygons: [{ ...base.polygons[0], authorship: "tool" }],
    });
    const derived = shapes.find((s) => s.kind === "box")!;
    expect(derived).toMatchObject({
      dashed: true,
      dash_kind: "derived", // the derived box's own pattern, never the tool's dots
      label: "subject_a, tool",
      authorship: "tool",
    });
  });

  it("box mode includes the selected polygon and the rubber-band box", () => {
    const shapes = buildAnnotateShapes({
      ...base,
      mode: "box",
      boxes: [],
      selectedPolygonIdx: 1,
      drawingBox: { x1: 50, y1: 50, x2: 40, y2: 60 },
    });
    expect(shapes.some((s) => s.kind === "polygon" && s.color === "#00BFFF")).toBe(true);
    const rubber = shapes.find((s) => s.tag === "in_progress")!;
    expect(rubber).toMatchObject({ kind: "box", xyxy: [40, 50, 50, 60], dashed: true });
  });
});

describe("buildReviewShapes", () => {
  const matches = {
    img_width: 100,
    img_height: 80,
    n_tp: 1,
    n_fp: 1,
    n_fn: 1,
    detections: [
      {
        det_type: "tp",
        class_name: "subject_a",
        conf: 0.9,
        iou: 0.8,
        gt_idx: 0,
        pred_idx: 0,
        bbox: [0, 0, 10, 10],
        reviewed: true,
        reviewed_action: "accepted",
      },
      {
        det_type: "fp",
        class_name: "subject_a",
        conf: 0.7,
        iou: null,
        gt_idx: null,
        pred_idx: 1,
        bbox: [20, 20, 30, 30],
        reviewed: false,
        reviewed_action: null,
      },
      {
        det_type: "fn",
        class_name: "subject_a",
        conf: null,
        iou: null,
        gt_idx: 1,
        pred_idx: null,
        bbox: [40, 40, 50, 50],
        reviewed: false,
        reviewed_action: null,
      },
    ],
    gt: [
      { subject: "subject_a", bbox: [0, 0, 10, 10], attributes: {} },
      { subject: "subject_a", bbox: [40, 40, 50, 50], attributes: {} },
    ],
    preds: [
      { subject: "subject_a", bbox: [1, 1, 11, 11], attributes: {}, score: 0.9 },
      { subject: "subject_a", bbox: [20, 20, 30, 30], attributes: {}, score: 0.7 },
    ],
    image_status: "started",
  } as unknown as MatchesResponse;

  it("mirrors the review symbology: outcome colors, focused dashed-active, reviewed wash", () => {
    const shapes = buildReviewShapes(matches, COLORS, 1);
    const tp = shapes.find((s) => s.tag === "tp")!;
    const fp = shapes.find((s) => s.tag === "fp")!;
    const fn = shapes.find((s) => s.tag === "fn")!;
    expect(tp).toMatchObject({ color: COLORS.tp, fill: true }); // reviewed → washed
    expect(fp).toMatchObject({ color: COLORS.active, dashed: true, label: "subject_a 0.70" });
    expect(fn).toMatchObject({ color: COLORS.fn });
  });

  it("the focused TP overlays its prediction dashed-active", () => {
    const shapes = buildReviewShapes(matches, COLORS, 0);
    const pred = shapes.find((s) => s.tag === "pred");
    expect(pred).toMatchObject({ color: COLORS.active, dashed: true });
  });

  it("honors the GT / Pred visibility toggles", () => {
    expect(
      buildReviewShapes(matches, COLORS, 1, { showPred: false }).some(
        (s) => s.tag === "fp" || s.tag === "pred",
      ),
    ).toBe(false);
    expect(
      buildReviewShapes(matches, COLORS, 1, { showGT: false }).some(
        (s) => s.tag === "tp" || s.tag === "fn",
      ),
    ).toBe(false);
  });

  it("renders both geometry kinds, a box and a polygon annotation each draw (no kind hidden)", () => {
    // Measurement-critical: a unified file may mix a bbox annotation and a polygon annotation.
    // Both must render by their own geometry; hiding a kind is an unreviewed false-negative.
    const mixed = {
      img_width: 100,
      img_height: 80,
      n_tp: 0,
      n_fp: 0,
      n_fn: 2,
      detections: [
        {
          det_type: "fn",
          class_name: "subject_a",
          conf: null,
          iou: null,
          gt_idx: 0,
          pred_idx: null,
          bbox: [0, 0, 10, 10],
          reviewed: false,
          reviewed_action: null,
        },
        {
          det_type: "fn",
          class_name: "leaf",
          conf: null,
          iou: null,
          gt_idx: 1,
          pred_idx: null,
          bbox: [40, 40, 60, 60],
          reviewed: false,
          reviewed_action: null,
        },
      ],
      gt: [
        { subject: "subject_a", bbox: [0, 0, 10, 10], attributes: {} },
        {
          subject: "leaf",
          rings: [
            [
              [40, 40],
              [60, 40],
              [60, 60],
            ],
          ],
          attributes: {},
        },
      ],
      preds: [],
      image_status: "started",
    } as unknown as MatchesResponse;
    const shapes = buildReviewShapes(mixed, COLORS, -1); // none focused
    expect(shapes.filter((s) => s.kind === "box")).toHaveLength(1);
    expect(shapes.filter((s) => s.kind === "polygon")).toHaveLength(1);
  });

  it("an occlusion-split prediction pushes every ring, in the same outcome colour", () => {
    // A verdict on a two-part prediction is a verdict on both parts, so the agent's mirror of the
    // review canvas has to show both: one shape entry per ring, not just the first.
    const split = {
      img_width: 100,
      img_height: 80,
      n_tp: 0,
      n_fp: 1,
      n_fn: 0,
      detections: [
        {
          det_type: "fp",
          class_name: "subject_a",
          conf: 0.7,
          iou: null,
          gt_idx: null,
          pred_idx: 0,
          bbox: [0, 0, 60, 60],
          reviewed: false,
          reviewed_action: null,
        },
      ],
      gt: [],
      preds: [
        {
          subject: "subject_a",
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
          attributes: {},
          score: 0.7,
        },
      ],
      image_status: "started",
    } as unknown as MatchesResponse;
    const shapes = buildReviewShapes(split, COLORS, -1); // not focused: plain outcome colour
    expect(shapes).toHaveLength(2);
    expect(
      shapes.every((s) => s.kind === "polygon" && s.color === COLORS.fp && s.tag === "fp"),
    ).toBe(true);
    expect(shapes.map((s) => s.points)).toEqual([
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
    ]);
  });

  it("draws the focused detection last so neighbours never bury it", () => {
    const shapes = buildReviewShapes(matches, COLORS, 0);
    expect(shapes.at(-1)!.tag).toBe("pred"); // the focused TP's overlay is on top
  });

  it("a point-carrying GT pushes a point shape, not a box, and not nothing", () => {
    // Review load responses can carry {point: [x, y]} on a GT/prediction. Dropping it would hide a
    // real annotation from the agent's mirror; boxing it would invent an extent.
    const withPoint = {
      img_width: 100,
      img_height: 80,
      n_tp: 0,
      n_fp: 0,
      n_fn: 1,
      detections: [
        {
          det_type: "fn",
          class_name: "tip",
          conf: null,
          iou: null,
          gt_idx: 0,
          pred_idx: null,
          bbox: [10, 20, 10, 20],
          reviewed: false,
          reviewed_action: null,
        },
      ],
      gt: [{ subject: "tip", point: [10.04, 20.06], attributes: {} }],
      preds: [],
      image_status: "started",
    } as unknown as MatchesResponse;
    const shapes = buildReviewShapes(withPoint, COLORS, -1);
    expect(shapes).toHaveLength(1);
    expect(shapes[0]).toMatchObject({ kind: "point", points: [[10, 20.1]], color: COLORS.fn });
    expect(shapes[0].xyxy).toBeUndefined();
  });
});

describe("pointShapeVisible", () => {
  // The Annotate canvas imports this predicate instead of restating it, so the GUI and the agent's
  // mirror cannot disagree about which points are on screen.
  it("shows active-subject points in point mode only", () => {
    expect(
      pointShapeVisible({
        mode: "point",
        subject: "subject_a",
        activeSubject: "subject_a",
        selected: false,
      }),
    ).toBe(true);
    expect(
      pointShapeVisible({
        mode: "point",
        subject: "other",
        activeSubject: "subject_a",
        selected: false,
      }),
    ).toBe(false);
    expect(
      pointShapeVisible({
        mode: "box",
        subject: "subject_a",
        activeSubject: "subject_a",
        selected: false,
      }),
    ).toBe(false);
  });

  it("always shows the selected point, whatever the mode or subject", () => {
    for (const mode of ["box", "polygon", "point"]) {
      expect(
        pointShapeVisible({ mode, subject: "other", activeSubject: "subject_a", selected: true }),
      ).toBe(true);
    }
  });
});

describe("createCanvasPusher", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  const body = (): CanvasStateBody => ({
    binding_generation: 1,
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

  it("with no options, a burst waits out the documented trailing debounce and then posts", () => {
    const posts: CanvasStateBody[] = [];
    const p = createCanvasPusher((b) => {
      posts.push(b);
    });
    p.schedule(body, true);
    vi.advanceTimersByTime(399);
    expect(posts).toHaveLength(0);
    vi.advanceTimersByTime(2);
    expect(posts).toHaveLength(1);
    p.dispose();
  });

  it("with no options, continuous activity surfaces at the documented maxWait", () => {
    const posts: CanvasStateBody[] = [];
    const p = createCanvasPusher((b) => {
      posts.push(b);
    });
    for (let i = 0; i < 4; i++) {
      p.schedule(body, false);
      vi.advanceTimersByTime(300); // re-arms faster than the default debounce can fire
    }
    p.schedule(body, false); // 1200 ms into the burst
    expect(posts).toHaveLength(0);
    vi.advanceTimersByTime(300); // 1500 ms: the send the maxWait ceiling owes
    expect(posts).toHaveLength(1);
    p.dispose();
  });

  it("a null build keeps the full flag pending", () => {
    const posts: CanvasStateBody[] = [];
    const p = createCanvasPusher(
      (b) => {
        posts.push(b);
      },
      { debounceMs: 100, maxWaitMs: 1000 },
    );
    let ready = false;
    const build = () => (ready ? body() : null);
    p.schedule(build, true);
    vi.advanceTimersByTime(150); // fires; builder returns null (mid-transition)
    expect(posts).toHaveLength(0);
    ready = true;
    p.schedule(build, false);
    vi.advanceTimersByTime(150);
    expect(posts[0].shapes).not.toBeNull(); // the owed geometry survived the null build
  });
});

describe("canvas state request", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("a request reaches every mounted tab's handler", () => {
    const annotateHandler = vi.fn();
    const reviewHandler = vi.fn();
    const offAnnotate = onCanvasStateRequest(annotateHandler);
    const offReview = onCanvasStateRequest(reviewHandler);
    notifyCanvasStateRequest();
    expect(annotateHandler).toHaveBeenCalledTimes(1);
    expect(reviewHandler).toHaveBeenCalledTimes(1);
    offAnnotate();
    offReview();
  });

  it("an unsubscribed handler stops receiving requests", () => {
    const unmounted = vi.fn();
    const mounted = vi.fn();
    const offUnmounted = onCanvasStateRequest(unmounted);
    const offMounted = onCanvasStateRequest(mounted);
    offUnmounted();
    notifyCanvasStateRequest();
    expect(unmounted).not.toHaveBeenCalled();
    expect(mounted).toHaveBeenCalledTimes(1);
    offMounted();
  });

  it("a request answered by flushing posts at once, without waiting out the debounce", () => {
    const posts: CanvasStateBody[] = [];
    const pusher = createCanvasPusher(
      (b) => {
        posts.push(b);
      },
      { debounceMs: 5000, maxWaitMs: 10000 },
    );
    const body = (): CanvasStateBody => ({
      binding_generation: 1,
      tab: "annotate",
      image_path: "/p/img.jpg",
      image: "img.jpg",
      img_width: 120,
      img_height: 90,
      viewport: null,
      classes: [],
      shapes: [{ kind: "box", xyxy: [4, 9, 22, 15], color: "#fff" }],
    });
    const off = onCanvasStateRequest(() => {
      pusher.schedule(body, true);
      pusher.flush();
    });
    notifyCanvasStateRequest();
    expect(posts).toHaveLength(1);
    expect(posts[0].shapes).not.toBeNull();
    off();
    pusher.dispose();
  });
});

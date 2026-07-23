import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  buildAnnotateShapes,
  buildReviewShapes,
  computeViewport,
  createCanvasPusher,
  type CanvasStateBody,
} from "@/lib/canvasSync";
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
        points: [
          [0, 0],
          [10, 0],
          [10, 10],
        ] as [number, number][],
        subject: "catkin",
        attributes: {},
        created_by: "user:zack",
      },
      {
        points: [
          [20, 20],
          [30, 20],
          [30, 30],
        ] as [number, number][],
        subject: "other",
        attributes: {},
      },
    ],
    currentPolygon: [] as [number, number][],
    selectedPolygonIdx: null,
    mode: "polygon",
    activeSubject: "catkin",
    visible: true,
    colorFor: (subject: string) => (subject === "catkin" ? "#FF0000" : "#00FF00"),
  };

  it("filters polygon mode to the active subject, keeps provenance, colors from the GUI", () => {
    const shapes = buildAnnotateShapes(base);
    expect(shapes).toHaveLength(1); // "other" filtered out (not selected)
    expect(shapes[0]).toMatchObject({
      kind: "polygon",
      color: "#FF0000",
      label: "catkin",
      tag: "gt",
      created_by: "user:zack",
    });
  });

  it("a selected polygon of another subject is included and highlighted", () => {
    const shapes = buildAnnotateShapes({ ...base, selectedPolygonIdx: 1 });
    expect(shapes).toHaveLength(2);
    expect(shapes[1].color).toBe("#00BFFF");
    expect(shapes[1].label).toBe("other");
  });

  it("an in-progress drawing rides along as a dashed polyline", () => {
    const shapes = buildAnnotateShapes({
      ...base,
      currentPolygon: [
        [1, 1],
        [2, 2],
      ],
    });
    expect(shapes.at(-1)).toMatchObject({ kind: "polyline", tag: "in_progress", dashed: true });
  });

  it("the labels toggle hides everything, exactly like the canvas", () => {
    expect(buildAnnotateShapes({ ...base, visible: false })).toEqual([]);
  });

  it("box mode renders the active-subject boxes (no polygon derivation)", () => {
    const shapes = buildAnnotateShapes({
      ...base,
      mode: "box",
      boxes: [
        { x1: 0, y1: 0, x2: 5, y2: 5, subject: "catkin", attributes: {} },
        { x1: 8, y1: 8, x2: 9, y2: 9, subject: "other", attributes: {} },
      ],
    });
    // Only the active subject's real box renders; polygons are not shown as boxes.
    expect(shapes).toHaveLength(1);
    expect(shapes[0]).toMatchObject({ kind: "box", xyxy: [0, 0, 5, 5], label: "catkin" });
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
        class_name: "catkin",
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
        class_name: "catkin",
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
        class_name: "catkin",
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
      { subject: "catkin", bbox: [0, 0, 10, 10], attributes: {} },
      { subject: "catkin", bbox: [40, 40, 50, 50], attributes: {} },
    ],
    preds: [
      { subject: "catkin", bbox: [1, 1, 11, 11], attributes: {}, score: 0.9 },
      { subject: "catkin", bbox: [20, 20, 30, 30], attributes: {}, score: 0.7 },
    ],
    image_status: "started",
  } as unknown as MatchesResponse;

  it("mirrors the review symbology: outcome colors, focused dashed-active, reviewed wash", () => {
    const shapes = buildReviewShapes(matches, COLORS, 1);
    const tp = shapes.find((s) => s.tag === "tp")!;
    const fp = shapes.find((s) => s.tag === "fp")!;
    const fn = shapes.find((s) => s.tag === "fn")!;
    expect(tp).toMatchObject({ color: COLORS.tp, fill: true }); // reviewed → washed
    expect(fp).toMatchObject({ color: COLORS.active, dashed: true, label: "catkin 0.70" });
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

  it("renders BOTH geometry kinds — a box and a polygon annotation each draw (no kind hidden)", () => {
    // Measurement-critical: a unified file may mix a bbox annotation and a polygon annotation.
    // Both must render by their OWN geometry; hiding a kind is an unreviewed false-negative.
    const mixed = {
      img_width: 100,
      img_height: 80,
      n_tp: 0,
      n_fp: 0,
      n_fn: 2,
      detections: [
        {
          det_type: "fn",
          class_name: "catkin",
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
        { subject: "catkin", bbox: [0, 0, 10, 10], attributes: {} },
        {
          subject: "leaf",
          points: [
            [40, 40],
            [60, 40],
            [60, 60],
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

  it("draws the focused detection last so neighbours never bury it", () => {
    const shapes = buildReviewShapes(matches, COLORS, 0);
    expect(shapes.at(-1)!.tag).toBe("pred"); // the focused TP's overlay is on top
  });
});

describe("createCanvasPusher", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  const body = (): CanvasStateBody => ({
    schema_version: 1,
    project_root: "/p",
    tab: "annotate",
    image_path: "/p/img.jpg",
    image: "img.jpg",
    img_width: 100,
    img_height: 80,
    viewport: null,
    classes: [],
    shapes: [{ kind: "box", xyxy: [0, 0, 1, 1], color: "#fff" }],
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

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
    boxes: [],
    polygons: [
      {
        points: [
          [0, 0],
          [10, 0],
          [10, 10],
        ] as [number, number][],
        class_id: 0,
        created_by: "user:zack",
      },
      {
        points: [
          [20, 20],
          [30, 20],
          [30, 30],
        ] as [number, number][],
        class_id: 1,
      },
    ],
    currentPolygon: [] as [number, number][],
    selectedPolygonIdx: null,
    mode: "polygon",
    activeClass: 0,
    visible: true,
    colorFor: (cid: number) => (cid === 0 ? "#FF0000" : "#00FF00"),
    nameFor: (cid: number) => (cid === 0 ? "catkin" : "other"),
  };

  it("filters polygon mode to the active class, keeps provenance, colors from the GUI", () => {
    const shapes = buildAnnotateShapes(base);
    expect(shapes).toHaveLength(1); // class 1 filtered out (not selected)
    expect(shapes[0]).toMatchObject({
      kind: "polygon",
      color: "#FF0000",
      tag: "gt",
      created_by: "user:zack",
    });
  });

  it("a selected polygon of another class is included and highlighted", () => {
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

  it("box mode renders polygon-DERIVED boxes of the active class, not the stale box store", () => {
    const shapes = buildAnnotateShapes({
      ...base,
      mode: "box",
      boxes: [{ x1: 0, y1: 0, x2: 5, y2: 5, class_id: 0 }], // stale loaded detect layer
    });
    // polygons exist -> derived boxes; only class 0 renders (class-1 polygon filtered).
    expect(shapes).toHaveLength(1);
    expect(shapes[0]).toMatchObject({ kind: "box", xyxy: [0, 0, 10, 10] }); // bbox of polygon 0
  });

  it("box mode includes the selected polygon and the rubber-band box", () => {
    const shapes = buildAnnotateShapes({
      ...base,
      mode: "box",
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
        class_id: 0,
        conf: 0.9,
        iou: 0.8,
        gt_type: "box",
        gt_idx: 0,
        pred_type: "box",
        pred_idx: 0,
        bbox: [0, 0, 10, 10],
        reviewed: true,
        reviewed_action: "accepted",
      },
      {
        det_type: "fp",
        class_id: 0,
        conf: 0.7,
        iou: null,
        gt_type: null,
        gt_idx: null,
        pred_type: "box",
        pred_idx: 1,
        bbox: [20, 20, 30, 30],
        reviewed: false,
        reviewed_action: null,
      },
      {
        det_type: "fn",
        class_id: 0,
        conf: null,
        iou: null,
        gt_type: "box",
        gt_idx: 1,
        pred_type: null,
        pred_idx: null,
        bbox: [40, 40, 50, 50],
        reviewed: false,
        reviewed_action: null,
      },
    ],
    gt_boxes: [
      { x1: 0, y1: 0, x2: 10, y2: 10, class_id: 0 },
      { x1: 40, y1: 40, x2: 50, y2: 50, class_id: 0 },
    ],
    gt_polygons: [],
    pred_boxes: [
      { x1: 1, y1: 1, x2: 11, y2: 11, class_id: 0, confidence: 0.9 },
      { x1: 20, y1: 20, x2: 30, y2: 30, class_id: 0, confidence: 0.7 },
    ],
    pred_polygons: [],
    image_status: "started",
  } as unknown as MatchesResponse;
  const nameFor = () => "catkin";

  it("mirrors the review symbology: outcome colors, focused dashed-active, reviewed wash", () => {
    const shapes = buildReviewShapes(matches, COLORS, 1, nameFor);
    const tp = shapes.find((s) => s.tag === "tp")!;
    const fp = shapes.find((s) => s.tag === "fp")!;
    const fn = shapes.find((s) => s.tag === "fn")!;
    expect(tp).toMatchObject({ color: COLORS.tp, fill: true }); // reviewed → washed
    expect(fp).toMatchObject({ color: COLORS.active, dashed: true, label: "catkin 0.70" });
    expect(fn).toMatchObject({ color: COLORS.fn });
  });

  it("the focused TP overlays its prediction dashed-active", () => {
    const shapes = buildReviewShapes(matches, COLORS, 0, nameFor);
    const pred = shapes.find((s) => s.tag === "pred");
    expect(pred).toMatchObject({ color: COLORS.active, dashed: true });
  });

  it("honors the GT / Pred visibility toggles", () => {
    expect(
      buildReviewShapes(matches, COLORS, 1, nameFor, { showPred: false }).some(
        (s) => s.tag === "fp" || s.tag === "pred",
      ),
    ).toBe(false);
    expect(
      buildReviewShapes(matches, COLORS, 1, nameFor, { showGT: false }).some(
        (s) => s.tag === "tp" || s.tag === "fn",
      ),
    ).toBe(false);
  });

  it("renders only the kind under review — derived box twins must not double-render", () => {
    const mixed = {
      ...matches,
      // A polygon-kind detection for the same object (the derived-twin scenario): preds are
      // boxes, so reviewKind = box and the polygon-kind detection must be skipped.
      detections: [
        ...matches.detections,
        {
          det_type: "fn",
          class_id: 0,
          conf: null,
          iou: null,
          gt_type: "polygon",
          gt_idx: 0,
          pred_type: null,
          pred_idx: null,
          bbox: [0, 0, 10, 10],
          reviewed: false,
          reviewed_action: null,
        },
      ],
      gt_polygons: [
        {
          points: [
            [0, 0],
            [10, 0],
            [10, 10],
          ] as [number, number][],
          class_id: 0,
        },
      ],
    } as unknown as MatchesResponse;
    const shapes = buildReviewShapes(mixed, COLORS, 1, nameFor);
    expect(shapes.some((s) => s.kind === "polygon")).toBe(false); // box kind only
    expect(shapes).toHaveLength(3); // tp + fp + fn boxes, no phantom polygon twin
  });

  it("draws the focused detection last so neighbours never bury it", () => {
    const shapes = buildReviewShapes(matches, COLORS, 0, nameFor);
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

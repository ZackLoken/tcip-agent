/**
 * Live canvas-state sync: lets the agent see exactly what the canvas shows.
 *
 * Hybrid cadence so dense images (thousands of polygons) never jank the UI:
 *   - heartbeat (shapes: null — image, viewport, classes, counts; a few KB) on view/meta changes
 *   - full geometry only when shapes actually change (draw / edit / delete / image load), and
 *     never mid-drag/stream — the tab downgrades to heartbeats while a pointer interaction is
 *     live and pushes once on release, so committed geometry is never re-serialized per tick
 *   - the agent can ping "canvas_state_request" through the panel-event hub; the mounted tab
 *     answers with an immediate full push (see onCanvasStateRequest).
 *
 * Shapes are display-resolved AND display-filtered: each carries the exact hex color / dash /
 * label the GUI renders, and the builders reproduce the canvas's own visibility rules (mode
 * filters, active-class filter, derived detect boxes, the labels toggle, review's single-kind
 * rule) — so the server-side render (visualize_canvas) is faithful by construction.
 */

import { polygonBbox } from "@/lib/polygonGeometry";
import type { ReviewColors } from "@/lib/reviewColors";
import type { Box, MatchesResponse, PolygonShape } from "@/store/types";

export interface CanvasViewport {
  x: number;
  y: number;
  w: number;
  h: number;
  scale: number;
}

export interface CanvasShape {
  kind: "box" | "polygon" | "polyline";
  xyxy?: [number, number, number, number];
  points?: [number, number][];
  color: string;
  fill?: boolean;
  dashed?: boolean;
  label?: string;
  tag?: string; // gt | tp | fp | fn | pred | in_progress
  created_by?: string | null;
  accepted_by?: string | null;
}

export interface CanvasStateBody {
  schema_version: 1;
  project_root: string;
  tab: "annotate" | "review";
  image_path: string;
  image: string;
  img_width: number;
  img_height: number;
  viewport: CanvasViewport | null;
  mode?: string;
  active_class?: number;
  dirty?: boolean;
  user?: string;
  classes: { id: number; name: string; color: string }[];
  legend?: Record<string, string> | null;
  counts?: Record<string, number>;
  /** null = heartbeat (backend keeps the last pushed geometry for this image). */
  shapes: CanvasShape[] | null;
}

/** 0.1-px precision is beyond what any render needs; rounding cuts dense payloads ~2-3×. */
const r1 = (n: number): number => Math.round(n * 10) / 10;
const rPts = (pts: [number, number][]): [number, number][] => pts.map(([x, y]) => [r1(x), r1(y)]);

/** The visible image region (image coords) from the pan/zoom view + canvas host size. */
export function computeViewport(
  view: { scale: number; offset_x: number; offset_y: number },
  host: { w: number; h: number },
  imgW: number,
  imgH: number,
): CanvasViewport | null {
  const s = view.scale || 1;
  if (host.w <= 1 || host.h <= 1 || !imgW || !imgH) return null;
  const x = -view.offset_x / s;
  const y = -view.offset_y / s;
  const x1 = Math.max(0, x);
  const y1 = Math.max(0, y);
  const x2 = Math.min(imgW, x + host.w / s);
  const y2 = Math.min(imgH, y + host.h / s);
  if (x2 - x1 < 1 || y2 - y1 < 1) return null;
  return { x: r1(x1), y: r1(y1), w: r1(x2 - x1), h: r1(y2 - y1), scale: s };
}

/** Size of the mounted canvas host (CanvasStage tags its wrapper with data-canvas-host). */
export function measureCanvasHost(): { w: number; h: number } | null {
  const el = document.querySelector("[data-canvas-host]");
  if (!el) return null;
  const r = el.getBoundingClientRect();
  return r.width > 1 && r.height > 1 ? { w: r.width, h: r.height } : null;
}

/** Annotate-tab shapes, mirroring the canvas render rules exactly: the labels toggle hides
 *  everything; polygon mode shows polygons of the active class plus the selection (outline
 *  only, like the GUI); box mode shows the active-class boxes — the polygon-DERIVED boxes when
 *  polygons exist — plus the selected polygon and the in-flight rubber-band box. */
export function buildAnnotateShapes(args: {
  boxes: Box[];
  polygons: PolygonShape[];
  currentPolygon: [number, number][];
  drawingBox?: { x1: number; y1: number; x2: number; y2: number } | null;
  selectedPolygonIdx: number | null;
  selectedBoxIdx?: number | null;
  mode: string;
  activeClass: number;
  visible: boolean;
  colorFor: (classId: number) => string;
  nameFor: (classId: number) => string;
}): CanvasShape[] {
  if (!args.visible) return []; // the GUI's labels toggle hides every committed shape

  const shapes: CanvasShape[] = [];
  const pushPolygon = (p: PolygonShape, selected: boolean) => {
    shapes.push({
      kind: "polygon",
      points: rPts(p.points),
      color: selected ? "#00BFFF" : args.colorFor(p.class_id),
      label: args.nameFor(p.class_id),
      tag: "gt",
      created_by: p.created_by ?? null,
      accepted_by: p.accepted_by ?? null,
    });
  };

  if (args.mode === "polygon") {
    args.polygons.forEach((p, i) => {
      const selected = i === args.selectedPolygonIdx;
      if (!selected && p.class_id !== args.activeClass) return;
      pushPolygon(p, selected);
    });
    if (args.currentPolygon.length > 0) {
      shapes.push({
        kind: "polyline",
        points: rPts(args.currentPolygon),
        color: "#FFE7B1",
        dashed: true,
        label: "drawing",
        tag: "in_progress",
      });
    }
    return shapes;
  }

  // Box mode: the detect layer is derived from polygons whenever any exist (read-only,
  // recomputed live), else the real boxes. Only the active class renders.
  const derived = args.polygons.length > 0;
  const boxesToRender: Box[] = derived
    ? args.polygons.map((p) => {
        const [x1, y1, x2, y2] = polygonBbox(p.points);
        return { x1, y1, x2, y2, class_id: p.class_id };
      })
    : args.boxes;
  boxesToRender.forEach((b, i) => {
    if (b.class_id !== args.activeClass) return;
    const selected = !derived && i === (args.selectedBoxIdx ?? null);
    shapes.push({
      kind: "box",
      xyxy: [r1(b.x1), r1(b.y1), r1(b.x2), r1(b.y2)],
      color: selected ? "#00BFFF" : args.colorFor(b.class_id),
      label: args.nameFor(b.class_id),
      tag: "gt",
      created_by: b.created_by ?? null,
      accepted_by: b.accepted_by ?? null,
    });
  });
  // Box mode still shows the selected polygon (the shape being inspected).
  const sel = args.selectedPolygonIdx;
  if (sel !== null && args.polygons[sel]) pushPolygon(args.polygons[sel], true);
  if (args.drawingBox) {
    const d = args.drawingBox;
    shapes.push({
      kind: "box",
      xyxy: [
        r1(Math.min(d.x1, d.x2)),
        r1(Math.min(d.y1, d.y2)),
        r1(Math.max(d.x1, d.x2)),
        r1(Math.max(d.y1, d.y2)),
      ],
      color: args.colorFor(args.activeClass),
      dashed: true,
      tag: "in_progress",
    });
  }
  return shapes;
}

/** Review-tab shapes, mirroring the Review canvas rules: only the kind under review renders
 *  (preds decide, GT breaks the tie — an image's derived box twins must not double-render),
 *  FP = its prediction (dashed blue when focused), TP/FN = the ground truth (focused FN goes
 *  active-blue; reviewed shapes washed), the focused TP overlays its prediction dashed, and
 *  the focused detection draws last so neighbours never bury it. */
export function buildReviewShapes(
  matches: MatchesResponse,
  colors: ReviewColors,
  focusedIdx: number,
  nameFor: (classId: number) => string,
  vis: { showGT?: boolean; showPred?: boolean } = {},
): CanvasShape[] {
  const showGT = vis.showGT ?? true;
  const showPred = vis.showPred ?? true;
  const reviewKind: "box" | "polygon" = (() => {
    if (matches.pred_boxes.length || matches.pred_polygons.length) {
      return matches.pred_polygons.length && !matches.pred_boxes.length ? "polygon" : "box";
    }
    return matches.gt_polygons.length && !matches.gt_boxes.length ? "polygon" : "box";
  })();

  const rest: CanvasShape[] = [];
  const focused: CanvasShape[] = [];
  matches.detections.forEach((d, i) => {
    if ((d.gt_type ?? d.pred_type) !== reviewKind) return; // one annotation kind at a time
    const active = i === focusedIdx;
    const out = active ? focused : rest;
    const outcome = colors[d.det_type as "tp" | "fp" | "fn"] ?? "#ffffff";
    const label = active
      ? `${nameFor(d.class_id)}${d.conf != null ? ` ${d.conf.toFixed(2)}` : ""}`
      : undefined;

    const push = (
      geom: {
        box?: Box | { x1: number; y1: number; x2: number; y2: number };
        poly?: PolygonShape | { points: [number, number][] };
      },
      color: string,
      opts: { dashed?: boolean; fill?: boolean; tag: string },
    ) => {
      if (geom.box) {
        out.push({
          kind: "box",
          xyxy: [r1(geom.box.x1), r1(geom.box.y1), r1(geom.box.x2), r1(geom.box.y2)],
          color,
          dashed: opts.dashed,
          fill: opts.fill,
          label,
          tag: opts.tag,
        });
      } else if (geom.poly) {
        out.push({
          kind: "polygon",
          points: rPts(geom.poly.points),
          color,
          dashed: opts.dashed,
          fill: opts.fill,
          label,
          tag: opts.tag,
        });
      }
    };

    if (d.det_type === "fp") {
      if (!showPred) return;
      const b =
        d.pred_type === "box" && d.pred_idx != null ? matches.pred_boxes[d.pred_idx] : undefined;
      const p =
        d.pred_type === "polygon" && d.pred_idx != null
          ? matches.pred_polygons[d.pred_idx]
          : undefined;
      push({ box: b, poly: p }, active ? colors.active : outcome, {
        dashed: active,
        fill: true,
        tag: "fp",
      });
    } else {
      const activeFn = active && d.det_type === "fn";
      if (showGT) {
        const b = d.gt_type === "box" && d.gt_idx != null ? matches.gt_boxes[d.gt_idx] : undefined;
        const p =
          d.gt_type === "polygon" && d.gt_idx != null ? matches.gt_polygons[d.gt_idx] : undefined;
        push({ box: b, poly: p }, activeFn ? colors.active : outcome, {
          fill: activeFn || d.reviewed,
          tag: d.det_type,
        });
      }
      if (active && d.det_type === "tp" && showPred) {
        const pb =
          d.pred_type === "box" && d.pred_idx != null ? matches.pred_boxes[d.pred_idx] : undefined;
        const pp =
          d.pred_type === "polygon" && d.pred_idx != null
            ? matches.pred_polygons[d.pred_idx]
            : undefined;
        push({ box: pb, poly: pp }, colors.active, { dashed: true, fill: true, tag: "pred" });
      }
    }
  });
  return rest.concat(focused);
}

/* ── agent "push now" request (visualize_canvas refresh ping) ─────────────── */

const requestListeners = new Set<() => void>();

/** Register the mounted tab's "flush a full push now" handler; returns an unsubscribe. */
export function onCanvasStateRequest(cb: () => void): () => void {
  requestListeners.add(cb);
  return () => requestListeners.delete(cb);
}

export function notifyCanvasStateRequest(): void {
  requestListeners.forEach((cb) => cb());
}

/* ── hybrid pusher: trailing debounce + maxWait, heartbeat vs full ────────── */

export interface CanvasPusher {
  /** Register the freshest state builder; full=true marks geometry as changed. */
  schedule(build: () => CanvasStateBody | null, full: boolean): void;
  /** Send immediately (used for the agent's refresh ping). */
  flush(): void;
  dispose(): void;
}

export function createCanvasPusher(
  post: (body: CanvasStateBody) => void | Promise<unknown>,
  opts: { debounceMs?: number; maxWaitMs?: number } = {},
): CanvasPusher {
  const debounceMs = opts.debounceMs ?? 400;
  const maxWaitMs = opts.maxWaitMs ?? 1500;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let firstAt: number | null = null;
  let fullPending = false;
  let builder: (() => CanvasStateBody | null) | null = null;

  const fire = () => {
    if (timer) clearTimeout(timer);
    timer = null;
    firstAt = null;
    const full = fullPending;
    fullPending = false;
    const body = builder ? builder() : null;
    if (!body) {
      fullPending = fullPending || full; // nothing sent — geometry is still owed
      return;
    }
    if (!full) body.shapes = null; // heartbeat — backend keeps the last geometry for this image
    try {
      const res = post(body);
      if (res && typeof (res as Promise<unknown>).catch === "function") {
        // A dropped full push must not let later heartbeats masquerade as fresh geometry.
        void (res as Promise<unknown>).catch(() => {
          fullPending = fullPending || full;
        });
      }
    } catch {
      fullPending = fullPending || full;
    }
  };

  return {
    schedule(build, full) {
      builder = build;
      fullPending = fullPending || full;
      const now = Date.now();
      if (firstAt === null) firstAt = now;
      if (now - firstAt >= maxWaitMs) {
        fire(); // continuous activity (pan / stream) must still surface at maxWait cadence
        return;
      }
      if (timer) clearTimeout(timer);
      timer = setTimeout(fire, Math.min(debounceMs, firstAt + maxWaitMs - now));
    },
    flush() {
      if (builder) fire();
    },
    dispose() {
      if (timer) clearTimeout(timer);
      timer = null;
      builder = null;
    },
  };
}

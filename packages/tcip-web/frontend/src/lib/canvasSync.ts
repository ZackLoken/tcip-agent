/**
 * Live canvas-state sync: lets the agent see exactly what the canvas shows.
 *
 * Hybrid cadence so dense images (thousands of polygons) never jank the UI:
 *   - heartbeat (shapes: null; image, viewport, classes, counts; a few KB) on view/meta changes
 *   - full geometry only when shapes actually change (draw / edit / delete / image load), and
 *     never mid-drag/stream: the tab downgrades to heartbeats while a pointer interaction is
 *     live and pushes once on release, so committed geometry is never re-serialized per tick
 *   - the agent can ping "canvas_state_request" through the panel-event hub; the mounted tab
 *     answers with an immediate full push (see onCanvasStateRequest).
 *
 * Shapes are display-resolved and display-filtered: each carries the exact hex color / dash /
 * label the GUI renders, and the builders reproduce the canvas's own visibility rules (mode
 * filters, active-class filter, derived detect boxes, the labels toggle, the undecided
 * proposals), so the server-side render (capture_live_canvas) is faithful by construction.
 */

import { authorshipLabel } from "@/lib/authorshipSymbology";
import { annotationsToCanvas } from "@/lib/labelSerde";
import { ringsBbox } from "@/lib/polygonGeometry";
import type { Box, PointShape, PolygonShape, Proposal, TabName } from "@/store/types";

export interface CanvasViewport {
  x: number;
  y: number;
  w: number;
  h: number;
  scale: number;
}

/** One drawn path. A multi-ring polygon annotation contributes one shape per ring (the render
 *  contract `render_canvas_state` reads is one path per entry), all sharing the instance's color /
 *  dash / tag, with the label on the first so the instance is still named once. A `point` carries a
 *  single coordinate in `points` and is rendered as a mark, never as a path or a derived box. */
export interface CanvasShape {
  kind: "box" | "polygon" | "polyline" | "point";
  xyxy?: [number, number, number, number];
  points?: [number, number][];
  color: string;
  fill?: boolean;
  dashed?: boolean;
  // Which pattern a dashed shape draws (a tool's own unaccepted shape vs. a derived box).
  dash_kind?: "tool" | "derived";
  // On a proposal: whether the bucket's validated count operating point admits it.
  admitted?: boolean;
  label?: string;
  tag?: string; // gt | proposal | in_progress
  // The load route's authorship classification (person | tool | tool_accepted | unattributed).
  authorship?: string | null;
}

export interface CanvasStateBody {
  /** The id of the project this body was built for; the backend writes it only while that
   *  project is the one it has open. */
  project_id: string;
  tab: Extract<TabName, "annotate">;
  image_path: string;
  image: string;
  img_width: number;
  img_height: number;
  viewport: CanvasViewport | null;
  mode?: string;
  active_subject?: string;
  // Stays true across a completed cut and a refusal alike (the flag is sticky), unlike the
  // pending-segment polyline, which clears on both.
  cut_armed?: boolean;
  dirty?: boolean;
  user?: string;
  // The dataset's subjects with their GUI-local colors (the registry stores no color). Sent
  // under the backend's ``classes`` key, which stores the list verbatim for capture_live_canvas.
  classes: { name: string; color: string }[];
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

/** Whether a placed point draws in the current mode: in point mode, every point of the active
 *  subject; in any mode, the selected one (a selection survives a mode switch, so the shape being
 *  inspected stays on screen, the same rule box mode applies to the selected polygon). The
 *  Annotate canvas imports this rather than restating it, so the agent's mirror and the GUI cannot
 *  disagree about which points are on screen. */
export function pointShapeVisible(args: {
  mode: string;
  subject: string;
  activeSubject: string;
  selected: boolean;
}): boolean {
  if (args.selected) return true;
  return args.mode === "point" && args.subject === args.activeSubject;
}

/** Annotate-tab shapes, mirroring the canvas render rules exactly: the labels toggle hides
 *  everything; polygon mode shows polygons of the active subject plus the selection (outline
 *  only, like the GUI); box mode shows the active-subject boxes plus the selected polygon and the
 *  in-flight rubber-band box; points follow pointShapeVisible. Each shape's label is its subject
 *  name; its color is GUI-local. */
export function buildAnnotateShapes(args: {
  boxes: Box[];
  polygons: PolygonShape[];
  points?: PointShape[];
  currentPolygon: [number, number][];
  drawingBox?: { x1: number; y1: number; x2: number; y2: number } | null;
  selectedPolygonIdx: number | null;
  selectedBoxIdx?: number | null;
  selectedPointIdx?: number | null;
  mode: string;
  activeSubject: string;
  visible: boolean;
  colorFor: (subject: string) => string;
  // The cut tool's pending first click, in the selected polygon's own color, and the cursor for
  // its dashed tail (or none, once the start is placed but the pointer hasn't moved yet).
  cutStart?: { point: [number, number]; color: string } | null;
  cursor?: [number, number] | null;
  proposals?: Proposal[];
  selectedProposal?: number | null;
}): CanvasShape[] {
  if (!args.visible) return []; // the GUI's labels toggle hides every committed shape

  const shapes: CanvasShape[] = [];
  const pushPolygon = (p: PolygonShape, selected: boolean) => {
    const isTool = p.authorship === "tool";
    p.rings.forEach((ring, i) => {
      shapes.push({
        kind: "polygon",
        points: rPts(ring),
        color: selected ? "#00BFFF" : args.colorFor(p.subject),
        ...(isTool ? { dashed: true, dash_kind: "tool" as const } : {}),
        label: i === 0 ? authorshipLabel(p.subject, p.authorship) : undefined,
        tag: "gt",
        authorship: p.authorship ?? null,
      });
    });
  };
  // A point is one mark at one coordinate: one shape entry carrying a single position, never a
  // path and never a derived box (a fabricated box would read downstream as a real detection).
  const pushPoints = () => {
    (args.points ?? []).forEach((p, i) => {
      const selected = i === (args.selectedPointIdx ?? null);
      if (
        !pointShapeVisible({
          mode: args.mode,
          subject: p.subject,
          activeSubject: args.activeSubject,
          selected,
        })
      )
        return;
      const isTool = p.authorship === "tool";
      shapes.push({
        kind: "point",
        points: [[r1(p.x), r1(p.y)]],
        color: selected ? "#00BFFF" : args.colorFor(p.subject),
        ...(isTool ? { dashed: true, dash_kind: "tool" as const } : {}),
        label: authorshipLabel(p.subject, p.authorship),
        tag: "gt",
        authorship: p.authorship ?? null,
      });
    });
  };
  // The proposals the canvas shows, its dotted tool stroke; the selected one in the selection color.
  const pushProposals = () => {
    (args.proposals ?? []).forEach((p) => {
      const color = p.index === args.selectedProposal ? "#00BFFF" : args.colorFor(p.subject);
      const base = { color, dashed: true, dash_kind: "tool" as const, tag: "proposal" };
      const extra = { label: `${p.subject} proposal`, admitted: p.admitted };
      const { boxes, polygons } = annotationsToCanvas([p]);
      boxes.forEach((b) =>
        shapes.push({
          kind: "box",
          xyxy: [r1(b.x1), r1(b.y1), r1(b.x2), r1(b.y2)],
          ...base,
          ...extra,
        }),
      );
      polygons.forEach((poly) =>
        poly.rings.forEach((ring, i) =>
          shapes.push({ kind: "polygon", points: rPts(ring), ...base, ...(i === 0 ? extra : {}) }),
        ),
      );
    });
  };

  if (args.mode === "polygon") {
    args.polygons.forEach((p, i) => {
      const selected = i === args.selectedPolygonIdx;
      if (!selected && p.subject !== args.activeSubject) return;
      pushPolygon(p, selected);
    });
    if (args.currentPolygon.length > 0) {
      shapes.push({
        kind: "polyline",
        points: rPts(args.currentPolygon),
        // Mirrors the canvas's own InProgressPolygon stroke exactly: the active subject's color,
        // amber only in the edge case where nothing is selected (drawing is otherwise blocked).
        color: args.activeSubject ? args.colorFor(args.activeSubject) : "#FFE7B1",
        dashed: true,
        label: "drawing",
        tag: "in_progress",
      });
    }
    if (args.cutStart) {
      const pts: [number, number][] = args.cursor
        ? [args.cutStart.point, args.cursor]
        : [args.cutStart.point];
      shapes.push({
        kind: "polyline",
        points: rPts(pts),
        color: args.cutStart.color,
        dashed: true,
        label: "cut",
        tag: "in_progress",
      });
    }
    pushPoints();
    pushProposals();
    return shapes;
  }

  // Box mode: the active subject's editable boxes render solid. Point mode draws no box or
  // derived box, only its own points and the selection carried in from another mode.
  const boxMode = args.mode === "box";
  args.boxes.forEach((b, i) => {
    if (!boxMode || b.subject !== args.activeSubject) return;
    const selected = i === (args.selectedBoxIdx ?? null);
    const isTool = b.authorship === "tool";
    shapes.push({
      kind: "box",
      xyxy: [r1(b.x1), r1(b.y1), r1(b.x2), r1(b.y2)],
      color: selected ? "#00BFFF" : args.colorFor(b.subject),
      ...(isTool ? { dashed: true, dash_kind: "tool" as const } : {}),
      label: authorshipLabel(b.subject, b.authorship),
      tag: "gt",
      authorship: b.authorship ?? null,
    });
  });
  // ...plus each active-subject polygon's read-only derived box, mirroring the canvas so the capture
  // stays faithful. Derived from ringsBbox here (the same min/max the loader
  // re-derives, over every ring), never a stored box, so it can't be double-counted as its own
  // annotation. Dashed distinguishes it from a real editable box (solid), the same convention the
  // in-progress/under-review shapes already use for "not a committed, directly-editable annotation."
  args.polygons.forEach((p) => {
    if (!boxMode || p.subject !== args.activeSubject) return;
    const [x1, y1, x2, y2] = ringsBbox(p.rings);
    shapes.push({
      kind: "box",
      xyxy: [r1(x1), r1(y1), r1(x2), r1(y2)],
      color: args.colorFor(p.subject),
      dashed: true,
      dash_kind: "derived",
      label: authorshipLabel(p.subject, p.authorship),
      tag: "gt",
      authorship: p.authorship ?? null,
    });
  });
  // The other modes still show the selected polygon (the shape being inspected).
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
      color: args.colorFor(args.activeSubject),
      dashed: true,
      tag: "in_progress",
    });
  }
  pushPoints();
  pushProposals();
  return shapes;
}

/* ── agent "push now" request (capture_live_canvas refresh ping) ─────────────── */

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
      fullPending = fullPending || full; // nothing sent, geometry is still owed
      return;
    }
    if (!full) body.shapes = null; // heartbeat: backend keeps the last geometry for this image
    try {
      const res = post(body);
      if (res && typeof (res as Promise<unknown>).then === "function") {
        // A dropped full push (rejected, or resolved as a conflict) must not let later
        // heartbeats masquerade as fresh geometry.
        void (res as Promise<{ status?: string } | unknown>).then(
          (r) => {
            if (r && typeof r === "object" && (r as { status?: string }).status === "conflict") {
              fullPending = fullPending || full;
            }
          },
          () => {
            fullPending = fullPending || full;
          },
        );
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

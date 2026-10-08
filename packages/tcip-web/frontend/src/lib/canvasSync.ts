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
 * label the GUI renders, read from the one symbology module the overlays draw with, and the
 * builders ask the canvas's own visibility and focus rules (mode filters, active-class filter,
 * derived detect boxes, the labels toggle, the shown proposals), so the server-side render
 * (capture_live_canvas) draws what the canvas shows; the render's line widths, tick marks and
 * vertex dots are its own, scaled to the frame it draws on.
 */

import type { CanvasStatePayload } from "@/api/types.generated";
import { annotationsToCanvas } from "@/lib/labelSerde";
import { ringsBbox } from "@/lib/polygonGeometry";
import { boxDraft, type DraftStroke } from "@/lib/draftStrokes";
import {
  focusesAnnotation,
  focusesProposal,
  type MatchTypes,
  NO_MATCHES,
  proposalMatch,
  type ReviewItem,
} from "@/lib/reviewItems";
import {
  authorshipLabel,
  type DashUnits,
  dashUnitsFor,
  DRAFT_DASH,
  FLAG_MARK,
  FOCUS_HALO,
  lineStyleOf,
  outlineColor,
  proposalLabel,
} from "@/lib/symbology";
import type { DrawingBox, Flag, Proposal, TabName, ToolContent } from "@/store/types";

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
  // The stroke's (on, off) dash in stroke widths: symbology's DOTTED_DASH or DRAFT_DASH.
  dash?: DashUnits;
  // On the focused item: the halo the render draws under its stroke (symbology's FOCUS_HALO).
  halo?: { color: string; opacity: number; width_factor: number };
  label?: string;
  // False on a draft's tail, which draws no dot at its ends; a polyline dots its vertices otherwise.
  vertices?: boolean;
  tag?: string; // gt | proposal | in_progress | flag
}

/** The push body: the backend's own model, with the display geometry it stores uninterpreted typed
 *  here. ``classes`` carries the subjects with their GUI-local colors (the registry stores none);
 *  ``shapes`` is null for a heartbeat, where the backend keeps the last pushed geometry. */
export type CanvasStateBody = Omit<
  CanvasStatePayload,
  "tab" | "viewport" | "classes" | "counts" | "shapes"
> & {
  tab: Extract<TabName, "annotate">;
  viewport: CanvasViewport | null;
  classes: { name: string; color: string }[];
  counts?: Record<string, number>;
  shapes: CanvasShape[] | null;
};

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

/** Whether a committed shape draws: only in its own tool mode, and there when it is of the active
 *  subject or is the focused item (a step can land on another subject's shape). A `derived` box
 *  is a polygon's read-only bounds, drawn with the boxes. The Annotate canvas and the agent's
 *  mirror both ask here, so they cannot disagree about what is on screen. */
export function shapeVisible(args: {
  kind: "box" | "derived" | "polygon" | "point";
  mode: string;
  subject: string;
  activeSubject: string;
  focused: boolean;
}): boolean {
  const ownMode = args.kind === "derived" ? "box" : args.kind;
  return args.mode === ownMode && (args.focused || args.subject === args.activeSubject);
}

/** The line style and focus halo a mirrored shape carries, from the symbology's own rules. */
function strokeOf(authorship: string | null | undefined, focused: boolean) {
  const dash = dashUnitsFor(lineStyleOf(authorship));
  return {
    ...(dash ? { dash } : {}),
    ...(focused
      ? {
          halo: {
            color: FOCUS_HALO.color,
            opacity: FOCUS_HALO.opacity,
            width_factor: FOCUS_HALO.widthFactor,
          },
        }
      : {}),
  };
}

/** Annotate-tab shapes, mirroring the canvas: the labels toggle hides every committed shape and
 *  never a draft, each committed shape draws by `shapeVisible`, and polygon mode adds the drawing
 *  in progress and the pending cut. Colors, line styles, the halo and labels come from the symbology module: the focused item
 *  and the unpaired proposals are the labeled ones. */
export function buildAnnotateShapes(
  args: ToolContent & {
    /** The strokes of the polygon in progress and the pending cut (`draftStrokes`). */
    draft?: DraftStroke[];
    /** The box being dragged out, in its own subject's color as on the canvas. */
    drawingBox?: DrawingBox | null;
    /** The one focused item (an annotation or a shown proposal), or none. */
    focused?: ReviewItem | null;
    /** Each array's match types (`matchTypes`); every entry null while not reviewing. */
    matches?: MatchTypes;
    /** Each open flag that has a place on the image, with that place (`flagPlaces`). */
    flagMarks?: { at: [number, number]; flag: Flag }[];
    mode: string;
    activeSubject: string;
    visible: boolean;
    /** A subject's color, for the box being dragged out. */
    colorFor: (subject: string) => string;
    /** The proposals the canvas shows. */
    proposals?: Proposal[];
  },
): CanvasShape[] {
  const d = args.drawingBox ? boxDraft(args.drawingBox, args.colorFor) : null;
  const drafts: CanvasShape[] = [
    ...(args.draft ?? []).map(({ points, color, vertices, label }) => ({
      kind: "polyline" as const,
      points: rPts(points),
      color,
      dash: DRAFT_DASH,
      ...(vertices ? { label } : { vertices: false }),
      tag: "in_progress",
    })),
    ...(d
      ? [
          {
            kind: "box" as const,
            xyxy: d.bounds.map(r1) as [number, number, number, number],
            color: d.color,
            dash: DRAFT_DASH,
            tag: "in_progress",
          },
        ]
      : []),
  ];
  // The labels toggle hides every committed shape; a draft is never gated by it, as on the canvas.
  if (!args.visible) return drafts;

  const statuses = args.matches ?? NO_MATCHES;
  const focused = args.focused ?? null;
  const isFocused = (shape: ReviewItem["shape"], i: number) => focusesAnnotation(focused, shape, i);
  const shapes: CanvasShape[] = [];
  const visible = (kind: "box" | "derived" | "polygon" | "point", subject: string, at: boolean) =>
    shapeVisible({
      kind,
      mode: args.mode,
      subject,
      activeSubject: args.activeSubject,
      focused: at,
    });

  args.boxes.forEach((b, i) => {
    const boxFocused = isFocused("box", i);
    if (!visible("box", b.subject, boxFocused)) return;
    shapes.push({
      kind: "box",
      xyxy: [r1(b.x1), r1(b.y1), r1(b.x2), r1(b.y2)],
      color: outlineColor(b.subject, statuses.boxes[i] ?? null),
      ...strokeOf(b.authorship, boxFocused),
      label: boxFocused ? authorshipLabel(b.subject, b.authorship) : undefined,
      tag: "gt",
    });
  });
  // A polygon's read-only derived box: ringsBbox, never a stored box, never focused or labeled.
  args.polygons.forEach((p, i) => {
    if (!visible("derived", p.subject, false)) return;
    const [x1, y1, x2, y2] = ringsBbox(p.rings);
    shapes.push({
      kind: "box",
      xyxy: [r1(x1), r1(y1), r1(x2), r1(y2)],
      color: outlineColor(p.subject, statuses.polygons[i] ?? null),
      ...strokeOf(p.authorship, false),
      tag: "gt",
    });
  });
  args.polygons.forEach((p, i) => {
    const polygonFocused = isFocused("polygon", i);
    if (!visible("polygon", p.subject, polygonFocused)) return;
    p.rings.forEach((ring, ri) => {
      shapes.push({
        kind: "polygon",
        points: rPts(ring),
        color: outlineColor(p.subject, statuses.polygons[i] ?? null),
        ...strokeOf(p.authorship, polygonFocused),
        label: ri === 0 && polygonFocused ? authorshipLabel(p.subject, p.authorship) : undefined,
        tag: "gt",
      });
    });
  });
  // A point is one mark at one coordinate: one shape entry carrying a single position, never a
  // path and never a derived box (a fabricated box would read downstream as a real detection).
  args.points.forEach((p, i) => {
    const pointFocused = isFocused("point", i);
    if (!visible("point", p.subject, pointFocused)) return;
    shapes.push({
      kind: "point",
      points: [[r1(p.x), r1(p.y)]],
      color: outlineColor(p.subject, statuses.points[i] ?? null),
      ...strokeOf(p.authorship, pointFocused),
      label: pointFocused ? authorshipLabel(p.subject, p.authorship) : undefined,
      tag: "gt",
    });
  });
  (args.proposals ?? []).forEach((p) => {
    const proposalFocused = focusesProposal(focused, p.index);
    const base = {
      color: outlineColor(p.subject, proposalMatch(p)),
      ...strokeOf("tool", proposalFocused),
      tag: "proposal",
    };
    const label = proposalLabel(p, proposalFocused);
    const extra = label !== null ? { label } : {};
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
  (args.flagMarks ?? []).forEach(({ at, flag }) =>
    shapes.push({
      kind: "point",
      points: [[r1(at[0]), r1(at[1])]],
      color: FLAG_MARK.color,
      label: `${FLAG_MARK.glyph} ${flag.text}`,
      tag: "flag",
    }),
  );
  return [...shapes, ...drafts];
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

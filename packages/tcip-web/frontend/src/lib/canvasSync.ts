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

import type {
  Authorship,
  CanvasClass,
  CanvasStatePayload,
  CaptureImage,
  ServedFlag,
  Viewport,
  ViewState,
} from "@/api/types.generated";
import { boxDraft, type DraftStroke } from "@/lib/draftStrokes";
import { annotationsToCanvas } from "@/lib/labelSerde";
import { ringsBbox } from "@/lib/polygonGeometry";
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
  CASING,
  dashUnitsFor,
  DRAFT_DASH,
  FLAG_MARK,
  FOCUS_HALO,
  lineStyleOf,
  outlineColor,
  proposalLabel,
} from "@/lib/symbology";
import type { CanvasState } from "@/store/slices/canvas";
import type {
  CanvasContent,
  DrawingBox,
  Mode,
  Proposal,
  TabName,
  ToolContent,
} from "@/store/types";

/** One drawn path. A multi-ring polygon annotation contributes one shape per ring (the render
 *  contract `render_canvas_state` reads is one path per entry), all sharing the instance's color /
 *  dash / tag, with the label on the first so the instance is still named once. A `point` carries a
 *  single coordinate in `points` and is rendered as a mark, never as a path or a derived box. */
export type CanvasShape = NonNullable<CanvasStatePayload["shapes"]>[number];

/** The push body: the backend's own model. ``classes`` carries the subjects with their GUI-local
 *  colors (the registry stores none); ``shapes`` is null for a heartbeat, where the backend keeps
 *  the last pushed geometry. */
export type CanvasStateBody = Omit<CanvasStatePayload, "tab"> & {
  tab: Extract<TabName, "annotate">;
};

/** The push body of the Annotate canvas holding `canvas` over `image` of the project
 *  `projectId`, its counts read off the canvas; a heartbeat when `shapes` is null. */
export function canvasStateBody(args: {
  projectId: string;
  image: CaptureImage;
  canvas: CanvasContent & Pick<CanvasState, "dirty" | "currentPolygon">;
  viewport: Viewport | null;
  mode: Mode;
  activeSubject: string | null;
  cutArmed: boolean;
  user: string;
  classes: CanvasClass[];
  shapes: CanvasShape[] | null;
}): CanvasStateBody {
  const { canvas } = args;
  return {
    project_id: args.projectId,
    tab: "annotate",
    image_path: args.image.path,
    viewport: args.viewport,
    mode: args.mode,
    active_subject: args.activeSubject,
    cut_armed: args.cutArmed,
    dirty: canvas.dirty,
    user: args.user,
    classes: args.classes,
    counts: {
      boxes: canvas.boxes.length,
      polygons: canvas.polygons.length,
      points: canvas.points.length,
      image_ratings: canvas.imageAnnotations.length,
      drawing_points: canvas.currentPolygon.length,
    },
    shapes: args.shapes,
  };
}

/** 0.1-px precision is beyond what any render needs; rounding cuts dense payloads ~2-3×. */
const r1 = (n: number): number => Math.round(n * 10) / 10;
const rPts = (pts: [number, number][]): [number, number][] => pts.map(([x, y]) => [r1(x), r1(y)]);

/** The visible image region (image coords) from the pan/zoom view + canvas host size. */
export function computeViewport(
  view: ViewState,
  host: { w: number; h: number },
  imgW: number,
  imgH: number,
): Viewport | null {
  const s = view.scale;
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
  activeSubject: string | null;
  focused: boolean;
}): boolean {
  const ownMode = args.kind === "derived" ? "box" : args.kind;
  return args.mode === ownMode && (args.focused || args.subject === args.activeSubject);
}

/** The line style, casing and focus halo a mirrored shape carries, from the symbology's own
 *  rules. */
function strokeOf(authorship: Authorship | undefined, focused: boolean) {
  return {
    dash: dashUnitsFor(lineStyleOf(authorship)) ?? null,
    casing: { color: CASING.color, opacity: CASING.opacity, width_factor: CASING.widthFactor },
    halo: focused
      ? {
          color: FOCUS_HALO.color,
          opacity: FOCUS_HALO.opacity,
          width_factor: FOCUS_HALO.widthFactor,
        }
      : null,
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
    flagMarks?: { at: [number, number]; flag: ServedFlag }[];
    mode: string;
    activeSubject: string | null;
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
      casing: null,
      halo: null,
      label: label ?? null,
      vertices,
      tag: "in_progress",
    })),
    ...(d
      ? [
          {
            kind: "box" as const,
            xyxy: d.bounds.map(r1) as [number, number, number, number],
            color: d.color,
            dash: DRAFT_DASH,
            casing: null,
            halo: null,
            label: null,
            fill: false,
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
      label: boxFocused ? authorshipLabel(b.subject, b.authorship) : null,
      fill: false,
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
      label: null,
      fill: false,
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
        label: ri === 0 && polygonFocused ? authorshipLabel(p.subject, p.authorship) : null,
        fill: false,
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
      label: pointFocused ? authorshipLabel(p.subject, p.authorship) : null,
      tag: "gt",
    });
  });
  (args.proposals ?? []).forEach((p) => {
    const proposalFocused = focusesProposal(focused, p.index, p.bucket);
    const base = {
      color: outlineColor(p.subject, proposalMatch(p)),
      ...strokeOf("tool", proposalFocused),
      tag: "proposal",
    };
    const label = proposalLabel(p, proposalFocused);
    const { boxes, polygons, points } = annotationsToCanvas([p]);
    boxes.forEach((b) =>
      shapes.push({
        kind: "box",
        xyxy: [r1(b.x1), r1(b.y1), r1(b.x2), r1(b.y2)],
        ...base,
        fill: false,
        label,
      }),
    );
    polygons.forEach((poly) =>
      poly.rings.forEach((ring, i) =>
        shapes.push({
          kind: "polygon",
          points: rPts(ring),
          ...base,
          fill: false,
          label: i === 0 ? label : null,
        }),
      ),
    );
    points.forEach((pt) =>
      shapes.push({ kind: "point", points: [[r1(pt.x), r1(pt.y)]], ...base, label }),
    );
  });
  (args.flagMarks ?? []).forEach(({ at, flag }) =>
    shapes.push({
      kind: "point",
      points: [[r1(at[0]), r1(at[1])]],
      color: FLAG_MARK.color,
      dash: null,
      casing: null,
      halo: null,
      label: `${FLAG_MARK.glyph} ${flag.request.text}`,
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

/** A state builder: with `full` false it is a heartbeat, which carries `shapes: null` and builds
 *  no geometry. */
export type CanvasBodyBuilder = (full: boolean) => CanvasStateBody | null;

export interface CanvasPusher {
  /** Register the freshest state builder; full=true marks geometry as changed. */
  schedule(build: CanvasBodyBuilder, full: boolean): void;
  /** Send immediately (used for the agent's refresh ping). */
  flush(): void;
  dispose(): void;
}

/** A pusher of the freshest canvas state through `post`, whose promise rejects for a push the
 *  backend refused. */
export function createCanvasPusher(
  post: (body: CanvasStateBody) => void | Promise<unknown>,
  opts: { debounceMs?: number; maxWaitMs?: number } = {},
): CanvasPusher {
  const debounceMs = opts.debounceMs ?? 400;
  const maxWaitMs = opts.maxWaitMs ?? 1500;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let firstAt: number | null = null;
  let fullPending = false;
  let builder: CanvasBodyBuilder | null = null;

  const fire = () => {
    if (timer) clearTimeout(timer);
    timer = null;
    firstAt = null;
    const full = fullPending;
    fullPending = false;
    const body = builder ? builder(full) : null;
    if (!body) {
      fullPending = fullPending || full; // nothing sent, geometry is still owed
      return;
    }
    try {
      const res = post(body);
      if (res) {
        // A refused full push must not let later heartbeats masquerade as fresh geometry.
        void res.catch(() => {
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

/**
 * The strokes of a polygon being drawn or a cut being placed, resolved once for the canvas overlay
 * and for the agent's mirror: the laid vertices (each dotted) and the tail from the last one to
 * the cursor (never dotted). Neither consumer is gated by the labels toggle.
 */

import { NO_SUBJECT_DRAFT_COLOR } from "@/lib/symbology";
import type { DrawingBox } from "@/store/types";

export interface DraftStroke {
  points: [number, number][];
  color: string;
  /** Whether each point is a laid vertex that draws a dot. */
  vertices: boolean;
  /** What the mirror names the stroke, when it names it. */
  label?: string;
}

/** A box's corners as bounds in ascending order, whichever corner the drag started at. */
export function boxBounds(d: DrawingBox): [number, number, number, number] {
  return [Math.min(d.x1, d.x2), Math.min(d.y1, d.y2), Math.max(d.x1, d.x2), Math.max(d.y1, d.y2)];
}

/** The box being dragged out: its bounds in ascending order and its own subject's color. */
export function boxDraft(
  d: DrawingBox,
  colorFor: (subject: string) => string,
): { bounds: [number, number, number, number]; color: string } {
  return { bounds: boxBounds(d), color: colorFor(d.subject) };
}

export function draftStrokes(args: {
  mode: string;
  currentPolygon: [number, number][];
  /** The subject whose color the polygon in progress draws in; none draws the neutral draft color. */
  polygonColor: string | null;
  cutStart: { point: [number, number]; color: string } | null;
  cursor: [number, number] | null;
}): DraftStroke[] {
  if (args.mode !== "polygon") return [];
  const strokes: DraftStroke[] = [];
  const add = (laid: [number, number][], color: string, label: string) => {
    strokes.push({ points: laid, color, vertices: true, label });
    if (args.cursor) {
      strokes.push({ points: [laid[laid.length - 1], args.cursor], color, vertices: false });
    }
  };
  if (args.currentPolygon.length > 0) {
    add(args.currentPolygon, args.polygonColor ?? NO_SUBJECT_DRAFT_COLOR, "drawing");
  }
  if (args.cutStart) add([args.cutStart.point], args.cutStart.color, "cut");
  return strokes;
}

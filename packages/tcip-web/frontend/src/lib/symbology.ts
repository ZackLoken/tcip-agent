/**
 * The Annotate canvas' symbology, one rule per channel, stated once for the overlays, the legend
 * and the agent's canvas mirror: line style says whose shape it is, outline color says review
 * status while proposals are shown and subject otherwise, focus is a halo under the item's own
 * stroke, and line widths derive from the zoom.
 */

import { subjectColor } from "@/api/subjects";

/** Whose shape it is: a person's (or one a person accepted) draws solid, a tool's unaccepted
 *  shape and every proposal draw dotted. */
export type LineStyle = "solid" | "dotted";

export function lineStyleOf(authorship: string | null | undefined): LineStyle {
  return authorship === "tool" ? "dotted" : "solid";
}

/** The dash array for a style at a stroke width; none for a solid stroke. */
export function dashFor(style: LineStyle, width: number): number[] | undefined {
  return style === "dotted" ? [width, 3 * width] : undefined;
}

/** An item's standing in the review of the shown bucket: a proposal awaiting a decision (and the
 *  annotation it pairs with), an annotation a decision confirmed, or an annotation the bucket
 *  proposes nothing for. */
export type ReviewStatus = "undecided" | "confirmed" | "unproposed";

export const REVIEW_STATUSES: readonly ReviewStatus[] = ["undecided", "confirmed", "unproposed"];

/** Status colors, set by eye against the canvas surface and apart from each other; provisional. */
export const STATUS_COLORS: Record<ReviewStatus, string> = {
  undecided: "#FFD54A",
  confirmed: "#5BD17A",
  unproposed: "#C77DFF",
};

export const STATUS_WORDS: Record<ReviewStatus, string> = {
  undecided: "Awaiting a decision",
  confirmed: "Confirmed by a decision",
  unproposed: "Nothing proposed for it",
};

/** The focus halo: a wider translucent stroke under the focused item's own; set by eye,
 *  provisional. */
export const FOCUS_HALO = { color: "#FFFFFF", opacity: 0.55, widthFactor: 3 } as const;

/** An item's outline color: its review status while proposals are shown, its subject otherwise. */
export function outlineColor(subject: string, status: ReviewStatus | null): string {
  return status === null ? subjectColor(subject) : STATUS_COLORS[status];
}

/** The focused item's label: the subject name, with ", tool" appended for a shape a tool drew
 *  and no person has accepted and ", accepted tool" for one a person has since accepted. */
export function authorshipLabel(subject: string, authorship?: string | null): string {
  if (authorship === "tool") return `${subject}, tool`;
  if (authorship === "tool_accepted") return `${subject}, accepted tool`;
  return subject;
}

/** The label a proposal carries, or none: the focused proposal and every unpaired one are
 *  labeled, with the subject and score, naming the pairing when there is one. */
export function proposalLabel(
  p: { subject: string; score?: number | null; paired: number | null },
  focused: boolean,
): string | null {
  if (!focused && p.paired !== null) return null;
  const scored = p.score != null ? ` ${p.score.toFixed(2)}` : "";
  return `${p.subject} proposal${scored}${p.paired !== null ? " (pairs with an annotation)" : ""}`;
}

/** The stroke of a shape being drawn while no subject is active. */
export const NO_SUBJECT_DRAFT_COLOR = "#FFE7B1";

/** Screen-pixel reach of a placed point's mark, which is also its grab radius. */
export const POINT_HIT_CANVAS = 11;

const VERTEX_HANDLE_RADIUS = 4;

/** Every stroke, handle and label size at zoom `scale`, resolved in screen pixels and then
 *  divided by the scale once, so a width grows mildly with zoom on screen and never with the
 *  image. */
export function strokeWidths(scale: number) {
  const s = scale || 1;
  const scaleLineW = 1 / s;
  const vertScreen = Math.max(3, Math.min(VERTEX_HANDLE_RADIUS * (1.6 - s * 0.2), 12));
  const selScreen = Math.max(vertScreen + 1, Math.min(VERTEX_HANDLE_RADIUS * (2.2 - s * 0.2), 16));
  return {
    scaleLineW,
    boxStroke: Math.max(1, Math.min(2 + s * 0.5, 6)) * scaleLineW,
    polyStroke: Math.max(1, Math.min(2.5 + s * 0.5, 7)) * scaleLineW,
    vertR: vertScreen * scaleLineW,
    selVertR: selScreen * scaleLineW,
    labelSize: Math.max(8, Math.min(Math.round(9 * (0.6 + s * 0.4)), 18)) * scaleLineW,
    pointCoreR: 3.5 * scaleLineW,
    pointSelCoreR: 5 * scaleLineW,
    pointTickInner: 6.5 * scaleLineW,
    pointTickOuter: POINT_HIT_CANVAS * scaleLineW,
  };
}

/** One legend row: a swatch drawn with the same color and line style the canvas uses. */
export interface LegendRow {
  color: string;
  style: LineStyle;
  /** The focus halo row draws the halo around a neutral swatch. */
  halo?: boolean;
  text: string;
  /** The subject a row names, when it names one (the row then opens its color picker). */
  subject?: string;
}

/** The legend's rows, from the constants above: the review statuses while proposals are shown,
 *  else each subject in its color; then the line styles and the focus halo. */
export function legendRows(subjects: string[], reviewing: boolean): LegendRow[] {
  const colors: LegendRow[] = reviewing
    ? REVIEW_STATUSES.map((status) => ({
        color: STATUS_COLORS[status],
        style: "solid",
        text: STATUS_WORDS[status],
      }))
    : subjects.map((name) => ({
        color: subjectColor(name),
        style: "solid",
        text: name,
        subject: name,
      }));
  return [
    ...colors,
    { color: "currentColor", style: "solid", text: "Solid: drawn or accepted by a person" },
    { color: "currentColor", style: "dotted", text: "Dotted: drawn by a tool, not yet accepted" },
    { color: "currentColor", style: "solid", halo: true, text: "Halo: the focused item" },
  ];
}

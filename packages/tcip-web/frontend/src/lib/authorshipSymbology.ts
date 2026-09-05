/**
 * The Annotate canvas' dash symbology, one table shared by the box, polygon and point overlays so
 * a "derived" pattern (a polygon's own read-only bounding box) and a "tool" pattern (a shape a
 * tool drew that no person has accepted) never drift apart between them. The classification itself
 * comes from the backend's `authorship_of` (carried onto each canvas shape by labelSerde); nothing
 * here re-derives it from a `created_by` string.
 */

export type DashKind = "derived" | "tool";

/** The dash array for one kind, in stroke-width units. */
export function dashPattern(kind: DashKind, width: number): number[] {
  return kind === "tool" ? [width, 3 * width] : [6 * width, 4 * width];
}

/** A shape's hover label: the subject name, with ", tool" appended for one a tool drew and no
 *  person has accepted, so the same fact the dotted stroke shows is also named on hover. */
export function authorshipLabel(subject: string, authorship?: string | null): string {
  return authorship === "tool" ? `${subject}, tool` : subject;
}

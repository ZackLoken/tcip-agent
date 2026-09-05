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
 *  person has accepted (the dotted stroke's own fact, also named on hover) and ", accepted tool"
 *  for one a person has since accepted (no stroke change; the solid stroke a person's own shape
 *  also draws stays unchanged, so the suffix is the only mark of the acceptance). */
export function authorshipLabel(subject: string, authorship?: string | null): string {
  if (authorship === "tool") return `${subject}, tool`;
  if (authorship === "tool_accepted") return `${subject}, accepted tool`;
  return subject;
}

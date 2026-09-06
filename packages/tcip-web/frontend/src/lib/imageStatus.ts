import type { ImageStatus } from "@/api/classes";

/** What a hydrate reconcile does with one image name: write its derived token to the store, or
 *  hold it back because a human already confirmed a different one. */
export interface ImageStatusReconcile {
  /** Names outside `confirmed` whose derived token differs from the stored one: safe to write,
   *  since nobody has asserted anything about them yet. */
  writes: Record<string, ImageStatus>;
  /** Confirmed names (stored `complete` or `negative`) whose derived token now disagrees, in
   *  either direction: the label file's content changed since a human finished the image, so the
   *  mark needs a fresh look rather than a silent rewrite. A name whose content is unchanged but
   *  whose subject's attribute schema moved under it is a separate, digest-stale cause the
   *  hydrate hook unions in from the status route, not this reconcile. */
  staleMarks: string[];
}

/** Reconciles a dataset's stored per-image statuses against what the label files derive to now,
 *  without ever overwriting a human's confirmation.
 *
 *  `stored` is what the status store holds; `derived` is what `deriveImageStatus` returns for the
 *  same image list, honoring `confirmed` as its `complete_override`; `confirmed` is every name in
 *  `stored` whose value is `"complete"` or `"negative"`. A name outside `confirmed` heals freely
 *  (its status was never a human assertion); a confirmed name that now derives differently is
 *  never rewritten, only flagged for the breeder to re-confirm. */
export function reconcileImageStatuses(
  stored: Record<string, ImageStatus>,
  derived: Record<string, ImageStatus>,
  confirmed: string[],
): ImageStatusReconcile {
  const confirmedSet = new Set(confirmed);
  const writes: Record<string, ImageStatus> = {};
  const staleMarks: string[] = [];
  for (const [name, status] of Object.entries(derived)) {
    if (stored[name] === status) continue;
    if (confirmedSet.has(name)) staleMarks.push(name);
    else writes[name] = status;
  }
  return { writes, staleMarks: staleMarks.sort() };
}

/** A shape carrying a `subject` field: a box, polygon, point or image-level rating. */
interface SubjectScoped {
  subject: string;
}

/** The canvas content a subject-scoped check reads: every shape kind a save or a status write
 *  considers, image-level ratings included. */
export interface SubjectScopedShapes {
  boxes: SubjectScoped[];
  polygons: SubjectScoped[];
  points: SubjectScoped[];
  imageAnnotations: SubjectScoped[];
}

/** Whether the canvas holds at least one shape (of any kind, an image-level rating included)
 *  authored for `subject`. The one predicate the Complete toggle, the stale re-confirm action and
 *  the save path all call to decide whether an image carries this subject's content, so a status
 *  write can't read a different subject's shapes as this one's. A null subject holds nothing. */
export function canvasHoldsSubject(shapes: SubjectScopedShapes, subject: string | null): boolean {
  if (!subject) return false;
  return (
    shapes.boxes.some((s) => s.subject === subject) ||
    shapes.polygons.some((s) => s.subject === subject) ||
    shapes.points.some((s) => s.subject === subject) ||
    shapes.imageAnnotations.some((s) => s.subject === subject)
  );
}

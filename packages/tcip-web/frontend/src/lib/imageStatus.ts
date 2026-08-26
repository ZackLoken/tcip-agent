import type { ImageStatus } from "@/api/classes";

/** What a hydrate reconcile does with one image name: write its derived token to the store, or
 *  hold it back because a human already confirmed a different one. */
export interface ImageStatusReconcile {
  /** Names outside `confirmed` whose derived token differs from the stored one: safe to write,
   *  since nobody has asserted anything about them yet. */
  writes: Record<string, ImageStatus>;
  /** Confirmed names (stored `complete` or `negative`) whose derived token now disagrees, in
   *  either direction: the label file changed since a human finished the image, so the mark
   *  needs a fresh look rather than a silent rewrite. */
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

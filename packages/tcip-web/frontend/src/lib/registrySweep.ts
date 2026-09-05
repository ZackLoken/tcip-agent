import type { SchemaChangeSweep } from "@/api/classes";

/** The toast a registry save's `schema_change_sweep` earns, shared by every door that grows the
 *  registry (the attribute panel's declare paths, the toolbar's subject add): names each affected
 *  subject and how many of its confirmations now predate the vocabulary change, plus any warning
 *  the sweep itself raised. Null when nothing was stamped, nothing predates the change, and
 *  nothing warned. States only the fact and says nothing about what happens next: a stale negative
 *  is quarantined by the training carry today, a stale complete confirmation is not, and this
 *  toast does not speak to that difference. */
export function schemaChangeSweepToast(sweep: SchemaChangeSweep): string | null {
  const parts: string[] = [];
  for (const [subject, count] of Object.entries(sweep.newly_stamped)) {
    if (count > 0) {
      parts.push(
        `${count} of ${subject}'s confirmations now predate this vocabulary change and must be ` +
          "re-reviewed before they train.",
      );
    }
  }
  for (const [subject, count] of Object.entries(sweep.predating_vocabulary)) {
    if (count > 0) {
      parts.push(
        `${count} confirmed images of ${subject} were confirmed under its previous vocabulary.`,
      );
    }
  }
  if (sweep.warning) parts.push(sweep.warning);
  return parts.length ? parts.join(" ") : null;
}

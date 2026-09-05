import type { SchemaChangeSweep } from "@/api/classes";

/** The toast a registry save's `schema_change_sweep` earns, shared by every door that grows the
 *  registry (the attribute panel's declare paths, the toolbar's subject add): names each affected
 *  subject and how many of its confirmations now predate the vocabulary change, plus any warning
 *  the sweep itself raised. Null when nothing was stamped and nothing warned. */
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
  if (sweep.warning) parts.push(sweep.warning);
  return parts.length ? parts.join(" ") : null;
}

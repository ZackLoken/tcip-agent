import type { SchemaChangeSweep } from "@/api/classes";

/** The toast a registry save's `schema_change_sweep` earns, shared by every door that grows the
 *  registry (the attribute panel's declare paths, the toolbar's subject add): names each affected
 *  subject and how many of its confirmations, complete or negative alike, are quarantined from
 *  training until re-confirmed, plus any warning the sweep itself raised. Reads only
 *  `predating_vocabulary`: a newly stamped confirmation is, by construction, counted there too
 *  once the sweep completes, so looping `newly_stamped` as well would report the same image twice
 *  under two different sentences; it still rides the response and the audit line for the
 *  operator. Null when nothing predates the change and nothing warned. */
export function schemaChangeSweepToast(sweep: SchemaChangeSweep): string | null {
  const parts: string[] = [];
  for (const [subject, count] of Object.entries(sweep.predating_vocabulary)) {
    if (count > 0) {
      parts.push(
        `${count} confirmed image(s) of ${subject} were confirmed under its previous vocabulary ` +
          "and will train again once re-confirmed.",
      );
    }
  }
  if (sweep.warning) parts.push(sweep.warning);
  return parts.length ? parts.join(" ") : null;
}

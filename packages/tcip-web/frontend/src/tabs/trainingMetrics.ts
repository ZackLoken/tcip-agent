/** Metric-stream helpers for the Training tab (kept out of the .tsx so they're unit-testable). */

import type { MetricRow } from "@/api/training";

/** The run list's own poll cadence, shared by TrainingTab's row refresh and RunComparison's
 * comparison refresh so the two surfaces poll on one cadence rather than two literals. */
export const RUN_REFRESH_MS = 4000;

/** The row's identity for de-duplication: epoch if present, else step, else none. */
export function metricKey(row: MetricRow): number | undefined {
  if (typeof row.epoch === "number") return row.epoch;
  if (typeof row.step === "number") return row.step;
  return undefined;
}

/**
 * Upsert a streamed metric row by epoch/step instead of appending. The training WS
 * replays every row from the start on each (re)connect, so a plain append would
 * double-plot points after a reconnect (and the old seed-GET + replay double-loaded
 * on first open). Rows without an epoch/step key can't be deduped, so they append.
 */
export function mergeMetric(prev: MetricRow[], row: MetricRow): MetricRow[] {
  const key = metricKey(row);
  if (key === undefined) return [...prev, row];
  const idx = prev.findIndex((r) => metricKey(r) === key);
  if (idx < 0) return [...prev, row];
  const next = prev.slice();
  next[idx] = row;
  return next;
}

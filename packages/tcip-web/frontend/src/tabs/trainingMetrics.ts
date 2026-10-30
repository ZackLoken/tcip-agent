/** Reading the metric keys and values of a run's epoch rows. */

import { METRICS_ROW_STAMPS, type MetricsRow, NOT_FINITE_SUFFIX } from "@/api/types.generated";

/** The poll cadence of a run listing, in milliseconds. */
export const RUN_REFRESH_MS = 4000;

const NON_METRIC_KEYS = new Set<string>(METRICS_ROW_STAMPS);

/** The keys of ``metrics`` holding a number, other than a row's epoch, step and instant stamps
 * and a key's non-finite state companion. */
export function numericMetricKeys(metrics: Record<string, unknown> | null | undefined): string[] {
  if (!metrics) return [];
  return Object.keys(metrics).filter(
    (k) =>
      !NON_METRIC_KEYS.has(k) && !k.endsWith(NOT_FINITE_SUFFIX) && typeof metrics[k] === "number",
  );
}

/** Every key `numericMetricKeys` admits in any of `records`, in the order first met. */
export function unionMetricKeys(
  records: Iterable<Record<string, unknown> | null | undefined>,
): string[] {
  const keys = new Set<string>();
  for (const record of records) numericMetricKeys(record).forEach((k) => keys.add(k));
  return Array.from(keys);
}

/** ``prev`` with the streamed epoch row ``row`` in place of the row of its epoch, or appended
 * when ``prev`` holds none; each frame the stream sends is its epoch's whole row. */
export function mergeMetric(prev: MetricsRow[], row: MetricsRow): MetricsRow[] {
  const idx = prev.findIndex((r) => r.epoch === row.epoch);
  if (idx < 0) return [...prev, row];
  const next = prev.slice();
  next[idx] = row;
  return next;
}

/** The streamed per-batch row ``batch`` while its epoch is still in progress, that is while
 * ``epochs`` holds no row of its epoch; null otherwise. */
export function inProgressBatch(epochs: MetricsRow[], batch: MetricsRow | null): MetricsRow | null {
  if (!batch) return null;
  return epochs.some((r) => r.epoch === batch.epoch) ? null : batch;
}

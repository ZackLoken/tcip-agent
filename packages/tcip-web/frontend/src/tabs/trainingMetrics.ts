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

/** Bookkeeping fields a metrics record carries that are never themselves a metric. */
const NON_METRIC_KEYS = new Set(["epoch", "step", "timestamp"]);

/** The suffix a stamped metric's own non-finite state companion carries (e.g. `train_loss_state`
 * beside `train_loss`); a companion key, never itself offered as a metric. */
export const METRIC_STATE_SUFFIX = "_state";

/** The prefix every validation metric is stamped with (``VAL_METRIC_PREFIX`` in
 * evaluation.py): ``evaluation.HIGHER_IS_BETTER_BY_METRIC``'s own keys are bare, and a run's
 * own resolved selection-metric name is bare too, so both need this prepended to match a
 * stamped metric key. */
export const VAL_METRIC_PREFIX = "val_";

/**
 * Keys eligible to plot or rank as a metric: present with a numeric value, not one of a row's
 * own bookkeeping fields, and not another key's `_state` companion. The one filter the
 * logged-metrics table and the rank chooser both apply, so a string-valued key (a selection
 * label, say) is never offered as something to rank by.
 */
export function numericMetricKeys(metrics: Record<string, unknown> | null | undefined): string[] {
  if (!metrics) return [];
  return Object.keys(metrics).filter(
    (k) =>
      !NON_METRIC_KEYS.has(k) && !k.endsWith(METRIC_STATE_SUFFIX) && typeof metrics[k] === "number",
  );
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

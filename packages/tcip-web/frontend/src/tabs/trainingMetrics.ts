/** Metric-stream helpers for the Training tab (kept out of the .tsx so they're unit-testable). */

import type { MetricRow } from "@/api/training";

/** The run list's own poll cadence, shared by TrainingTab's row refresh and RunComparison's
 * comparison refresh so the two surfaces poll on one cadence rather than two literals. */
export const RUN_REFRESH_MS = 4000;

/**
 * The sentence both the Training and Tuning list headers state their own ordering with, so the
 * two tabs never drift into naming a different scope (a browser window, say) for the same
 * process-level fact: the runs this running process itself launched list first, in launch
 * order, then every other recorded item follows, sorted by the given field. A row's own launcher
 * sentence names who started it, not which group it sorts into: a restart moves every run this
 * process held, whoever launched it, into the second group, so two rows reading the same
 * launcher can sit in different groups. `itemNoun` is the singular ("run", "sweep").
 */
export function runOrderLine(itemNoun: string, sortField: string): string {
  const plural = `${itemNoun}s`;
  const capitalized = plural.charAt(0).toUpperCase() + plural.slice(1);
  return `${capitalized} this running process itself launched come first, in launch order; every other recorded ${itemNoun} follows, sorted by ${sortField}.`;
}

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

/** The chart's default series: which of `metricKeys` it plots, each key's legend/accessible-name
 * label, and whether the all-numeric-keys fallback applied. */
export interface ChartSeries {
  keys: string[];
  labels: Record<string, string>;
  /** True only when the log carries neither a key ending in "loss" nor "selection": every
   * numeric key is plotted, a documented rule rather than a fallback hiding a choice. */
  allKeys: boolean;
}

function latestSelectionMetric(rows: MetricRow[]): string | null {
  for (let i = rows.length - 1; i >= 0; i--) {
    const value = rows[i]?.selection_metric;
    if (typeof value === "string" && value) return value;
  }
  return null;
}

/**
 * The chart's default series: every key of `metricKeys` ending in "loss", in that order, plus
 * `selection` when the log carries it. When the run's selection metric is itself a plotted loss
 * (its own name ends in "loss"), the two are merged into one series named for the loss key it
 * duplicates, resolved as `val_<metric>` when that key is plotted, else the bare metric name
 * when that key is plotted, else `train_loss`, so the chart never draws the same line twice. The
 * limit is stated, never widened: a bespoke non-loss key sits in the table, and the all-keys
 * fallback below applies only when the log carries neither a loss-suffixed key nor `selection`.
 * The merge fires only when a row names the selection metric, which the stock trainer always
 * does alongside the value; a bespoke loop that logs `selection` without `selection_metric`
 * plots it as its own duplicate line.
 */
export function defaultChartSeries(metricKeys: string[], rows: MetricRow[]): ChartSeries {
  const lossKeys = metricKeys.filter((k) => k.endsWith("loss"));
  const hasSelection = metricKeys.includes("selection");
  if (lossKeys.length === 0 && !hasSelection) {
    return { keys: metricKeys, labels: {}, allKeys: true };
  }

  const selectionMetric = hasSelection ? latestSelectionMetric(rows) : null;
  let duplicateLossKey: string | null = null;
  if (selectionMetric && selectionMetric.endsWith("loss")) {
    const valKey = VAL_METRIC_PREFIX + selectionMetric;
    if (lossKeys.includes(valKey)) duplicateLossKey = valKey;
    else if (lossKeys.includes(selectionMetric)) duplicateLossKey = selectionMetric;
    else if (lossKeys.includes("train_loss")) duplicateLossKey = "train_loss";
  }

  const keys = hasSelection && !duplicateLossKey ? [...lossKeys, "selection"] : lossKeys;
  const labels: Record<string, string> = {};
  for (const key of keys) {
    if (key === duplicateLossKey) labels[key] = `${key} (selection)`;
    else if (key === "selection") {
      labels[key] = selectionMetric ? `selection (${selectionMetric})` : "selection";
    }
  }
  return { keys, labels, allKeys: false };
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

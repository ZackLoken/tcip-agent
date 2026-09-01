/** Overlay-chart helper for the Training tab's run comparison (kept out of the .tsx so it's
 * unit-testable). Joins several runs' own metric streams into one set of points for a single
 * metric, keyed by the tab's one row identity (metricKey: epoch, else step). */

import type { MetricRow } from "@/api/training";
import { metricKey } from "@/tabs/trainingMetrics";

export interface RunSeries {
  runId: string;
  rows: MetricRow[];
}

export interface JoinedPoint {
  x: number;
  [runId: string]: number | undefined;
}

/**
 * One row per distinct epoch/step present in any run's own rows, sorted ascending, each carrying
 * every run's own finite value for `metric` under that run's id (absent, not zero, for a run
 * with no row at that key or a non-finite value there), so a chart line for one run ends where
 * its own finite values end rather than dropping to zero or connecting across the gap. A row
 * with neither epoch nor step (metricKey undefined) contributes nothing here; the caller counts
 * how many were dropped per run itself, from the same rows this was given.
 */
export function joinRunSeries(series: RunSeries[], metric: string): JoinedPoint[] {
  const byX = new Map<number, JoinedPoint>();
  for (const { runId, rows } of series) {
    for (const row of rows) {
      const x = metricKey(row);
      if (x === undefined) continue;
      const value = row[metric];
      if (typeof value !== "number" || !Number.isFinite(value)) continue;
      let point = byX.get(x);
      if (!point) {
        point = { x };
        byX.set(x, point);
      }
      point[runId] = value;
    }
  }
  return Array.from(byX.values()).sort((a, b) => a.x - b.x);
}

/** Every metric key present in any of the given rows, excluding the epoch/step identity fields
 * themselves, sorted, for a metric chooser to offer over the marked runs' own combined history. */
export function metricKeysAcross(series: RunSeries[]): string[] {
  const keys = new Set<string>();
  for (const { rows } of series) {
    for (const row of rows) {
      for (const [k, v] of Object.entries(row)) {
        if (typeof v === "number" && k !== "epoch" && k !== "step") keys.add(k);
      }
    }
  }
  return Array.from(keys).sort();
}

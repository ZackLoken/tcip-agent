import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { CompareExperiment, CompareResult, CompareSplit, MetricRow } from "@/api/training";
import { openTrainingStream, trainingApi } from "@/api/training";
import { joinRunSeries, metricKeysAcross, type RunSeries } from "@/lib/joinRunSeries";
import { CHART, CHART_LINE_COLORS } from "@/tabs/chartTheme";

/** One marked run: the id the tab tracks it by, and the experiment id compare_experiments
 * and select_best_model take. Only a run with a resolved experiment_id is ever marked. */
export interface MarkedRun {
  runId: string;
  experimentId: string;
}

/** The columns the detail region fits at 1440px, at the tokens' minimum column width: the
 * layout-derived default the marked set is capped at, stated here so the cap and the layout
 * that motivates it live beside each other. */
export const MAX_MARKED_RUNS = 4;

const REFRESH_MS = 4000;
const NOT_FINITE_SUFFIX = "_state";
const UNRECORDED = "unrecorded";

function cellText(value: unknown): string {
  if (value === null || value === undefined) return UNRECORDED;
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(4);
  return String(value);
}

function splitLine(split: CompareSplit | undefined): string {
  if (!split) return UNRECORDED;
  if (split.case === "bound") {
    return `bound to ${split.manifest_dir}${split.seed != null ? ` (seed ${split.seed})` : ""}`;
  }
  if (split.case === "drawn") return `drawn again (seed ${split.seed ?? UNRECORDED})`;
  if (split.case === "error") return `unreadable: ${split.error}`;
  return "no split record";
}

function fingerprintNote(experiments: CompareExperiment[]): string {
  const unreadable = experiments.filter((e) => e.error).map((e) => e.experiment_id);
  if (unreadable.length > 0) {
    return `not comparable: ${unreadable.join(", ")} could not be read`;
  }
  const noRecord = experiments.filter((e) => !e.dataset_fingerprint).map((e) => e.experiment_id);
  const predatesFormula = experiments
    .filter((e) => e.fingerprint_formula_unrecorded)
    .map((e) => e.experiment_id);
  const parts: string[] = [];
  if (noRecord.length > 0) parts.push(`${noRecord.join(", ")} carry no fingerprint`);
  if (predatesFormula.length > 0)
    parts.push(`${predatesFormula.join(", ")} predate the fingerprint formula`);
  return parts.length > 0 ? `not comparable: ${parts.join("; ")}` : "not comparable";
}

function registryMetricKeys(experiments: CompareExperiment[]): string[] {
  const keys = new Set<string>();
  for (const exp of experiments) {
    for (const entry of exp.registry ?? []) {
      for (const k of Object.keys(entry.metrics ?? {})) {
        if (!k.endsWith(NOT_FINITE_SUFFIX)) keys.add(k);
      }
    }
  }
  return Array.from(keys).sort();
}

function loggedMetricKeys(experiments: CompareExperiment[]): string[] {
  const keys = new Set<string>();
  for (const exp of experiments) {
    for (const k of Object.keys(exp.last_logged_metrics ?? {})) {
      if (k === "epoch" || k === "step" || k === "timestamp" || k.endsWith(NOT_FINITE_SUFFIX))
        continue;
      keys.add(k);
    }
  }
  return Array.from(keys).sort();
}

function loggedMetricCell(row: MetricRow | undefined, key: string): string {
  if (!row) return UNRECORDED;
  const state = row[`${key}${NOT_FINITE_SUFFIX}`];
  if (typeof state === "string") return state;
  return cellText(row[key]);
}

/**
 * Side-by-side detail for two to four marked runs: one column per run, every value labelled by
 * the record it came from (never a placeholder for one the platform never recorded), the last
 * logged row, the runs' own registered checkpoints, an overlay chart over their live streams,
 * and one grouped rank control wrapping the platform's own best-model derivation over exactly
 * this marked set. Nothing here reaches a delivered number, a weight or a deletion, and ranking
 * never creates a registry.
 */
export function RunComparison({
  marked,
  projectRoot,
}: {
  marked: MarkedRun[];
  projectRoot: string | null;
}) {
  const [result, setResult] = useState<CompareResult | null>(null);
  const [compareError, setCompareError] = useState<string | null>(null);
  const [seriesByRun, setSeriesByRun] = useState<Record<string, MetricRow[]>>({});

  const experimentIds = useMemo(() => marked.map((m) => m.experimentId), [marked]);

  useEffect(() => {
    let cancelled = false;
    async function refresh() {
      try {
        const r = await trainingApi.compare(experimentIds);
        if (!cancelled) {
          setResult(r);
          setCompareError(null);
        }
      } catch (e) {
        if (!cancelled) setCompareError(e instanceof Error ? e.message : String(e));
      }
    }
    void refresh();
    const t = setInterval(() => void refresh(), REFRESH_MS);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, [experimentIds]);

  useEffect(() => {
    setSeriesByRun({});
    if (!projectRoot) return;
    const stops = marked.map(({ runId }) =>
      openTrainingStream(projectRoot, runId, (msg) => {
        if (msg.type !== "metric" || !msg.row) return;
        setSeriesByRun((prev) => {
          const merged = [...(prev[runId] ?? []), msg.row as MetricRow];
          return { ...prev, [runId]: merged };
        });
      }),
    );
    return () => stops.forEach((stop) => stop());
    // marked's own identity (run ids) is what a stream subscribes to; re-derive on every change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [marked.map((m) => m.runId).join(","), projectRoot]);

  const byExperimentId = useMemo(() => {
    const map = new Map<string, CompareExperiment>();
    (result?.experiments ?? []).forEach((e) => map.set(e.experiment_id, e));
    return map;
  }, [result]);

  const columns = marked.map((m) => ({ ...m, exp: byExperimentId.get(m.experimentId) }));

  const metricRows = loggedMetricKeys(result?.experiments ?? []);
  const rankMetricOptions = registryMetricKeys(result?.experiments ?? []);

  const runSeries: RunSeries[] = marked.map((m) => ({
    runId: m.runId,
    rows: seriesByRun[m.runId] ?? [],
  }));
  const overlayMetricOptions = metricKeysAcross(runSeries);
  const [overlayMetric, setOverlayMetric] = useState("");
  useEffect(() => {
    if (overlayMetric && overlayMetricOptions.includes(overlayMetric)) return;
    setOverlayMetric(overlayMetricOptions[0] ?? "");
    // Re-pick only when the option set itself changes, not on every streamed row.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [overlayMetricOptions.join(",")]);
  const chartData = overlayMetric ? joinRunSeries(runSeries, overlayMetric) : [];
  const droppedByRun = Object.fromEntries(
    marked.map((m) => [
      m.runId,
      (seriesByRun[m.runId] ?? []).filter((r) => r.epoch === undefined && r.step === undefined)
        .length,
    ]),
  );

  const [rankMetric, setRankMetric] = useState("");
  const [rankDirection, setRankDirection] = useState<boolean | null>(null);
  const [includeUnverified, setIncludeUnverified] = useState(false);
  const [needsDirection, setNeedsDirection] = useState(false);
  const [needsUnverifiedOption, setNeedsUnverifiedOption] = useState(false);
  const [rankResult, setRankResult] = useState<Awaited<
    ReturnType<typeof trainingApi.compareBest>
  > | null>(null);
  const [rankError, setRankError] = useState<string | null>(null);

  function onRankMetricChange(metric: string) {
    setRankMetric(metric);
    setRankDirection(null);
    setIncludeUnverified(false);
    setNeedsDirection(false);
    setNeedsUnverifiedOption(false);
    setRankResult(null);
    setRankError(null);
  }

  async function onRank() {
    setRankError(null);
    try {
      const res = await trainingApi.compareBest({
        experiment_ids: experimentIds,
        metric: rankMetric,
        higher_is_better: needsDirection ? (rankDirection ?? undefined) : undefined,
        include_unverified: includeUnverified,
      });
      setRankResult(res);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      setRankResult(null);
      setRankError(message);
      if (message.includes("no declared ranking direction")) setNeedsDirection(true);
      if (message.includes("unverified")) setNeedsUnverifiedOption(true);
    }
  }

  if (compareError) {
    return (
      <div className="text-[11px] text-tcip-fp">Could not load the comparison: {compareError}</div>
    );
  }
  if (!result) {
    return <div className="text-[11px] text-tcip-muted">Loading comparison...</div>;
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="overflow-x-auto tcip-panel">
        <table className="w-full text-[11px]">
          <thead>
            <tr className="border-b border-tcip-border">
              <th className="tcip-th" />
              {columns.map((c) => (
                <th key={c.runId} className="tcip-th font-mono">
                  {c.runId}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-tcip-border">
              <th className="tcip-th">Experiment</th>
              {columns.map((c) => (
                <td key={c.runId} className="px-2 py-1 font-mono">
                  {c.experimentId}
                </td>
              ))}
            </tr>
            {columns.some((c) => c.exp?.error) && (
              <tr className="border-b border-tcip-border">
                <th className="tcip-th">Read error</th>
                {columns.map((c) => (
                  <td key={c.runId} className="px-2 py-1 text-tcip-fp">
                    {c.exp?.error ?? ""}
                  </td>
                ))}
              </tr>
            )}
            <tr className="border-b border-tcip-border">
              <th className="tcip-th">Builder</th>
              {columns.map((c) => (
                <td key={c.runId} className="px-2 py-1">
                  {c.exp?.model ?? "no builder recorded"}
                </td>
              ))}
            </tr>
            <tr className="border-b border-tcip-border">
              <th className="tcip-th">Task / subject</th>
              {columns.map((c) => (
                <td key={c.runId} className="px-2 py-1">
                  {c.exp?.task ?? UNRECORDED} / {c.exp?.subject ?? UNRECORDED}
                </td>
              ))}
            </tr>
            <tr className="border-b border-tcip-border">
              <th className="tcip-th">State</th>
              {columns.map((c) => (
                <td key={c.runId} className="px-2 py-1">
                  {c.exp?.state ?? UNRECORDED}
                  {c.exp?.status_error ? (
                    <span className="block text-tcip-fp">{c.exp.status_error}</span>
                  ) : null}
                </td>
              ))}
            </tr>
            <tr className="border-b border-tcip-border">
              <th className="tcip-th">Epochs logged</th>
              {columns.map((c) => (
                <td key={c.runId} className="px-2 py-1 tabular-nums">
                  {c.exp?.n_epochs ?? UNRECORDED}
                </td>
              ))}
            </tr>
            <tr className="border-b border-tcip-border">
              <th className="tcip-th">Images</th>
              <td colSpan={columns.length} className="px-2 py-1 text-tcip-muted" />
            </tr>
            <tr className="border-b border-tcip-border">
              <th className="tcip-th" />
              <td colSpan={columns.length} className="px-2 py-1">
                {result.same_dataset_fingerprint === true
                  ? "same source images"
                  : result.same_dataset_fingerprint === false
                    ? "not the same source images"
                    : fingerprintNote(result.experiments)}
              </td>
            </tr>
            <tr className="border-b border-tcip-border">
              <th className="tcip-th">Partition</th>
              {columns.map((c) => (
                <td key={c.runId} className="px-2 py-1">
                  {splitLine(c.exp?.split)}
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>

      <div className="overflow-x-auto tcip-panel">
        <div className="tcip-heading px-2 pt-2">
          Logged metrics (last logged, not a verified result)
        </div>
        <table className="w-full text-[11px]">
          <thead>
            <tr className="border-b border-tcip-border">
              <th className="tcip-th" />
              {columns.map((c) => (
                <th key={c.runId} className="tcip-th font-mono">
                  {c.runId}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {metricRows.length === 0 ? (
              <tr>
                <td colSpan={columns.length + 1} className="px-2 py-1 text-tcip-muted">
                  No run has logged a metric yet.
                </td>
              </tr>
            ) : (
              metricRows.map((key) => (
                <tr key={key} className="border-b border-tcip-border">
                  <th className="tcip-th">{key}</th>
                  {columns.map((c) => (
                    <td key={c.runId} className="px-2 py-1 tabular-nums">
                      {loggedMetricCell(c.exp?.last_logged_metrics, key)}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      <div className="overflow-x-auto tcip-panel">
        <div className="tcip-heading px-2 pt-2">Registered checkpoints</div>
        <table className="w-full text-[11px]">
          <thead>
            <tr className="border-b border-tcip-border">
              <th className="tcip-th" />
              {columns.map((c) => (
                <th key={c.runId} className="tcip-th font-mono">
                  {c.runId}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr>
              <th className="tcip-th">Entries</th>
              {columns.map((c) => (
                <td key={c.runId} className="px-2 py-1 align-top">
                  {c.exp?.registry_error ? (
                    <span className="text-tcip-fp">
                      registry unreadable: {c.exp.registry_error}
                    </span>
                  ) : c.exp?.registry && c.exp.registry.length > 0 ? (
                    <ul className="space-y-1">
                      {c.exp.registry.map((entry) => (
                        <li key={entry.name}>
                          <span className="block font-mono">{entry.name}</span>
                          <span className="block text-tcip-muted">
                            {entry.metrics_source ?? "no metrics"}
                            {entry.registered_at
                              ? ` (${new Date(entry.registered_at).toLocaleString()})`
                              : ""}
                          </span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <span className="text-tcip-muted">none registered</span>
                  )}
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>

      <div className="h-[32vh] min-h-[220px] shrink-0 tcip-panel p-2">
        <div className="flex items-center gap-2 mb-1">
          <span className="tcip-heading">Overlay</span>
          {overlayMetricOptions.length > 0 && (
            <select
              className="tcip-select text-[11px]"
              value={overlayMetric}
              onChange={(e) => setOverlayMetric(e.target.value)}
            >
              {overlayMetricOptions.map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
          )}
        </div>
        {overlayMetric && chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height="88%">
            <LineChart data={chartData}>
              <CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" />
              <XAxis dataKey="x" stroke={CHART.axis} style={{ fontSize: 11 }} />
              <YAxis stroke={CHART.axis} style={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  background: CHART.tooltipBg,
                  border: `1px solid ${CHART.tooltipBorder}`,
                  borderRadius: 4,
                  fontSize: 11,
                }}
              />
              <Legend wrapperStyle={{ fontSize: 11, color: CHART.legendText }} />
              {marked.map((m, i) => (
                <Line
                  key={m.runId}
                  type="monotone"
                  dataKey={m.runId}
                  name={m.runId}
                  stroke={CHART_LINE_COLORS[i % CHART_LINE_COLORS.length]}
                  dot={false}
                  strokeWidth={1.5}
                  isAnimationActive={false}
                  connectNulls={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex items-center justify-center h-full text-tcip-muted text-[12px]">
            Waiting for metrics...
          </div>
        )}
        {Object.entries(droppedByRun).some(([, n]) => n > 0) && (
          <div className="text-[10px] text-tcip-muted mt-1">
            {Object.entries(droppedByRun)
              .filter(([, n]) => n > 0)
              .map(
                ([runId, n]) =>
                  `${runId}: ${n} row(s) with no epoch/step, dropped from the overlay`,
              )
              .join(" / ")}
          </div>
        )}
      </div>

      <div className="tcip-panel p-2">
        <div className="tcip-heading mb-2">Rank</div>
        <div className="flex flex-wrap items-center gap-2">
          <select
            className="tcip-select text-[11px]"
            value={rankMetric}
            onChange={(e) => onRankMetricChange(e.target.value)}
          >
            <option value="">Choose a metric...</option>
            {rankMetricOptions.map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
          {needsDirection && (
            <div className="inline-flex rounded border border-tcip-border overflow-hidden">
              <button
                type="button"
                className={`px-2 py-1 text-[11px] ${rankDirection === true ? "bg-tcip-accent text-white" : ""}`}
                onClick={() => setRankDirection(true)}
              >
                higher is better
              </button>
              <button
                type="button"
                className={`px-2 py-1 text-[11px] ${rankDirection === false ? "bg-tcip-accent text-white" : ""}`}
                onClick={() => setRankDirection(false)}
              >
                lower is better
              </button>
            </div>
          )}
          {needsUnverifiedOption && (
            <label className="inline-flex items-center gap-1 text-[11px]">
              <input
                type="checkbox"
                checked={includeUnverified}
                onChange={(e) => setIncludeUnverified(e.target.checked)}
              />
              include checkpoints the trainer did not stamp
            </label>
          )}
          <button
            type="button"
            className="tcip-btn text-[11px]"
            disabled={!rankMetric || (needsDirection && rankDirection === null)}
            onClick={() => void onRank()}
          >
            Rank
          </button>
        </div>
        {rankError && <div className="mt-2 text-[11px] text-tcip-fp">{rankError}</div>}
        {rankResult && (
          <div className="mt-2 text-[11px]">
            <span className="font-mono">{rankResult.name}</span>
            {rankResult.experiment_id && (
              <span className="text-tcip-muted"> ({rankResult.experiment_id})</span>
            )}
            <span className="block text-tcip-muted">
              {rankResult.metrics_source ?? "no source"} / {rankResult.direction_source} direction
              {rankResult.excluded_unverified.length > 0
                ? ` / excluded: ${rankResult.excluded_unverified.map((e) => e.name).join(", ")}`
                : ""}
            </span>
          </div>
        )}
      </div>

      {result.experiments.some(
        (e) => (e.refused_mutations && e.refused_mutations.length > 0) || e.rows_after_end,
      ) && (
        <div className="tcip-panel p-2 text-[11px] text-tcip-muted">
          {result.experiments.map((e) =>
            (e.refused_mutations && e.refused_mutations.length > 0) || e.rows_after_end ? (
              <div key={e.experiment_id}>
                <span className="font-mono">{e.experiment_id}</span>:{" "}
                {e.refused_mutations && e.refused_mutations.length > 0
                  ? `${e.refused_mutations.length} refused mutation(s)`
                  : ""}
                {e.rows_after_end ? ` / ${e.rows_after_end} row(s) logged after the run ended` : ""}
              </div>
            ) : null,
          )}
        </div>
      )}
    </div>
  );
}

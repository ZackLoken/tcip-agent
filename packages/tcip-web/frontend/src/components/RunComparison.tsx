/** The Training tab's side-by-side run comparison: RunComparison and its cell helpers. */
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

import { StructuredRefusalError } from "@/api/http";
import type { CompareExperiment, CompareResult, CompareSplit, MetricRow } from "@/api/training";
import { openTrainingStream, trainingApi } from "@/api/training";
import { joinRunSeries, metricKeysAcross, type RunSeries } from "@/lib/joinRunSeries";
import { CHART, CHART_LINE_COLORS } from "@/tabs/chartTheme";
import {
  mergeMetric,
  METRIC_STATE_SUFFIX,
  numericMetricKeys,
  RUN_REFRESH_MS,
  VAL_METRIC_PREFIX,
} from "@/tabs/trainingMetrics";

/** One marked run: the id the tab tracks it by, and the experiment id compare_experiments
 * and rank_registered_models take. Only a run with a resolved experiment_id is ever marked. */
export interface MarkedRun {
  runId: string;
  experimentId: string;
}

/** Columns the detail region fits at 1440px, the tokens' minimum column width; the design
 * record's own layout-derived default (docs/audit/remediation/batch9/u3-comparison-design.md). */
export const MAX_MARKED_RUNS = 4;

const UNRECORDED = "unrecorded";
/** A column whose experiment id the returned answer never carried at all (never a still-loading
 * state: the component's own early return covers that case before any column renders). */
const NOT_IN_ANSWER = "not in the comparison answer";
const NO_REGISTERED_CHECKPOINT = "no registered checkpoint";
/** The tool's own wording for a marked set with nothing at all to rank. */
const NO_MARKED_CHECKPOINT = "none of the marked experiments registered a checkpoint";
const NOT_RANKED_NO_CHECKPOINT = `not ranked: ${NO_REGISTERED_CHECKPOINT}`;
/** Shown beside a disabled Rank when a marked column's own registry can't be read, in place of
 * a chooser built from the columns that could: ranking never silently drops the unreadable one. */
const REGISTRY_UNREADABLE_FOR_RANK =
  "registry unreadable for a marked run; see the checkpoints above";
const OPEN_PROJECT_FOR_METRICS = "open the project to stream metrics";
const RANK_REASON_ID = "rank-disabled-reason";

function cellText(value: unknown): string {
  if (value === null || value === undefined) return UNRECORDED;
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
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

// The rank chooser and the logged-metrics table share this one filter (numericMetricKeys) so a
// bookkeeping or string-valued key, a selection label say, is never offered as something to rank.
function registryMetricKeys(experiments: CompareExperiment[]): string[] {
  const keys = new Set<string>();
  for (const exp of experiments) {
    for (const entry of exp.registry ?? []) {
      for (const k of numericMetricKeys(entry.metrics)) keys.add(k);
    }
  }
  return Array.from(keys).sort();
}

function loggedMetricKeys(experiments: CompareExperiment[]): string[] {
  const keys = new Set<string>();
  for (const exp of experiments) {
    for (const k of numericMetricKeys(exp.last_logged_metrics)) keys.add(k);
  }
  return Array.from(keys).sort();
}

function loggedMetricCell(row: MetricRow | undefined, key: string): string {
  if (!row) return UNRECORDED;
  const state = row[`${key}${METRIC_STATE_SUFFIX}`];
  if (typeof state === "string") return state;
  return cellText(row[key]);
}

/** The wording for a marked run whose registered checkpoint(s) exist but never stamped the
 * metric being ranked by, so it never just silently drops out of the answer. */
function notRankedNoMetricStamped(metric: string): string {
  return `not ranked: no ${metric} stamped`;
}

/** Side-by-side detail for two to four marked runs: one column per run labelled by the record it
 * came from, an overlay chart, and one rank control over the platform's best-model derivation. */
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

  const [rankMetric, setRankMetric] = useState("");
  const [rankDirection, setRankDirection] = useState<boolean | null>(null);
  const [includeUnverified, setIncludeUnverified] = useState(false);
  const [needsUnverifiedOption, setNeedsUnverifiedOption] = useState(false);
  const [rankResult, setRankResult] = useState<Awaited<
    ReturnType<typeof trainingApi.compareBest>
  > | null>(null);
  // The metric rankResult actually ranked by, kept separate from rankMetric so a chooser change
  // after a successful rank never relabels the answer already on screen.
  const [rankedMetric, setRankedMetric] = useState("");
  const [rankError, setRankError] = useState<string | null>(null);
  const [higherIsBetterByMetric, setHigherIsBetterByMetric] = useState<Record<string, boolean>>({});

  useEffect(() => {
    let cancelled = false;
    trainingApi
      .metricDirections()
      .then((r) => {
        if (!cancelled) setHigherIsBetterByMetric(r.higher_is_better);
      })
      .catch(() => {
        if (!cancelled) setHigherIsBetterByMetric({});
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const markedKey = marked.map((m) => m.experimentId).join(",");

  useEffect(() => {
    let cancelled = false;
    const ids = marked.map((m) => m.experimentId);
    // The marked set just changed identity: the previous answer, and any rank computed from it,
    // must not linger as if they already described the new set.
    setResult(null);
    setCompareError(null);
    setRankResult(null);
    setRankedMetric("");
    setRankError(null);
    setNeedsUnverifiedOption(false);
    setRankMetric("");
    async function refresh() {
      try {
        const r = await trainingApi.compare(ids);
        if (!cancelled) {
          setResult(r);
          setCompareError(null);
        }
      } catch (e) {
        if (!cancelled) setCompareError(e instanceof Error ? e.message : String(e));
      }
    }
    void refresh();
    const t = setInterval(() => void refresh(), RUN_REFRESH_MS);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [markedKey]);

  useEffect(() => {
    setSeriesByRun({});
    if (!projectRoot) return;
    const stops = marked.map(({ runId }) =>
      openTrainingStream(projectRoot, runId, (msg) => {
        if (msg.type !== "metric" || !msg.row) return;
        setSeriesByRun((prev) => ({
          ...prev,
          [runId]: mergeMetric(prev[runId] ?? [], msg.row as MetricRow),
        }));
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

  const columns = marked.map((m) => ({
    ...m,
    exp: byExperimentId.get(m.experimentId),
    hasEntry: byExperimentId.has(m.experimentId),
  }));

  const metricRows = loggedMetricKeys(result?.experiments ?? []);
  const rankMetricOptions = registryMetricKeys(result?.experiments ?? []);

  const registryEntriesCount = columns.reduce((sum, c) => sum + (c.exp?.registry?.length ?? 0), 0);
  const allColumnsLoaded = columns.length > 0 && columns.every((c) => c.hasEntry);
  const hasRegistryError = columns.some((c) => c.hasEntry && c.exp?.registry_error);
  const noMarkedCheckpoint = allColumnsLoaded && !hasRegistryError && registryEntriesCount === 0;
  const columnsWithNoCheckpoint = columns.filter(
    (c) => c.hasEntry && !c.exp?.registry_error && (c.exp?.registry?.length ?? 0) === 0,
  );
  // A column that registered a checkpoint but never stamped the ranked metric never just drops
  // out of the answer silently; it is named the same way a column with no checkpoint at all is.
  const columnsMissingRankedMetric = columns.filter(
    (c) =>
      c.hasEntry &&
      !c.exp?.registry_error &&
      (c.exp?.registry?.length ?? 0) > 0 &&
      !(c.exp?.registry ?? []).some((entry) => typeof entry.metrics?.[rankedMetric] === "number"),
  );

  function hasDeclaredDirection(metric: string): boolean {
    const bare = metric.startsWith(VAL_METRIC_PREFIX)
      ? metric.slice(VAL_METRIC_PREFIX.length)
      : metric;
    return higherIsBetterByMetric[bare] !== undefined;
  }
  const declaredMetricOptions = rankMetricOptions.filter(hasDeclaredDirection);
  const undeclaredMetricOptions = rankMetricOptions.filter((k) => !hasDeclaredDirection(k));
  // Derived straight from the declared-direction table already fetched on mount, so the direction
  // choice appears as soon as the breeder picks an undeclared metric, never only after a refusal.
  const needsDirection = rankMetric !== "" && !hasDeclaredDirection(rankMetric);
  const rankDisabledReason = noMarkedCheckpoint
    ? NO_MARKED_CHECKPOINT
    : hasRegistryError
      ? REGISTRY_UNREADABLE_FOR_RANK
      : !rankMetric
        ? "choose a metric before ranking"
        : needsDirection && rankDirection === null
          ? "choose a ranking direction before ranking"
          : null;

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
  // droppedByRun counts a keyless row regardless of which metric is currently charted, so this
  // is called even before a metric is picked (an empty metric key then just plots nothing).
  const { points: chartData, droppedByRun } = joinRunSeries(runSeries, overlayMetric);

  function onRankMetricChange(metric: string) {
    setRankMetric(metric);
    setRankDirection(null);
    setIncludeUnverified(false);
    setNeedsUnverifiedOption(false);
    setRankResult(null);
    setRankedMetric("");
    setRankError(null);
  }

  // Choosing a direction answers whatever the pending refusal was asking for, so it clears that
  // refusal rather than leaving it to read as still describing the choice just made.
  function onChooseDirection(higherIsBetter: boolean) {
    setRankDirection(higherIsBetter);
    setRankError(null);
  }

  async function onRank() {
    setRankError(null);
    try {
      const res = await trainingApi.compareBest({
        experiment_ids: marked.map((m) => m.experimentId),
        metric: rankMetric,
        higher_is_better: needsDirection ? (rankDirection ?? undefined) : undefined,
        include_unverified: includeUnverified,
      });
      setRankResult(res);
      setRankedMetric(rankMetric);
    } catch (e) {
      setRankResult(null);
      setRankedMetric("");
      // Branches on the tool's own refusal fields, never on matching the error text: the route
      // now carries rank_registered_models's whole error dict as the refusal's structured detail.
      if (e instanceof StructuredRefusalError) {
        setRankError(typeof e.detail.error === "string" ? e.detail.error : e.message);
        if (e.detail.all_unverified === true) setNeedsUnverifiedOption(true);
      } else {
        setRankError(e instanceof Error ? e.message : String(e));
      }
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
            {columns.some((c) => c.experimentId !== c.runId) && (
              <tr className="border-b border-tcip-border">
                <th className="tcip-th">Experiment</th>
                {columns.map((c) => (
                  <td key={c.runId} className="px-2 py-1 font-mono">
                    {c.experimentId}
                  </td>
                ))}
              </tr>
            )}
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
                  {c.hasEntry ? (c.exp?.model ?? UNRECORDED) : NOT_IN_ANSWER}
                </td>
              ))}
            </tr>
            <tr className="border-b border-tcip-border">
              <th className="tcip-th">Task / subject</th>
              {columns.map((c) => (
                <td key={c.runId} className="px-2 py-1">
                  {c.hasEntry
                    ? `${c.exp?.task ?? UNRECORDED} / ${c.exp?.subject ?? UNRECORDED}`
                    : NOT_IN_ANSWER}
                </td>
              ))}
            </tr>
            <tr className="border-b border-tcip-border">
              <th className="tcip-th">State</th>
              {columns.map((c) => (
                <td key={c.runId} className="px-2 py-1">
                  {c.hasEntry ? (c.exp?.state ?? UNRECORDED) : NOT_IN_ANSWER}
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
                  {c.hasEntry ? (c.exp?.n_epochs ?? UNRECORDED) : NOT_IN_ANSWER}
                </td>
              ))}
            </tr>
            <tr className="border-b border-tcip-border">
              <th className="tcip-th">Images</th>
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
                  {c.hasEntry ? splitLine(c.exp?.split) : NOT_IN_ANSWER}
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
                      {c.hasEntry
                        ? loggedMetricCell(c.exp?.last_logged_metrics, key)
                        : NOT_IN_ANSWER}
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
                  {!c.hasEntry ? (
                    <span className="text-tcip-muted">{NOT_IN_ANSWER}</span>
                  ) : c.exp?.registry_error ? (
                    <span className="text-tcip-fp">{c.exp.registry_error}</span>
                  ) : c.exp?.registry && c.exp.registry.length > 0 ? (
                    <ul className="space-y-1">
                      {c.exp.registry.map((entry) => (
                        <li key={entry.name}>
                          <span className="block font-mono">{entry.name}</span>
                          <span className="block text-tcip-muted">
                            {entry.metrics_source ?? UNRECORDED}
                            {entry.registered_at
                              ? ` (${new Date(entry.registered_at).toLocaleString()})`
                              : ""}
                          </span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <span className="text-tcip-muted">{NO_REGISTERED_CHECKPOINT}</span>
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
              aria-label="Overlay metric"
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
              <XAxis
                dataKey="x"
                stroke={CHART.axis}
                style={{ fontSize: 11 }}
                label={{
                  value: "epoch/step",
                  position: "insideBottom",
                  offset: -5,
                  fill: CHART.axis,
                }}
              />
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
            {projectRoot ? "Waiting for metrics..." : OPEN_PROJECT_FOR_METRICS}
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
          {noMarkedCheckpoint ? (
            <span id={RANK_REASON_ID} className="text-[11px] text-tcip-muted">
              {NO_MARKED_CHECKPOINT}
            </span>
          ) : hasRegistryError ? (
            <span id={RANK_REASON_ID} className="text-[11px] text-tcip-muted">
              {REGISTRY_UNREADABLE_FOR_RANK}
            </span>
          ) : (
            <>
              <select
                aria-label="Rank by metric"
                className="tcip-select text-[11px]"
                value={rankMetric}
                onChange={(e) => onRankMetricChange(e.target.value)}
              >
                <option value="">Choose a metric...</option>
                {declaredMetricOptions.map((k) => (
                  <option key={k} value={k}>
                    {k}
                  </option>
                ))}
                {undeclaredMetricOptions.length > 0 && (
                  <optgroup label="no declared direction">
                    {undeclaredMetricOptions.map((k) => (
                      <option key={k} value={k}>
                        {k}
                      </option>
                    ))}
                  </optgroup>
                )}
              </select>
              {rankDisabledReason && (
                <span id={RANK_REASON_ID} className="sr-only">
                  {rankDisabledReason}
                </span>
              )}
            </>
          )}
          {needsDirection && (
            <div
              role="group"
              aria-label="Ranking direction"
              className="inline-flex rounded border border-tcip-border overflow-hidden"
            >
              <button
                type="button"
                aria-pressed={rankDirection === true}
                className={`px-2 py-1 text-[11px] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-tcip-accent/70 ${
                  rankDirection === true ? "bg-tcip-accent text-white" : "hover:bg-tcip-hover"
                }`}
                onClick={() => onChooseDirection(true)}
              >
                higher is better
              </button>
              <button
                type="button"
                aria-pressed={rankDirection === false}
                className={`px-2 py-1 text-[11px] border-l border-tcip-border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-tcip-accent/70 ${
                  rankDirection === false ? "bg-tcip-accent text-white" : "hover:bg-tcip-hover"
                }`}
                onClick={() => onChooseDirection(false)}
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
            disabled={rankDisabledReason !== null}
            aria-describedby={rankDisabledReason !== null ? RANK_REASON_ID : undefined}
            onClick={() => void onRank()}
          >
            Rank
          </button>
        </div>
        {rankError && (
          <div role="status" className="mt-2 text-[11px] text-tcip-fp">
            {rankError}
          </div>
        )}
        {rankResult && (
          <div role="status" className="mt-2 text-[11px]">
            <span className="font-mono">{rankResult.name}</span>
            {rankResult.experiment_id && rankResult.experiment_id !== rankResult.name && (
              <span className="text-tcip-muted"> ({rankResult.experiment_id})</span>
            )}
            {rankedMetric && (
              <span className="text-tcip-muted">
                {" "}
                / {rankedMetric}: {cellText(rankResult.metrics[rankedMetric])}
              </span>
            )}
            <span className="block text-tcip-muted">
              {[
                rankResult.metrics_source ?? UNRECORDED,
                `${rankResult.direction_source} direction`,
                rankResult.excluded_unverified.length > 0
                  ? `excluded: ${rankResult.excluded_unverified.map((e) => e.name).join(", ")}`
                  : null,
                columnsWithNoCheckpoint.length > 0
                  ? columnsWithNoCheckpoint
                      .map((c) => `${c.experimentId}: ${NOT_RANKED_NO_CHECKPOINT}`)
                      .join(", ")
                  : null,
                columnsMissingRankedMetric.length > 0
                  ? columnsMissingRankedMetric
                      .map((c) => `${c.experimentId}: ${notRankedNoMetricStamped(rankedMetric)}`)
                      .join(", ")
                  : null,
              ]
                .filter(Boolean)
                .join(" / ")}
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

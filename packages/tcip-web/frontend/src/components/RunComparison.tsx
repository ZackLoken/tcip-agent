/** The Training tab's side-by-side run comparison: RunComparison and its cell helpers. */
import { useEffect, useMemo, useState, type ReactNode } from "react";

import { messageOf, StructuredRefusalError } from "@/api/http";
import { trainingApi } from "@/api/training";
import {
  NOT_FINITE_SUFFIX,
  type ComparedRun,
  type Comparison,
  type MetricsRow,
  type Skipped,
  type SplitColumn,
} from "@/api/types.generated";
import { formatNumber } from "@/lib/numberFormat";
import { RUN_REFRESH_MS, unionMetricKeys } from "@/tabs/trainingMetrics";

/** One marked run: the one id it is tracked by, compared, ranked and streamed under. */
export interface MarkedRun {
  experimentId: string;
}

/** Columns the detail region fits at 1440px, the tokens' minimum column width. */
export const MAX_MARKED_RUNS = 4;

const UNRECORDED = "unrecorded";
/** A column whose experiment id the returned answer does not carry. */
const NOT_IN_ANSWER = "not in the comparison answer";
const NO_REGISTERED_CHECKPOINT = "no registered checkpoint";
/** What the Rank control says when no column of the comparison carries a registered checkpoint. */
export const NO_MARKED_CHECKPOINT = "nothing to rank: no marked run has a registered checkpoint";
const RANK_REASON_ID = "rank-disabled-reason";

function cellText(value: unknown): string {
  if (value === null || value === undefined) return UNRECORDED;
  if (typeof value === "number") return formatNumber(value);
  return typeof value === "string" ? value : JSON.stringify(value);
}

/** One id's column of a comparison: its run read, or the reason it could not be. */
type CompareEntry = Comparison["experiments"][number];

/** Whether the comparison read `entry`'s run. */
function isRead(entry: CompareEntry): entry is ComparedRun {
  return "run" in entry;
}

/** The experiment id a comparison entry answers for. */
function entryId(entry: CompareEntry): string {
  return isRead(entry) ? entry.run.experiment_id : entry.experiment_id;
}

/** A marked run's column: absent from the answer, unread with the reason, or read. */
type Column = MarkedRun &
  ({ kind: "absent" } | { kind: "unread"; error: string } | { kind: "read"; exp: ComparedRun });

/** A column's cell: the absence wording, the unrecorded wording for a run the comparison could
 * not read, or `of` its read run. */
function cell(column: Column, of: (exp: ComparedRun) => ReactNode): ReactNode {
  if (column.kind === "absent") return NOT_IN_ANSWER;
  return column.kind === "unread" ? UNRECORDED : of(column.exp);
}

function splitLine(split: SplitColumn): string {
  if (split.case === "bound") {
    return `bound to ${split.selection_dir}${split.seed != null ? ` (seed ${split.seed})` : ""}`;
  }
  if (split.case === "drawn") return `drawn again (seed ${split.seed ?? UNRECORDED})`;
  if (split.case === "spatial") return "a within-image spatial split";
  return "no split record";
}

/** What a comparison says of its runs' source images, naming the runs that left it unknown. */
function fingerprintNote({ dataset_fingerprints: match, experiments }: Comparison): string {
  if (match === "same") return "same source images";
  if (match === "different") return "not the same source images";
  if (match === "unread") {
    return `not comparable: ${experiments
      .filter((e) => !isRead(e))
      .map(entryId)
      .join(", ")} could not be read`;
  }
  const unset = experiments.filter(isRead).filter((e) => e.dataset_fingerprint === null);
  return `not comparable: ${unset.map(entryId).join(", ")} carry no fingerprint`;
}

/** What the ranking says of a checkpoint it left out ranking by `metric`: every fact it carries. */
function skippedText(skipped: Skipped, metric: string): string {
  const facts = [
    skipped.unverified ? "unverified, its metrics are not the trainer's" : null,
    skipped.unstamped ? `no ${metric} stamped` : null,
  ];
  return `${skipped.name}: not ranked: ${facts.filter(Boolean).join("; ")}`;
}

function loggedMetricKeys(experiments: CompareEntry[]): string[] {
  return unionMetricKeys(experiments.filter(isRead).map((exp) => exp.last_logged_metrics)).sort();
}

function loggedMetricCell(row: MetricsRow | null, key: string): string {
  if (!row) return UNRECORDED;
  const state = row[`${key}${NOT_FINITE_SUFFIX}`];
  if (typeof state === "string") return state;
  return cellText(row[key]);
}

/** Side-by-side detail for two to four marked runs: one column per run labeled by the record it
 * came from, and one rank control over the platform's best-model derivation. */
export function RunComparison({ marked }: { marked: MarkedRun[] }) {
  const [result, setResult] = useState<Comparison | null>(null);
  const [compareError, setCompareError] = useState<string | null>(null);

  const [rankMetric, setRankMetric] = useState("");
  const [rankDirection, setRankDirection] = useState<boolean | null>(null);
  const [includeUnverified, setIncludeUnverified] = useState(false);
  const [needsUnverifiedOption, setNeedsUnverifiedOption] = useState(false);
  const [rankResult, setRankResult] = useState<Awaited<
    ReturnType<typeof trainingApi.compareBest>
  > | null>(null);
  const [rankError, setRankError] = useState<string | null>(null);

  const markedKey = marked.map((m) => m.experimentId).join(",");

  useEffect(() => {
    let canceled = false;
    const ids = marked.map((m) => m.experimentId);
    // The marked set just changed identity: the previous answer, and any rank computed from it,
    // must not linger as if they already described the new set.
    setResult(null);
    setCompareError(null);
    setRankResult(null);
    setRankError(null);
    setNeedsUnverifiedOption(false);
    setRankMetric("");
    async function refresh() {
      try {
        const r = await trainingApi.compare({ experiment_ids: ids });
        if (!canceled) {
          setResult(r);
          setCompareError(null);
        }
      } catch (e) {
        if (!canceled) setCompareError(messageOf(e));
      }
    }
    void refresh();
    const t = setInterval(() => void refresh(), RUN_REFRESH_MS);
    return () => {
      canceled = true;
      clearInterval(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [markedKey]);

  const byExperimentId = useMemo(() => {
    const map = new Map<string, CompareEntry>();
    (result?.experiments ?? []).forEach((e) => map.set(entryId(e), e));
    return map;
  }, [result]);

  const columns = marked.map((m): Column => {
    const entry = byExperimentId.get(m.experimentId);
    if (entry === undefined) return { ...m, kind: "absent" };
    return isRead(entry)
      ? { ...m, kind: "read", exp: entry }
      : { ...m, kind: "unread", error: entry.error };
  });
  const registries = columns.map((c) => (c.kind === "read" ? c.exp.registry : []));

  const metricRows = loggedMetricKeys(result?.experiments ?? []);
  const rankMetricOptions = Object.keys(result?.higher_is_better ?? {});

  const allColumnsLoaded = columns.length > 0 && columns.every((c) => c.kind !== "absent");
  const noMarkedCheckpoint = allColumnsLoaded && registries.every((r) => r.length === 0);

  function hasDeclaredDirection(metric: string): boolean {
    return (result?.higher_is_better[metric] ?? null) !== null;
  }
  const declaredMetricOptions = rankMetricOptions.filter(hasDeclaredDirection);
  const undeclaredMetricOptions = rankMetricOptions.filter((k) => !hasDeclaredDirection(k));
  // Read off the comparison's own directions, as soon as the breeder picks a metric.
  const needsDirection = rankMetric !== "" && !hasDeclaredDirection(rankMetric);
  const rankDisabledReason = noMarkedCheckpoint
    ? NO_MARKED_CHECKPOINT
    : !rankMetric
      ? "choose a metric before ranking"
      : needsDirection && rankDirection === null
        ? "choose a ranking direction before ranking"
        : null;

  function onRankMetricChange(metric: string) {
    setRankMetric(metric);
    setRankDirection(null);
    setIncludeUnverified(false);
    setNeedsUnverifiedOption(false);
    setRankResult(null);
    setRankError(null);
  }

  // Choosing a direction answers the pending refusal, so it clears it.
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
        higher_is_better: needsDirection ? rankDirection : null,
        include_unverified: includeUnverified,
      });
      setRankResult(res);
    } catch (e) {
      setRankResult(null);
      // Branches on the ranking's own refusal fields.
      setRankError(messageOf(e));
      if (e instanceof StructuredRefusalError && e.detail.all_unverified === true) {
        setNeedsUnverifiedOption(true);
      }
    }
  }

  if (compareError) {
    return (
      <div className="text-[13px] tcip-problem">Could not load the comparison: {compareError}</div>
    );
  }
  if (!result) {
    return <div className="text-[13px] text-tcip-muted">Loading comparison...</div>;
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="overflow-x-auto tcip-panel">
        <table className="w-full text-[13px]">
          <thead>
            <tr className="border-b border-tcip-border">
              <th className="tcip-th" />
              {columns.map((c) => (
                <th key={c.experimentId} className="tcip-th font-mono">
                  {c.experimentId}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {columns.some((c) => c.kind === "unread") && (
              <tr className="border-b border-tcip-border">
                <th className="tcip-th">Read error</th>
                {columns.map((c) => (
                  <td key={c.experimentId} className="px-2 py-1 tcip-problem">
                    {c.kind === "unread" && c.error}
                  </td>
                ))}
              </tr>
            )}
            <tr className="border-b border-tcip-border">
              <th className="tcip-th">Builder</th>
              {columns.map((c) => (
                <td key={c.experimentId} className="px-2 py-1">
                  {cell(c, (exp) => exp.run.builder)}
                </td>
              ))}
            </tr>
            <tr className="border-b border-tcip-border">
              <th className="tcip-th">Task / subject</th>
              {columns.map((c) => (
                <td key={c.experimentId} className="px-2 py-1">
                  {cell(c, (exp) => `${exp.run.task} / ${exp.subject ?? UNRECORDED}`)}
                </td>
              ))}
            </tr>
            <tr className="border-b border-tcip-border">
              <th className="tcip-th">State</th>
              {columns.map((c) => (
                <td key={c.experimentId} className="px-2 py-1">
                  {cell(c, (exp) => (
                    <>
                      {exp.run.state}
                      {exp.run.status_error !== null && (
                        <span className="block tcip-problem">{exp.run.status_error}</span>
                      )}
                    </>
                  ))}
                </td>
              ))}
            </tr>
            <tr className="border-b border-tcip-border">
              <th className="tcip-th">Epochs logged</th>
              {columns.map((c) => (
                <td key={c.experimentId} className="px-2 py-1 tabular-nums">
                  {cell(c, (exp) => exp.n_epochs)}
                </td>
              ))}
            </tr>
            <tr className="border-b border-tcip-border">
              <th className="tcip-th">Images</th>
              <td colSpan={columns.length} className="px-2 py-1">
                {fingerprintNote(result)}
              </td>
            </tr>
            <tr className="border-b border-tcip-border">
              <th className="tcip-th">Partition</th>
              {columns.map((c) => (
                <td key={c.experimentId} className="px-2 py-1">
                  {cell(c, (exp) => splitLine(exp.split))}
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>

      <div className="overflow-x-auto tcip-panel">
        <div className="tcip-heading px-2 pt-2">Last logged metrics</div>
        <table className="w-full text-[13px]">
          <thead>
            <tr className="border-b border-tcip-border">
              <th className="tcip-th" />
              {columns.map((c) => (
                <th key={c.experimentId} className="tcip-th font-mono">
                  {c.experimentId}
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
                    <td key={c.experimentId} className="px-2 py-1 tabular-nums">
                      {cell(c, (exp) => loggedMetricCell(exp.last_logged_metrics, key))}
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
        <table className="w-full text-[13px]">
          <thead>
            <tr className="border-b border-tcip-border">
              <th className="tcip-th" />
              {columns.map((c) => (
                <th key={c.experimentId} className="tcip-th font-mono">
                  {c.experimentId}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr>
              <th className="tcip-th">Entries</th>
              {columns.map((c, i) => (
                <td key={c.experimentId} className="px-2 py-1 align-top">
                  {c.kind === "absent" ? (
                    <span className="text-tcip-muted">{NOT_IN_ANSWER}</span>
                  ) : registries[i].length > 0 ? (
                    <ul className="space-y-1">
                      {registries[i].map(({ entry, facts }) => (
                        <li key={entry.name}>
                          <span className="block font-mono">{entry.name}</span>
                          <span className="block text-tcip-muted">
                            {facts.metrics_source ?? UNRECORDED}
                            {` (${new Date(entry.registered_at).toLocaleString()})`}
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

      <div className="tcip-panel p-2">
        <div className="tcip-heading mb-2">Rank</div>
        <div className="flex flex-wrap items-center gap-2">
          {noMarkedCheckpoint ? (
            <span id={RANK_REASON_ID} className="text-[13px] text-tcip-muted">
              {NO_MARKED_CHECKPOINT}
            </span>
          ) : (
            <>
              <select
                aria-label="Rank by metric"
                className="tcip-select text-[13px]"
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
                className={`px-2 py-1 text-[13px] transition-colors focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-tcip-accent/70 ${
                  rankDirection === true ? "bg-tcip-accent text-white" : "hover:bg-tcip-hover"
                }`}
                onClick={() => onChooseDirection(true)}
              >
                higher is better
              </button>
              <button
                type="button"
                aria-pressed={rankDirection === false}
                className={`px-2 py-1 text-[13px] border-l border-tcip-border transition-colors focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-tcip-accent/70 ${
                  rankDirection === false ? "bg-tcip-accent text-white" : "hover:bg-tcip-hover"
                }`}
                onClick={() => onChooseDirection(false)}
              >
                lower is better
              </button>
            </div>
          )}
          {needsUnverifiedOption && (
            <label className="inline-flex items-center gap-1 text-[13px]">
              <input
                type="checkbox"
                checked={includeUnverified}
                onChange={(e) => setIncludeUnverified(e.target.checked)}
              />
              include unverified checkpoints
            </label>
          )}
          <button
            type="button"
            className="tcip-btn text-[13px]"
            disabled={rankDisabledReason !== null}
            aria-describedby={rankDisabledReason !== null ? RANK_REASON_ID : undefined}
            onClick={() => void onRank()}
          >
            Rank
          </button>
        </div>
        {rankError && (
          <div role="status" className="mt-2 text-[13px] tcip-problem">
            {rankError}
          </div>
        )}
        {rankResult && (
          <div role="status" className="mt-2 text-[13px]">
            <span className="font-mono">{rankResult.entry.name}</span>
            {rankResult.entry.experiment_id !== null &&
              rankResult.entry.experiment_id !== rankResult.entry.name && (
                <span className="text-tcip-muted"> ({rankResult.entry.experiment_id})</span>
              )}
            <span className="text-tcip-muted">
              {" "}
              / {rankResult.metric}: {cellText(rankResult.facts.metrics[rankResult.metric])}
            </span>
            <span className="block text-tcip-muted">
              {[
                rankResult.facts.metrics_source ?? UNRECORDED,
                `${rankResult.direction_source} direction`,
                ...rankResult.skipped.map((s) => skippedText(s, rankResult.metric)),
              ].join(" / ")}
            </span>
          </div>
        )}
      </div>

      {result.experiments.filter(isRead).some((e) => e.rows_after_end) && (
        <div className="tcip-panel p-2 text-[13px] text-tcip-muted">
          {result.experiments.filter(isRead).map((e) =>
            e.rows_after_end ? (
              <div key={e.run.experiment_id}>
                <span className="font-mono">{e.run.experiment_id}</span>:{" "}
                {`${e.rows_after_end} row(s) logged after the run ended`}
              </div>
            ) : null,
          )}
        </div>
      )}
    </div>
  );
}

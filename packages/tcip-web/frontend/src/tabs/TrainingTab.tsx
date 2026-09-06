import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
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
import { openTrainingStream, trainingApi } from "@/api/training";
import type { LaunchableConfig, MetricRow, SplitChoices, TrainingRunSummary } from "@/api/training";
import { DisclosureChevron } from "@/components/CollapsibleSection";
import { EmbeddedTool } from "@/components/EmbeddedTool";
import { LaunchPicker, type DataPicker, type LaunchPickerRow } from "@/components/LaunchPicker";
import { MAX_MARKED_RUNS, RunComparison, type MarkedRun } from "@/components/RunComparison";
import { TabHeading } from "@/components/TabHeading";
import { useDisclosure } from "@/hooks/useDisclosure";
import { useEditableAgentRequest } from "@/hooks/useEditableAgentRequest";
import { useEmbeddedToolRetry, type EmbeddedToolStepResult } from "@/hooks/useEmbeddedToolRetry";
import { UNSET_GLYPH } from "@/lib/glyphs";
import { TERMINAL_STATUSES } from "@/lib/runStatus";
import { useStore } from "@/store";
import { defaultTrainingRequest } from "@/tabs/agentPrompts";
import { CHART, CHART_LINE_COLORS } from "@/tabs/chartTheme";
import { RunMonitorEmpty, RunMonitorLayout } from "@/tabs/RunMonitorLayout";
import {
  defaultChartSeries,
  mergeMetric,
  numericMetricKeys,
  runOrderLine,
  RUN_REFRESH_MS,
} from "@/tabs/trainingMetrics";

// Runs can only be stopped while still active; terminal/historical runs show no button.
const TRAINING_CANCELLABLE: ReadonlySet<string> = new Set(["created", "running"]);

/** Why a row's Compare toggle is disabled, when it is: the two reasons a run's own
 * experiment_id/experiment_error name, the one implementation both the row and the toggle's
 * own click handler consult. Markable (neither reason) returns null. */
function unmarkableReason(run: TrainingRunSummary): string | null {
  if (run.experiment_error) return `experiment tracking failed: ${run.experiment_error}`;
  if (!run.experiment_id) return "experiment not resolved yet";
  return null;
}

function messageOf(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** What a run's launched_by field says about who started it, from the record alone: never a
 * guess, and never derived from process locality (external is a separate, unrendered fact). */
function launcherSentence(launchedBy: TrainingRunSummary["launched_by"]): string {
  const launcher = typeof launchedBy?.launcher === "string" ? launchedBy.launcher : null;
  if (launcher === null) return "launcher not recorded";
  if (launcher === "gui") return "started through this app";
  if (launcher === "agent") return "started by the agent";
  if (launcher === "process") return "started by another process";
  return `started by ${launcher}`;
}

/** The longer sentence behind a row's launcher mark, reachable by assistive technology through
 * aria-describedby: what the mark means, with the declared client named for an agent launch. */
function launcherDescription(launchedBy: TrainingRunSummary["launched_by"]): string {
  const launcher = typeof launchedBy?.launcher === "string" ? launchedBy.launcher : null;
  if (launcher === null) {
    return "This run's record carries no launcher: its tracking never reached the stamp, or the stamp did not land.";
  }
  if (launcher === "gui") {
    return "This run was launched through this app's own route.";
  }
  if (launcher === "agent") {
    const name =
      typeof launchedBy?.agent_client_name === "string" ? launchedBy.agent_client_name : null;
    const version =
      typeof launchedBy?.agent_client_version === "string" ? launchedBy.agent_client_version : null;
    const client = name ? (version ? `${name} ${version}` : name) : null;
    return client
      ? `This run was launched by an agent through the MCP door, declared as ${client}.`
      : "This run was launched by an agent through the MCP door.";
  }
  if (launcher === "process") {
    return "This run was launched by a process with no browser and no agent handshake: a script or a test.";
  }
  return `This run's record names its own launcher: ${launcher}.`;
}

/** The row's own select control name: id, its experiment id when the two differ (exactly as the
 * visible row states it), status and the record's own launcher sentence, with the best value
 * and its metric appended exactly as the record carries them when both are present. */
function runRowLabel(run: TrainingRunSummary): string {
  const idPart =
    run.experiment_id && run.experiment_id !== run.run_id
      ? `${run.run_id} · ${run.experiment_id}`
      : run.run_id;
  const base = `${idPart} ${run.status}, ${launcherSentence(run.launched_by)}`;
  if (run.best_metric === undefined || run.best_metric === null || !run.best_metric_name) {
    return base;
  }
  return `${base}, best ${run.best_metric_name} ${run.best_metric}`;
}

const NO_OTHER_PARTITION =
  "this listing found no other recorded partition the config can bind to; the agent can draw one.";

export function dataPickerFor(choices: SplitChoices | undefined): DataPicker | undefined {
  if (!choices) return undefined;
  return {
    asRecordedLine: choices.as_recorded.line,
    asRecordedDisabled: !choices.as_recorded.compatible,
    asRecordedReason: choices.as_recorded.reason ?? undefined,
    absenceMessage: NO_OTHER_PARTITION,
    choices: choices.manifests.map((m) => ({
      manifestDir: m.manifest_dir,
      disabled: !m.enabled,
      reason: m.reason ?? undefined,
      replacedSplitKeys: m.replaced_split_keys,
      label: (
        <>
          <span className="block font-mono">{m.manifest_dir}</span>
          <span className="block text-tcip-muted">
            seed {m.seed ?? "unrecorded"} · {m.group_by ?? "unrecorded grouping"} · train {m.train}{" "}
            · val {m.val} · calibration {m.calibration}
            {m.other_dates > 0 ? ` · ${m.other_dates} member(s) under other dates` : ""}
          </span>
        </>
      ),
    })),
  };
}

function configRow(
  cfg: LaunchableConfig,
  choices: SplitChoices | undefined,
  dataLoading: boolean,
  dataError: string | undefined,
  onStart: (splitManifestDir: string | null) => Promise<void>,
): LaunchPickerRow {
  const pristine = cfg.state === "created";
  return {
    key: cfg.experiment_id,
    content: (
      <>
        <span className="block font-mono text-[11px]">{cfg.experiment_id}</span>
        <span className="block text-[10px] text-tcip-muted">
          {cfg.builder ?? "unknown builder"}
          {cfg.task ? ` · ${cfg.task}` : ""}
          {cfg.subject ? ` · ${cfg.subject}` : ""}
        </span>
        <span className="block text-[10px] text-tcip-muted">
          {cfg.created ? new Date(cfg.created).toLocaleString() : "no creation date recorded"}
          {" · "}
          {cfg.state}
          {cfg.parent_experiment ? ` · parent ${cfg.parent_experiment}` : ""}
        </span>
      </>
    ),
    branchLine: pristine
      ? "Its first run, on the data paths it names"
      : "A new run of this config on the data paths it names, as they are now, with the recorded seed",
    branchLineForData: pristine
      ? "Its first run, on the partition you chose"
      : "A new run of this config on the partition you chose, with the recorded seed",
    data: dataPickerFor(choices),
    dataLoading,
    dataError,
    onStart,
  };
}

// Training is launched from a config already recorded in this project, or described fresh to
// the agent; this tab tracks the runs those launches produce and their live metrics.
export function TrainingTab() {
  const projectRoot = useStore((s) => s.gui.dataset.project_root);
  const datasetRoot = useStore((s) => s.gui.dataset.dataset_root);
  const subject = useStore((s) => s.gui.dataset.subject);

  const { request, setRequest } = useEditableAgentRequest(
    defaultTrainingRequest(datasetRoot, subject),
  );

  const [pickerOpen, setPickerOpen] = useState(false);
  const [configs, setConfigs] = useState<LaunchableConfig[]>([]);
  const [configsError, setConfigsError] = useState<string | null>(null);
  const [splitChoicesById, setSplitChoicesById] = useState<Record<string, SplitChoices>>({});
  const [splitChoicesLoadingId, setSplitChoicesLoadingId] = useState<string | null>(null);
  const [splitChoiceErrors, setSplitChoiceErrors] = useState<Record<string, string>>({});
  const [runs, setRuns] = useState<TrainingRunSummary[]>([]);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<MetricRow[]>([]);
  // Non-null once the run's own TensorBoard refusal names no_logs; its error is the run's own
  // recorded status.error, null when the run produced no logs but recorded no reason either.
  const [tbNoLogs, setTbNoLogs] = useState<{ error: string | null } | null>(null);
  const [tbAttempt, setTbAttempt] = useState(0);
  const [markedRunIds, setMarkedRunIds] = useState<Set<string>>(new Set());
  // Cancel in flight, by run id: disables that row's own Cancel button with a pending label,
  // and a failure lands in cancelErrors rather than only a toast.
  const [pendingCancel, setPendingCancel] = useState<ReadonlySet<string>>(new Set());
  const [cancelErrors, setCancelErrors] = useState<Record<string, string>>({});
  const streamRef = useRef<(() => void) | null>(null);
  const { open: chartTableOpen, toggle: toggleChartTable } = useDisclosure();
  const chartHeadingId = useId();
  const chartNameId = useId();
  const chartTableId = useId();

  const marked: MarkedRun[] = runs
    .filter((r) => markedRunIds.has(r.run_id) && !unmarkableReason(r))
    .map((r) => ({ runId: r.run_id, experimentId: r.experiment_id as string }));
  const comparing = marked.length >= 2;

  function toggleMarked(run: TrainingRunSummary) {
    setMarkedRunIds((prev) => {
      const next = new Set(prev);
      if (next.has(run.run_id)) {
        next.delete(run.run_id);
        return next;
      }
      if (unmarkableReason(run)) return prev;
      // The same markable count the header prints, not the raw id set: a lingering id for a
      // run that turned unmarkable since must not count toward the cap.
      const markableCount = runs.filter((r) => next.has(r.run_id) && !unmarkableReason(r)).length;
      if (markableCount >= MAX_MARKED_RUNS) {
        useStore
          .getState()
          .pushToast(`Compare fits at most ${MAX_MARKED_RUNS} runs at once; unmark one first.`);
        return prev;
      }
      next.add(run.run_id);
      return next;
    });
  }

  // A run's own marked state, never "No run selected" left over: once the marked set settles
  // at exactly one, that run becomes the one the detail region shows.
  const soleMarkedRunId = marked.length === 1 ? marked[0].runId : null;
  useEffect(() => {
    if (soleMarkedRunId) setSelectedRun(soleMarkedRunId);
  }, [soleMarkedRunId]);

  const refreshRuns = useCallback(async () => {
    try {
      const r = await trainingApi.listRuns();
      const nextRuns = r.runs ?? [];
      setRuns(nextRuns);
      setRunsError(null);
      // A run that leaves the list must also leave the marked set, or the cap (which counts
      // markedRunIds itself) can read full while the header (runs still present) shows fewer.
      const stillPresent = new Set(nextRuns.map((run) => run.run_id));
      setMarkedRunIds((prev) => {
        const pruned = new Set(Array.from(prev).filter((id) => stillPresent.has(id)));
        return pruned.size === prev.size ? prev : pruned;
      });
    } catch (e) {
      setRunsError(`Could not load training runs: ${e instanceof Error ? e.message : String(e)}`);
    }
  }, []);

  const refreshConfigs = useCallback(async () => {
    try {
      const r = await trainingApi.listConfigs();
      setConfigs(r.configs ?? []);
      setConfigsError(null);
    } catch (e) {
      setConfigs([]);
      setConfigsError(`Could not load configs: ${e instanceof Error ? e.message : String(e)}`);
    }
    setSplitChoicesById({});
    setSplitChoiceErrors({});
  }, []);

  // Fetched only for the row the breeder actually opens: list_split_choices re-enumerates
  // every experiment in the project, so fetching it for every row on open would cost O(n^2).
  const loadSplitChoices = useCallback(async (experimentId: string) => {
    setSplitChoicesLoadingId(experimentId);
    try {
      const choices = await trainingApi.listSplitChoices(experimentId);
      setSplitChoicesById((prev) => ({ ...prev, [experimentId]: choices }));
      setSplitChoiceErrors((prev) => {
        const { [experimentId]: _drop, ...rest } = prev;
        return rest;
      });
    } catch (e) {
      setSplitChoiceErrors((prev) => ({
        ...prev,
        [experimentId]: `Could not load its data choices: ${e instanceof Error ? e.message : String(e)}`,
      }));
    } finally {
      setSplitChoicesLoadingId((current) => (current === experimentId ? null : current));
    }
  }, []);

  useEffect(() => {
    void refreshRuns();
    const t = setInterval(refreshRuns, RUN_REFRESH_MS);
    return () => clearInterval(t);
  }, [refreshRuns]);

  useEffect(() => {
    if (pickerOpen) void refreshConfigs();
  }, [pickerOpen, refreshConfigs]);

  async function startFromConfig(experimentId: string, splitManifestDir: string | null) {
    const result = await trainingApi.relaunch(experimentId, splitManifestDir);
    setPickerOpen(false);
    void refreshRuns();
    if (typeof result.run_id === "string") setSelectedRun(result.run_id);
  }

  function sendToAgent() {
    useStore.getState().sendToAgentTerminal(request);
    setPickerOpen(false);
  }

  // The run list's own poll keeps a status per run independent of the metrics stream; a ref
  // (not a dependency) so reading it doesn't reopen the stream on every poll.
  const runsRef = useRef<TrainingRunSummary[]>(runs);
  useEffect(() => {
    runsRef.current = runs;
  }, [runs]);

  useEffect(() => {
    // Comparing owns its own per-run streams; suspend this one so the stream count stays the marked count.
    if (!selectedRun || !projectRoot || comparing) return;
    // Clear the previous run's curve; the stream replays this run from the start, so a
    // seed GET would just double-load the same rows. The WS is the single source now.
    setMetrics([]);
    streamRef.current?.();
    // A run already terminal when this stream opened is a rediscovery, not a transition the
    // breeder is watching; only a run still live at open time toasts on its own terminal frame.
    const knownAtOpen = runsRef.current.find((r) => r.run_id === selectedRun)?.status;
    const alreadyTerminal = TERMINAL_STATUSES.has(knownAtOpen ?? "");
    streamRef.current = openTrainingStream(projectRoot, selectedRun, (msg) => {
      if (msg.type === "metric" && msg.row) {
        setMetrics((prev) => mergeMetric(prev, msg.row as MetricRow));
      } else if (msg.type === "status") {
        // Terminal frame: an unknown run carries error and no status; a known run carries its
        // status report and no error.
        if (msg.error) {
          useStore.getState().pushToast(`Training stream error: ${msg.error}`);
          return;
        }
        const st = msg.status?.status;
        if (typeof st === "string" && !alreadyTerminal)
          useStore.getState().pushToast(`Training ${selectedRun}: ${st}`, "info");
        void refreshRuns();
      }
    });
    return () => streamRef.current?.();
  }, [selectedRun, projectRoot, refreshRuns, comparing]);

  // tbNoLogs is a step side effect below, not part of the hook's own outcome, since its text
  // overrides tbError's; reset it on the same triggers the hook itself resets url/error on.
  useEffect(() => {
    setTbNoLogs(null);
  }, [selectedRun, tbAttempt]);

  // Adopt the TensorBoard already serving this run, or start one, retrying on a timer while
  // the run is live: useEmbeddedToolRetry, the loop the Tuning tab's own panel shares.
  const tbStep = useCallback(async (): Promise<EmbeddedToolStepResult> => {
    if (!selectedRun) return { url: null, error: null, done: true };
    const runId = selectedRun;
    let detail;
    try {
      detail = await trainingApi.getRun(runId);
    } catch (e) {
      return { url: null, error: messageOf(e), done: true };
    }

    let url = detail.tensorboard_url ?? null;
    let failure: string | null = null;
    let noLogs = false;
    if (!url) {
      try {
        const launched = await trainingApi.launchTensorboard(runId);
        url = launched.url ?? null;
        if (launched.error) {
          failure = launched.output ? `${launched.error}: ${launched.output}` : launched.error;
        }
      } catch (e) {
        if (e instanceof StructuredRefusalError && e.detail.no_logs === true) {
          noLogs = true;
        } else {
          failure = messageOf(e);
        }
      }
    }

    if (url) {
      setTbNoLogs(null);
      return { url, error: null, done: true };
    }
    const terminal = TERMINAL_STATUSES.has(detail.status ?? "");
    if (!terminal) return { url: null, error: null, done: false };
    if (noLogs) {
      setTbNoLogs({ error: detail.error ?? null });
      return { url: null, error: null, done: true };
    }
    return { url: null, error: failure ?? "No TensorBoard is serving this run.", done: true };
  }, [selectedRun]);

  const { url: tbUrl, error: tbError } = useEmbeddedToolRetry(
    selectedRun,
    !!selectedRun,
    tbAttempt,
    tbStep,
  );

  async function onCancel(runId: string) {
    setPendingCancel((prev) => new Set(prev).add(runId));
    setCancelErrors((prev) => {
      const { [runId]: _drop, ...rest } = prev;
      return rest;
    });
    try {
      await trainingApi.cancel(runId);
      void refreshRuns();
    } catch (e) {
      const message = `Cancel failed: ${messageOf(e)}`;
      useStore.getState().pushToast(message);
      setCancelErrors((prev) => ({ ...prev, [runId]: message }));
    } finally {
      setPendingCancel((prev) => {
        const next = new Set(prev);
        next.delete(runId);
        return next;
      });
    }
  }

  const chartData = useMemo((): (MetricRow & { step: number })[] => {
    return metrics.map((m, i) => ({ ...m, step: m.epoch ?? m.step ?? i }));
  }, [metrics]);

  // Whether metrics[i] itself carried an ordinal, checked against the raw row since chartData's
  // own step field has by then been overwritten with the derived value (real or index).
  const hasOrdinal = useCallback(
    (i: number) => typeof metrics[i]?.epoch === "number" || typeof metrics[i]?.step === "number",
    [metrics],
  );

  const selectedRunSummary = runs.find((r) => r.run_id === selectedRun);
  const selectedRunTerminal = TERMINAL_STATUSES.has(selectedRunSummary?.status ?? "");

  const noLogsMessage = tbNoLogs
    ? tbNoLogs.error
      ? `This run failed: ${tbNoLogs.error}. It produced no logs.`
      : "This run produced no logs."
    : null;

  const metricKeys = useMemo(() => {
    const keys = new Set<string>();
    metrics.forEach((row) => {
      numericMetricKeys(row).forEach((k) => keys.add(k));
    });
    return Array.from(keys);
  }, [metrics]);

  const chartSeries = useMemo(() => defaultChartSeries(metricKeys, metrics), [metricKeys, metrics]);

  return (
    <>
      <TabHeading tab="training" />
      <RunMonitorLayout
        title="Runs"
        headerRight={
          <button
            type="button"
            aria-expanded={pickerOpen}
            className={pickerOpen ? "tcip-btn text-[11px]" : "tcip-btn-primary text-[11px]"}
            onClick={() => setPickerOpen((open) => !open)}
          >
            Start a run
          </button>
        }
        detailHeader={
          comparing ? (
            <>
              <span className="tcip-heading">Comparing</span>
              <span className="text-[11px] text-tcip-muted">
                {marked.length} of {MAX_MARKED_RUNS} runs
              </span>
            </>
          ) : selectedRun ? (
            <>
              <h2 id={chartHeadingId} className="tcip-heading">
                Live metrics
              </h2>
              <span className="font-mono text-[12px] text-tcip-fg">{selectedRun}</span>
              {chartData.length > 0 && (
                <>
                  <span className="flex-1" />
                  <button
                    type="button"
                    onClick={toggleChartTable}
                    aria-expanded={chartTableOpen}
                    aria-controls={chartTableId}
                    className="flex items-center gap-1 text-[11px] text-tcip-muted hover:text-tcip-fg"
                  >
                    <DisclosureChevron open={chartTableOpen} />
                    as table
                  </button>
                </>
              )}
            </>
          ) : (
            <span className="tcip-heading">Select a run to view metrics</span>
          )
        }
        detail={
          comparing ? (
            <RunComparison marked={marked} projectRoot={projectRoot} />
          ) : (
            <div className="flex flex-col gap-4">
              <div className="h-[38vh] min-h-[220px] shrink-0">
                {selectedRun && chartData.length > 0 ? (
                  <figure
                    role="img"
                    aria-labelledby={`${chartHeadingId} ${chartNameId}`}
                    className="m-0 h-full"
                  >
                    <span id={chartNameId} className="sr-only">
                      {chartSeries.allKeys
                        ? `for ${selectedRun}: all logged metrics: ${chartSeries.keys.join(", ")}`
                        : `for ${selectedRun}: ${chartSeries.keys
                            .map((key) => chartSeries.labels[key] ?? key)
                            .join(", ")}${
                            chartSeries.keys.length < metricKeys.length
                              ? '; every other logged metric is behind the "as table" toggle'
                              : ""
                          }`}
                    </span>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData}>
                        <CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" />
                        <XAxis
                          dataKey="step"
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
                        {chartSeries.keys.map((key, i) => (
                          <Line
                            key={key}
                            type="monotone"
                            dataKey={key}
                            name={chartSeries.labels[key] ?? key}
                            stroke={CHART_LINE_COLORS[i % CHART_LINE_COLORS.length]}
                            dot={false}
                            strokeWidth={1.5}
                            isAnimationActive={false}
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </figure>
                ) : (
                  <div
                    role="status"
                    className="flex items-center justify-center h-full text-tcip-muted text-[12px]"
                  >
                    {!selectedRun
                      ? "No run selected."
                      : selectedRunTerminal
                        ? "This run recorded no metrics."
                        : "Waiting for metrics…"}
                  </div>
                )}
              </div>

              {selectedRun && chartData.length > 0 && (
                <div
                  id={chartTableId}
                  hidden={!chartTableOpen}
                  className="overflow-auto max-h-64 shrink-0"
                >
                  <table className="w-full text-[11px]">
                    <caption className="sr-only">{`${selectedRun} metrics as a table`}</caption>
                    <thead>
                      <tr className="border-b border-tcip-border">
                        <th className="tcip-th">epoch/step</th>
                        {metricKeys.map((key) => (
                          <th key={key} className="tcip-th">
                            {key}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {chartData.map((row, i) => (
                        <tr key={i} className="border-t border-tcip-border first:border-t-0">
                          <td className="py-1 pr-3 tabular-nums">
                            {hasOrdinal(i) ? row.step : UNSET_GLYPH}
                          </td>
                          {metricKeys.map((key) => (
                            <td key={key} className="pr-3 tabular-nums">
                              {typeof row[key] === "number" ? row[key] : UNSET_GLYPH}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {selectedRun && (
                <div className="h-[60vh] min-h-[360px] shrink-0">
                  <EmbeddedTool
                    title="TensorBoard"
                    url={tbUrl}
                    loading={!tbUrl && !tbError && !tbNoLogs}
                    error={noLogsMessage ?? tbError}
                    onRetry={tbNoLogs ? undefined : () => setTbAttempt((n) => n + 1)}
                  />
                </div>
              )}
            </div>
          )
        }
      >
        {pickerOpen && (
          <div className="mb-3 pb-3 border-b border-tcip-border">
            <LaunchPicker
              list={{
                title: "Configs in this project",
                emptyMessage: "No config exists in this project yet.",
                error: configsError ?? undefined,
                onRetry: () => void refreshConfigs(),
                rows: configs.map((cfg) =>
                  configRow(
                    cfg,
                    splitChoicesById[cfg.experiment_id],
                    splitChoicesLoadingId === cfg.experiment_id,
                    splitChoiceErrors[cfg.experiment_id],
                    (dir) => startFromConfig(cfg.experiment_id, dir),
                  ),
                ),
              }}
              composerLabel="Describe a new one to the agent"
              request={request}
              onRequestChange={setRequest}
              onSend={sendToAgent}
              onSelect={(key) => void loadSplitChoices(key)}
            />
          </div>
        )}

        {runsError && (
          <div className="text-[11px] text-tcip-fp mb-2">
            {runsError}{" "}
            <button className="tcip-btn text-[11px] ml-1" onClick={() => void refreshRuns()}>
              Retry
            </button>
          </div>
        )}
        {runs.length === 0 && !runsError && (
          <RunMonitorEmpty>No runs yet. Use "Start a run" above.</RunMonitorEmpty>
        )}
        {runs.length > 0 && (
          <div className="text-[10px] text-tcip-muted mb-1">
            {runOrderLine("run", "experiment id")}
          </div>
        )}
        <ul className="space-y-1">
          {runs.map((r) => {
            const isMarked = markedRunIds.has(r.run_id);
            const reason = unmarkableReason(r);
            const cancelling = pendingCancel.has(r.run_id);
            const cancelError = cancelErrors[r.run_id];
            return (
              <li key={r.run_id}>
                <div
                  className={`flex items-start gap-1 p-2 rounded border transition-colors ${
                    selectedRun === r.run_id && !isMarked
                      ? "border-tcip-accent bg-tcip-accent/10"
                      : "border-tcip-border hover:border-tcip-border-hover hover:bg-tcip-hover"
                  }`}
                >
                  <button
                    type="button"
                    aria-pressed={selectedRun === r.run_id}
                    aria-label={runRowLabel(r)}
                    aria-describedby={`origin-mark-${r.run_id}`}
                    className="flex-1 min-w-0 text-left"
                    onClick={() => setSelectedRun(r.run_id)}
                  >
                    <div className="font-mono text-[11px]">
                      {r.run_id}
                      {r.experiment_id && r.experiment_id !== r.run_id && (
                        <span className="text-tcip-muted break-all"> · {r.experiment_id}</span>
                      )}
                    </div>
                    <div className="text-[10px] text-tcip-muted flex justify-between">
                      <span>
                        {r.status}
                        <span title={launcherDescription(r.launched_by)}>
                          {` · ${launcherSentence(r.launched_by)}`}
                        </span>
                        <span id={`origin-mark-${r.run_id}`} className="sr-only">
                          {launcherDescription(r.launched_by)}
                        </span>
                      </span>
                      {r.best_metric !== undefined &&
                        r.best_metric !== null &&
                        r.best_metric_name && (
                          <span className="tabular-nums">
                            best {r.best_metric_name} {r.best_metric}
                          </span>
                        )}
                    </div>
                  </button>
                  <div className="flex flex-col items-end gap-1 shrink-0">
                    <div
                      role="group"
                      aria-label="Run actions"
                      className="inline-flex rounded border border-tcip-border overflow-hidden"
                    >
                      <button
                        type="button"
                        aria-pressed={isMarked}
                        aria-label={`Compare ${r.run_id}`}
                        aria-describedby={reason ? `compare-reason-${r.run_id}` : undefined}
                        disabled={!isMarked && !!reason}
                        className={`px-2 py-1 text-[10px] transition-colors disabled:opacity-40 disabled:cursor-not-allowed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-tcip-accent/70 ${
                          isMarked ? "bg-tcip-accent text-white" : "hover:bg-tcip-hover"
                        }`}
                        onClick={() => toggleMarked(r)}
                      >
                        Compare
                      </button>
                      {TRAINING_CANCELLABLE.has(r.status) && (
                        <button
                          type="button"
                          aria-label={`Cancel ${r.run_id}`}
                          aria-describedby={cancelError ? `cancel-error-${r.run_id}` : undefined}
                          disabled={cancelling}
                          className="px-2 py-1 text-[10px] border-l border-tcip-border hover:bg-tcip-hover transition-colors disabled:opacity-40 disabled:cursor-not-allowed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-tcip-accent/70"
                          onClick={() => void onCancel(r.run_id)}
                        >
                          {cancelling ? "Cancelling…" : "Cancel"}
                        </button>
                      )}
                    </div>
                    {reason && (
                      <span
                        id={`compare-reason-${r.run_id}`}
                        className="text-[10px] text-tcip-muted text-right max-w-[150px]"
                      >
                        {reason}
                      </span>
                    )}
                    {cancelError && (
                      <span
                        id={`cancel-error-${r.run_id}`}
                        className="text-[10px] text-tcip-fp text-right max-w-[150px]"
                      >
                        {cancelError}
                      </span>
                    )}
                  </div>
                </div>
              </li>
            );
          })}
        </ul>
      </RunMonitorLayout>
    </>
  );
}

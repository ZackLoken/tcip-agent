import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
import { EmbeddedTool } from "@/components/EmbeddedTool";
import { LaunchPicker, type DataPicker, type LaunchPickerRow } from "@/components/LaunchPicker";
import { MAX_MARKED_RUNS, RunComparison, type MarkedRun } from "@/components/RunComparison";
import { TabHeading } from "@/components/TabHeading";
import { useEditableAgentRequest } from "@/hooks/useEditableAgentRequest";
import { TERMINAL_STATUSES } from "@/lib/runStatus";
import { useStore } from "@/store";
import { defaultTrainingRequest } from "@/tabs/agentPrompts";
import { CHART, CHART_LINE_COLORS } from "@/tabs/chartTheme";
import { RunMonitorEmpty, RunMonitorLayout } from "@/tabs/RunMonitorLayout";
import { mergeMetric, RUN_REFRESH_MS } from "@/tabs/trainingMetrics";

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

const TENSORBOARD_RETRY_MS = 3000;

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
  const [tbUrl, setTbUrl] = useState<string | null>(null);
  const [tbError, setTbError] = useState<string | null>(null);
  const [tbNoLogs, setTbNoLogs] = useState(false);
  const [tbAttempt, setTbAttempt] = useState(0);
  const [markedRunIds, setMarkedRunIds] = useState<Set<string>>(new Set());
  const streamRef = useRef<(() => void) | null>(null);

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

  useEffect(() => {
    // Comparing owns its own per-run streams; suspend this one so the stream count stays the marked count.
    if (!selectedRun || !projectRoot || comparing) return;
    // Clear the previous run's curve; the stream replays this run from the start, so a
    // seed GET would just double-load the same rows. The WS is the single source now.
    setMetrics([]);
    streamRef.current?.();
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
        if (typeof st === "string")
          useStore.getState().pushToast(`Training ${selectedRun}: ${st}`, "info");
        void refreshRuns();
      }
    });
    return () => streamRef.current?.();
  }, [selectedRun, projectRoot, refreshRuns, comparing]);

  // Adopt the TensorBoard already serving this run, or start one. A run that has not written its
  // log directory yet has nothing to serve, so a failed start is retried while the run is live.
  useEffect(() => {
    setTbUrl(null);
    setTbError(null);
    setTbNoLogs(false);
    if (!selectedRun) return;
    const runId = selectedRun;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const attach = async () => {
      let detail;
      try {
        detail = await trainingApi.getRun(runId);
      } catch (e) {
        if (!cancelled) setTbError(e instanceof Error ? e.message : String(e));
        return;
      }
      if (cancelled) return;

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
            failure = e instanceof Error ? e.message : String(e);
          }
        }
      }
      if (cancelled) return;

      if (url) {
        setTbUrl(url);
        setTbError(null);
        setTbNoLogs(false);
        return;
      }
      if (TERMINAL_STATUSES.has(detail.status ?? "")) {
        if (noLogs) {
          setTbNoLogs(true);
          setTbError(null);
        } else {
          setTbError(failure ?? "No TensorBoard is serving this run.");
        }
        return;
      }
      setTbError(null);
      timer = setTimeout(() => void attach(), TENSORBOARD_RETRY_MS);
    };

    void attach();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [selectedRun, tbAttempt]);

  async function onCancel(runId: string) {
    try {
      await trainingApi.cancel(runId);
      void refreshRuns();
    } catch (e) {
      useStore.getState().pushToast(`Cancel failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  const chartData = useMemo(() => {
    return metrics.map((m, i) => ({ step: m.epoch ?? m.step ?? i, ...m }));
  }, [metrics]);

  const metricKeys = useMemo(() => {
    const keys = new Set<string>();
    metrics.forEach((row) => {
      Object.entries(row).forEach(([k, v]) => {
        if (typeof v === "number" && k !== "epoch" && k !== "step") keys.add(k);
      });
    });
    return Array.from(keys);
  }, [metrics]);

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
              <span className="tcip-heading">Live metrics</span>
              <span className="font-mono text-[12px] text-tcip-fg">{selectedRun}</span>
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
                      {metricKeys.map((key, i) => (
                        <Line
                          key={key}
                          type="monotone"
                          dataKey={key}
                          stroke={CHART_LINE_COLORS[i % CHART_LINE_COLORS.length]}
                          dot={false}
                          strokeWidth={1.5}
                          isAnimationActive={false}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex items-center justify-center h-full text-tcip-muted text-[12px]">
                    {selectedRun ? "Waiting for metrics…" : "No run selected."}
                  </div>
                )}
              </div>

              {selectedRun && (
                <div className="h-[60vh] min-h-[360px] shrink-0">
                  <EmbeddedTool
                    title="TensorBoard"
                    url={tbUrl}
                    loading={!tbUrl && !tbError && !tbNoLogs}
                    error={tbNoLogs ? "This run produced no logs." : tbError}
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
            Runs this window launched first, in launch order; every other recorded run follows,
            sorted by experiment id.
          </div>
        )}
        <ul className="space-y-1">
          {runs.map((r) => {
            const isMarked = markedRunIds.has(r.run_id);
            const reason = unmarkableReason(r);
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
                    className="flex-1 text-left"
                    onClick={() => setSelectedRun(r.run_id)}
                  >
                    <div className="font-mono text-[11px]">
                      {r.run_id}
                      {r.experiment_id && r.experiment_id !== r.run_id && (
                        <span className="text-tcip-muted"> · {r.experiment_id}</span>
                      )}
                    </div>
                    <div className="text-[10px] text-tcip-muted flex justify-between">
                      <span>
                        {r.status}
                        {r.external && r.status === "running" ? " · agent" : ""}
                      </span>
                      {r.best_metric !== undefined &&
                        r.best_metric !== null &&
                        r.best_metric_name && (
                          <span className="tabular-nums">
                            best {r.best_metric_name} {Number(r.best_metric).toFixed(3)}
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
                          className="px-2 py-1 text-[10px] border-l border-tcip-border hover:bg-tcip-hover transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-tcip-accent/70"
                          onClick={() => void onCancel(r.run_id)}
                        >
                          Cancel
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

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { messageOf, refusalOf } from "@/api/http";
import { openTrainingStream, trainingApi } from "@/api/training";
import {
  type MetricsRow,
  NO_LOGS,
  type NoLogsRefusal,
  TENSORBOARD_RELOAD_SELECTOR,
  type RunRow,
  type SplitChoices,
  type SweepGroup,
  type TrainingListing,
} from "@/api/types.generated";
import { EmbeddedTool } from "@/components/EmbeddedTool";
import { LaunchPicker, type LaunchPickerRow } from "@/components/LaunchPicker";
import { MAX_MARKED_RUNS, RunComparison, type MarkedRun } from "@/components/RunComparison";
import { TabHeading } from "@/components/TabHeading";
import { useEmbeddedToolRetry, type EmbeddedToolStepResult } from "@/hooks/useEmbeddedToolRetry";
import { UNSET_GLYPH } from "@/lib/glyphs";
import { formatNumber } from "@/lib/numberFormat";
import { TERMINAL_STATES } from "@/lib/runStatus";
import { useStore } from "@/store";
import { declaredClient } from "@/store/slices/agentActivity";
import { selectProjectRoot } from "@/store/slices/gui";
import { RunMonitorEmpty, RunMonitorLayout } from "@/tabs/RunMonitorLayout";
import {
  inProgressBatch,
  mergeMetric,
  numericMetricKeys,
  RUN_REFRESH_MS,
  unionMetricKeys,
} from "@/tabs/trainingMetrics";

const NO_LISTING: TrainingListing = { runs: [], sweeps: [] };

/** What a run's launch event says about who started it. */
function launcherSentence(launch: RunRow["launch"]): string {
  if (!launch) return "no launch event recorded";
  return declaredClient(launch) ? "started by the agent" : "started with no agent declared";
}

/** The longer sentence behind a row's launcher mark, reachable by assistive technology through
 * aria-describedby: what the mark means, with the declared client named for an agent launch. */
function launcherDescription(launch: RunRow["launch"]): string {
  if (!launch) return "No launch event names this run.";
  const client = declaredClient(launch);
  return client ? `Launched by the agent ${client}.` : "Launched with no agent named.";
}

/** How long ago a status record's own heartbeat instant was stamped, in whole minutes. No
 * process id is persisted anywhere, so this is the one liveness signal a "running" row has. */
function heartbeatAge(heartbeat: string): string {
  const mins = Math.max(0, Math.round((Date.now() - Date.parse(heartbeat)) / 60000));
  return mins < 1 ? "last heartbeat under a minute ago" : `last heartbeat ${mins} min ago`;
}

/** A row's state, with a running row's own heartbeat age. */
function stateLine(state: string, heartbeat: string): string {
  const age = state === "running" ? heartbeatAge(heartbeat) : null;
  return age ? `${state}, ${age}` : state;
}

/** The row's own select control name: id, state, and the record's own launcher sentence, with
 * the best value's metric appended exactly as the record carries them when present; the same
 * order the visible row itself reads in. */
function runRowLabel(run: RunRow): string {
  const base = `${run.experiment_id} ${stateLine(run.state, run.heartbeat)}, ${launcherSentence(run.launch)}`;
  if (run.best_metric === null || !run.best_metric_name) return base;
  return `${base}, best ${run.best_metric_name} ${formatNumber(run.best_metric)}`;
}

/** What a sweep's trials amount to so far, under the selection metric it ranks them by. */
function sweepOutcomeLine(sweep: SweepGroup): string {
  const { best_params, best_value, best_value_state } = sweep.outcome;
  return best_params === null
    ? "no completed trial yet"
    : `best ${sweep.selection_metric} ${best_value === null ? best_value_state : formatNumber(best_value)}`;
}

function runLaunchRow(
  run: RunRow,
  choices: SplitChoices | undefined,
  dataLoading: boolean,
  dataError: string | undefined,
  onStart: (selectionDir: string | null) => Promise<void>,
): LaunchPickerRow {
  return {
    key: run.experiment_id,
    content: (
      <>
        <span className="block font-mono text-[13px]">{run.experiment_id}</span>
        <span className="block text-[12px] text-tcip-muted">
          {run.builder} · {run.task}
          {run.sweep ? ` · trial of ${run.sweep}` : ""}
        </span>
        <span className="block text-[12px] text-tcip-muted">
          {new Date(run.created).toLocaleString()} · {run.state}
          {run.relaunched_from ? ` · from ${run.relaunched_from}` : ""}
        </span>
      </>
    ),
    branchLine:
      "A new run of this config on the data paths it names, as they are now, with the recorded seed",
    branchLineForData:
      "A new run of this config on the partition you chose, with the recorded seed",
    data: choices,
    dataLoading,
    dataError,
    onStart,
  };
}

function sweepLaunchRow(sweep: SweepGroup, onStart: () => Promise<void>): LaunchPickerRow {
  return {
    key: sweep.sweep_id,
    content: (
      <>
        <span className="block font-mono text-[13px]">{sweep.sweep_id}</span>
        <span className="block text-[12px] text-tcip-muted">
          sweep of {sweep.input.n_trials} trials · {String(sweep.input.search_alg)} · {sweep.state}
        </span>
      </>
    ),
    branchLine: "A new sweep over this sweep's recorded config, search and seed",
    onStart,
  };
}

function RunItem({
  run,
  selected,
  marked,
  canceling,
  cancelError,
  onSelect,
  onToggleMarked,
  onCancel,
}: {
  run: RunRow;
  selected: boolean;
  marked: boolean;
  canceling: boolean;
  cancelError: string | undefined;
  onSelect: () => void;
  onToggleMarked: () => void;
  onCancel: () => void;
}) {
  const id = run.experiment_id;
  return (
    <div
      className={`flex items-start gap-1 p-2 rounded border transition-colors ${
        selected && !marked
          ? "border-tcip-accent bg-tcip-accent/10"
          : "border-tcip-border hover:border-tcip-border-hover hover:bg-tcip-hover"
      }`}
    >
      <button
        type="button"
        aria-pressed={selected}
        aria-label={runRowLabel(run)}
        aria-describedby={`origin-mark-${id}`}
        className="flex-1 min-w-0 text-left"
        onClick={onSelect}
      >
        <div className="font-mono text-[13px]">{id}</div>
        <div className="text-[12px] text-tcip-muted flex justify-between">
          <span>
            {stateLine(run.state, run.heartbeat)}
            <span title={launcherDescription(run.launch)}>
              {` · ${launcherSentence(run.launch)}`}
            </span>
            <span id={`origin-mark-${id}`} className="sr-only">
              {launcherDescription(run.launch)}
            </span>
          </span>
          {run.best_metric !== null && run.best_metric_name && (
            <span className="tabular-nums">
              best {run.best_metric_name} {formatNumber(run.best_metric)}
            </span>
          )}
        </div>
        {run.status_error && <div className="text-[12px] tcip-problem">{run.status_error}</div>}
      </button>
      <div className="flex flex-col items-end gap-1 shrink-0">
        <div
          role="group"
          aria-label="Run actions"
          className="inline-flex rounded border border-tcip-border overflow-hidden"
        >
          <button
            type="button"
            aria-pressed={marked}
            aria-label={`Compare ${id}`}
            className={`px-2 py-1 text-[12px] transition-colors disabled:opacity-40 disabled:cursor-not-allowed focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-tcip-accent/70 ${
              marked ? "bg-tcip-accent text-white" : "hover:bg-tcip-hover"
            }`}
            onClick={onToggleMarked}
          >
            Compare
          </button>
          <CancelButton
            id={id}
            state={run.state}
            canceling={canceling}
            cancelError={cancelError}
            onCancel={onCancel}
          />
        </div>
        <CancelError id={id} cancelError={cancelError} />
      </div>
    </div>
  );
}

function CancelButton({
  id,
  state,
  canceling,
  cancelError,
  onCancel,
}: {
  id: string;
  state: string;
  canceling: boolean;
  cancelError: string | undefined;
  onCancel: () => void;
}) {
  if (TERMINAL_STATES.has(state)) return null;
  return (
    <button
      type="button"
      title="Reaches a live process only; a stale running row keeps this control until its heartbeat window lapses."
      aria-label={`Cancel ${id}`}
      aria-describedby={cancelError ? `cancel-error-${id}` : undefined}
      disabled={canceling}
      className="px-2 py-1 text-[12px] border-l border-tcip-border first:border-l-0 hover:bg-tcip-hover transition-colors disabled:opacity-40 disabled:cursor-not-allowed focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-tcip-accent/70"
      onClick={onCancel}
    >
      {canceling ? "Canceling…" : "Cancel"}
    </button>
  );
}

function CancelError({ id, cancelError }: { id: string; cancelError: string | undefined }) {
  if (!cancelError) return null;
  return (
    <span id={`cancel-error-${id}`} className="text-[12px] tcip-problem text-right max-w-[150px]">
      {cancelError}
    </span>
  );
}

// Training is launched from a run or sweep already recorded in this project, or described fresh
// to the agent; this tab tracks the runs and sweeps those launches produce and their metrics.
export function TrainingTab() {
  const projectRoot = useStore(selectProjectRoot);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [splitChoicesById, setSplitChoicesById] = useState<Record<string, SplitChoices>>({});
  const [splitChoicesLoadingId, setSplitChoicesLoadingId] = useState<string | null>(null);
  const [splitChoiceErrors, setSplitChoiceErrors] = useState<Record<string, string>>({});
  const [listing, setListing] = useState<TrainingListing>(NO_LISTING);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<MetricsRow[]>([]);
  const [latestBatch, setLatestBatch] = useState<MetricsRow | null>(null);
  // The run's own TensorBoard refusal for a board holding no logs, once one answers.
  const [tbNoLogs, setTbNoLogs] = useState<NoLogsRefusal | null>(null);
  const [tbAttempt, setTbAttempt] = useState(0);
  const [markedExperimentIds, setMarkedExperimentIds] = useState<Set<string>>(new Set());
  // Cancel in flight, by run or sweep id: disables that row's own Cancel button with a pending
  // label, and a failure lands in cancelErrors.
  const [pendingCancel, setPendingCancel] = useState<ReadonlySet<string>>(new Set());
  const [cancelErrors, setCancelErrors] = useState<Record<string, string>>({});
  const streamRef = useRef<(() => void) | null>(null);

  const rows = useMemo(
    () => [...listing.runs, ...listing.sweeps.flatMap((s) => s.trials)],
    [listing],
  );
  const selectedSweep = listing.sweeps.find((s) => s.sweep_id === selectedRun);

  const marked: MarkedRun[] = rows
    .filter((r) => markedExperimentIds.has(r.experiment_id))
    .map((r) => ({ experimentId: r.experiment_id }));
  const comparing = marked.length >= 2;

  function toggleMarked(run: RunRow) {
    setMarkedExperimentIds((prev) => {
      const next = new Set(prev);
      if (next.has(run.experiment_id)) {
        next.delete(run.experiment_id);
        return next;
      }
      if (next.size >= MAX_MARKED_RUNS) {
        useStore
          .getState()
          .pushToast(`Compare fits at most ${MAX_MARKED_RUNS} runs at once; unmark one first.`);
        return prev;
      }
      next.add(run.experiment_id);
      return next;
    });
  }

  // Once the marked set settles at exactly one run, that run becomes the one the detail region
  // shows.
  const soleMarkedExperimentId = marked.length === 1 ? marked[0].experimentId : null;
  useEffect(() => {
    if (soleMarkedExperimentId) setSelectedRun(soleMarkedExperimentId);
  }, [soleMarkedExperimentId]);

  const refreshRuns = useCallback(async () => {
    try {
      const next = await trainingApi.listRuns();
      setListing(next);
      setRunsError(null);
      // A run that leaves the list must also leave the marked set, or the cap (which counts
      // markedExperimentIds itself) can read full while the header (runs still present) shows fewer.
      const stillPresent = new Set([
        ...next.runs.map((run) => run.experiment_id),
        ...next.sweeps.flatMap((s) => [s.sweep_id, ...s.trials.map((t) => t.experiment_id)]),
      ]);
      setMarkedExperimentIds((prev) => {
        const pruned = new Set(Array.from(prev).filter((id) => stillPresent.has(id)));
        return pruned.size === prev.size ? prev : pruned;
      });
      // A selected run that left the list (another project opened, say) must give up its
      // stream too, or a stale selection keeps reconnecting behind a run this list never shows.
      setSelectedRun((prev) => (prev !== null && !stillPresent.has(prev) ? null : prev));
    } catch (e) {
      setRunsError(`Could not load training runs: ${messageOf(e)}`);
    }
  }, []);

  // Fetched only for the row the breeder opens.
  const loadSplitChoices = useCallback(
    async (key: string) => {
      if (listing.sweeps.some((s) => s.sweep_id === key)) return;
      setSplitChoicesLoadingId(key);
      try {
        const choices = await trainingApi.listSplitChoices(key);
        setSplitChoicesById((prev) => ({ ...prev, [key]: choices }));
        setSplitChoiceErrors((prev) => {
          const { [key]: _drop, ...rest } = prev;
          return rest;
        });
      } catch (e) {
        setSplitChoiceErrors((prev) => ({
          ...prev,
          [key]: `Could not load its data choices: ${messageOf(e)}`,
        }));
      } finally {
        setSplitChoicesLoadingId((current) => (current === key ? null : current));
      }
    },
    [listing],
  );

  useEffect(() => {
    void refreshRuns();
    const t = setInterval(refreshRuns, RUN_REFRESH_MS);
    return () => clearInterval(t);
  }, [refreshRuns]);

  useEffect(() => {
    if (!pickerOpen) return;
    setSplitChoicesById({});
    setSplitChoiceErrors({});
  }, [pickerOpen]);

  async function startFrom(source: string, selectionDir: string | null) {
    const result = await trainingApi.relaunch({
      relaunched_from: source,
      selection_dir: selectionDir,
      user: useStore.getState().user,
    });
    setPickerOpen(false);
    void refreshRuns();
    setSelectedRun("sweep_id" in result ? result.sweep_id : result.experiment_id);
  }

  // The run list's own poll keeps a state per run, which the metrics stream reads without reopening.
  const rowsRef = useRef<RunRow[]>(rows);
  useEffect(() => {
    rowsRef.current = rows;
  }, [rows]);

  const streaming = selectedRun !== null && !selectedSweep;
  useEffect(() => {
    // Comparing owns the detail region; a sweep has no metrics log of its own.
    if (!selectedRun || !streaming || !projectRoot || comparing) return;
    // The stream replays this run from the start; it is the rows' one source.
    setMetrics([]);
    setLatestBatch(null);
    streamRef.current?.();
    // A run already terminal when this stream opened is a rediscovery, not a transition the
    // breeder is watching; only a run still live at open time toasts on its own terminal frame.
    const knownAtOpen = rowsRef.current.find((r) => r.experiment_id === selectedRun);
    const alreadyTerminal = knownAtOpen !== undefined && TERMINAL_STATES.has(knownAtOpen.state);
    // A run selected at its launch moment can be unknown to the backend for a few reconnects;
    // the toast names that once per selection.
    let errorToasted = false;
    streamRef.current = openTrainingStream(selectedRun, (msg) => {
      if (msg.type === "metric") {
        setMetrics((prev) => mergeMetric(prev, msg.row));
      } else if (msg.type === "batch") {
        setLatestBatch(msg.row);
      } else if (msg.type === "status") {
        // A known run carries its row and no error; an unknown run carries error and no row,
        // is not terminal, and the socket keeps reconnecting behind it.
        if (msg.error) {
          if (!errorToasted) {
            errorToasted = true;
            useStore.getState().pushToast(`Training stream error: ${msg.error}`);
          }
          return;
        }
        if (msg.status !== null && !alreadyTerminal)
          useStore.getState().pushToast(`Training ${selectedRun}: ${msg.status.state}`, "info");
        void refreshRuns();
      }
    });
    return () => streamRef.current?.();
  }, [selectedRun, streaming, projectRoot, refreshRuns, comparing]);

  // tbNoLogs is a step side effect below, not part of the hook's own outcome, since its text
  // overrides tbError's; reset it on the same triggers the hook itself resets url/error on.
  useEffect(() => {
    setTbNoLogs(null);
  }, [selectedRun, tbAttempt]);

  // Adopt the TensorBoard already serving this run, trial or sweep, or start one, retrying on a
  // timer while it is live.
  const tbStep = useCallback(async (): Promise<EmbeddedToolStepResult> => {
    if (!selectedRun) return { url: null, error: null, done: true };
    const experimentId = selectedRun;
    let detail;
    try {
      detail = await trainingApi.getRun(experimentId);
    } catch (e) {
      return { url: null, error: messageOf(e), done: true };
    }

    let url = detail.tensorboard?.url ?? null;
    let failure: string | null = null;
    let noLogs: NoLogsRefusal | null = null;
    if (!url) {
      try {
        const launched = await trainingApi.launchTensorboard(experimentId);
        if ("url" in launched) {
          url = launched.url;
        } else {
          failure = launched.output ? `${launched.error}: ${launched.output}` : launched.error;
        }
      } catch (e) {
        noLogs = refusalOf<NoLogsRefusal>(e, NO_LOGS);
        if (!noLogs) failure = messageOf(e);
      }
    }

    if (url) {
      setTbNoLogs(null);
      return { url, error: null, done: true };
    }
    // The route answers a detail carrying the run or the sweep its id names.
    const status = (detail.run ?? detail.sweep)!;
    if (!TERMINAL_STATES.has(status.state)) return { url: null, error: null, done: false };
    if (noLogs) {
      setTbNoLogs(noLogs);
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

  // TensorBoard's own header button reloads its data in place, keeping the person's run and groups.
  const tbFrame = useRef<HTMLIFrameElement>(null);
  const epochRows = metrics.length;
  useEffect(() => {
    if (epochRows === 0) return;
    const reload = tbFrame.current?.contentDocument?.querySelector<HTMLButtonElement>(
      TENSORBOARD_RELOAD_SELECTOR,
    );
    if (reload && !reload.disabled) reload.click();
  }, [epochRows]);

  async function onCancel(id: string) {
    setPendingCancel((prev) => new Set(prev).add(id));
    setCancelErrors((prev) => {
      const { [id]: _drop, ...rest } = prev;
      return rest;
    });
    try {
      await trainingApi.cancel(id, { user: useStore.getState().user });
      void refreshRuns();
    } catch (e) {
      const message = `Cancel failed: ${messageOf(e)}`;
      useStore.getState().pushToast(message);
      setCancelErrors((prev) => ({ ...prev, [id]: message }));
    } finally {
      setPendingCancel((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
    }
  }

  const selectedRow = rows.find((r) => r.experiment_id === selectedRun);
  const selectedTerminal = selectedRow !== undefined && TERMINAL_STATES.has(selectedRow.state);

  const noLogsMessage = tbNoLogs?.message ?? null;

  const metricKeys = useMemo(() => unionMetricKeys(metrics), [metrics]);
  const batch = inProgressBatch(metrics, latestBatch);

  function runItem(run: RunRow) {
    const id = run.experiment_id;
    return (
      <li key={id}>
        <RunItem
          run={run}
          selected={selectedRun === id}
          marked={markedExperimentIds.has(id)}
          canceling={pendingCancel.has(id)}
          cancelError={cancelErrors[id]}
          onSelect={() => setSelectedRun(id)}
          onToggleMarked={() => toggleMarked(run)}
          onCancel={() => void onCancel(id)}
        />
      </li>
    );
  }

  const launchRows: LaunchPickerRow[] = [
    ...rows.map((run) =>
      runLaunchRow(
        run,
        splitChoicesById[run.experiment_id],
        splitChoicesLoadingId === run.experiment_id,
        splitChoiceErrors[run.experiment_id],
        (dir) => startFrom(run.experiment_id, dir),
      ),
    ),
    ...listing.sweeps.map((sweep) => sweepLaunchRow(sweep, () => startFrom(sweep.sweep_id, null))),
  ];
  const empty = listing.runs.length === 0 && listing.sweeps.length === 0;

  return (
    <>
      <TabHeading tab="training" />
      <RunMonitorLayout
        title="Runs"
        headerRight={
          <button
            type="button"
            aria-expanded={pickerOpen}
            className={pickerOpen ? "tcip-btn text-[13px]" : "tcip-btn-primary text-[13px]"}
            onClick={() => setPickerOpen((open) => !open)}
          >
            Start a run
          </button>
        }
        detailHeader={
          comparing ? (
            <>
              <span className="tcip-heading">Comparing</span>
              <span className="text-[13px] text-tcip-muted">
                {marked.length} of {MAX_MARKED_RUNS} runs
              </span>
            </>
          ) : selectedRun ? (
            <>
              <h2 className="tcip-heading">{selectedSweep ? "Sweep" : "Metrics"}</h2>
              <span className="font-mono text-[13px] text-tcip-fg">{selectedRun}</span>
            </>
          ) : (
            <span className="tcip-heading">Metrics</span>
          )
        }
        detail={
          comparing ? (
            <RunComparison marked={marked} />
          ) : (
            <div className="flex flex-col gap-4">
              {streaming &&
                (metrics.length > 0 || batch ? (
                  <div className="overflow-auto max-h-64 shrink-0">
                    <table className="w-full text-[13px]">
                      <caption className="sr-only">{`${selectedRun} metrics by epoch`}</caption>
                      <thead className="sticky top-0 bg-tcip-panel">
                        <tr className="border-b border-tcip-border">
                          <th className="tcip-th">epoch</th>
                          {metricKeys.map((key) => (
                            <th key={key} className="tcip-th">
                              {key.replaceAll("_", " ")}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {metrics.map((row, i) => (
                          <tr key={i} className="border-t border-tcip-border first:border-t-0">
                            <td className="py-1 pr-3 tabular-nums">{row.epoch}</td>
                            {metricKeys.map((key) => (
                              <td key={key} className="pr-3 tabular-nums">
                                {typeof row[key] === "number"
                                  ? formatNumber(row[key])
                                  : UNSET_GLYPH}
                              </td>
                            ))}
                          </tr>
                        ))}
                        {batch && (
                          <tr className="border-t border-tcip-border text-tcip-muted">
                            <td className="py-1 pr-3 tabular-nums">{batch.epoch}</td>
                            <td
                              colSpan={Math.max(1, metricKeys.length)}
                              className="pr-3 tabular-nums"
                            >
                              {`${selectedTerminal ? "stopped at" : "in progress ·"} step ${String(batch.step)}`}
                              {numericMetricKeys(batch).map(
                                (key) => ` · ${key} ${String(batch[key])}`,
                              )}
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div role="status" className="text-tcip-muted text-[13px]">
                    {selectedTerminal ? "This run recorded no metrics." : "Waiting for metrics…"}
                  </div>
                ))}
              {!selectedRun && (
                <div role="status" className="text-tcip-muted text-[13px]">
                  No run selected.
                </div>
              )}

              {selectedRun && (
                <div className="h-[60vh] min-h-[360px] shrink-0">
                  <EmbeddedTool
                    title="TensorBoard"
                    url={tbUrl}
                    frameRef={tbFrame}
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
                title: "Runs and sweeps in this project",
                emptyMessage: "No runs.",
                error: runsError ?? undefined,
                onRetry: () => void refreshRuns(),
                rows: launchRows,
              }}
              onSelect={(key) => void loadSplitChoices(key)}
            />
          </div>
        )}

        {runsError && (
          <div className="text-[13px] tcip-problem mb-2">
            {runsError}{" "}
            <button className="tcip-btn text-[13px] ml-1" onClick={() => void refreshRuns()}>
              Retry
            </button>
          </div>
        )}
        {empty && !runsError && <RunMonitorEmpty>No runs.</RunMonitorEmpty>}
        <ul className="space-y-1">
          {listing.runs.map(runItem)}
          {listing.sweeps.map((sweep) => (
            <li key={sweep.sweep_id}>
              <div
                role="group"
                aria-label={`Sweep ${sweep.sweep_id}`}
                className="p-2 rounded border border-tcip-border"
              >
                <div className="flex items-start gap-1">
                  <button
                    type="button"
                    aria-pressed={selectedRun === sweep.sweep_id}
                    className="flex-1 min-w-0 text-left"
                    onClick={() => setSelectedRun(sweep.sweep_id)}
                  >
                    <div className="font-mono text-[13px]">{sweep.sweep_id}</div>
                    <div className="text-[12px] text-tcip-muted">
                      sweep · {stateLine(sweep.state, sweep.heartbeat)} · {sweepOutcomeLine(sweep)}
                    </div>
                  </button>
                  <div className="flex flex-col items-end gap-1 shrink-0">
                    <div className="inline-flex rounded border border-tcip-border overflow-hidden empty:hidden">
                      <CancelButton
                        id={sweep.sweep_id}
                        state={sweep.state}
                        canceling={pendingCancel.has(sweep.sweep_id)}
                        cancelError={cancelErrors[sweep.sweep_id]}
                        onCancel={() => void onCancel(sweep.sweep_id)}
                      />
                    </div>
                    <CancelError id={sweep.sweep_id} cancelError={cancelErrors[sweep.sweep_id]} />
                  </div>
                </div>
                {sweep.status_error && (
                  <div className="text-[12px] tcip-problem">{sweep.status_error}</div>
                )}
                <ul className="mt-1 pl-3 space-y-1">{sweep.trials.map(runItem)}</ul>
              </div>
            </li>
          ))}
        </ul>
      </RunMonitorLayout>
    </>
  );
}

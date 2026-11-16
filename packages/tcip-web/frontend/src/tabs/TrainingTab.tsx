import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { messageOf, refusalOf } from "@/api/http";
import { openTrainingStream, trainingApi } from "@/api/training";
import {
  type MetricsRow,
  NO_LOGS,
  type NoLogsRefusal,
  RANGE_TEST,
  TENSORBOARD_RELOAD_SELECTOR,
  type RunRow,
  type SplitChoices,
  type SweepGroup,
  TRIAL,
} from "@/api/types.generated";
import { EmbeddedTool } from "@/components/EmbeddedTool";
import { ErrorRetry } from "@/components/ErrorRetry";
import { LaunchPicker, type LaunchPickerRow } from "@/components/LaunchPicker";
import { NotificationItem } from "@/components/Notifications";
import { MAX_MARKED_RUNS, RunComparison, type MarkedRun } from "@/components/RunComparison";
import { TabHeading } from "@/components/TabHeading";
import {
  type EmbeddedToolOutcome,
  type EmbeddedToolStepResult,
  PENDING,
  useEmbeddedToolRetry,
} from "@/hooks/useEmbeddedToolRetry";
import { UNSET_GLYPH } from "@/lib/glyphs";
import { formatNumber } from "@/lib/numberFormat";
import { TERMINAL_STATES } from "@/lib/runStatus";
import { tell } from "@/lib/tell";
import { trainingKey } from "@/lib/trainingKeys";
import { useStore } from "@/store";
import { clientName } from "@/store/slices/agentActivity";
import { selectProjectRoot } from "@/store/slices/gui";
import { selectStanding } from "@/store/slices/notifications";
import { RunMonitorEmpty, RunMonitorLayout } from "@/tabs/RunMonitorLayout";
import {
  inProgressBatch,
  mergeMetric,
  numericMetricKeys,
  unionMetricKeys,
} from "@/tabs/trainingMetrics";

/** Ask the shell's reader to list the runs again now. */
const listRunsNow = () => useStore.getState().askListing("training");

/** What a run's launch event says about who started it. */
function launcherSentence(launch: RunRow["launch"]): string {
  if (launch.kind === "no_launch_line") return "no launch event recorded";
  return launch.kind === "agent" ? "started by the agent" : "started with no agent declared";
}

/** The longer sentence behind a row's launcher mark, reachable by assistive technology through
 * aria-describedby: what the mark means, with the declared client named for an agent launch. */
function launcherDescription(launch: RunRow["launch"]): string {
  if (launch.kind === "no_launch_line") return "No launch event names this run.";
  return launch.kind === "agent"
    ? `Launched by the agent ${clientName(launch.agent_client_name, launch.agent_client_version)}.`
    : "Launched with no agent named.";
}

/** A row's best value under its selection metric, or nothing before one. */
function bestLine(run: RunRow): string | null {
  return run.best && `best ${run.best.metric} ${formatNumber(run.best.value)}`;
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
  const best = bestLine(run);
  return best ? `${base}, ${best}` : base;
}

/** What a sweep's trials amount to so far, under the selection metric it ranks them by. */
function sweepOutcomeLine(sweep: SweepGroup): string {
  const { outcome } = sweep;
  if (outcome.kind === "no_best") return "no completed trial yet";
  const value =
    outcome.kind === "finite_best" ? formatNumber(outcome.best_value) : outcome.best_value_state;
  return `best ${sweep.selection_metric} ${value}`;
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
          {run.kind === TRIAL ? ` · trial of ${run.sweep_id}` : ""}
          {run.kind === RANGE_TEST ? " · learning-rate range test" : ""}
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

/** The key the Compare cap's refusal stands on in the project `projectId`. */
const compareCapKey = (projectId: string | undefined) => `training-compare-cap:${projectId}`;

/** The id of the control that selects the run or sweep `id`, where focus returns once its Cancel's
 *  failure is dismissed. */
const selectControlId = (id: string) => `select-${id}`;

/** The id of the Compare toggle of the run `id`. */
const compareToggleId = (id: string) => `compare-${id}`;

/** The id of the text of the failure standing on the Cancel of the run or sweep `id`, which
 *  describes that Cancel button; `id` is encoded so one holding a space stays one token of the
 *  button's `aria-describedby`. */
const cancelFailureId = (id: string) => `cancel-error-${encodeURIComponent(id)}`;

function RunItem({
  run,
  selected,
  marked,
  canceling,
  cancelKey,
  onSelect,
  onToggleMarked,
  onCancel,
}: {
  run: RunRow;
  selected: boolean;
  marked: boolean;
  canceling: boolean;
  cancelKey: string;
  onSelect: () => void;
  onToggleMarked: () => void;
  onCancel: () => void;
}) {
  const id = run.experiment_id;
  return (
    <div
      className={`p-2 rounded border transition-colors ${
        selected && !marked
          ? "border-tcip-accent bg-tcip-accent/10"
          : "border-tcip-border hover:border-tcip-border-hover hover:bg-tcip-hover"
      }`}
    >
      <div className="flex items-start gap-1">
        <button
          type="button"
          id={selectControlId(id)}
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
            {run.best && <span className="tabular-nums">{bestLine(run)}</span>}
          </div>
          {run.status_error && <div className="text-[12px] tcip-problem">{run.status_error}</div>}
        </button>
        <div
          role="group"
          aria-label="Run actions"
          className="inline-flex shrink-0 rounded border border-tcip-border overflow-hidden"
        >
          <button
            type="button"
            id={compareToggleId(id)}
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
            cancelKey={cancelKey}
            onCancel={onCancel}
          />
        </div>
      </div>
      <CancelFailure id={id} cancelKey={cancelKey} />
    </div>
  );
}

function CancelButton({
  id,
  state,
  canceling,
  cancelKey,
  onCancel,
}: {
  id: string;
  state: string;
  canceling: boolean;
  cancelKey: string;
  onCancel: () => void;
}) {
  const failed = useStore((s) => selectStanding(cancelKey)(s) !== null);
  if (TERMINAL_STATES.has(state)) return null;
  return (
    <button
      type="button"
      title="Reaches a live process only; a stale running row keeps this control until its heartbeat window lapses."
      aria-label={`Cancel ${id}`}
      aria-describedby={failed ? cancelFailureId(id) : undefined}
      disabled={canceling}
      className="px-2 py-1 text-[12px] border-l border-tcip-border first:border-l-0 hover:bg-tcip-hover transition-colors disabled:opacity-40 disabled:cursor-not-allowed focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-tcip-accent/70"
      onClick={onCancel}
    >
      {canceling ? "Canceling…" : "Cancel"}
    </button>
  );
}

/** The failure standing on the Cancel of the run or sweep `id`, in place under its row. */
function CancelFailure({ id, cancelKey }: { id: string; cancelKey: string }) {
  return (
    <div className="mt-1 empty:hidden">
      <NotificationItem
        placement="in place"
        entryKey={cancelKey}
        textId={cancelFailureId(id)}
        afterDismiss={() => document.getElementById(selectControlId(id))?.focus()}
      />
    </div>
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
  const listing = useStore((s) => s.jobListings.training.answer);
  const listFailure = useStore((s) => s.jobListings.training.failure);
  const runsError = listFailure && `Could not load training runs: ${listFailure}`;
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<MetricsRow[]>([]);
  const [latestBatch, setLatestBatch] = useState<MetricsRow | null>(null);
  const [tbAttempt, setTbAttempt] = useState(0);
  const [markedExperimentIds, setMarkedExperimentIds] = useState<Set<string>>(new Set());
  // Cancel in flight, by run or sweep id: disables that row's own Cancel button with a pending
  // label.
  const [pendingCancel, setPendingCancel] = useState<ReadonlySet<string>>(new Set());
  const projectId = useStore((s) => s.openProject?.id);
  const capKey = compareCapKey(projectId);
  // The run whose Compare toggle the cap last refused, where focus returns once that is dismissed.
  const capRefused = useRef<string | null>(null);
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
    const id = run.experiment_id;
    if (!markedExperimentIds.has(id) && markedExperimentIds.size >= MAX_MARKED_RUNS) {
      capRefused.current = id;
      tell(
        "failure",
        "error",
        `Compare fits at most ${MAX_MARKED_RUNS} runs at once; unmark one first.`,
        "Compare",
        capKey,
      );
      return;
    }
    setMarkedExperimentIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  // The cap's refusal stands while the marked set is full, and ends once it is not or the tab
  // leaves, which empties the set.
  const markedCount = markedExperimentIds.size;
  useEffect(() => {
    const release = () => useStore.getState().releaseNotification(capKey);
    if (markedCount < MAX_MARKED_RUNS) release();
    return release;
  }, [markedCount, capKey]);

  // Once the marked set settles at exactly one run, that run becomes the one the detail region
  // shows.
  const soleMarkedExperimentId = marked.length === 1 ? marked[0].experimentId : null;
  useEffect(() => {
    if (soleMarkedExperimentId) setSelectedRun(soleMarkedExperimentId);
  }, [soleMarkedExperimentId]);

  useEffect(() => {
    // A run that leaves the list must also leave the marked set, or the cap (which counts
    // markedExperimentIds itself) can read full while the header (runs still present) shows fewer.
    const stillPresent = new Set([
      ...listing.runs.map((run) => run.experiment_id),
      ...listing.sweeps.flatMap((s) => [s.sweep_id, ...s.trials.map((t) => t.experiment_id)]),
    ]);
    setMarkedExperimentIds((prev) => {
      const pruned = new Set(Array.from(prev).filter((id) => stillPresent.has(id)));
      return pruned.size === prev.size ? prev : pruned;
    });
    // A selected run that left the list (another project opened, say) must give up its
    // stream too, or a stale selection keeps reconnecting behind a run this list never shows.
    setSelectedRun((prev) => (prev !== null && !stillPresent.has(prev) ? null : prev));
  }, [listing]);

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
    const target = result.kind === "sweep" ? result.sweep_id : result.experiment_id;
    for (const warning of result.warnings) tell("notice", "warning", warning, target);
    if (result.kind === "run" && result.tensorboard.kind === "failed") {
      const { error } = result.tensorboard;
      tell("notice", "warning", `TensorBoard did not start with the launch: ${error}`, target);
    }
    setPickerOpen(false);
    listRunsNow();
    setSelectedRun(target);
  }

  const streaming = selectedRun !== null && !selectedSweep;
  useEffect(() => {
    // Comparing owns the detail region; a sweep has no metrics log of its own.
    if (!selectedRun || !streaming || !projectRoot || comparing) return;
    // The stream replays this run from the start; it is the rows' one source.
    setMetrics([]);
    setLatestBatch(null);
    streamRef.current?.();
    // The stream's error stands on its key until the stream delivers anything else or closes; the
    // backend's frames name the stream's own run.
    const errorKey = trainingKey("stream", projectId, selectedRun);
    const release = () => useStore.getState().releaseNotification(errorKey);
    const stop = openTrainingStream(selectedRun, (msg) => {
      if (msg.type === "error") {
        // A run selected at its launch can be unknown to the backend for a few reconnects, each
        // sending this frame again, which counts on the one entry.
        tell("failure", "error", `Training stream error: ${msg.error}`, selectedRun, errorKey);
        return;
      }
      release();
      if (msg.type === "metric") {
        setMetrics((prev) => mergeMetric(prev, msg.row));
      } else if (msg.type === "batch") {
        setLatestBatch(msg.row);
      } else {
        // The run's end is told from the listing this read answers.
        listRunsNow();
      }
    });
    streamRef.current = () => {
      stop();
      release();
    };
    return () => streamRef.current?.();
  }, [selectedRun, streaming, projectRoot, projectId, comparing]);

  // Adopt the TensorBoard already serving this run, trial or sweep, or start one, retrying on a
  // timer while it is live.
  const tbStep = useCallback(async (): Promise<EmbeddedToolStepResult> => {
    if (!selectedRun) return { outcome: PENDING, done: true };
    const experimentId = selectedRun;
    const unreached = (e: unknown): EmbeddedToolOutcome => ({
      kind: "unreached",
      message: messageOf(e),
    });
    let detail;
    try {
      detail = await trainingApi.getRun(experimentId);
    } catch (e) {
      return { outcome: unreached(e), done: true };
    }

    let outcome: EmbeddedToolOutcome;
    try {
      outcome =
        detail.tensorboard.kind === "serving"
          ? detail.tensorboard
          : await trainingApi.launchTensorboard(experimentId);
    } catch (e) {
      outcome = refusalOf<NoLogsRefusal>(e, NO_LOGS) ?? unreached(e);
    }
    if (outcome.kind === "serving") return { outcome, done: true };
    const status = detail.kind === "run" ? detail.run : detail.sweep;
    if (!TERMINAL_STATES.has(status.state)) return { outcome: PENDING, done: false };
    return { outcome, done: true };
  }, [selectedRun]);

  const tbOutcome = useEmbeddedToolRetry(selectedRun, !!selectedRun, tbAttempt, tbStep);

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
    const key = trainingKey("cancel", projectId, id);
    setPendingCancel((prev) => new Set(prev).add(id));
    useStore.getState().releaseNotification(key);
    try {
      await trainingApi.cancel(id, { user: useStore.getState().user });
      listRunsNow();
    } catch (e) {
      tell("failure", "error", `Cancel failed: ${messageOf(e)}`, id, key);
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
          cancelKey={trainingKey("cancel", projectId, id)}
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
                    outcome={tbOutcome}
                    frameRef={tbFrame}
                    onRetry={() => setTbAttempt((n) => n + 1)}
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
                onRetry: listRunsNow,
                rows: launchRows,
              }}
              onSelect={(key) => void loadSplitChoices(key)}
            />
          </div>
        )}

        {runsError && !pickerOpen && (
          <ErrorRetry message={runsError} onRetry={listRunsNow} className="text-[13px] mb-2" />
        )}
        {empty && !runsError && <RunMonitorEmpty>No runs.</RunMonitorEmpty>}
        <div className="mb-2 empty:hidden">
          <NotificationItem
            placement="in place"
            entryKey={capKey}
            afterDismiss={() =>
              document.getElementById(compareToggleId(capRefused.current ?? ""))?.focus()
            }
          />
        </div>
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
                    id={selectControlId(sweep.sweep_id)}
                    aria-pressed={selectedRun === sweep.sweep_id}
                    className="flex-1 min-w-0 text-left"
                    onClick={() => setSelectedRun(sweep.sweep_id)}
                  >
                    <div className="font-mono text-[13px]">{sweep.sweep_id}</div>
                    <div className="text-[12px] text-tcip-muted">
                      sweep · {stateLine(sweep.state, sweep.heartbeat)} · {sweepOutcomeLine(sweep)}
                    </div>
                  </button>
                  <div className="inline-flex shrink-0 rounded border border-tcip-border overflow-hidden empty:hidden">
                    <CancelButton
                      id={sweep.sweep_id}
                      state={sweep.state}
                      canceling={pendingCancel.has(sweep.sweep_id)}
                      cancelKey={trainingKey("cancel", projectId, sweep.sweep_id)}
                      onCancel={() => void onCancel(sweep.sweep_id)}
                    />
                  </div>
                </div>
                <CancelFailure
                  id={sweep.sweep_id}
                  cancelKey={trainingKey("cancel", projectId, sweep.sweep_id)}
                />
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

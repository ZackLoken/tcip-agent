import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { StructuredRefusalError } from "@/api/http";
import { openTrainingStream, trainingApi } from "@/api/training";
import type { MetricRow, SplitChoices } from "@/api/training";
import {
  EPOCH_KEY,
  STEP_KEY,
  type RunRow,
  type SweepGroup,
  type TrainingListing,
} from "@/api/types.generated";
import { EmbeddedTool } from "@/components/EmbeddedTool";
import { LaunchPicker, type DataPicker, type LaunchPickerRow } from "@/components/LaunchPicker";
import { MAX_MARKED_RUNS, RunComparison, type MarkedRun } from "@/components/RunComparison";
import { TabHeading } from "@/components/TabHeading";
import { useEditableAgentRequest } from "@/hooks/useEditableAgentRequest";
import { useEmbeddedToolRetry, type EmbeddedToolStepResult } from "@/hooks/useEmbeddedToolRetry";
import { UNSET_GLYPH } from "@/lib/glyphs";
import { TERMINAL_STATES } from "@/lib/runStatus";
import { useStore } from "@/store";
import { declaredClient } from "@/store/slices/agentActivity";
import { selectProjectRoot } from "@/store/slices/gui";
import { defaultTrainingRequest } from "@/tabs/agentPrompts";
import { RunMonitorEmpty, RunMonitorLayout } from "@/tabs/RunMonitorLayout";
import {
  inProgressBatch,
  mergeMetric,
  numericMetricKeys,
  RUN_REFRESH_MS,
  unionMetricKeys,
} from "@/tabs/trainingMetrics";

const NO_LISTING: TrainingListing = { runs: [], sweeps: [] };

function messageOf(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** What a run's launch event says about who started it: never a guess. */
function launcherSentence(launch: RunRow["launch"]): string {
  if (!launch) return "no launch event recorded";
  return declaredClient(launch) ? "started by the agent" : "started with no agent declared";
}

/** The longer sentence behind a row's launcher mark, reachable by assistive technology through
 * aria-describedby: what the mark means, with the declared client named for an agent launch. */
function launcherDescription(launch: RunRow["launch"]): string {
  if (!launch) {
    return "No launch event in this project's audit log names this run.";
  }
  const client = declaredClient(launch);
  return client
    ? `This run was launched by an agent through the MCP door, declared as ${client}.`
    : "This run's launch event carries no agent identity: no agent declared one to the process that launched it.";
}

/** How long ago a status record's own heartbeat instant was stamped, in whole minutes; null
 * for a missing or unparseable instant so a caller shows nothing rather than a wrong age. No
 * process id is persisted anywhere, so this is the one liveness signal a "running" row has. */
function heartbeatAge(heartbeat: string | null | undefined): string | null {
  if (typeof heartbeat !== "string") return null;
  const then = Date.parse(heartbeat);
  if (Number.isNaN(then)) return null;
  const mins = Math.max(0, Math.round((Date.now() - then) / 60000));
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
  return `${base}, best ${run.best_metric_name} ${run.best_metric}`;
}

/** What a sweep's trials amount to so far, under the objective its record states. */
function sweepOutcomeLine(sweep: SweepGroup): string {
  const selected = sweep.objective.selection_metric;
  const metric = typeof selected === "string" ? selected : "objective";
  return sweep.outcome.best_params == null
    ? "no completed trial yet"
    : `best ${metric} ${String(sweep.outcome.best_value)}`;
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
    choices: choices.selections.map((s) => ({
      selectionDir: s.selection_dir,
      disabled: !s.enabled,
      reason: s.reason ?? undefined,
      replacedSplitKeys: s.replaced_split_keys,
      label: (
        <>
          <span className="block font-mono">{s.selection_dir}</span>
          <span className="block text-tcip-muted">
            seed {s.seed ?? "unrecorded"} · {s.group_by ?? "unrecorded grouping"} · train {s.train}{" "}
            · val {s.val} · calibration {s.calibration}
          </span>
        </>
      ),
    })),
  };
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
        <span className="block font-mono text-[11px]">{run.experiment_id}</span>
        <span className="block text-[10px] text-tcip-muted">
          {run.builder} · {run.task}
          {run.sweep ? ` · trial of ${run.sweep}` : ""}
        </span>
        <span className="block text-[10px] text-tcip-muted">
          {new Date(run.created).toLocaleString()} · {run.state}
          {run.relaunched_from ? ` · from ${run.relaunched_from}` : ""}
        </span>
      </>
    ),
    branchLine:
      "A new run of this config on the data paths it names, as they are now, with the recorded seed",
    branchLineForData:
      "A new run of this config on the partition you chose, with the recorded seed",
    data: dataPickerFor(choices),
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
        <span className="block font-mono text-[11px]">{sweep.sweep_id}</span>
        <span className="block text-[10px] text-tcip-muted">
          sweep of {String(sweep.input.n_trials)} trials · {String(sweep.input.search_alg)} ·{" "}
          {sweep.state}
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
        <div className="font-mono text-[11px]">{id}</div>
        <div className="text-[10px] text-tcip-muted flex justify-between">
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
              best {run.best_metric_name} {run.best_metric}
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
            aria-pressed={marked}
            aria-label={`Compare ${id}`}
            className={`px-2 py-1 text-[10px] transition-colors disabled:opacity-40 disabled:cursor-not-allowed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-tcip-accent/70 ${
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
      className="px-2 py-1 text-[10px] border-l border-tcip-border first:border-l-0 hover:bg-tcip-hover transition-colors disabled:opacity-40 disabled:cursor-not-allowed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-tcip-accent/70"
      onClick={onCancel}
    >
      {canceling ? "Canceling…" : "Cancel"}
    </button>
  );
}

function CancelError({ id, cancelError }: { id: string; cancelError: string | undefined }) {
  if (!cancelError) return null;
  return (
    <span id={`cancel-error-${id}`} className="text-[10px] text-tcip-fp text-right max-w-[150px]">
      {cancelError}
    </span>
  );
}

// Training is launched from a run or sweep already recorded in this project, or described fresh
// to the agent; this tab tracks the runs and sweeps those launches produce and their metrics.
export function TrainingTab() {
  const projectRoot = useStore(selectProjectRoot);
  const datasetRoot = useStore((s) => s.gui.dataset.dataset_root);
  const subject = useStore((s) => s.gui.dataset.subject);

  const { request, setRequest } = useEditableAgentRequest(
    defaultTrainingRequest(datasetRoot, subject),
  );

  const [pickerOpen, setPickerOpen] = useState(false);
  const [splitChoicesById, setSplitChoicesById] = useState<Record<string, SplitChoices>>({});
  const [splitChoicesLoadingId, setSplitChoicesLoadingId] = useState<string | null>(null);
  const [splitChoiceErrors, setSplitChoiceErrors] = useState<Record<string, string>>({});
  const [listing, setListing] = useState<TrainingListing>(NO_LISTING);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<MetricRow[]>([]);
  const [latestBatch, setLatestBatch] = useState<MetricRow | null>(null);
  // Non-null once the run's own TensorBoard refusal names no_logs; its error is the run's own
  // recorded status_error, null when the run produced no logs but recorded no reason either.
  const [tbNoLogs, setTbNoLogs] = useState<{ statusError: string | null } | null>(null);
  const [tbAttempt, setTbAttempt] = useState(0);
  const [markedExperimentIds, setMarkedExperimentIds] = useState<Set<string>>(new Set());
  // Cancel in flight, by run or sweep id: disables that row's own Cancel button with a pending
  // label, and a failure lands in cancelErrors rather than only a toast.
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

  // A run's own marked state, never "No run selected" left over: once the marked set settles
  // at exactly one, that run becomes the one the detail region shows.
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

  // Fetched only for the row the breeder actually opens: list_split_choices re-enumerates
  // every experiment in the project, so fetching it for every row on open would cost O(n^2).
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
    const result = await trainingApi.relaunch(source, useStore.getState().user, selectionDir);
    setPickerOpen(false);
    void refreshRuns();
    const started = result.experiment_id ?? result.sweep_id;
    if (typeof started === "string") setSelectedRun(started);
  }

  function sendToAgent() {
    useStore.getState().sendToAgentTerminal(request);
    setPickerOpen(false);
  }

  // The run list's own poll keeps a state per run independent of the metrics stream; a ref
  // (not a dependency) so reading it doesn't reopen the stream on every poll.
  const rowsRef = useRef<RunRow[]>(rows);
  useEffect(() => {
    rowsRef.current = rows;
  }, [rows]);

  const streaming = selectedRun !== null && !selectedSweep;
  useEffect(() => {
    // Comparing owns the detail region; a sweep has no metrics log of its own.
    if (!selectedRun || !streaming || !projectRoot || comparing) return;
    // The stream replays this run from the start, so a seed GET would just double-load the
    // same rows. The WS is the single source.
    setMetrics([]);
    setLatestBatch(null);
    streamRef.current?.();
    // A run already terminal when this stream opened is a rediscovery, not a transition the
    // breeder is watching; only a run still live at open time toasts on its own terminal frame.
    const knownAtOpen = rowsRef.current.find((r) => r.experiment_id === selectedRun)?.state;
    const alreadyTerminal = TERMINAL_STATES.has(knownAtOpen ?? "");
    // A run selected at its launch moment can be unknown to the backend for a few reconnects;
    // the toast names that once per selection, not once per silent retry.
    let errorToasted = false;
    streamRef.current = openTrainingStream(selectedRun, (msg) => {
      if (msg.type === "metric" && msg.row) {
        setMetrics((prev) => mergeMetric(prev, msg.row as MetricRow));
      } else if (msg.type === "batch" && msg.row) {
        setLatestBatch(msg.row as MetricRow);
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
        const st = msg.status?.state;
        if (typeof st === "string" && !alreadyTerminal)
          useStore.getState().pushToast(`Training ${selectedRun}: ${st}`, "info");
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

    let url = detail.tensorboard_url;
    let failure: string | null = null;
    let noLogs = false;
    if (!url) {
      try {
        const launched = await trainingApi.launchTensorboard(experimentId);
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
    // The route answers a detail carrying the run or the sweep its id names, never neither.
    const status = (detail.run ?? detail.sweep)!;
    if (!TERMINAL_STATES.has(status.state)) return { url: null, error: null, done: false };
    if (noLogs) {
      setTbNoLogs({ statusError: status.status_error });
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

  async function onCancel(id: string) {
    setPendingCancel((prev) => new Set(prev).add(id));
    setCancelErrors((prev) => {
      const { [id]: _drop, ...rest } = prev;
      return rest;
    });
    try {
      await trainingApi.cancel(id, useStore.getState().user);
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
  const selectedTerminal = TERMINAL_STATES.has(selectedRow?.state ?? "");

  const noLogsMessage = tbNoLogs
    ? tbNoLogs.statusError
      ? `This run failed: ${tbNoLogs.statusError}. It produced no logs.`
      : "This run produced no logs."
    : null;

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
              <h2 className="tcip-heading">{selectedSweep ? "Sweep" : "Metrics"}</h2>
              <span className="font-mono text-[12px] text-tcip-fg">{selectedRun}</span>
            </>
          ) : (
            <span className="tcip-heading">Select a run to view metrics</span>
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
                    <table className="w-full text-[11px]">
                      <caption className="sr-only">{`${selectedRun} metrics by epoch`}</caption>
                      <thead>
                        <tr className="border-b border-tcip-border">
                          <th className="tcip-th">epoch</th>
                          {metricKeys.map((key) => (
                            <th key={key} className="tcip-th">
                              {key}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {metrics.map((row, i) => (
                          <tr key={i} className="border-t border-tcip-border first:border-t-0">
                            <td className="py-1 pr-3 tabular-nums">
                              {typeof row[EPOCH_KEY] === "number" ? row[EPOCH_KEY] : UNSET_GLYPH}
                            </td>
                            {metricKeys.map((key) => (
                              <td key={key} className="pr-3 tabular-nums">
                                {typeof row[key] === "number" ? row[key] : UNSET_GLYPH}
                              </td>
                            ))}
                          </tr>
                        ))}
                        {batch && (
                          <tr className="border-t border-tcip-border text-tcip-muted">
                            <td className="py-1 pr-3 tabular-nums">
                              {typeof batch[EPOCH_KEY] === "number"
                                ? batch[EPOCH_KEY]
                                : UNSET_GLYPH}
                            </td>
                            <td
                              colSpan={Math.max(1, metricKeys.length)}
                              className="pr-3 tabular-nums"
                            >
                              {`${selectedTerminal ? "stopped at" : "in progress ·"} step ${String(batch[STEP_KEY])}`}
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
                  <div role="status" className="text-tcip-muted text-[12px]">
                    {selectedTerminal ? "This run recorded no metrics." : "Waiting for metrics…"}
                  </div>
                ))}
              {!selectedRun && (
                <div role="status" className="text-tcip-muted text-[12px]">
                  No run selected.
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
                title: "Runs and sweeps in this project",
                emptyMessage: "No run or sweep exists in this project yet.",
                error: runsError ?? undefined,
                onRetry: () => void refreshRuns(),
                rows: launchRows,
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
        {empty && !runsError && (
          <RunMonitorEmpty>No runs yet. Use &quot;Start a run&quot; above.</RunMonitorEmpty>
        )}
        {!empty && (
          <div className="text-[10px] text-tcip-muted mb-1">
            Every recorded run, sorted by experiment id, then every sweep with its trials.
          </div>
        )}
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
                    <div className="font-mono text-[11px]">{sweep.sweep_id}</div>
                    <div className="text-[10px] text-tcip-muted">
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
                  <div className="text-[10px] text-tcip-fp">{sweep.status_error}</div>
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

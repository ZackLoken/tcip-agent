import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { StructuredRefusalError } from "@/api/http";
import { tuningApi, type Sweep, type SweepDetail, type SweepTrial } from "@/api/tuning";
import type { TensorboardLaunch } from "@/api/training";
import { DisclosureChevron } from "@/components/CollapsibleSection";
import { EmbeddedTool } from "@/components/EmbeddedTool";
import { LaunchPicker } from "@/components/LaunchPicker";
import { TabHeading } from "@/components/TabHeading";
import { useEditableAgentRequest } from "@/hooks/useEditableAgentRequest";
import { useEmbeddedToolRetry, type EmbeddedToolStepResult } from "@/hooks/useEmbeddedToolRetry";
import { TERMINAL_STATUSES } from "@/lib/runStatus";
import { useStore } from "@/store";
import { defaultSweepRequest } from "@/tabs/agentPrompts";
import { RunMonitorEmpty, RunMonitorLayout } from "@/tabs/RunMonitorLayout";
import { runOrderLine, RUN_REFRESH_MS } from "@/tabs/trainingMetrics";

function hasContent(value: unknown): boolean {
  return typeof value === "object" && value !== null && Object.keys(value).length > 0;
}

/** A launch either serves a url straight away or says why it could not. */
function launchOutcome(launched: TensorboardLaunch): { url: string | null; error: string | null } {
  if (launched.url) return { url: launched.url, error: null };
  const error = launched.error
    ? launched.output
      ? `${launched.error}: ${launched.output}`
      : launched.error
    : "TensorBoard did not start.";
  return { url: null, error };
}

function messageOf(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function cellText(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return String(value);
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

/** A cancelled sweep whose record carries no reason: shown identically in the row and the
 * detail pane, from this one wording. */
const NO_CANCEL_REASON = "no reason recorded";

/** Shown while the selected sweep's own detail answers with ``has_manifest: false``: the
 * pre-manifest window before ``run_hpo`` writes a sweep's first manifest. The relaunch route
 * registers the job before it answers, so this is never a 404; it is the detail itself saying
 * the record is not written yet. A sweep refused before it ever wrote one is served by the
 * listing's own terminal status instead, and never reaches this. */
const SWEEP_NO_RECORD_YET = "This sweep's record is not written yet.";

/** The one sweep status Cancel is offered on, the same explicit-allowlist shape
 * TrainingTab's TRAINING_CANCELLABLE uses rather than inferring it from TERMINAL_STATUSES:
 * a sweep the backend derives as "interrupted" is done, not merely non-terminal. */
const SWEEP_CANCELLABLE: ReadonlySet<string> = new Set(["running"]);

export function TuningTab() {
  const datasetRoot = useStore((s) => s.gui.dataset.dataset_root);
  const { request, setRequest } = useEditableAgentRequest(defaultSweepRequest(datasetRoot));

  const [pickerOpen, setPickerOpen] = useState(false);
  const [sweeps, setSweeps] = useState<Sweep[]>([]);
  const [sweepsError, setSweepsError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<SweepDetail | null>(null);
  // A failure reading the selected sweep's own detail: shown, not swallowed, so a retry in
  // progress is visible instead of a silent stall on whatever was last on screen.
  const [detailError, setDetailError] = useState<string | null>(null);
  const [trials, setTrials] = useState<SweepTrial[]>([]);
  const [selectedTrialId, setSelectedTrialId] = useState<string | null>(null);
  const [trialMetrics, setTrialMetrics] = useState<Record<string, unknown>[]>([]);
  const [rayUrl, setRayUrl] = useState<string | null>(null);
  const [rayError, setRayError] = useState<string | null>(null);
  const [rayAttempt, setRayAttempt] = useState(0);
  const [sweepTbAttempt, setSweepTbAttempt] = useState(0);
  const [trialTb, setTrialTb] = useState<{ url: string | null; error: string | null }>({
    url: null,
    error: null,
  });
  const [trialTbAttempt, setTrialTbAttempt] = useState(0);
  // Cancel/Run again in flight, by sweep id: disables that row's own button with a pending
  // label. A failure lands in actionErrors too, not only a toast.
  const [pendingActions, setPendingActions] = useState<
    Readonly<Record<string, "cancel" | "relaunch">>
  >({});
  const [actionErrors, setActionErrors] = useState<Record<string, string>>({});

  // Read fresh inside the interval tick below, so one function can refresh whichever sweep is
  // selected without recreating the interval on every selection.
  const selectedIdRef = useRef<string | null>(null);
  useEffect(() => {
    selectedIdRef.current = selectedId;
  }, [selectedId]);

  const refreshSweeps = useCallback(async () => {
    try {
      const r = await tuningApi.listSweeps();
      setSweeps(r.sweeps ?? []);
      setSweepsError(null);
    } catch (e) {
      setSweepsError(`Could not load sweeps: ${messageOf(e)}`);
    }
  }, []);

  // The selected sweep's own detail and its trials, read for the expanded sweep only: a
  // listing per sweep per tick would walk the whole HPO root every few seconds.
  const refreshDetail = useCallback(async (id: string) => {
    try {
      const d = await tuningApi.getSweep(id);
      if (selectedIdRef.current !== id) return;
      setDetail(d);
      setDetailError(null);
      try {
        const t = await tuningApi.listTrials(id);
        if (selectedIdRef.current === id) setTrials(t.trials ?? []);
      } catch {
        // A sweep launched here has no trial directory until its first trial writes one.
        if (selectedIdRef.current === id) setTrials([]);
      }
    } catch (e) {
      if (selectedIdRef.current === id) setDetailError(messageOf(e));
    }
  }, []);

  // One poll drives the listing and the selected sweep's own detail together, on the same
  // tick, so the row and the detail never read two different moments of the same sweep.
  const refresh = useCallback(async () => {
    await refreshSweeps();
    if (selectedIdRef.current) await refreshDetail(selectedIdRef.current);
  }, [refreshSweeps, refreshDetail]);

  useEffect(() => {
    void refresh();
    const t = setInterval(refresh, RUN_REFRESH_MS);
    return () => clearInterval(t);
  }, [refresh]);

  // On a fresh selection, drop the previous sweep's detail/trials/error and fetch this one
  // right away rather than waiting for the shared interval's next tick.
  useEffect(() => {
    setDetail(null);
    setDetailError(null);
    setTrials([]);
    if (!selectedId) return;
    void refreshDetail(selectedId);
  }, [selectedId, refreshDetail]);

  useEffect(() => {
    if (!selectedId || !selectedTrialId) {
      setTrialMetrics([]);
      return;
    }
    let cancelled = false;
    void tuningApi.getTrialMetrics(selectedId, selectedTrialId).then(
      (r) => {
        if (!cancelled) setTrialMetrics(r.metrics ?? []);
      },
      () => {
        if (!cancelled) setTrialMetrics([]);
      },
    );
    return () => {
      cancelled = true;
    };
  }, [selectedId, selectedTrialId]);

  // Ray holds one cluster per process and tears it down with the sweep that started it, so the
  // dashboard is read fresh whenever this panel comes up rather than cached across selections.
  useEffect(() => {
    setRayUrl(null);
    setRayError(null);
    if (!selectedId) return;
    let cancelled = false;
    void tuningApi.getRayDashboard().then(
      (r) => {
        if (cancelled) return;
        setRayUrl(r.url);
        setRayError(r.url ? null : "Ray's dashboard isn't running right now.");
      },
      (e) => {
        if (!cancelled) setRayError(messageOf(e));
      },
    );
    return () => {
      cancelled = true;
    };
  }, [selectedId, rayAttempt]);

  // Both three-valued: null while the detail hasn't answered for this selection yet, else the
  // detail's own reading. The trial list below keys its own text on hasManifest the same way.
  const detailForSelection = detail && detail.sweep_id === selectedId ? detail : null;
  const hasManifest = detailForSelection ? detailForSelection.has_manifest : null;
  const sweepNonTerminal =
    detailForSelection === null || !TERMINAL_STATUSES.has(detailForSelection.status);

  // One TensorBoard over the whole sweep directory: this route's own 404 is how the
  // pre-manifest window presents here, so it is retried silently, never shown as a failure.
  const sweepTbStep = useCallback(async (): Promise<EmbeddedToolStepResult> => {
    if (!selectedId) return { url: null, error: null, done: true };
    try {
      const launched = await tuningApi.launchSweepTensorboard(selectedId);
      const outcome = launchOutcome(launched);
      return { ...outcome, done: outcome.url != null || !sweepNonTerminal };
    } catch (e) {
      if (e instanceof StructuredRefusalError && e.status === 404) {
        return { url: null, error: null, done: !sweepNonTerminal };
      }
      return { url: null, error: messageOf(e), done: !sweepNonTerminal };
    }
  }, [selectedId, sweepNonTerminal]);

  const sweepTb = useEmbeddedToolRetry(selectedId, !!selectedId, sweepTbAttempt, sweepTbStep);

  // Every trial's TensorBoard costs a port out of a bounded range, so moving off a trial stops
  // the one it was showing.
  useEffect(() => {
    setTrialTb({ url: null, error: null });
    if (!selectedId || !selectedTrialId) return;
    const sweepId = selectedId;
    const trialId = selectedTrialId;
    let cancelled = false;
    void tuningApi.launchTrialTensorboard(sweepId, trialId).then(
      (launched) => {
        if (!cancelled) setTrialTb(launchOutcome(launched));
      },
      (e) => {
        if (!cancelled) setTrialTb({ url: null, error: messageOf(e) });
      },
    );
    return () => {
      cancelled = true;
      void tuningApi.stopTrialTensorboard(sweepId, trialId).catch(() => {
        /* the trial's TensorBoard is already gone */
      });
    };
  }, [selectedId, selectedTrialId, trialTbAttempt]);

  const metricColumns = useMemo(() => {
    const keys: string[] = [];
    trialMetrics.forEach((row) => {
      Object.keys(row).forEach((k) => {
        if (!keys.includes(k)) keys.push(k);
      });
    });
    return keys;
  }, [trialMetrics]);

  const selectedTrial = trials.find((t) => t.trial_id === selectedTrialId) ?? null;

  function toggleSweep(sweepId: string) {
    setSelectedTrialId(null);
    setSelectedId((current) => (current === sweepId ? null : sweepId));
  }

  async function onCancelSweep(sweepId: string) {
    setPendingActions((prev) => ({ ...prev, [sweepId]: "cancel" }));
    setActionErrors((prev) => {
      const { [sweepId]: _drop, ...rest } = prev;
      return rest;
    });
    try {
      await tuningApi.cancel(sweepId);
      void refresh();
    } catch (e) {
      const message = `Cancel failed: ${messageOf(e)}`;
      useStore.getState().pushToast(message);
      setActionErrors((prev) => ({ ...prev, [sweepId]: message }));
    } finally {
      setPendingActions((prev) => {
        const { [sweepId]: _drop, ...rest } = prev;
        return rest;
      });
    }
  }

  async function onRelaunchSweep(sweepId: string) {
    setPendingActions((prev) => ({ ...prev, [sweepId]: "relaunch" }));
    setActionErrors((prev) => {
      const { [sweepId]: _drop, ...rest } = prev;
      return rest;
    });
    try {
      const result = await tuningApi.relaunch(sweepId);
      void refresh();
      if (typeof result.sweep_id === "string") setSelectedId(result.sweep_id);
    } catch (e) {
      const message = `Relaunch failed: ${messageOf(e)}`;
      useStore.getState().pushToast(message);
      setActionErrors((prev) => ({ ...prev, [sweepId]: message }));
    } finally {
      setPendingActions((prev) => {
        const { [sweepId]: _drop, ...rest } = prev;
        return rest;
      });
    }
  }

  function sendToAgent() {
    useStore.getState().sendToAgentTerminal(request);
    setPickerOpen(false);
  }

  // Cancel while running, gone once "stop requested" except for an external sweep; "Run again"
  // once terminal and relaunchable. Either disables with a pending label while in flight.
  function sweepAction(s: Sweep) {
    const pending = pendingActions[s.sweep_id];
    const actionError = actionErrors[s.sweep_id];
    const describedBy = actionError ? `sweep-action-error-${s.sweep_id}` : undefined;
    if (SWEEP_CANCELLABLE.has(s.status)) {
      if (s.cancel_requested && !s.external) return null;
      return (
        <button
          type="button"
          aria-label={`Cancel ${s.sweep_id}`}
          aria-describedby={describedBy}
          disabled={pending === "cancel"}
          className="tcip-btn text-[10px] shrink-0 mt-2 disabled:opacity-60"
          onClick={(e) => {
            e.stopPropagation();
            void onCancelSweep(s.sweep_id);
          }}
        >
          {pending === "cancel" ? "Cancelling…" : "Cancel"}
        </button>
      );
    }
    if (s.relaunchable) {
      return (
        <button
          type="button"
          aria-label={`Run again ${s.sweep_id}`}
          aria-describedby={describedBy}
          disabled={pending === "relaunch"}
          className="tcip-btn text-[10px] shrink-0 mt-2 disabled:opacity-60"
          onClick={(e) => {
            e.stopPropagation();
            void onRelaunchSweep(s.sweep_id);
          }}
        >
          {pending === "relaunch" ? "Starting…" : "Run again"}
        </button>
      );
    }
    return null;
  }

  return (
    <>
      <TabHeading tab="tuning" />
      <RunMonitorLayout
        title="Sweeps"
        headerRight={
          <button
            type="button"
            aria-expanded={pickerOpen}
            className={pickerOpen ? "tcip-btn text-[11px]" : "tcip-btn-primary text-[11px]"}
            onClick={() => setPickerOpen((open) => !open)}
          >
            Start a sweep
          </button>
        }
        detailHeader={
          selectedTrial ? (
            <>
              <span className="tcip-heading">Trial</span>
              <span className="font-mono text-[12px] text-tcip-fg">{selectedTrial.trial_id}</span>
              <span className="text-[11px] text-tcip-muted">of {detail?.sweep_id}</span>
            </>
          ) : detail ? (
            <>
              <span className="tcip-heading">Sweep</span>
              <span className="font-mono text-[12px] text-tcip-fg">{detail.sweep_id}</span>
              <span className="text-[11px] text-tcip-muted">({detail.status})</span>
            </>
          ) : selectedId ? (
            <>
              <span className="tcip-heading">Sweep</span>
              <span className="font-mono text-[12px] text-tcip-fg">{selectedId}</span>
            </>
          ) : (
            <span className="tcip-heading">Select a sweep</span>
          )
        }
        detail={
          selectedTrial ? (
            <div className="flex flex-col gap-4">
              <div className="h-[52vh] min-h-[320px] shrink-0">
                <EmbeddedTool
                  title="Trial TensorBoard"
                  url={trialTb.url}
                  loading={!trialTb.url && !trialTb.error}
                  error={trialTb.error}
                  onRetry={() => setTrialTbAttempt((n) => n + 1)}
                />
              </div>
              <div>
                <div className="tcip-heading mb-1">Parameters</div>
                {Object.keys(selectedTrial.params).length === 0 ? (
                  <div className="text-[11px] text-tcip-muted">
                    This trial has not written a resolved config yet.
                  </div>
                ) : (
                  <pre className="text-[11px] font-mono p-3 tcip-panel overflow-auto">
                    {JSON.stringify(selectedTrial.params, null, 2)}
                  </pre>
                )}
                {selectedTrial.unconsumed_params.length > 0 && (
                  <div className="mt-1 text-[11px] text-tcip-fp">
                    Swept but not read by the training config:{" "}
                    {selectedTrial.unconsumed_params.join(", ")}
                  </div>
                )}
              </div>
              <div>
                <div className="tcip-heading mb-1">Metrics</div>
                {trialMetrics.length === 0 ? (
                  <div className="text-[11px] text-tcip-muted">
                    No metrics rows from this trial yet.
                  </div>
                ) : (
                  <div className="tcip-panel overflow-auto">
                    <table className="w-full text-[11px] font-mono">
                      <thead>
                        <tr className="text-tcip-muted">
                          {metricColumns.map((k) => (
                            <th key={k} className="px-2 py-1 text-left font-normal">
                              {k}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {trialMetrics.map((row, i) => (
                          <tr key={i} className="border-t border-tcip-border">
                            {metricColumns.map((k) => (
                              <td key={k} className="px-2 py-1 tabular-nums">
                                {cellText(row[k])}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          ) : detail ? (
            <div className="flex flex-col gap-3">
              {detailError && (
                <div className="text-[11px] text-tcip-fp">
                  Could not refresh this sweep's record: {detailError}
                </div>
              )}
              {!detail.has_manifest ? (
                <div role="status" aria-live="polite" className="text-[11px] text-tcip-muted">
                  {SWEEP_NO_RECORD_YET}
                </div>
              ) : detail.status === "cancelled" ? (
                <div className="text-[11px] text-tcip-muted">
                  Cancelled: {detail.error ?? NO_CANCEL_REASON}
                </div>
              ) : hasContent(detail.result) ? (
                <>
                  {detail.error ? (
                    <div className="text-[11px] text-tcip-fp">{detail.error}</div>
                  ) : null}
                  <pre className="max-h-[24vh] text-[11px] font-mono p-3 tcip-panel overflow-auto">
                    {JSON.stringify(detail.result, null, 2)}
                  </pre>
                </>
              ) : !TERMINAL_STATUSES.has(detail.status) ? (
                <div className="text-[11px] text-tcip-muted">
                  The best config appears here once the sweep finishes. Pick one of its trials to
                  follow that trial while it runs.
                </div>
              ) : (
                <div className="text-[11px] text-tcip-muted">
                  No result recorded; the sweep {detail.status}
                  {detail.error ? `: ${detail.error}` : "."}
                </div>
              )}
              <div className="h-[46vh] min-h-[300px] shrink-0">
                <EmbeddedTool
                  title="Ray dashboard"
                  url={rayUrl}
                  loading={!rayUrl && !rayError}
                  error={rayError}
                  onRetry={() => setRayAttempt((n) => n + 1)}
                />
              </div>
              <div className="h-[38vh] min-h-[260px] shrink-0">
                <EmbeddedTool
                  title="Sweep TensorBoard"
                  url={sweepTb.url}
                  loading={!sweepTb.url && !sweepTb.error && sweepNonTerminal}
                  error={sweepTb.error}
                  onRetry={() => setSweepTbAttempt((n) => n + 1)}
                />
              </div>
            </div>
          ) : detailError ? (
            <div className="text-[11px] text-tcip-fp">{detailError}</div>
          ) : selectedId ? (
            <div className="text-[11px] text-tcip-muted">Reading this sweep's record…</div>
          ) : (
            <div className="text-[11px] text-tcip-muted">
              Select a sweep to see its trials and its result.
            </div>
          )
        }
      >
        {pickerOpen && (
          <div className="mb-3 pb-3 border-b border-tcip-border">
            <LaunchPicker
              composerLabel="Describe a new one to the agent"
              request={request}
              onRequestChange={setRequest}
              onSend={sendToAgent}
            />
          </div>
        )}

        {sweepsError && (
          <div className="text-[11px] text-tcip-fp mb-2">
            {sweepsError}{" "}
            <button className="tcip-btn text-[11px] ml-1" onClick={() => void refresh()}>
              Retry
            </button>
          </div>
        )}
        {sweeps.length === 0 && !sweepsError && (
          <RunMonitorEmpty>No sweeps yet. Use "Start a sweep" above.</RunMonitorEmpty>
        )}
        {sweeps.length > 0 && (
          <div className="text-[10px] text-tcip-muted mb-1">
            {runOrderLine("sweep", "sweep id")}
          </div>
        )}
        {sweeps.length > 0 && (
          <ul className="space-y-1">
            {sweeps.map((s) => {
              const expanded = selectedId === s.sweep_id;
              const running = SWEEP_CANCELLABLE.has(s.status);
              const searchLine = [
                s.n_trials != null
                  ? `${s.n_trials} trial${s.n_trials === 1 ? "" : "s"} planned${
                      s.split_draws != null && s.split_draws > 1
                        ? `, ${s.split_draws} draws each`
                        : ""
                    }`
                  : null,
                s.search_alg ? `search ${s.search_alg}` : null,
                s.scheduler
                  ? s.scheduler === "none"
                    ? "no scheduler"
                    : `scheduler ${s.scheduler}`
                  : null,
                s.param_space_keys && s.param_space_keys.length > 0
                  ? `axes ${s.param_space_keys.join(", ")}`
                  : null,
              ]
                .filter((part): part is string => !!part)
                .join(" · ");
              const describedBy = [
                s.relaunched_from ? `sweep-relaunch-${s.sweep_id}` : null,
                s.error ? `sweep-error-${s.sweep_id}` : null,
              ]
                .filter((part): part is string => !!part)
                .join(" ");
              return (
                <li key={s.sweep_id}>
                  <div
                    className={`flex items-start gap-1 p-2 rounded border transition-colors ${
                      expanded && !selectedTrialId
                        ? "border-tcip-accent bg-tcip-accent/10"
                        : "border-tcip-border hover:border-tcip-border-hover hover:bg-tcip-hover"
                    }`}
                  >
                    <button
                      type="button"
                      aria-expanded={expanded}
                      aria-label={`${s.sweep_id} ${s.status}${
                        running && s.cancel_requested ? ", stop requested" : ""
                      }`}
                      aria-describedby={describedBy || undefined}
                      className="flex-1 flex items-start gap-2 text-left"
                      onClick={() => toggleSweep(s.sweep_id)}
                    >
                      <span className="mt-[3px] text-tcip-muted">
                        <DisclosureChevron open={expanded} />
                      </span>
                      <span className="flex-1">
                        <span className="block font-mono text-[11px]">{s.sweep_id}</span>
                        <span className="block text-[10px] text-tcip-muted">
                          {s.status}
                          {running && s.cancel_requested ? " · stop requested" : ""}
                        </span>
                        {s.relaunched_from && (
                          <span
                            id={`sweep-relaunch-${s.sweep_id}`}
                            className="block text-[10px] text-tcip-muted"
                          >
                            relaunched from {s.relaunched_from}
                          </span>
                        )}
                        {s.error ? (
                          <span
                            id={`sweep-error-${s.sweep_id}`}
                            className="block text-[10px] text-tcip-fp"
                          >
                            {s.error}
                          </span>
                        ) : (
                          s.status === "cancelled" && (
                            <span className="block text-[10px] text-tcip-muted">
                              {NO_CANCEL_REASON}
                            </span>
                          )
                        )}
                        {!s.relaunchable && s.reason && (
                          <span className="block text-[10px] text-tcip-muted">{s.reason}</span>
                        )}
                      </span>
                    </button>
                    {sweepAction(s)}
                  </div>
                  {actionErrors[s.sweep_id] && (
                    <div
                      id={`sweep-action-error-${s.sweep_id}`}
                      className="mt-1 ml-5 text-[10px] text-tcip-fp"
                    >
                      {actionErrors[s.sweep_id]}
                    </div>
                  )}
                  {expanded && (
                    <div className="mt-1 ml-5">
                      {searchLine && (
                        <div className="text-[10px] text-tcip-muted mb-1">{searchLine}</div>
                      )}
                      <ul className="space-y-1">
                        {trials.length === 0 ? (
                          // hasManifest === false: the detail pane already carries
                          // SWEEP_NO_RECORD_YET; nothing renders here so it appears once.
                          hasManifest === true ? (
                            <li className="text-[10px] text-tcip-muted">No trials on disk yet.</li>
                          ) : hasManifest === null ? (
                            <li className="text-[10px] text-tcip-muted">
                              Reading this sweep's record…
                            </li>
                          ) : null
                        ) : (
                          trials.map((t) => (
                            <li key={t.trial_id}>
                              <button
                                type="button"
                                aria-pressed={selectedTrialId === t.trial_id}
                                className={`w-full p-2 rounded border text-left transition-colors ${
                                  selectedTrialId === t.trial_id
                                    ? "border-tcip-accent bg-tcip-accent/10"
                                    : "border-tcip-border hover:border-tcip-border-hover hover:bg-tcip-hover"
                                }`}
                                onClick={() => setSelectedTrialId(t.trial_id)}
                              >
                                <span className="block font-mono text-[11px]">{t.trial_id}</span>
                                <span className="block text-[10px] text-tcip-muted">
                                  {t.has_metrics ? "metrics recorded" : "no metrics yet"}
                                  {Object.keys(t.params).length > 0 && (
                                    <>
                                      {" · "}
                                      {Object.entries(t.params).map(([k, v], i) => (
                                        <span key={k}>
                                          {i > 0 ? ", " : ""}
                                          {k}={cellText(v)}
                                        </span>
                                      ))}
                                    </>
                                  )}
                                </span>
                              </button>
                            </li>
                          ))
                        )}
                      </ul>
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </RunMonitorLayout>
    </>
  );
}
